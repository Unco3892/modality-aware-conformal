"""SEMF on SRED — full experiment: train + multi-regime missing-modality eval.

Pipeline:
  1. Load SRED tabular + cached embeddings; PCA-reduce text/image.
  2. Train original SEMF once on the full feature matrix.
  3. For each of {full, no_text, no_img, tab_only}, mask the corresponding
     feature blocks at inference time, get MC samples on valid + test.
  4. Apply CQR, density-conformal, Mondrian-CQR (strata = modality regime).
  5. Save trained SEMF + raw samples + all evaluation tables.

Run:
  python src/sred/run_full_experiment.py --R 10 --R-infer 50 --max-it 20
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "sred"))

from semf.preprocessing import DataPreprocessor  # noqa: E402
from semf.semf import SEMF  # noqa: E402
from multimodal_calibration.reproducibility import (  # noqa: E402
    attach_run_provenance,
    atomic_write_binary,
    atomic_write_json,
    thread_identity,
    validate_campaign_tag,
    write_campaign_manifest,
)
from multimodal_calibration.experiment_config import (  # noqa: E402
    DATASET_MODALITY_FIELDS,
    PAPER_CPU_DEVICE,
    PAPER_SEEDS,
    SEMF_IMPLEMENTATION_POLICY,
    SEMF_PAPER_RUN_CONFIG,
    SRED_TABULAR_COLUMNS,
    require_canonical_run_config,
)

from conformal import (  # noqa: E402
    conformal_quantile,
    cqr,
    density_conformal,
    DensityConformalConfig,
    evaluate as eval_intervals,
    mondrian_cqr,
)
# tuned_models monkey-patch is no longer needed: MultiXGBs in the semf package
# now natively supports xgb_learning_rate / subsample / colsample_bytree.


# ---------------------------------------------------------------------------
# data helpers


SRED = ROOT / "data" / "sred"
META = SRED / "metadata"
CACHE = SRED / "embeddings"
RESULTS = ROOT / "results" / "sred_semf"
SEMF_PAPER_TEXT_SLUG = SEMF_PAPER_RUN_CONFIG["features"]["text_slug"]
SEMF_PAPER_IMAGE_SLUG = SEMF_PAPER_RUN_CONFIG["features"]["image_slug"]


def _optional_int(value: str) -> int | None:
    return None if value.lower() in {"none", "null"} else int(value)


def semf_run_config(args: argparse.Namespace) -> dict:
    return {
        "experiment": "sred_semf_v1",
        "alpha": float(args.alpha),
        "features": {
            "text_pca": int(args.text_pca),
            "image_pca": int(args.image_pca),
            "text_slug": args.text_slug,
            "image_slug": args.image_slug,
            "image_types": list(args.image_types),
        },
        "semf": {
            "R": int(args.R),
            "R_infer": int(args.R_infer),
            "max_it": int(args.max_it),
            "nodes_per_feature": int(args.nodes_per_feature),
            "x_group_size": int(args.x_group_size),
            "z_norm_sd": str(args.z_norm_sd),
            "model_class": args.model_class,
            "weight_alignment": args.weight_alignment,
            "outer_patience": int(args.patience),
            "allow_partial_recovery": bool(args.allow_partial_recovery),
        },
        "tree": {
            "n_estimators": int(args.tree_n_estimators),
            "max_depth": args.xgb_max_depth,
            "patience": int(args.xgb_patience),
            "learning_rate": float(args.xgb_learning_rate),
            "subsample": float(args.xgb_subsample),
            "colsample_bytree": float(args.xgb_colsample),
        },
        "regimes": list(args.regimes),
        "data_valid_fraction": SEMF_PAPER_RUN_CONFIG[
            "data_valid_fraction"
        ],
        "models_valid_fraction": SEMF_PAPER_RUN_CONFIG[
            "models_valid_fraction"
        ],
    }


def validate_paper_cache_contract(text_slug: str, image_slug: str) -> None:
    """Enforce the SEMF-specific 36-D cache contract.

    This is intentionally different from the cross-dataset SRED experiments,
    which require e5-large and DINOv2-B/14.
    """
    if (
        text_slug != SEMF_PAPER_TEXT_SLUG
        or image_slug != SEMF_PAPER_IMAGE_SLUG
    ):
        raise ValueError(
            "SEMF paper runs require "
            f"{SEMF_PAPER_TEXT_SLUG!r} and {SEMF_PAPER_IMAGE_SLUG!r}; "
            "e5-large/ViT-B14 is the separate cross-dataset cache contract"
        )


def load_sred() -> tuple[pd.DataFrame, pd.DataFrame]:
    tr = pd.read_csv(META / "train_data_with_text.csv", encoding="latin-1")
    te = pd.read_csv(META / "test_data_with_text.csv", encoding="latin-1")
    return tr, te


def load_cached(split: str, kind: str, slug: str) -> np.ndarray:
    p = CACHE / f"{split}_{kind}_{slug}.npy"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found — run baseline.py first")
    return np.load(p)


def pca_block(tr: np.ndarray, te: np.ndarray, n: int, prefix: str, seed: int):
    n = min(n, tr.shape[1])
    pca = PCA(n_components=n, random_state=seed)
    Z_tr = pca.fit_transform(tr).astype(np.float32)
    Z_te = pca.transform(te).astype(np.float32)
    cols = [f"{prefix}_{i}" for i in range(n)]
    return pd.DataFrame(Z_tr, columns=cols), pd.DataFrame(Z_te, columns=cols)


# ---------------------------------------------------------------------------
# artifact naming


def _stem(base: str, tag: str) -> str:
    """Build an artifact filename stem, inserting `tag` only when set."""
    return f"{base}_{tag}" if tag else base


# ---------------------------------------------------------------------------
# masking


def mask_features(df: pd.DataFrame, mask_blocks: list[str], block_cols: dict[str, list[str]]) -> pd.DataFrame:
    """Zero out (in already-standardised feature space, mean=0) the listed blocks."""
    out = df.copy()
    for b in mask_blocks:
        for c in block_cols[b]:
            out[c] = 0.0
    return out


# ---------------------------------------------------------------------------
# main


def main():
    ap = argparse.ArgumentParser()
    features = SEMF_PAPER_RUN_CONFIG["features"]
    semf_config = SEMF_PAPER_RUN_CONFIG["semf"]
    tree_config = SEMF_PAPER_RUN_CONFIG["tree"]
    ap.add_argument("--text-pca", type=int, default=features["text_pca"])
    ap.add_argument("--image-pca", type=int, default=features["image_pca"])
    ap.add_argument("--text-slug", default=features["text_slug"],
                    help="text-encoder slug used to locate cached embeddings; matches baseline.py output")
    ap.add_argument("--image-slug", default=features["image_slug"],
                    help="image-encoder slug used to locate cached embeddings; matches baseline.py output")
    ap.add_argument("--image-types", nargs="+", default=list(features["image_types"]))
    ap.add_argument("--R", type=int, default=semf_config["R"])
    ap.add_argument("--R-infer", type=int, default=semf_config["R_infer"])
    ap.add_argument("--max-it", type=int, default=semf_config["max_it"])
    ap.add_argument("--nodes-per-feature", type=int, default=semf_config["nodes_per_feature"],
                    help="latent dim per group")
    ap.add_argument("--x-group-size", type=int, default=semf_config["x_group_size"],
                    help="features per phi-head group; set >1 to reduce # phi-heads (big speedup)")
    ap.add_argument("--z-norm-sd", default=semf_config["z_norm_sd"],
                    help="fixed scalar (e.g. '0.1') or 'train_residual_models'")
    ap.add_argument("--alpha", type=float, default=SEMF_PAPER_RUN_CONFIG["alpha"])
    ap.add_argument("--seed", type=int, default=PAPER_SEEDS[0])
    ap.add_argument("--device", default=PAPER_CPU_DEVICE)
    ap.add_argument("--tag", default="",
                    help="optional artifact tag to distinguish experimental runs; "
                         "when set, outputs are named semf_<tag>_seed<n>.* instead "
                         "of the default semf_seed<n>.*. predictor_agnostic_mondrian.py "
                         "and testsplit_semf_calibration.py must be given a matching "
                         "--semf-tag to read these files")
    ap.add_argument("--results-dir", type=Path, default=RESULTS)
    ap.add_argument(
        "--paper-run",
        action="store_true",
        help="require a tag, exact paper cache slugs, and campaign provenance",
    )
    ap.add_argument(
        "--campaign-seeds",
        nargs="+",
        type=int,
        default=list(PAPER_SEEDS),
        help="complete seed grid represented by the shared campaign directory",
    )
    ap.add_argument(
        "--allow-partial-recovery",
        action="store_true",
        help="diagnostic only: return an explicitly incomplete SEMF after an error",
    )
    ap.add_argument("--regimes", nargs="+",
                    default=list(SEMF_PAPER_RUN_CONFIG["regimes"]))
    ap.add_argument("--model-class", default=semf_config["model_class"],
                    choices=["MultiXGBs", "MultiETs", "MultiMLPs"])
    ap.add_argument(
        "--weight-alignment",
        choices=["legacy", "aligned"],
        default=semf_config["weight_alignment"],
        help=(
            "SEMF replication-weight layout. 'aligned' pairs replicated row "
            "r*n+i with w_R[i,r]; 'legacy' reproduces historical artifacts."
        ),
    )
    ap.add_argument("--patience", type=int, default=semf_config["outer_patience"],
                    help="SEMF early-stopping patience")
    ap.add_argument("--tree-n-estimators", type=int,
                    default=tree_config["n_estimators"])
    ap.add_argument("--xgb-max-depth", type=_optional_int,
                    default=tree_config["max_depth"])
    ap.add_argument("--xgb-patience", type=int, default=tree_config["patience"])
    ap.add_argument("--xgb-learning-rate", type=float,
                    default=tree_config["learning_rate"])
    ap.add_argument("--xgb-subsample", type=float,
                    default=tree_config["subsample"])
    ap.add_argument("--xgb-colsample", type=float,
                    default=tree_config["colsample_bytree"])
    args = ap.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.tag = validate_campaign_tag(args.tag, required=args.paper_run) or ""
    run_config = semf_run_config(args)
    if args.paper_run:
        try:
            validate_paper_cache_contract(args.text_slug, args.image_slug)
        except ValueError as error:
            ap.error(str(error))
        if tuple(sorted(set(args.campaign_seeds))) != PAPER_SEEDS:
            ap.error(
                "--paper-run requires canonical --campaign-seeds "
                + " ".join(str(seed) for seed in PAPER_SEEDS)
            )
        if args.seed not in args.campaign_seeds:
            ap.error("--seed must belong to --campaign-seeds")
        if args.device != PAPER_CPU_DEVICE:
            ap.error("--paper-run requires --device cpu")
        try:
            require_canonical_run_config("semf", run_config)
        except ValueError as error:
            ap.error(str(error))
        write_campaign_manifest(
            args.results_dir,
            ROOT,
            campaign_tag=args.tag,
            datasets=("sred",),
            seeds=args.campaign_seeds,
            methods=("semf",),
            requested_device=args.device,
            run_config=run_config,
            producer_threads=thread_identity(),
        )

    print("=" * 78)
    print("SEMF SRED full experiment — config:")
    print(json.dumps(vars(args), indent=2, default=str))
    print("=" * 78)

    # ---- features ------------------------------------------------------
    tr_meta, te_meta = load_sred()
    tab_cols = list(SRED_TABULAR_COLUMNS)
    tab_tr = tr_meta[tab_cols].reset_index(drop=True).astype(np.float32)
    tab_te = te_meta[tab_cols].reset_index(drop=True).astype(np.float32)
    y_tr_raw = np.log(tr_meta["price"].to_numpy(dtype=np.float64))
    y_te_raw = np.log(te_meta["price"].to_numpy(dtype=np.float64))

    text_slug = args.text_slug
    img_slug = args.image_slug

    txt_parts_tr, txt_parts_te = [], []
    text_fields = DATASET_MODALITY_FIELDS["sred"]["text"]
    for f in text_fields:
        ptr, pte = pca_block(load_cached("train", f, text_slug),
                             load_cached("test", f, text_slug),
                             args.text_pca, prefix=f"txt_{f}", seed=args.seed)
        txt_parts_tr.append(ptr)
        txt_parts_te.append(pte)

    img_parts_tr, img_parts_te = [], []
    for k in args.image_types:
        ptr, pte = pca_block(load_cached("train", k, img_slug),
                             load_cached("test", k, img_slug),
                             args.image_pca, prefix=f"img_{k}", seed=args.seed)
        img_parts_tr.append(ptr)
        img_parts_te.append(pte)

    X_tr = pd.concat([tab_tr] + txt_parts_tr + img_parts_tr, axis=1).reset_index(drop=True)
    X_te = pd.concat([tab_te] + txt_parts_te + img_parts_te, axis=1).reset_index(drop=True)

    # block -> column-name lists (for masking)
    block_cols = {"tab": tab_cols}
    block_cols.update({f"txt_{f}": txt_parts_tr[i].columns.tolist()
                       for i, f in enumerate(text_fields)})
    block_cols.update({f"img_{k}": img_parts_tr[i].columns.tolist()
                       for i, k in enumerate(args.image_types)})

    # regime -> blocks to *mask*
    text_blocks = [f"txt_{field}" for field in text_fields]
    image_blocks = [f"img_{kind}" for kind in args.image_types]
    regime_blocks = {
        "full": [],
        "no_text": text_blocks,
        "no_img": image_blocks,
        "tab_only": text_blocks + image_blocks,
    }
    print("regime -> masked blocks:", regime_blocks)
    print("X shape:", X_tr.shape, X_te.shape)

    # ---- preprocess ----------------------------------------------------
    df_tr_full = X_tr.copy(); df_tr_full["log_price"] = y_tr_raw
    df_te_full = X_te.copy(); df_te_full["log_price"] = y_te_raw

    pre = DataPreprocessor(
        df=df_tr_full,
        y_col="log_price",
        train_size=SEMF_IMPLEMENTATION_POLICY[
            "preprocessing_train_fraction"
        ],
        valid_size=SEMF_PAPER_RUN_CONFIG["data_valid_fraction"],
        seed=args.seed,
    )
    pre.split_data(df_test=df_te_full)
    pre.scale_data(scale_output=SEMF_IMPLEMENTATION_POLICY["scale_output"])
    df_train, df_valid, df_test = pre.get_train_valid_test()
    X_train, y_train = pre.split_X_y(df_train)
    X_valid, y_valid = pre.split_X_y(df_valid)
    X_test, y_test = pre.split_X_y(df_test)
    print(f"sizes — train={len(df_train)} valid={len(df_valid)} test={len(df_test)}")

    # ---- train SEMF ----------------------------------------------------
    n_feat = X_train.shape[1]
    n_groups = int(np.ceil(n_feat / args.x_group_size))
    nodes_per_feature = np.array([args.nodes_per_feature] * n_groups)

    # Parse z_norm_sd: try float, else string.
    try:
        z_norm_sd = float(args.z_norm_sd)
    except ValueError:
        z_norm_sd = args.z_norm_sd

    semf = SEMF(
        data_preprocessor=pre,
        R=args.R,
        nodes_per_feature=nodes_per_feature,
        model_class=args.model_class,
        z_norm_sd=z_norm_sd,
        x_group_size=args.x_group_size,
        return_mean_default=SEMF_IMPLEMENTATION_POLICY[
            "return_mean_default"
        ],
        stopping_metric=SEMF_IMPLEMENTATION_POLICY["stopping_metric"],
        tree_config={"tree_n_estimators": args.tree_n_estimators,
                     "xgb_max_depth": args.xgb_max_depth,
                     "xgb_patience": args.xgb_patience,
                     "xgb_learning_rate": args.xgb_learning_rate,
                     "xgb_subsample": args.xgb_subsample,
                     "xgb_colsample_bytree": args.xgb_colsample,
                     "et_max_depth": SEMF_IMPLEMENTATION_POLICY[
                         "extra_trees_max_depth"
                     ]},
        nn_config={
            "nn_batch_size": SEMF_IMPLEMENTATION_POLICY["mlp"]["batch_size"],
            "nn_epochs": SEMF_IMPLEMENTATION_POLICY["mlp"]["epochs"],
            "nn_lr": SEMF_IMPLEMENTATION_POLICY["mlp"]["learning_rate"],
            "nn_patience": SEMF_IMPLEMENTATION_POLICY["mlp"]["patience"],
        } if args.model_class == "MultiMLPs" else None,
        models_val_split=SEMF_PAPER_RUN_CONFIG["models_valid_fraction"],
        stopping_patience=args.patience,
        max_it=args.max_it,
        verbose=False,
        seed=args.seed,
        device=args.device,
        weight_alignment=args.weight_alignment,
        allow_partial_recovery=args.allow_partial_recovery,
    )
    print(f"training SEMF on {n_feat} features in {n_groups} groups (group_size={args.x_group_size}), "
          f"total Z dim={int(nodes_per_feature.sum())}, z_norm_sd={z_norm_sd!r} ...")
    t0 = time.time()
    semf.train_semf()
    fit_s = time.time() - t0
    if not semf.training_complete:
        raise RuntimeError(
            "SEMF returned an incomplete diagnostic model; paper artifacts "
            "will not be written"
        )
    print(f"trained in {fit_s:.1f}s")

    pkl_path = args.results_dir / f"{_stem('semf', args.tag)}_seed{args.seed}.pkl"
    try:
        pkl_artifact = atomic_write_binary(
            pkl_path,
            lambda stream: pickle.dump(semf, stream),
        )
        print(f"saved trained SEMF to {pkl_path}")
    except Exception as e:
        raise RuntimeError(f"failed to save required SEMF checkpoint: {e}") from e

    # ---- per-regime inference -----------------------------------------
    y_test_arr = y_test.values.squeeze()
    y_valid_arr = y_valid.values.squeeze()
    al = args.alpha

    point_full = semf.infer_semf(X_test, return_type="point")
    point_metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test_arr, point_full))),
        "mae": float(mean_absolute_error(y_test_arr, point_full)),
        "r2": float(r2_score(y_test_arr, point_full)),
    }
    print(f"point (full): {point_metrics}")

    samples_by_regime: dict[str, dict[str, np.ndarray]] = {}
    for regime in args.regimes:
        masks = regime_blocks[regime]
        Xv = mask_features(X_valid, masks, block_cols)
        Xt = mask_features(X_test, masks, block_cols)
        sv = np.asarray(semf.infer_semf(Xv, return_type="interval", R=args.R_infer))
        st = np.asarray(semf.infer_semf(Xt, return_type="interval", R=args.R_infer))
        if isinstance(sv, tuple) or sv.ndim != 2 or st.ndim != 2:
            raise ValueError(
                f"regime={regime} produced invalid samples: "
                f"valid={getattr(sv, 'shape', '?')}, test={getattr(st, 'shape', '?')}"
            )
        samples_by_regime[regime] = {"valid": sv, "test": st}
        print(f"regime={regime:<10} valid {sv.shape}, test {st.shape}, "
              f"raw test mean width = {float((np.percentile(st,(1-al/2)*100,axis=1) - np.percentile(st,al/2*100,axis=1)).mean()):.4f}")

    # ---- conformal eval per regime + Mondrian -------------------------
    table: list[dict] = []
    for regime, s in samples_by_regime.items():
        sv = s["valid"]; st = s["test"]
        # raw
        lo_q = (al / 2) * 100; hi_q = (1 - al / 2) * 100
        l_raw = np.percentile(st, lo_q, axis=1); u_raw = np.percentile(st, hi_q, axis=1)
        m_raw = eval_intervals(y_test_arr, l_raw, u_raw); m_raw["method"] = "raw"; m_raw["regime"] = regime
        # cqr (calibrated on same-regime valid)
        l_c, u_c, info_c = cqr(y_valid_arr, sv, st, alpha=al)
        m_cqr = eval_intervals(y_test_arr, l_c, u_c); m_cqr.update({"method": "cqr", "regime": regime, **info_c})
        # density-conformal
        l_d, u_d, info_d = density_conformal(y_valid_arr, sv, st, alpha=al)
        m_d = eval_intervals(y_test_arr, l_d, u_d); m_d.update({"method": "density", "regime": regime, **info_d})
        for r in (m_raw, m_cqr, m_d):
            table.append(r)

    # Global CQR: q_hat fitted on `full`-regime validation, applied uniformly to
    # all test regimes. Compares to Mondrian (per-regime q_hat).
    if "full" in samples_by_regime:
        sv_full = samples_by_regime["full"]["valid"]
        lo_q = (al / 2) * 100; hi_q = (1 - al / 2) * 100
        l_vf = np.percentile(sv_full, lo_q, axis=1); u_vf = np.percentile(sv_full, hi_q, axis=1)
        cal_scores_full = np.maximum(l_vf - y_valid_arr, y_valid_arr - u_vf)
        q_hat_global = conformal_quantile(cal_scores_full, al)
        for regime, s in samples_by_regime.items():
            st = s["test"]
            l_t = np.percentile(st, lo_q, axis=1) - q_hat_global
            u_t = np.percentile(st, hi_q, axis=1) + q_hat_global
            m = eval_intervals(y_test_arr, l_t, u_t)
            m.update({"method": "global_cqr", "regime": regime, "q_hat": q_hat_global})
            table.append(m)

    # Mondrian CQR: pool all regimes, label by regime int, calibrate per stratum.
    if "full" in samples_by_regime and len(samples_by_regime) > 1:
        regimes = list(samples_by_regime.keys())
        regime2id = {r: i for i, r in enumerate(regimes)}
        sv_pool = np.concatenate([samples_by_regime[r]["valid"] for r in regimes], axis=0)
        st_pool = np.concatenate([samples_by_regime[r]["test"] for r in regimes], axis=0)
        yv_pool = np.tile(y_valid_arr, len(regimes))
        yt_pool = np.tile(y_test_arr, len(regimes))
        v_strata = np.concatenate([np.full(samples_by_regime[r]["valid"].shape[0], regime2id[r]) for r in regimes])
        t_strata = np.concatenate([np.full(samples_by_regime[r]["test"].shape[0], regime2id[r]) for r in regimes])
        l_m, u_m, info_m = mondrian_cqr(yv_pool, sv_pool, st_pool, v_strata, t_strata, alpha=al)
        # per-stratum eval
        for r, rid in regime2id.items():
            mask_t = t_strata == rid
            stratum_y = yt_pool[mask_t]
            m = eval_intervals(stratum_y, l_m[mask_t], u_m[mask_t])
            m.update({"method": "mondrian_cqr", "regime": r, "q_hat": info_m["q_hats"].get(rid)})
            table.append(m)

    df_table = pd.DataFrame(table).set_index(["method", "regime"])[["picp", "mpiw", "crps_uniform"]].round(4)
    print("\n=== conformal × regime table ===")
    print(df_table.to_string())

    # Publish binary samples first and the JSON completion record last. The
    # recorded hashes prevent a current JSON from authenticating stale bytes
    # left by an interrupted rerun.
    npz_path = args.results_dir / f"{_stem('semf', args.tag)}_seed{args.seed}_samples.npz"
    npz_arrays = {
        "y_valid": y_valid_arr,
        "y_test": y_test_arr,
        **{
            f"valid_{r}": samples_by_regime[r]["valid"]
            for r in samples_by_regime
        },
        **{
            f"test_{r}": samples_by_regime[r]["test"]
            for r in samples_by_regime
        },
    }
    npz_artifact = atomic_write_binary(
        npz_path,
        lambda stream: np.savez_compressed(stream, **npz_arrays),
    )

    out_path = args.results_dir / f"{_stem('semf', args.tag)}_seed{args.seed}.json"
    payload = {
        "config": {**vars(args), "run_config": run_config},
        "fit_seconds": fit_s,
        "training_complete": semf.training_complete,
        "last_successful_iteration": semf.last_successful_iteration,
        "point_metrics_full": point_metrics,
        "n_features": n_feat,
        "block_cols": block_cols,
        "regime_blocks": regime_blocks,
        "results": [{**r, "q_hat": (None if isinstance(r.get("q_hat"), dict) else r.get("q_hat"))}
                    for r in table],
        "artifacts": {
            "checkpoint": pkl_artifact,
            "samples": npz_artifact,
        },
    }
    attach_run_provenance(
        payload,
        ROOT,
        seed=args.seed,
        campaign_tag=args.tag or None,
        requested_device=args.device,
    )
    atomic_write_json(
        out_path,
        payload,
        default=lambda o: (
            float(o)
            if isinstance(o, (np.floating, np.integer))
            else str(o)
        ),
    )
    print(f"\nwrote {out_path}")
    print(f"wrote {npz_path}")
    return 0


if __name__ == "__main__":
    rc = main() or 0
    # Hard-exit to dodge the Windows joblib resource-tracker hang; outputs are
    # already flushed before this point.
    os._exit(rc)
