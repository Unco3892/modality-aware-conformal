"""SEMF extension: augmented theta(x, z) on SRED.

Hypothesis: SEMF loses ~16 pp R^2 vs XGBoost-on-features because z is a lossy
projection of x. If we re-train theta to consume *both* x and z, the gap should
close while still letting us run conformal calibration over MC samples drawn
from the SEMF latent posterior p(z | x).

This script:
  1. Loads a trained SEMF pickle.
  2. Reconstructs the same train/valid/test splits used during SEMF training
     (via the pickled `data_preprocessor`).
  3. Calls `semf.predict_phi_models(...)` on each split to obtain z-means
     (per row, concatenated across phi groups) -- the conditional expectation
     E[z | x] under SEMF's learned phi heads.
  4. Trains a fresh XGBoost regressor `theta_aug` on `concat([X, Z_mean]) -> y`.
     Variants:
       - aug_full: theta_aug([X_full, Z_mean])
       - aug_tab : theta_aug([X_tab_only_4, Z_mean])  (partial augmentation)
       - aug_z   : theta_aug([Z_mean])                (sanity check; ~ vanilla SEMF
                   theta but on z-means rather than per-row z draws)
  5. For MC inference: tiles X by R_infer and concatenates with R_infer z draws
     sampled via SEMF's exact path (`semf.simulate_complete_data`). This gives a
     (n, R_infer) y-sample matrix for raw/CQR/density-conformal intervals.
  6. Outputs a JSON summary per seed.

Usage:
  python src/sred/augmented_theta.py --seed 0
  python src/sred/augmented_theta.py --seed 0 1 2 3 4
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
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "sred"))

from semf import utils as semf_utils  # noqa: E402

from conformal import (  # noqa: E402
    cqr,
    density_conformal,
    evaluate as eval_intervals,
)
from multimodal_calibration.reproducibility import (  # noqa: E402
    artifact_identity,
    attach_run_provenance,
    atomic_write_binary,
    atomic_write_json,
    validate_campaign_payloads,
    validate_campaign_tag,
    validate_artifact_identity,
)
from multimodal_calibration.experiment_config import (  # noqa: E402
    AUGMENTED_PAPER_RUN_CONFIG,
    AUGMENTED_IMPLEMENTATION_POLICY,
    PAPER_CPU_DEVICE,
    PAPER_SEEDS,
    SRED_TABULAR_COLUMNS,
    canonical_run_config,
    require_canonical_run_config,
)
from sred.run_full_experiment import (  # noqa: E402
    SEMF_PAPER_RUN_CONFIG,
)

RESULTS = ROOT / "results" / "sred_semf"
TAB_COLS = list(SRED_TABULAR_COLUMNS)


def augmented_run_config(R_infer: int, alpha: float, n_jobs: int) -> dict:
    config = canonical_run_config("augmented")
    config["R_infer"] = int(R_infer)
    config["alpha"] = float(alpha)
    config["n_jobs"] = int(n_jobs)
    return config


def validate_semf_companions(
    results_dir: Path,
    seeds: list[int] | tuple[int, ...],
    tag: str,
) -> list[dict]:
    payloads = []
    errors = []
    for seed in seeds:
        path = results_dir / f"{_stem('semf', tag)}_seed{seed}.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing companion SEMF JSON: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        cfg = payload.get("config", {})
        if (
            int(cfg.get("seed", -1)) != seed
            or cfg.get("tag") != tag
            or cfg.get("device") != PAPER_CPU_DEVICE
            or payload.get("training_complete") is not True
        ):
            errors.append(path.name)
        artifacts = payload.get("artifacts", {})
        validate_artifact_identity(
            results_dir / f"{_stem('semf', tag)}_seed{seed}.pkl",
            artifacts.get("checkpoint"),
        )
        validate_artifact_identity(
            results_dir / f"{_stem('semf', tag)}_seed{seed}_samples.npz",
            artifacts.get("samples"),
        )
        payloads.append((path.name, payload))
    if errors:
        raise RuntimeError(
            "invalid/incomplete companion SEMF JSONs: " + ", ".join(errors)
        )
    validate_campaign_payloads(
        payloads,
        ROOT,
        campaign_tag=tag,
        requested_device=PAPER_CPU_DEVICE,
        run_config=SEMF_PAPER_RUN_CONFIG,
    )
    return [payload for _, payload in payloads]


def validate_augmented_companions(
    results_dir: Path,
    seeds: list[int] | tuple[int, ...],
    tag: str,
    *,
    semf_payloads: list[dict] | None = None,
) -> list[dict]:
    if semf_payloads is None:
        semf_payloads = validate_semf_companions(results_dir, seeds, tag)
    semf_by_seed = {
        int(payload.get("config", {}).get("seed", -1)): payload
        for payload in semf_payloads
    }
    payloads = []
    errors = []
    for seed in seeds:
        path = results_dir / f"{_stem('aug_theta', tag)}_seed{seed}.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing companion augmented JSON: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        cfg = payload.get("config", {})
        if (
            int(cfg.get("seed", -1)) != seed
            or cfg.get("campaign_tag") != tag
            or cfg.get("semf_tag") != tag
        ):
            errors.append(path.name)
        inputs = payload.get("inputs", {})
        validate_artifact_identity(
            results_dir / f"{_stem('semf', tag)}_seed{seed}.pkl",
            inputs.get("semf_checkpoint"),
        )
        sample_artifacts = payload.get("artifacts", {}).get("samples", {})
        sample_paths = {
            variant: (
                results_dir
                / f"{_stem('aug_theta', tag)}_seed{seed}_{variant}_samples.npz"
            )
            for variant in AUGMENTED_PAPER_RUN_CONFIG["variants"]
        }
        for variant, sample_path in sample_paths.items():
            validate_artifact_identity(
                sample_path,
                sample_artifacts.get(variant),
            )

        semf_payload = semf_by_seed.get(seed)
        if semf_payload is None:
            raise RuntimeError(
                f"missing validated SEMF payload for seed {seed}"
            )
        semf_samples = (
            results_dir / f"{_stem('semf', tag)}_seed{seed}_samples.npz"
        )
        validate_artifact_identity(
            semf_samples,
            semf_payload.get("artifacts", {}).get("samples"),
        )
        with np.load(semf_samples, allow_pickle=False) as semf_npz:
            y_valid = np.asarray(semf_npz["y_valid"])
            y_test = np.asarray(semf_npz["y_test"])
        for variant, sample_path in sample_paths.items():
            with np.load(sample_path, allow_pickle=False) as augmented_npz:
                labels_match = (
                    np.asarray(augmented_npz["y_valid"]).shape
                    == y_valid.shape
                    and np.allclose(
                        np.asarray(augmented_npz["y_valid"]),
                        y_valid,
                        rtol=0.0,
                        atol=1e-6,
                    )
                    and np.asarray(augmented_npz["y_test"]).shape
                    == y_test.shape
                    and np.allclose(
                        np.asarray(augmented_npz["y_test"]),
                        y_test,
                        rtol=0.0,
                        atol=1e-6,
                    )
                )
            if not labels_match:
                raise RuntimeError(
                    f"{sample_path}: target labels do not match SEMF samples"
                )
        payloads.append((path.name, payload))
    if errors:
        raise RuntimeError(
            "invalid companion augmented JSONs: " + ", ".join(errors)
        )
    validate_campaign_payloads(
        payloads,
        ROOT,
        campaign_tag=tag,
        requested_device=PAPER_CPU_DEVICE,
        run_config=AUGMENTED_PAPER_RUN_CONFIG,
    )
    return [payload for _, payload in payloads]


# ---------------------------------------------------------------------------
# helpers


def _stem(base: str, tag: str) -> str:
    """Build an artifact filename stem, inserting `tag` only when set."""
    return f"{base}_{tag}" if tag else base


def _as_df(X) -> pd.DataFrame:
    """SEMF's predict_phi_models calls .iloc on input -> needs DataFrame."""
    if isinstance(X, pd.DataFrame):
        return X
    return pd.DataFrame(np.asarray(X))


def get_z_means(semf, X) -> np.ndarray:
    """Return per-row mean of z under SEMF's phi heads, shape (n, hidden_z)."""
    Xdf = _as_df(X)
    preds = semf.predict_phi_models(data_to_predict=Xdf, model=semf.modPhi_p)
    parts = []
    for p in preds:
        p = np.asarray(p)
        if p.ndim == 1:
            p = p[:, None]
        parts.append(p)
    return np.concatenate(parts, axis=1).astype(np.float32)


def sample_z_R(semf, X, R: int, seed: int) -> np.ndarray:
    """Return SEMF MC samples of z, shape (n, hidden_z, R).

    Uses the published `simulate_complete_data` pathway so we match SEMF's
    exact noise model (per-group sigma).
    """
    Xdf = _as_df(X)
    semf_utils.set_seed(seed)
    z_R_sep = semf.simulate_complete_data(
        data_to_predict=Xdf, input_length=Xdf.shape[0], R=R
    )
    # each entry is (n, n_outcomes_p, R); concat along axis=1.
    return np.concatenate(z_R_sep, axis=1).astype(np.float32)


def fit_xgb(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    seed: int,
    n_jobs: int = AUGMENTED_PAPER_RUN_CONFIG["n_jobs"],
) -> xgb.XGBRegressor:
    config = AUGMENTED_PAPER_RUN_CONFIG["xgb"]
    model = xgb.XGBRegressor(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        learning_rate=config["learning_rate"],
        subsample=config["subsample"],
        colsample_bytree=config["colsample_bytree"],
        early_stopping_rounds=config["early_stopping_rounds"],
        eval_metric=config["eval_metric"],
        tree_method=config["tree_method"],
        n_jobs=n_jobs,
        random_state=seed,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return model


def mc_predict(
    model: xgb.XGBRegressor,
    X: np.ndarray,
    Z_R: np.ndarray,
    feature_slice: slice | None = None,
) -> np.ndarray:
    """For each row i and replication r, predict theta_aug([X[i, slice], Z_R[i, :, r]]).

    Returns (n, R).
    """
    n, _, R = Z_R.shape
    Xs = X if feature_slice is None else X[:, feature_slice]
    # build (n*R, d_x + d_z), batched
    # tile X along R -> (n, R, d_x); reorder to (n, R, d_x) and flatten.
    X_tiled = np.broadcast_to(Xs[:, None, :], (n, R, Xs.shape[1])).reshape(n * R, -1)
    Z_flat = np.transpose(Z_R, (0, 2, 1)).reshape(n * R, -1)
    feats = np.concatenate([X_tiled, Z_flat], axis=1).astype(np.float32)
    pred = model.predict(feats).reshape(n, R)
    return pred


def point_metrics(y, yhat) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, yhat))),
        "mae": float(mean_absolute_error(y, yhat)),
        "r2": float(r2_score(y, yhat)),
    }


# ---------------------------------------------------------------------------
# per-seed driver


def run_seed(seed: int, R_infer: int, alpha: float, n_jobs: int, tag: str,
             semf_tag: str = "", results_dir: Path = RESULTS) -> dict:
    pkl = results_dir / f"{_stem('semf', semf_tag)}_seed{seed}.pkl"
    print(f"\n=== seed {seed}: loading {pkl.name} ===")
    semf_input_artifact = artifact_identity(pkl)
    with open(pkl, "rb") as f:
        semf = pickle.load(f)

    # SEMF stores the (already-scaled) split arrays directly.
    Xtr = np.asarray(semf.x_train, dtype=np.float32)
    Xva = np.asarray(semf.x_valid, dtype=np.float32)
    Xte = np.asarray(semf.x_test, dtype=np.float32)
    ytr = np.asarray(semf.y_train, dtype=np.float32).squeeze()
    yva = np.asarray(semf.y_valid, dtype=np.float32).squeeze()
    yte = np.asarray(semf.y_test, dtype=np.float32).squeeze()

    # Leading columns are the configured tabular features; the SEMF driver
    # concatenates them before the encoded modality blocks.
    n_feat = Xtr.shape[1]
    tab_idx = np.arange(len(SRED_TABULAR_COLUMNS))

    print(
        f"  splits: train={len(Xtr)}, valid={len(Xva)}, "
        f"test={len(Xte)}; n_features={n_feat}; "
        f"hidden_z={int(np.sum(semf.n_z_outcomes))}"
    )

    # --- phi means (E[z | x]) for each split
    t0 = time.time()
    Z_tr = get_z_means(semf, Xtr)
    Z_va = get_z_means(semf, Xva)
    Z_te = get_z_means(semf, Xte)
    t_phi = time.time() - t0
    print(f"  z-means computed in {t_phi:.1f}s; shape {Z_tr.shape}")

    # --- MC z-samples (n, hidden_z, R_infer) for valid + test
    t0 = time.time()
    Z_va_R = sample_z_R(semf, Xva, R_infer, seed=seed)
    Z_te_R = sample_z_R(
        semf,
        Xte,
        R_infer,
        seed=(
            seed
            + AUGMENTED_IMPLEMENTATION_POLICY[
                "latent_sampling_seed_offset"
            ]
        ),
    )
    t_mc = time.time() - t0
    print(f"  z-MC samples drawn in {t_mc:.1f}s; shape {Z_te_R.shape}")

    # --- variants
    variant_definitions = {
        "aug_full": {"X_slice": slice(None), "feat_dim": Xtr.shape[1]},
        "aug_tab": {"X_slice": tab_idx, "feat_dim": len(tab_idx)},
        "aug_z": {"X_slice": np.array([], dtype=int), "feat_dim": 0},
    }
    expected_variants = tuple(AUGMENTED_PAPER_RUN_CONFIG["variants"])
    if set(variant_definitions) != set(expected_variants):
        raise ValueError(
            "augmented implementation variants differ from the canonical "
            f"contract: implemented={sorted(variant_definitions)}, "
            f"configured={sorted(expected_variants)}"
        )
    variants = {
        name: variant_definitions[name] for name in expected_variants
    }

    out: dict = {
        "config": {
            "seed": int(seed),
            "campaign_tag": tag or None,
            "semf_tag": semf_tag or None,
            "run_config": augmented_run_config(R_infer, alpha, n_jobs),
        },
        "seed": seed,
        "R_infer": R_infer,
        "alpha": alpha,
        "n_features": int(Xtr.shape[1]),
        "hidden_z": int(Z_tr.shape[1]),
        "phi_predict_seconds": t_phi,
        "z_mc_seconds": t_mc,
        "variants": {},
    }
    sample_artifacts: dict[str, dict[str, object]] = {}

    for name, cfg in variants.items():
        sl = cfg["X_slice"]

        def _gather(X, Z):
            if isinstance(sl, slice) and sl == slice(None):
                base = X
            elif isinstance(sl, np.ndarray) and sl.size == 0:
                base = np.zeros((X.shape[0], 0), dtype=np.float32)
            else:
                base = X[:, sl]
            return np.concatenate([base, Z], axis=1).astype(np.float32)

        F_tr = _gather(Xtr, Z_tr)
        F_va = _gather(Xva, Z_va)
        F_te = _gather(Xte, Z_te)

        t0 = time.time()
        model = fit_xgb(F_tr, ytr, F_va, yva, seed=seed, n_jobs=n_jobs)
        t_fit = time.time() - t0

        # point predictions on test (E[z|x] only -- not MC)
        yhat = model.predict(F_te)
        pm = point_metrics(yte, yhat)

        # also compute MC-mean point pred (theta_aug averaged over R draws)
        S_va = mc_predict(model, Xva, Z_va_R, feature_slice=(None if isinstance(sl, slice) and sl == slice(None) else sl))
        S_te = mc_predict(model, Xte, Z_te_R, feature_slice=(None if isinstance(sl, slice) and sl == slice(None) else sl))
        yhat_mc = S_te.mean(axis=1)
        pm_mc = point_metrics(yte, yhat_mc)

        # interval methods
        lo_q = (alpha / 2) * 100
        hi_q = (1 - alpha / 2) * 100
        l_raw = np.percentile(S_te, lo_q, axis=1)
        u_raw = np.percentile(S_te, hi_q, axis=1)
        m_raw = eval_intervals(yte, l_raw, u_raw)
        m_raw["method"] = "raw"

        l_c, u_c, info_c = cqr(yva, S_va, S_te, alpha=alpha)
        m_cqr = eval_intervals(yte, l_c, u_c)
        m_cqr.update({"method": "cqr", **info_c})

        l_d, u_d, info_d = density_conformal(yva, S_va, S_te, alpha=alpha)
        m_dens = eval_intervals(yte, l_d, u_d)
        m_dens.update({"method": "density", **info_d})
        interval_by_method = {
            metric["method"]: metric
            for metric in (m_raw, m_cqr, m_dens)
        }
        expected_calibrations = tuple(
            AUGMENTED_PAPER_RUN_CONFIG["calibrations"]
        )
        if set(interval_by_method) != set(expected_calibrations):
            raise ValueError(
                "augmented interval implementations differ from the "
                f"canonical contract: implemented={sorted(interval_by_method)}, "
                f"configured={sorted(expected_calibrations)}"
            )

        out["variants"][name] = {
            "feat_dim": int(cfg["feat_dim"]),
            "fit_seconds": t_fit,
            "best_iteration": int(getattr(model, "best_iteration", -1) or -1),
            "point_metrics": pm,
            "point_metrics_mc": pm_mc,
            "intervals": [
                interval_by_method[method]
                for method in expected_calibrations
            ],
        }

        # Dump MC sample matrix per variant for downstream stratified analysis.
        npz_path = results_dir / f"{_stem('aug_theta', tag)}_seed{seed}_{name}_samples.npz"
        sample_artifacts[name] = atomic_write_binary(
            npz_path,
            lambda stream, yva=yva, yte=yte, S_va=S_va, S_te=S_te: (
                np.savez_compressed(
                    stream,
                    y_valid=yva,
                    y_test=yte,
                    # Stable names for downstream saved-sample diagnostics.
                    valid_full=S_va,
                    test_full=S_te,
                )
            ),
        )
        print(
            f"  [{name:8s}] fit {t_fit:5.1f}s  "
            f"R2={pm['r2']:.4f} (mc R2={pm_mc['r2']:.4f})  "
            f"raw PICP={m_raw['picp']:.3f} W={m_raw['mpiw']:.3f}  "
            f"CQR PICP={m_cqr['picp']:.3f} W={m_cqr['mpiw']:.3f}  "
            f"DEN PICP={m_dens['picp']:.3f} W={m_dens['mpiw']:.3f}"
        )

    out["inputs"] = {"semf_checkpoint": semf_input_artifact}
    out["artifacts"] = {"samples": sample_artifacts}
    attach_run_provenance(
        out,
        ROOT,
        seed=seed,
        campaign_tag=tag or None,
        requested_device=PAPER_CPU_DEVICE,
    )
    out_path = results_dir / f"{_stem('aug_theta', tag)}_seed{seed}.json"
    atomic_write_json(
        out_path,
        out,
        default=lambda o: (
            float(o)
            if isinstance(o, (np.floating, np.integer))
            else str(o)
        ),
    )
    print(f"  wrote {out_path}")
    return out


# ---------------------------------------------------------------------------
# aggregation


def aggregate(per_seed: list[dict]) -> dict:
    if not per_seed:
        return {}
    variant_names = list(per_seed[0]["variants"].keys())
    agg: dict = {"n_seeds": len(per_seed), "seeds": [r["seed"] for r in per_seed],
                 "variants": {}}
    for v in variant_names:
        runs = [r["variants"][v] for r in per_seed]
        r2 = np.array([s["point_metrics"]["r2"] for s in runs])
        rmse = np.array([s["point_metrics"]["rmse"] for s in runs])
        mae = np.array([s["point_metrics"]["mae"] for s in runs])
        r2_mc = np.array([s["point_metrics_mc"]["r2"] for s in runs])
        ints: dict = {}
        for method in AUGMENTED_PAPER_RUN_CONFIG["calibrations"]:
            picp = np.array([
                next(m for m in s["intervals"] if m["method"] == method)["picp"]
                for s in runs
            ])
            mpiw = np.array([
                next(m for m in s["intervals"] if m["method"] == method)["mpiw"]
                for s in runs
            ])
            ints[method] = {
                "picp_mean": float(picp.mean()), "picp_std": float(picp.std(ddof=1)) if len(picp) > 1 else 0.0,
                "mpiw_mean": float(mpiw.mean()), "mpiw_std": float(mpiw.std(ddof=1)) if len(mpiw) > 1 else 0.0,
            }
        agg["variants"][v] = {
            "r2_mean": float(r2.mean()), "r2_std": float(r2.std(ddof=1)) if len(r2) > 1 else 0.0,
            "r2_mc_mean": float(r2_mc.mean()), "r2_mc_std": float(r2_mc.std(ddof=1)) if len(r2_mc) > 1 else 0.0,
            "rmse_mean": float(rmse.mean()), "rmse_std": float(rmse.std(ddof=1)) if len(rmse) > 1 else 0.0,
            "mae_mean": float(mae.mean()), "mae_std": float(mae.std(ddof=1)) if len(mae) > 1 else 0.0,
            "intervals": ints,
        }
    return agg


def print_agg_table(agg: dict) -> None:
    if not agg:
        return
    print("\n" + "=" * 90)
    print(f"Aggregated over {agg['n_seeds']} seeds: {agg['seeds']}")
    print("=" * 90)
    rows = []
    for v, s in agg["variants"].items():
        row = {
            "variant": v,
            "R2_pt": f"{s['r2_mean']:.4f} +- {s['r2_std']:.4f}",
            "R2_mc": f"{s['r2_mc_mean']:.4f} +- {s['r2_mc_std']:.4f}",
            "RMSE": f"{s['rmse_mean']:.4f} +- {s['rmse_std']:.4f}",
        }
        for method in AUGMENTED_PAPER_RUN_CONFIG["calibrations"]:
            label = method.upper()
            row[f"{label}_PICP"] = (
                f"{s['intervals'][method]['picp_mean']:.3f}"
            )
            row[f"{label}_W"] = (
                f"{s['intervals'][method]['mpiw_mean']:.3f}"
            )
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))


# ---------------------------------------------------------------------------
# main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--seed", type=int, nargs="+", default=[PAPER_SEEDS[0]]
    )
    ap.add_argument(
        "--R-infer",
        type=int,
        default=AUGMENTED_PAPER_RUN_CONFIG["R_infer"],
    )
    ap.add_argument(
        "--alpha", type=float, default=AUGMENTED_PAPER_RUN_CONFIG["alpha"]
    )
    ap.add_argument(
        "--n-jobs", type=int, default=AUGMENTED_PAPER_RUN_CONFIG["n_jobs"]
    )
    ap.add_argument("--tag", default="",
                    help="optional artifact tag for this script's own outputs; "
                         "when set, outputs are named aug_theta_<tag>_seed<n>* "
                         "instead of the default aug_theta_seed<n>*")
    ap.add_argument("--semf-tag", default="",
                    help="tag of the SEMF pickle to load; must match the --tag "
                         "used by run_full_experiment.py (default: untagged, "
                         "semf_seed{s}.pkl)")
    ap.add_argument(
        "--paper-run",
        action="store_true",
        help="require tagged inputs/outputs and the configured paper seed grid",
    )
    ap.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="exploratory only: skip missing SEMF checkpoints",
    )
    ap.add_argument("--results-dir", type=Path, default=RESULTS)
    args = ap.parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.tag = validate_campaign_tag(args.tag, required=args.paper_run) or ""
    args.semf_tag = validate_campaign_tag(
        args.semf_tag, required=args.paper_run
    ) or ""
    if args.paper_run:
        if not args.tag or not args.semf_tag:
            ap.error("--paper-run requires --tag and --semf-tag")
        if tuple(sorted(set(args.seed))) != PAPER_SEEDS:
            ap.error(
                "--paper-run requires canonical --seed "
                + " ".join(str(seed) for seed in PAPER_SEEDS)
            )
        if args.tag != args.semf_tag:
            ap.error("--paper-run requires matching --tag and --semf-tag")
        try:
            require_canonical_run_config(
                "augmented",
                augmented_run_config(args.R_infer, args.alpha, args.n_jobs),
            )
        except ValueError as error:
            ap.error(str(error))
        validate_semf_companions(
            args.results_dir, tuple(args.seed), args.semf_tag
        )

    per_seed = []
    for s in args.seed:
        try:
            per_seed.append(run_seed(s, R_infer=args.R_infer, alpha=args.alpha,
                                     n_jobs=args.n_jobs, tag=args.tag,
                                     semf_tag=args.semf_tag,
                                     results_dir=args.results_dir))
        except FileNotFoundError as e:
            if not args.allow_incomplete:
                raise
            print(f"  WARN: seed {s} skipped ({e})")

    agg = aggregate(per_seed)
    if args.paper_run and agg.get("n_seeds") != len(PAPER_SEEDS):
        raise RuntimeError("paper aggregation requires exactly five completed seeds")
    print_agg_table(agg)
    if agg:
        agg["config"] = {
            "campaign_tag": args.tag or None,
            "semf_tag": args.semf_tag or None,
            "run_config": augmented_run_config(
                args.R_infer, args.alpha, args.n_jobs
            ),
        }
        agg["per_seed_artifacts"] = {
            str(result["seed"]): {
                "inputs": result["inputs"],
                "artifacts": result["artifacts"],
            }
            for result in per_seed
        }
        attach_run_provenance(
            agg,
            ROOT,
            seed=None,
            campaign_tag=args.tag or None,
            requested_device=PAPER_CPU_DEVICE,
        )
        agg_path = args.results_dir / f"{_stem('aug_theta', args.tag)}_agg.json"
        atomic_write_json(agg_path, agg)
        print(f"\nwrote {agg_path}")


if __name__ == "__main__":
    rc = main() or 0
    # Hard-exit to dodge the Windows joblib resource-tracker hang; outputs are
    # already flushed before this point.
    os._exit(rc)
