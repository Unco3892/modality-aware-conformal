"""Cross-dataset gated-mixture benchmark for multi-modal regression.

Pipeline (per dataset, per seed):
  1. Load dataset via loaders/<name>.load and assemble per-modality feature
     blocks. For SRED, use the v4 cache (multilingual-e5-large + DINOv2-B/14).
     The paper path requires the dataset-specific cache contract below and
     fails if an expected cache is absent.
  2. Split train into disjoint "fit", "tune", and "calib" folds. The fit fold
     trains the per-modality predictors and the homogeneous-XGB baseline; the
     tune fold trains the composition layer (linear stacker / gated mixture)
     and fixes every data-driven calibration choice; the calib fold is
     reserved for the conformal scores.
  3. Per-modality predictors of different model classes:
       tab    -> XGB-quantile (gradient-boosted trees)
       text   -> MLP-pinball  (PyTorch on GPU; frozen sentence-transformer emb)
       image  -> MLP-pinball  (PyTorch on GPU; frozen DINOv2 emb)
     All return (lo, point, hi) per row.
  4. Composition: linear stacker and gated mixture (per-row softmax weights
     from a small MLP), both fit on the tune fold with pinball loss.
  5. Calibration: marginal CQR, disagreement-Mondrian (3 bins), and
     disagreement-weighted (locally adaptive).
  6. Comparison baselines:
       * solo-XGB-tab / solo-MLP-text / solo-MLP-image (single-modality solos)
       * homogeneous-XGB-concat (XGB-quantile on concat(tab, text, image))
  7. Evaluate point R2/RMSE/MAE plus interval PICP/MPIW/NCIW per
     calibration variant. Write JSON per (dataset, seed).

Usage:
    python src/multimodal_calibration/run_hetero_mixture.py \\
        --datasets sred mercari pawpularity imdb_wiki \\
        --seeds 0 1 2 3 4 --alpha 0.05
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
EXP = Path(__file__).resolve().parent
SRED_EXP = ROOT / "src" / "sred"
RESULTS = ROOT / "results" / "multimodal_calibration"
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(SRED_EXP))

from predictors import XGBQuantileRegressor, MLPQuantileRegressor  # noqa: E402
from composition import LinearStacker, GatedMixture, stack_preds  # noqa: E402
from calibration import (  # noqa: E402
    conformal_quantile, cqr_calibrate, disagreement_mondrian_cqr,
    disagreement_score,
)
from nciw import compute_nciw  # noqa: E402
from encoders import paper_cache_slugs  # noqa: E402
from reproducibility import (  # noqa: E402
    attach_run_provenance,
    thread_identity,
    validate_campaign_payloads,
    validate_campaign_tag,
    write_campaign_manifest,
)
from experiment_config import (  # noqa: E402
    AUXILIARY_GATE_BATCH_POLICY,
    AUXILIARY_PAPER_RUN_CONFIG,
    AUXILIARY_PREPROCESSING_POLICY,
    DATASET_MODALITY_FIELDS,
    PAPER_DATASETS,
    PAPER_DISAGREEMENT_BINS,
    PAPER_SEEDS,
    SRED_TABULAR_COLUMNS,
    alpha_result_tag,
    auxiliary_gate_epochs,
    auxiliary_xgb_control_estimators,
    auxiliary_xgb_quantile_estimators,
    require_canonical_run_config,
)
from result_grid import (  # noqa: E402
    AUXILIARY_CAMPAIGN_METHODS,
    auxiliary_campaign_scope,
    auxiliary_run_config,
)


# ---------------------------------------------------------------------------
# tabular preprocessing (one-hot top-K levels, standardize numerics)


def _fit_tab_preprocessor(
    tab_fit: pd.DataFrame | None,
    top_k: int = AUXILIARY_PREPROCESSING_POLICY["categorical_top_k"],
) -> dict:
    """Fit tabular preprocessing on the model-fit fold only."""
    if tab_fit is None or len(tab_fit) == 0:
        return {"num_cols": [], "cat_levels": {}, "num_mean": None, "num_mu": None, "num_sd": None}

    cat_cols, num_cols = [], []
    for c in tab_fit.columns:
        if pd.api.types.is_numeric_dtype(tab_fit[c]):
            num_cols.append(c)
        else:
            cat_cols.append(c)

    prep = {"num_cols": num_cols, "cat_levels": {}, "num_mean": None, "num_mu": None, "num_sd": None}
    if num_cols:
        x = tab_fit[num_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32).copy()
        col_mean = np.nanmean(x, axis=0)
        col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
        for j in range(len(num_cols)):
            x[np.isnan(x[:, j]), j] = col_mean[j]
        prep["num_mean"] = col_mean
        prep["num_mu"] = x.mean(axis=0)
        prep["num_sd"] = x.std(axis=0) + 1e-6

    for c in cat_cols:
        prep["cat_levels"][c] = tab_fit[c].astype(str).value_counts().head(top_k).index.tolist()
    return prep


def _transform_tab(tab: pd.DataFrame | None, prep: dict) -> np.ndarray:
    n = 0 if tab is None else len(tab)
    if tab is None or n == 0:
        return np.zeros((n, 0), dtype=np.float32)

    parts = []
    num_cols = prep["num_cols"]
    if num_cols:
        x = tab[num_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32).copy()
        col_mean = prep["num_mean"]
        for j in range(len(num_cols)):
            x[np.isnan(x[:, j]), j] = col_mean[j]
        x = (x - prep["num_mu"]) / prep["num_sd"]
        parts.append(x)

    for c, levels in prep["cat_levels"].items():
        values = tab[c].astype(str).to_numpy()
        for lv in levels:
            parts.append((values == lv).astype(np.float32)[:, None])

    return np.concatenate(parts, axis=1).astype(np.float32) if parts else np.zeros((n, 0), dtype=np.float32)


def _onehot_tab(tab_train: pd.DataFrame | None, tab_test: pd.DataFrame | None,
                top_k: int = AUXILIARY_PREPROCESSING_POLICY[
                    "categorical_top_k"
                ], tab_fit: pd.DataFrame | None = None):
    """Transform train/test tabular blocks using preprocessing fit on ``tab_fit``.

    ``tab_fit`` defaults to ``tab_train`` for backwards-compatible callers, but
    experiment entry points should pass the model-fit fold to avoid tune/cal
    leakage through means, standard deviations, and categorical level choice.
    """
    if tab_train is None or len(tab_train) == 0:
        n_te = 0 if tab_test is None else len(tab_test)
        return (np.zeros((len(tab_train) if tab_train is not None else 0, 0), dtype=np.float32),
                np.zeros((n_te, 0), dtype=np.float32))

    prep = _fit_tab_preprocessor(tab_train if tab_fit is None else tab_fit, top_k=top_k)
    return _transform_tab(tab_train, prep), _transform_tab(tab_test, prep)


# ---------------------------------------------------------------------------
# embedding loader: pull whatever is cached in data/<name>/embeddings/


# (text_field_list, image_field_list)
def _embedding_fields(name: str) -> tuple[list[str], list[str]]:
    fields = DATASET_MODALITY_FIELDS.get(name)
    if fields is None:
        return ([], [])
    return (list(fields["text"]), list(fields["image"]))


def _load_emb(
    name: str,
    split: str,
    *,
    text_slug: str | None = None,
    image_slug: str | None = None,
    allow_approximate_fallback: bool = False,
) -> dict[str, np.ndarray]:
    """Load cached text+image embeddings from data/<name>/embeddings/.

    With no overrides, paper runs use the audited dataset-specific mapping:
    SRED uses e5-large/ViT-B; Mercari uses MiniLM; Pawpularity uses
    DINOv2-S; IMDB-WIKI uses MiniLM/DINOv2-S. Alternative caches are considered
    only when ``allow_approximate_fallback=True`` is passed explicitly.
    """
    if name == "sred":
        emb_dir = ROOT / "data" / "sred" / "embeddings"
    else:
        emb_dir = ROOT / "data" / name / "embeddings"
    text_fields, image_fields = _embedding_fields(name)
    contract_text, contract_image = paper_cache_slugs(name)
    text_slug = text_slug or contract_text
    image_slug = image_slug or contract_image

    out: dict[str, np.ndarray] = {}

    if text_fields:
        if text_slug is None:
            raise ValueError(f"{name} has text fields but no text cache contract")
        candidates = [text_slug]
        if allow_approximate_fallback:
            candidates.append("paraphrase-multilingual-MiniLM-L12-v2")
        selected_text_slug = None
        for c in candidates:
            ok = all((emb_dir / f"{split}_{f}_{c}.npy").exists() for f in text_fields)
            if ok:
                selected_text_slug = c
                break
        if selected_text_slug is None:
            expected = ", ".join(
                str(emb_dir / f"{split}_{f}_{text_slug}.npy")
                for f in text_fields
            )
            raise FileNotFoundError(
                f"exact text cache missing for {name}/{split}; expected {expected}. "
                "Build/install that cache, or opt into the approximate fallback explicitly."
            )
        parts = [
            np.load(emb_dir / f"{split}_{f}_{selected_text_slug}.npy")
            for f in text_fields
        ]
        # mean-pool fields (matches v3 SRED convention) if multi-field;
        # otherwise concat (= just one block).
        if len(parts) == 1:
            out["text"] = parts[0].astype(np.float32)
        else:
            out["text"] = np.mean(np.stack(parts, axis=0), axis=0).astype(np.float32)
        out["text_slug"] = selected_text_slug  # type: ignore[assignment]

    if image_fields:
        if image_slug is None:
            raise ValueError(f"{name} has image fields but no image cache contract")
        candidates = [image_slug]
        if allow_approximate_fallback:
            candidates.append("dinov2-vits14")
        selected_image_slug = None
        for c in candidates:
            ok = all((emb_dir / f"{split}_{f}_{c}.npy").exists() for f in image_fields)
            if ok:
                selected_image_slug = c
                break
        if selected_image_slug is None:
            expected = ", ".join(
                str(emb_dir / f"{split}_{f}_{image_slug}.npy")
                for f in image_fields
            )
            raise FileNotFoundError(
                f"exact image cache missing for {name}/{split}; expected {expected}. "
                "Build/install that cache, or opt into the approximate fallback explicitly."
            )
        parts = [
            np.load(emb_dir / f"{split}_{f}_{selected_image_slug}.npy")
            for f in image_fields
        ]
        if len(parts) == 1:
            out["image"] = parts[0].astype(np.float32)
        else:
            out["image"] = np.mean(np.stack(parts, axis=0), axis=0).astype(np.float32)
        out["image_slug"] = selected_image_slug  # type: ignore[assignment]

    return out


# ---------------------------------------------------------------------------
# data assembly


def _load_dataset(name: str) -> tuple[dict, dict]:
    """Return (train, test) dicts via the loader registry."""
    if name == "sred":
        # SRED is special: load metadata directly from sred and bypass
        # the loaders/ folder (which doesn't have a sred entry).
        from data_io import load_metadata
        tr_df, te_df = load_metadata()
        tab_cols = list(SRED_TABULAR_COLUMNS)
        train = {
            "y": np.log(tr_df["price"].to_numpy(dtype=np.float32)),
            "tab": tr_df[tab_cols].copy(),
            "text": None, "image": None,
            "id": tr_df["listing_id"].to_numpy(),
        }
        test = {
            "y": np.log(te_df["price"].to_numpy(dtype=np.float32)),
            "tab": te_df[tab_cols].copy(),
            "text": None, "image": None,
            "id": te_df["listing_id"].to_numpy(),
        }
        return train, test

    from loaders import REGISTRY
    train = REGISTRY[name]("train")
    test = REGISTRY[name]("test")
    return train, test


def _build_blocks(name: str, train: dict, test: dict, fit_idx: np.ndarray | None = None) -> dict:
    """Return dict with X_tab_tr/te, X_text_tr/te, X_image_tr/te, y_tr/te,
    plus 'enc_info' describing which encoder caches were used."""
    tab_train = train.get("tab")
    tab_fit = None if tab_train is None or fit_idx is None else tab_train.iloc[fit_idx]
    Xtab_tr, Xtab_te = _onehot_tab(tab_train, test.get("tab"), tab_fit=tab_fit)

    text_fields, image_fields = _embedding_fields(name)
    Xtxt_tr = np.zeros((len(train["y"]), 0), dtype=np.float32)
    Xtxt_te = np.zeros((len(test["y"]), 0), dtype=np.float32)
    Ximg_tr = np.zeros((len(train["y"]), 0), dtype=np.float32)
    Ximg_te = np.zeros((len(test["y"]), 0), dtype=np.float32)
    text_slug = None
    image_slug = None

    if text_fields:
        emb_tr = _load_emb(name, "train")
        emb_te = _load_emb(name, "test")
        Xtxt_tr = emb_tr["text"]
        Xtxt_te = emb_te["text"]
        text_slug = emb_tr.get("text_slug")
    if image_fields:
        emb_tr = _load_emb(name, "train")
        emb_te = _load_emb(name, "test")
        Ximg_tr = emb_tr["image"]
        Ximg_te = emb_te["image"]
        image_slug = emb_tr.get("image_slug")

    return {
        "X_tab_tr": Xtab_tr, "X_tab_te": Xtab_te,
        "X_text_tr": Xtxt_tr, "X_text_te": Xtxt_te,
        "X_image_tr": Ximg_tr, "X_image_te": Ximg_te,
        "y_tr": train["y"].astype(np.float32),
        "y_te": test["y"].astype(np.float32),
        "enc_info": {"text_slug": text_slug, "image_slug": image_slug},
    }


def _split_calib(n: int, calib_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_calib = max(50, int(round(calib_frac * n)))
    n_calib = min(n_calib, n - 50)  # leave at least 50 fit rows
    calib_idx = perm[:n_calib]
    fit_idx = perm[n_calib:]
    return fit_idx, calib_idx


def _split_fit_tune_cal(
    n: int,
    calib_frac: float,
    tune_frac: float,
    seed: int,
    min_fold: int = AUXILIARY_PAPER_RUN_CONFIG["split"]["min_fold"],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train/tune/final-calibration split.

    The tune fold is used for mixture/gate fitting and local-adaptivity tuning;
    the final calibration fold is reserved for conformal quantiles.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    if n < 3 * min_fold:
        n_calib = max(1, int(round(calib_frac * n)))
        n_tune = max(1, int(round(tune_frac * n)))
    else:
        n_calib = max(min_fold, int(round(calib_frac * n)))
        n_tune = max(min_fold, int(round(tune_frac * n)))
    while n_calib + n_tune > n - min_fold and n_calib > 1 and n_tune > 1:
        if n_calib >= n_tune:
            n_calib -= 1
        else:
            n_tune -= 1
    calib_idx = perm[:n_calib]
    tune_idx = perm[n_calib:n_calib + n_tune]
    fit_idx = perm[n_calib + n_tune:]
    if len(fit_idx) == 0:
        raise ValueError(f"cannot split n={n} into nonempty fit/tune/calibration folds")
    return fit_idx, tune_idx, calib_idx


# ---------------------------------------------------------------------------
# helpers: y standardization (helpful for MLP heads)


def _standardize_y(y_fit: np.ndarray):
    mu = float(y_fit.mean())
    sd = float(y_fit.std() + 1e-6)
    return mu, sd


# ---------------------------------------------------------------------------
# disagreement-weighted CQR (locally-adaptive)


def disagreement_weighted_cqr(
    y_tune: np.ndarray,
    preds_tune: np.ndarray,
    y_calib: np.ndarray,
    preds_calib: np.ndarray,
    preds_test: np.ndarray,
    s_dis_tune: np.ndarray,
    s_dis_calib: np.ndarray,
    s_dis_test: np.ndarray,
    alpha: float = 0.1,
):
    """Locally-adaptive CQR: rescale CQR score by sqrt(a0 + a1 * s_dis^2).

    The adaptivity parameter is selected on a tune fold, then the conformal
    quantile is computed only on the final calibration fold. This preserves the
    split-conformal validity argument for the reported intervals.
    """
    lo_u, _, hi_u = preds_tune[:, 0], preds_tune[:, 1], preds_tune[:, 2]
    lo_v, _, hi_v = preds_calib[:, 0], preds_calib[:, 1], preds_calib[:, 2]
    lo_t, _, hi_t = preds_test[:, 0], preds_test[:, 1], preds_test[:, 2]
    # Pure-expansion CQR scores (paper eq. 5): clipped at zero so a negative
    # quantile cannot contract more where disagreement is larger.
    raw_tune = np.maximum(np.maximum(lo_u - y_tune, y_tune - hi_u), 0.0)
    raw_calib = np.maximum(np.maximum(lo_v - y_calib, y_calib - hi_v), 0.0)

    a1_grid = np.asarray(
        AUXILIARY_PAPER_RUN_CONFIG["heterogeneous"]["weighted_a1_grid"],
        dtype=np.float64,
    )
    best = (None, np.inf)
    for a1 in a1_grid:
        sigma_tune = np.sqrt(1.0 + a1 * (s_dis_tune ** 2))
        sc_tune = raw_tune / sigma_tune
        q_hat = conformal_quantile(sc_tune, alpha)
        lo_tune = lo_u - q_hat * sigma_tune
        hi_tune = hi_u + q_hat * sigma_tune
        cov = float(np.mean((y_tune >= lo_tune) & (y_tune <= hi_tune)))
        if cov >= (1 - alpha):
            mw = float((hi_tune - lo_tune).mean())
            if mw < best[1]:
                best = ((a1, q_hat, cov), mw)
    if best[0] is None:
        best = ((0.0, conformal_quantile(raw_tune, alpha), float("nan")), np.inf)
    a1_best, q_tune, cov_tune = best[0]
    sigma_calib = np.sqrt(1.0 + a1_best * (s_dis_calib ** 2))
    sc_calib = raw_calib / sigma_calib
    q_hat = conformal_quantile(sc_calib, alpha)
    sigma_test = np.sqrt(1.0 + a1_best * (s_dis_test ** 2))
    lo = lo_t - q_hat * sigma_test
    hi = hi_t + q_hat * sigma_test
    info = {
        "method": "dis_weighted",
        "a1": float(a1_best),
        "q_hat": q_hat,
        "q_tune": float(q_tune),
        "coverage_tune": float(cov_tune),
        "n_tune": int(len(y_tune)),
        "n_calib": int(len(y_calib)),
        "a1_grid": a1_grid.tolist(),
    }
    return lo, hi, info


# ---------------------------------------------------------------------------
# metrics packaging


def _point_metrics(y, point) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, point))),
        "mae": float(mean_absolute_error(y, point)),
        "r2": float(r2_score(y, point)),
    }


def _interval_metrics(y, lo, hi, alpha: float) -> dict:
    nciw_v, info = compute_nciw(y, lo, hi, alpha=alpha)
    return {
        "picp": float(np.mean((y >= lo) & (y <= hi))),
        "mpiw": float(info["base_mpiw"]),
        "niw": float(info["base_niw"]),
        "nciw": float(nciw_v),
        "c_test_cal": float(info["c_test_cal"]),
        "y_range": float(info["y_range"]),
    }


# ---------------------------------------------------------------------------
# the full per-(dataset, seed) experiment


def run_one(
    name: str,
    seed: int,
    alpha: float,
    calib_frac: float = AUXILIARY_PAPER_RUN_CONFIG["split"]["calib_frac"],
    tune_frac: float = AUXILIARY_PAPER_RUN_CONFIG["split"]["tune_frac"],
    verbose: bool = True,
) -> dict:
    if verbose:
        print(f"\n=== {name}  seed={seed}  alpha={alpha} ===")
    train, test = _load_dataset(name)
    n_tr = len(train["y"])
    fit_idx, tune_idx, calib_idx = _split_fit_tune_cal(n_tr, calib_frac, tune_frac, seed)
    blocks = _build_blocks(name, train, test, fit_idx=fit_idx)

    Xtab_tr = blocks["X_tab_tr"]; Xtab_te = blocks["X_tab_te"]
    Xtxt_tr = blocks["X_text_tr"]; Xtxt_te = blocks["X_text_te"]
    Ximg_tr = blocks["X_image_tr"]; Ximg_te = blocks["X_image_te"]
    y_tr_raw = blocks["y_tr"]; y_te = blocks["y_te"]
    if verbose:
        print(f"  shapes: tab={Xtab_tr.shape[1]} text={Xtxt_tr.shape[1]} image={Ximg_tr.shape[1]}")
        print(f"  n_fit={len(fit_idx)} n_tune={len(tune_idx)} n_calib={len(calib_idx)} n_test={len(y_te)}")
        print(f"  enc: text={blocks['enc_info']['text_slug']} image={blocks['enc_info']['image_slug']}")

    # standardize y on the fit fold for stable MLP training; un-scale later
    y_mu, y_sd = _standardize_y(y_tr_raw[fit_idx])
    y_fit = (y_tr_raw[fit_idx] - y_mu) / y_sd
    y_tune = (y_tr_raw[tune_idx] - y_mu) / y_sd
    y_calib = (y_tr_raw[calib_idx] - y_mu) / y_sd
    y_test = (y_te - y_mu) / y_sd

    def _splice(M):
        return M[fit_idx], M[tune_idx], M[calib_idx], M

    Xtab_fit, Xtab_tune, Xtab_calib, _ = _splice(Xtab_tr)
    Xtxt_fit, Xtxt_tune, Xtxt_calib, _ = _splice(Xtxt_tr)
    Ximg_fit, Ximg_tune, Ximg_calib, _ = _splice(Ximg_tr)

    # --- per-modality predictors ---------------------------------------
    preds_tune_per: dict[str, np.ndarray] = {}
    preds_calib_per: dict[str, np.ndarray] = {}
    preds_test_per: dict[str, np.ndarray] = {}
    info_per: dict[str, dict] = {}

    if Xtab_fit.shape[1] > 0:
        t0 = time.time()
        xgb_config = AUXILIARY_PAPER_RUN_CONFIG["heterogeneous"][
            "xgb_quantile"
        ]
        p = XGBQuantileRegressor(
            alpha=alpha,
            n_estimators=auxiliary_xgb_quantile_estimators(n_tr),
            max_depth=xgb_config["max_depth"],
            learning_rate=xgb_config["learning_rate"],
            seed=seed,
        )
        p.fit(Xtab_fit, y_fit)
        preds_tune_per["tab"] = p.predict(Xtab_tune)
        preds_calib_per["tab"] = p.predict(Xtab_calib)
        preds_test_per["tab"] = p.predict(Xtab_te)
        info_per["tab"] = {"kind": "xgb_quantile", "fit_seconds": time.time() - t0,
                            "in_dim": int(Xtab_fit.shape[1])}
        if verbose:
            print(f"  tab predictor (XGB-q): {info_per['tab']['fit_seconds']:.1f}s")

    if Xtxt_fit.shape[1] > 0:
        t0 = time.time()
        mlp_config = AUXILIARY_PAPER_RUN_CONFIG["heterogeneous"][
            "modality_mlp_quantile"
        ]
        p = MLPQuantileRegressor(
            alpha=alpha,
            hidden=mlp_config["hidden"],
            dropout=mlp_config["dropout"],
            epochs=mlp_config["epochs"],
            batch_size=mlp_config["batch_size"],
            lr=mlp_config["learning_rate"],
            weight_decay=mlp_config["weight_decay"],
            seed=seed,
        )
        p.fit(Xtxt_fit, y_fit)
        preds_tune_per["text"] = p.predict(Xtxt_tune)
        preds_calib_per["text"] = p.predict(Xtxt_calib)
        preds_test_per["text"] = p.predict(Xtxt_te)
        info_per["text"] = {"kind": "mlp_pinball", "fit_seconds": time.time() - t0,
                              "in_dim": int(Xtxt_fit.shape[1])}
        if verbose:
            print(f"  text predictor (MLP-pinball): {info_per['text']['fit_seconds']:.1f}s "
                  f"best_val={getattr(p, 'best_val_', float('nan')):.4f}")

    if Ximg_fit.shape[1] > 0:
        t0 = time.time()
        mlp_config = AUXILIARY_PAPER_RUN_CONFIG["heterogeneous"][
            "modality_mlp_quantile"
        ]
        p = MLPQuantileRegressor(
            alpha=alpha,
            hidden=mlp_config["hidden"],
            dropout=mlp_config["dropout"],
            epochs=mlp_config["epochs"],
            batch_size=mlp_config["batch_size"],
            lr=mlp_config["learning_rate"],
            weight_decay=mlp_config["weight_decay"],
            seed=seed,
        )
        p.fit(Ximg_fit, y_fit)
        preds_tune_per["image"] = p.predict(Ximg_tune)
        preds_calib_per["image"] = p.predict(Ximg_calib)
        preds_test_per["image"] = p.predict(Ximg_te)
        info_per["image"] = {"kind": "mlp_pinball", "fit_seconds": time.time() - t0,
                               "in_dim": int(Ximg_fit.shape[1])}
        if verbose:
            print(f"  image predictor (MLP-pinball): {info_per['image']['fit_seconds']:.1f}s "
                  f"best_val={getattr(p, 'best_val_', float('nan')):.4f}")

    if not preds_calib_per:
        raise RuntimeError(f"{name}: no modality predictors built")

    modality_order = list(preds_calib_per.keys())  # deterministic key order
    P_tune = stack_preds(preds_tune_per)           # (K, n_tune, 3)
    P_calib = stack_preds(preds_calib_per)         # (K, n_calib, 3)
    P_test = stack_preds(preds_test_per)           # (K, n_test, 3)
    K = P_calib.shape[0]

    # --- composition layer ---------------------------------------------
    composition_results = {}

    # (a) linear stacker
    stacker = LinearStacker(alpha=alpha, seed=seed).fit(P_tune, y_tune)
    p_tune_st = stacker.predict(P_tune)
    p_calib_st = stacker.predict(P_calib)
    p_test_st = stacker.predict(P_test)
    composition_results["linear_stacker"] = {
        "weights": dict(zip(modality_order, stacker.weights_.tolist())),
    }

    # (b) gated mixture: gate features = concat of all modality features
    # (use only blocks that exist; tab is fine even when low-dim)
    gate_parts_tune = []
    gate_parts_calib = []
    gate_parts_test = []
    if Xtab_fit.shape[1] > 0:
        gate_parts_tune.append(Xtab_tune); gate_parts_calib.append(Xtab_calib); gate_parts_test.append(Xtab_te)
    if Xtxt_fit.shape[1] > 0:
        gate_parts_tune.append(Xtxt_tune); gate_parts_calib.append(Xtxt_calib); gate_parts_test.append(Xtxt_te)
    if Ximg_fit.shape[1] > 0:
        gate_parts_tune.append(Ximg_tune); gate_parts_calib.append(Ximg_calib); gate_parts_test.append(Ximg_te)
    gate_tune = np.concatenate(gate_parts_tune, axis=1).astype(np.float32)
    gate_calib = np.concatenate(gate_parts_calib, axis=1).astype(np.float32)
    gate_test = np.concatenate(gate_parts_test, axis=1).astype(np.float32)

    # smaller datasets -> shorter gate training to avoid overfitting
    gate_config = AUXILIARY_PAPER_RUN_CONFIG["heterogeneous"][
        "gated_mixture"
    ]
    gate_batch = min(
        AUXILIARY_GATE_BATCH_POLICY["maximum"],
        max(
            AUXILIARY_GATE_BATCH_POLICY["minimum"],
            len(tune_idx) // AUXILIARY_GATE_BATCH_POLICY["tune_divisor"],
        ),
    )
    gate_epochs = auxiliary_gate_epochs(n_tr)
    gated = GatedMixture(
        alpha=alpha,
        seed=seed,
        epochs=gate_epochs,
        batch_size=gate_batch,
        lr=gate_config["learning_rate"],
        hidden=gate_config["hidden"],
    ).fit(P_tune, y_tune, gate_tune)
    p_tune_gt = gated.predict(P_tune, gate_tune)
    p_calib_gt = gated.predict(P_calib, gate_calib)
    p_test_gt = gated.predict(P_test, gate_test)
    gate_w_test = gated.get_gate_weights(gate_test)  # (n_test, K)
    gate_stats = {
        m: {
            "mean": float(gate_w_test[:, i].mean()),
            "std": float(gate_w_test[:, i].std()),
            "median": float(np.median(gate_w_test[:, i])),
            "p10": float(np.quantile(gate_w_test[:, i], 0.1)),
            "p90": float(np.quantile(gate_w_test[:, i], 0.9)),
        }
        for i, m in enumerate(modality_order)
    }
    composition_results["gated_mixture"] = {
        "gate_stats": gate_stats,
        "best_val_pinball": float(getattr(gated, "best_val_", float("nan"))),
    }

    # disagreement scores from per-modality point predictions (in standardized y)
    s_dis_tune = disagreement_score(np.stack([P_tune[i, :, 1] for i in range(K)], axis=0))
    s_dis_calib = disagreement_score(np.stack([P_calib[i, :, 1] for i in range(K)], axis=0))
    s_dis_test = disagreement_score(np.stack([P_test[i, :, 1] for i in range(K)], axis=0))

    # --- evaluate base configurations ---------------------------------
    def _eval_with_calibrators(p_tune_mix: np.ndarray, p_calib_mix: np.ndarray,
                               p_test_mix: np.ndarray, label: str):
        # raw (uncalibrated) interval
        lo_r, hi_r = p_test_mix[:, 0], p_test_mix[:, 2]
        # marginal CQR
        lo_c, hi_c, _ = cqr_calibrate(y_calib, p_calib_mix, p_test_mix, alpha=alpha)
        # disagreement-Mondrian CQR (3 bins)
        lo_m, hi_m, mon_info = disagreement_mondrian_cqr(
            y_calib, p_calib_mix, p_test_mix,
            s_dis_calib,
            s_dis_test,
            alpha=alpha,
            n_bins=PAPER_DISAGREEMENT_BINS,
            s_dis_ref=s_dis_tune,
        )
        # disagreement-weighted CQR
        lo_w, hi_w, dw_info = disagreement_weighted_cqr(
            y_tune, p_tune_mix,
            y_calib, p_calib_mix, p_test_mix,
            s_dis_tune,
            s_dis_calib, s_dis_test, alpha=alpha,
        )
        # un-standardize for reporting (multiplicative on widths/lo/hi/y)
        def _unstd(y_arr):
            return y_arr * y_sd + y_mu
        def _unstd_int(lo, hi):
            return _unstd(lo), _unstd(hi)
        y_te_orig = _unstd(y_test)
        point_orig = _unstd(p_test_mix[:, 1])
        lo_r_o, hi_r_o = _unstd_int(lo_r, hi_r)
        lo_c_o, hi_c_o = _unstd_int(lo_c, hi_c)
        lo_m_o, hi_m_o = _unstd_int(lo_m, hi_m)
        lo_w_o, hi_w_o = _unstd_int(lo_w, hi_w)

        return {
            "label": label,
            "point": _point_metrics(y_te_orig, point_orig),
            "raw":      _interval_metrics(y_te_orig, lo_r_o, hi_r_o, alpha),
            "cqr":      _interval_metrics(y_te_orig, lo_c_o, hi_c_o, alpha),
            "mondrian": _interval_metrics(y_te_orig, lo_m_o, hi_m_o, alpha),
            "dis_weighted": _interval_metrics(y_te_orig, lo_w_o, hi_w_o, alpha),
            "calibration_info": {
                "mondrian": mon_info, "dis_weighted": dw_info,
            },
        }

    base_results: dict[str, dict] = {}
    # solo per-modality
    for m in modality_order:
        base_results[f"solo_{m}"] = _eval_with_calibrators(
            preds_tune_per[m], preds_calib_per[m], preds_test_per[m], f"solo_{m}"
        )

    # heterogeneous mixtures
    base_results["hetero_linear_stacker"] = _eval_with_calibrators(
        p_tune_st, p_calib_st, p_test_st, "hetero_linear_stacker"
    )
    base_results["hetero_gated_mixture"] = _eval_with_calibrators(
        p_tune_gt, p_calib_gt, p_test_gt, "hetero_gated_mixture"
    )

    # --- homogeneous baselines ----------------------------------------
    # XGB-quantile on concat(tab, text, image)
    parts_fit = []; parts_tune = []; parts_calib = []; parts_test = []
    if Xtab_fit.shape[1] > 0:
        parts_fit.append(Xtab_fit); parts_tune.append(Xtab_tune); parts_calib.append(Xtab_calib); parts_test.append(Xtab_te)
    if Xtxt_fit.shape[1] > 0:
        parts_fit.append(Xtxt_fit); parts_tune.append(Xtxt_tune); parts_calib.append(Xtxt_calib); parts_test.append(Xtxt_te)
    if Ximg_fit.shape[1] > 0:
        parts_fit.append(Ximg_fit); parts_tune.append(Ximg_tune); parts_calib.append(Ximg_calib); parts_test.append(Ximg_te)
    Xall_fit = np.concatenate(parts_fit, axis=1)
    Xall_tune = np.concatenate(parts_tune, axis=1)
    Xall_calib = np.concatenate(parts_calib, axis=1)
    Xall_test = np.concatenate(parts_test, axis=1)

    t0 = time.time()
    xgb_config = AUXILIARY_PAPER_RUN_CONFIG["heterogeneous"]["xgb_quantile"]
    homo = XGBQuantileRegressor(
        alpha=alpha,
        n_estimators=auxiliary_xgb_quantile_estimators(n_tr),
        max_depth=xgb_config["max_depth"],
        learning_rate=xgb_config["learning_rate"],
        seed=seed,
    )
    homo.fit(Xall_fit, y_fit)
    p_homo_tune = homo.predict(Xall_tune)
    p_homo_calib = homo.predict(Xall_calib)
    p_homo_test = homo.predict(Xall_test)
    info_per["homo_xgb_concat"] = {"kind": "xgb_quantile_concat",
                                     "fit_seconds": time.time() - t0,
                                     "in_dim": int(Xall_fit.shape[1])}
    base_results["homo_xgb_concat"] = _eval_with_calibrators(
        p_homo_tune, p_homo_calib, p_homo_test, "homo_xgb_concat"
    )

    # ---- Per-modality XGB ensemble (homogeneous-class control) -------
    preds_tune_xgbper: dict[str, np.ndarray] = {}
    preds_calib_xgbper: dict[str, np.ndarray] = {}
    preds_test_xgbper: dict[str, np.ndarray] = {}
    if Xtab_fit.shape[1] > 0:
        preds_tune_xgbper["tab"] = preds_tune_per["tab"]  # already XGB
        preds_calib_xgbper["tab"] = preds_calib_per["tab"]  # already XGB
        preds_test_xgbper["tab"] = preds_test_per["tab"]
    for mod_name, X_fit_, X_tune_, X_cal_, X_te_ in (
        ("text", Xtxt_fit, Xtxt_tune, Xtxt_calib, Xtxt_te),
        ("image", Ximg_fit, Ximg_tune, Ximg_calib, Ximg_te),
    ):
        if X_fit_.shape[1] == 0:
            continue
        t0 = time.time()
        # XGB on high-dim embeddings: cap n_estimators a bit
        m = XGBQuantileRegressor(
            alpha=alpha,
            n_estimators=auxiliary_xgb_control_estimators(n_tr),
            max_depth=xgb_config["max_depth"],
            learning_rate=xgb_config["learning_rate"],
            seed=seed,
        )
        m.fit(X_fit_, y_fit)
        preds_tune_xgbper[mod_name] = m.predict(X_tune_)
        preds_calib_xgbper[mod_name] = m.predict(X_cal_)
        preds_test_xgbper[mod_name] = m.predict(X_te_)
        info_per[f"xgbcontrol_{mod_name}"] = {
            "kind": "xgb_quantile",
            "fit_seconds": time.time() - t0,
            "in_dim": int(X_fit_.shape[1]),
        }
    if len(preds_calib_xgbper) >= 2:
        Pu = stack_preds(preds_tune_xgbper)
        Pc = stack_preds(preds_calib_xgbper)
        Pt = stack_preds(preds_test_xgbper)
        # use the same gate input as before
        gat_xgb = GatedMixture(
            alpha=alpha,
            seed=seed,
            epochs=gate_epochs,
            batch_size=gate_batch,
            lr=gate_config["learning_rate"],
            hidden=gate_config["hidden"],
        ).fit(Pu, y_tune, gate_tune)
        p_xgb_tune_g = gat_xgb.predict(Pu, gate_tune)
        p_xgb_calib_g = gat_xgb.predict(Pc, gate_calib)
        p_xgb_test_g = gat_xgb.predict(Pt, gate_test)
        base_results["homog_xgb_gated"] = _eval_with_calibrators(
            p_xgb_tune_g, p_xgb_calib_g, p_xgb_test_g, "homog_xgb_gated"
        )

    # ---- assemble payload -------------------------------------------
    payload = {
        "config": {
            "dataset": name, "seed": seed, "alpha": alpha,
            "calib_frac": calib_frac, "tune_frac": tune_frac,
            "split_protocol": "fit_tune_calibration_test",
            "n_train_total": int(n_tr), "n_fit": int(len(fit_idx)),
            "n_tune": int(len(tune_idx)),
            "n_calib": int(len(calib_idx)), "n_test": int(len(y_te)),
            "tab_dim": int(Xtab_tr.shape[1]),
            "text_dim": int(Xtxt_tr.shape[1]),
            "image_dim": int(Ximg_tr.shape[1]),
            "encoders": blocks["enc_info"],
            "modality_order": modality_order,
        },
        "modality_predictors": info_per,
        "composition": composition_results,
        "results": base_results,
        "y_stats": {"train_mean": float(y_mu), "train_std": float(y_sd)},
    }
    return payload


def _alpha_tag(alpha: float) -> str:
    return alpha_result_tag(alpha)


def _tagged_path(
    prefix: str,
    name: str,
    seed: int,
    tag: str | None = None,
    results_dir: Path = RESULTS,
) -> Path:
    mid = f"{name}_{tag}" if tag else name
    return results_dir / f"{prefix}_{mid}_seed{seed}.json"


def _save(
    payload: dict,
    name: str,
    seed: int,
    tag: str | None = None,
    results_dir: Path = RESULTS,
    run_config: dict | None = None,
):
    if tag:
        payload.setdefault("config", {})["output_tag"] = tag
    payload.setdefault("config", {})["run_config"] = run_config or {}
    attach_run_provenance(
        payload,
        ROOT,
        seed=seed,
        campaign_tag=tag,
        requested_device="cuda" if torch.cuda.is_available() else "cpu",
    )
    p = _tagged_path("hetero", name, seed, tag, results_dir)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, p)
    print(f"  -> {p}")
    return p


def _valid_existing(
    path: Path,
    name: str,
    seed: int,
    alpha: float,
    tag: str,
    run_config: dict,
) -> bool:
    try:
        payload = json.loads(path.read_text())
        cfg = payload.get("config", {})
        valid = (
            cfg.get("dataset") == name
            and int(cfg.get("seed")) == seed
            and abs(float(cfg.get("alpha")) - alpha) < 1e-12
            and cfg.get("output_tag") == tag
            and bool(payload.get("results"))
        )
        if not valid:
            return False
        validate_campaign_payloads(
            [(path.name, payload)],
            ROOT,
            campaign_tag=tag,
            requested_device="cuda" if torch.cuda.is_available() else "cpu",
            run_config=run_config,
            producer_threads=thread_identity(),
        )
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(PAPER_DATASETS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(PAPER_SEEDS))
    ap.add_argument(
        "--alpha", type=float, default=AUXILIARY_PAPER_RUN_CONFIG["alpha"]
    )
    ap.add_argument(
        "--tune-frac",
        type=float,
        default=AUXILIARY_PAPER_RUN_CONFIG["split"]["tune_frac"],
    )
    ap.add_argument("--tag", default=None,
                    help="result filename tag; defaults to an alpha-derived tag")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip (dataset, seed) combos that already have a JSON")
    ap.add_argument("--out-dir", type=Path, default=RESULTS)
    ap.add_argument(
        "--paper-run",
        action="store_true",
        help="require an explicit campaign tag and write a source manifest",
    )
    ap.add_argument(
        "--campaign-datasets",
        nargs="+",
        default=None,
        help="full paper campaign scope when this worker runs a subset",
    )
    ap.add_argument(
        "--campaign-seeds",
        nargs="+",
        type=int,
        default=None,
        help="full paper campaign scope when this worker runs a subset",
    )
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.paper_run and args.tag is None:
        ap.error("--paper-run requires an explicit --tag")
    tag = args.tag if args.tag is not None else _alpha_tag(args.alpha)
    validate_campaign_tag(tag, required=args.paper_run)
    run_config = auxiliary_run_config(args.alpha, args.tune_frac)
    if args.paper_run:
        try:
            require_canonical_run_config("aux", run_config)
        except ValueError as error:
            ap.error(str(error))
        campaign_datasets, campaign_seeds = auxiliary_campaign_scope(
            args.datasets,
            args.seeds,
            campaign_datasets=args.campaign_datasets,
            campaign_seeds=args.campaign_seeds,
        )
        write_campaign_manifest(
            args.out_dir,
            ROOT,
            campaign_tag=tag,
            datasets=campaign_datasets,
            seeds=campaign_seeds,
            methods=AUXILIARY_CAMPAIGN_METHODS,
            requested_device="cuda" if torch.cuda.is_available() else "cpu",
            run_config=run_config,
            producer_threads=thread_identity(),
        )
    summary = []
    failures = 0
    for name in args.datasets:
        for seed in args.seeds:
            out_path = _tagged_path(
                "hetero", name, seed, tag, args.out_dir
            )
            if args.skip_existing and out_path.exists():
                if _valid_existing(
                    out_path, name, seed, args.alpha, tag, run_config
                ):
                    print(f"[skip] {out_path.name} exists and matches config")
                    continue
                if args.paper_run:
                    raise RuntimeError(
                        f"--skip-existing rejected stale/incompatible {out_path}"
                    )
            try:
                p = run_one(name, seed=seed, alpha=args.alpha, tune_frac=args.tune_frac)
                _save(
                    p,
                    name,
                    seed,
                    tag=tag,
                    results_dir=args.out_dir,
                    run_config=run_config,
                )
                summary.append({"dataset": name, "seed": seed, "ok": True,
                                 "hetero_gated_R2": p["results"]["hetero_gated_mixture"]["point"]["r2"],
                                 "homo_concat_R2": p["results"]["homo_xgb_concat"]["point"]["r2"]})
            except Exception as e:
                traceback.print_exc()
                failures += 1
                summary.append({"dataset": name, "seed": seed, "ok": False, "err": repr(e)})
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    sys.exit(main() or 0)
