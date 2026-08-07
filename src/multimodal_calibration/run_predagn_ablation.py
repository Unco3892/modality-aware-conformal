"""Cross-dataset predictor-agnostic disagreement calibration.

This driver answers a narrow paper question:

    Given the same fitted base predictor, does disagreement-aware calibration
    improve interval efficiency relative to marginal split conformal/CQR?

It deliberately avoids the heterogeneous gated mixture so that the calibration
wrapper is not confounded with a base model that already consumes source-wise
disagreement. For each dataset/seed it fits:

  * XGB point on concatenated modalities, calibrated by absolute residuals.
  * XGB quantile on concatenated modalities, calibrated by CQR.
  * Source-wise stacked point predictor, built from per-modality XGB points.

The modality disagreement score is the standard deviation across per-modality
point predictions. The score is fitted before final calibration: fit rows train
base predictors, tune rows choose the weighted scale and stacker, and final
calibration rows compute conformal quantiles.

Weighted and scaled paths use the paper's pure-expansion score max(e, 0).
Marginal CQR and unscaled Mondrian deliberately retain signed scores. The
stored scale choices and quantiles are checked by
paper/scripts/audit_weighted_quantiles.py.

The legacy artifact key ``difficulty_weighted`` denotes the XGBoost-learned
scale sensitivity comparator reported in the appendix. It fits a positive
log-score scale from fused tune-fold covariates and non-negative tune scores.
That model is frozen before final calibration, so it belongs to
``D_pre`` and preserves the same split-conformal validity argument.
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
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
EXP = Path(__file__).resolve().parent
SRED_EXP = ROOT / "src" / "sred"
RESULTS = ROOT / "results" / "multimodal_calibration"
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(SRED_EXP))

from calibration import conformal_quantile, crps_uniform, disagreement_mondrian_cqr  # noqa: E402
from experiment_config import (  # noqa: E402
    PAPER_CPU_DEVICE,
    PAPER_DATASETS,
    PAPER_DISAGREEMENT_BINS,
    PAPER_SEEDS,
    PAPER_XGB_POINT_OBJECTIVE,
    PREDAGN_BASE_PREDICTORS,
    PREDAGN_IMPLEMENTATION_POLICY,
    PREDAGN_PAPER_RUN_CONFIG,
    PREDAGN_SEED_POLICY,
    alpha_result_tag,
    canonical_run_config,
    require_canonical_run_config,
)
from result_aggregation import (  # noqa: E402
    summarize_predagn,
    summarize_predagn_bins,
)
from nciw import compute_nciw  # noqa: E402
from run_hetero_mixture import (  # noqa: E402
    _embedding_fields, _load_dataset, _load_emb, _split_fit_tune_cal,
)
from reproducibility import (  # noqa: E402
    attach_run_provenance,
    thread_identity,
    validate_campaign_payloads,
    validate_campaign_tag,
    write_campaign_manifest,
)
from result_grid import expected_seeds, paper_campaign_scope  # noqa: E402


DEFAULT_DATASETS = PAPER_DATASETS
MODALITIES = ("tab", "text", "image")


def point_xgb_run_config(n_estimators: int, max_depth: int) -> dict:
    """Shared fitted-XGB contract used by predagn and missingness runs."""
    config = canonical_run_config("predagn")["point_xgb"]
    config["n_estimators"] = int(n_estimators)
    config["max_depth"] = int(max_depth)
    return config


def predagn_run_config(
    *,
    alpha: float,
    calib_frac: float,
    tune_frac: float,
    n_estimators: int,
    q_n_estimators: int,
    max_depth: int,
) -> dict:
    config = canonical_run_config("predagn")
    config["alpha"] = float(alpha)
    config["split"]["calib_frac"] = float(calib_frac)
    config["split"]["tune_frac"] = float(tune_frac)
    config["point_xgb"] = point_xgb_run_config(n_estimators, max_depth)
    config["quantile_xgb"]["n_estimators"] = int(q_n_estimators)
    config["quantile_xgb"]["max_depth"] = int(max_depth)
    config["quantile_xgb"]["quantiles"] = [
        float(alpha / 2),
        0.5,
        float(1 - alpha / 2),
    ]
    return config


def _stack3(lo, mid, hi) -> np.ndarray:
    out = np.stack([lo, mid, hi], axis=1).astype(np.float32)
    return np.sort(out, axis=1)


def _concat_nonempty(arrays: list[np.ndarray]) -> np.ndarray:
    nonempty = [a for a in arrays if a.shape[1] > 0]
    if not nonempty:
        raise ValueError("no nonempty feature blocks")
    return np.concatenate(nonempty, axis=1).astype(np.float32)


def _fit_tab_preprocessor(
    tab_fit: pd.DataFrame | None,
    top_k: int = PREDAGN_IMPLEMENTATION_POLICY["categorical_top_k"],
) -> dict:
    if tab_fit is None or len(tab_fit) == 0:
        return {"num_cols": [], "cat_levels": {}, "num_mean": None, "num_mu": None, "num_sd": None}
    cat_cols, num_cols = [], []
    for col in tab_fit.columns:
        if pd.api.types.is_numeric_dtype(tab_fit[col]):
            num_cols.append(col)
        else:
            cat_cols.append(col)

    prep: dict = {"num_cols": num_cols, "cat_levels": {}, "num_mean": None, "num_mu": None, "num_sd": None}
    if num_cols:
        x = tab_fit[num_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32).copy()
        col_mean = np.nanmean(x, axis=0)
        col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
        for j in range(len(num_cols)):
            x[np.isnan(x[:, j]), j] = col_mean[j]
        prep["num_mean"] = col_mean
        prep["num_mu"] = x.mean(axis=0)
        prep["num_sd"] = x.std(axis=0) + 1e-6
    for col in cat_cols:
        prep["cat_levels"][col] = tab_fit[col].astype(str).value_counts().head(top_k).index.tolist()
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
    for col, levels in prep["cat_levels"].items():
        values = tab[col].astype(str).to_numpy()
        for level in levels:
            parts.append((values == level).astype(np.float32)[:, None])
    return np.concatenate(parts, axis=1).astype(np.float32) if parts else np.zeros((n, 0), dtype=np.float32)


def _build_blocks_split_aware(dataset: str, train: dict, test: dict, fit_idx: np.ndarray) -> dict:
    """Assemble feature blocks with tabular preprocessing fit on fit rows only."""
    tab_train = train.get("tab")
    tab_test = test.get("tab")
    tab_fit = None if tab_train is None else tab_train.iloc[fit_idx]
    prep = _fit_tab_preprocessor(tab_fit)
    Xtab_tr = _transform_tab(tab_train, prep)
    Xtab_te = _transform_tab(tab_test, prep)

    text_fields, image_fields = _embedding_fields(dataset)
    Xtxt_tr = np.zeros((len(train["y"]), 0), dtype=np.float32)
    Xtxt_te = np.zeros((len(test["y"]), 0), dtype=np.float32)
    Ximg_tr = np.zeros((len(train["y"]), 0), dtype=np.float32)
    Ximg_te = np.zeros((len(test["y"]), 0), dtype=np.float32)
    text_slug = None
    image_slug = None
    if text_fields:
        emb_tr = _load_emb(dataset, "train")
        emb_te = _load_emb(dataset, "test")
        Xtxt_tr = emb_tr["text"]
        Xtxt_te = emb_te["text"]
        text_slug = emb_tr.get("text_slug")
    if image_fields:
        emb_tr = _load_emb(dataset, "train")
        emb_te = _load_emb(dataset, "test")
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


def _block_dict(blocks: dict, idx: np.ndarray | None = None) -> dict[str, np.ndarray]:
    out = {
        "tab": blocks["X_tab_tr"],
        "text": blocks["X_text_tr"],
        "image": blocks["X_image_tr"],
    }
    if idx is not None:
        out = {k: v[idx] for k, v in out.items()}
    return out


def _test_block_dict(blocks: dict) -> dict[str, np.ndarray]:
    return {
        "tab": blocks["X_tab_te"],
        "text": blocks["X_text_te"],
        "image": blocks["X_image_te"],
    }


def _fit_xgb_point(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_tune: np.ndarray,
    y_tune: np.ndarray,
    seed: int,
    n_estimators: int,
    max_depth: int,
    device: str,
) -> xgb.XGBRegressor:
    config = point_xgb_run_config(n_estimators, max_depth)
    model = xgb.XGBRegressor(
        **config,
        device=device,
        random_state=seed,
    )
    model.fit(X_fit, y_fit, eval_set=[(X_tune, y_tune)], verbose=False)
    return model


def _fit_xgb_quantile(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    seed: int,
    alpha: float,
    n_estimators: int,
    max_depth: int,
    device: str,
) -> list[xgb.XGBRegressor]:
    config = canonical_run_config("predagn")["quantile_xgb"]
    config["n_estimators"] = int(n_estimators)
    config["max_depth"] = int(max_depth)
    quantiles = (
        float(alpha / 2),
        0.5,
        float(1 - alpha / 2),
    )
    config.pop("quantiles")
    models: list[xgb.XGBRegressor] = []
    for q in quantiles:
        model = xgb.XGBRegressor(
            **config,
            quantile_alpha=q,
            device=device,
            random_state=seed,
        )
        model.fit(X_fit, y_fit, verbose=False)
        models.append(model)
    return models


def _predict_quantile(models: list[xgb.XGBRegressor], X: np.ndarray) -> np.ndarray:
    preds = [m.predict(X) for m in models]
    return _stack3(preds[0], preds[1], preds[2])


def _fit_modality_points(
    X_fit: dict[str, np.ndarray],
    y_fit: np.ndarray,
    X_tune: dict[str, np.ndarray],
    y_tune: np.ndarray,
    X_cal: dict[str, np.ndarray],
    X_test: dict[str, np.ndarray],
    seed: int,
    n_estimators: int,
    max_depth: int,
    device: str,
) -> tuple[dict, dict[str, xgb.XGBRegressor]]:
    preds = {"tune": [], "calib": [], "test": []}
    models: dict[str, xgb.XGBRegressor] = {}
    for mod in MODALITIES:
        if X_fit[mod].shape[1] == 0:
            continue
        model = _fit_xgb_point(
            X_fit[mod], y_fit, X_tune[mod], y_tune,
            seed=(
                seed
                + PREDAGN_SEED_POLICY["modality_point_stride"]
                * (len(models) + 1)
            ),
            n_estimators=n_estimators,
            max_depth=max_depth,
            device=device,
        )
        models[mod] = model
        preds["tune"].append(model.predict(X_tune[mod]).astype(np.float32))
        preds["calib"].append(model.predict(X_cal[mod]).astype(np.float32))
        preds["test"].append(model.predict(X_test[mod]).astype(np.float32))

    if not preds["test"]:
        raise ValueError("no modality predictors were fitted")

    arrs = {k: np.stack(v, axis=0) for k, v in preds.items()}
    s_dis = {k: arrs[k].std(axis=0).astype(np.float32) for k in arrs}
    return {"per_modality": arrs, "s_dis": s_dis}, models


def _fit_modality_quantiles(
    X_fit: dict[str, np.ndarray],
    y_fit: np.ndarray,
    X_tune: dict[str, np.ndarray],
    X_cal: dict[str, np.ndarray],
    X_test: dict[str, np.ndarray],
    seed: int,
    alpha: float,
    n_estimators: int,
    max_depth: int,
    device: str,
) -> tuple[dict[str, np.ndarray], dict[str, list[xgb.XGBRegressor]]]:
    preds = {"tune": [], "calib": [], "test": []}
    models: dict[str, list[xgb.XGBRegressor]] = {}
    for mod in MODALITIES:
        if X_fit[mod].shape[1] == 0:
            continue
        q_models = _fit_xgb_quantile(
            X_fit[mod],
            y_fit,
            seed=(
                seed
                + PREDAGN_SEED_POLICY["modality_quantile_stride"]
                * (len(models) + 1)
            ),
            alpha=alpha,
            n_estimators=n_estimators, max_depth=max_depth, device=device,
        )
        models[mod] = q_models
        preds["tune"].append(_predict_quantile(q_models, X_tune[mod]))
        preds["calib"].append(_predict_quantile(q_models, X_cal[mod]))
        preds["test"].append(_predict_quantile(q_models, X_test[mod]))
    if not preds["test"]:
        raise ValueError("no modality quantile predictors were fitted")
    return {k: np.stack(v, axis=0).astype(np.float32) for k, v in preds.items()}, models


def _normalize_disagreement(s_tune: np.ndarray, s_cal: np.ndarray, s_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    q25, q75 = np.quantile(
        s_tune,
        PREDAGN_IMPLEMENTATION_POLICY["robust_scale_quantiles"],
    )
    scale = float(q75 - q25)
    scale_kind = "iqr"
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = float(np.std(s_tune))
        scale_kind = "std"
    if not np.isfinite(scale) or scale <= 1e-8:
        scale = 1.0
        scale_kind = "unit"
    info = {
        "raw_tune_mean": float(np.mean(s_tune)),
        "raw_calib_mean": float(np.mean(s_cal)),
        "raw_test_mean": float(np.mean(s_test)),
        "raw_test_std": float(np.std(s_test)),
        "scale": scale,
        "scale_kind": scale_kind,
    }
    return (
        (s_tune / scale).astype(np.float32),
        (s_cal / scale).astype(np.float32),
        (s_test / scale).astype(np.float32),
        info,
    )


def _tune_alpha01(scores_tune: np.ndarray, s_dis_tune: np.ndarray, alpha: float):
    """Choose a fixed scale before final calibration.

    The objective is a tune-fold interval-width proxy: q_hat times mean scale.
    """
    # Pure-expansion scores (paper eq. 5): a no-op for absolute residuals; for
    # CQR scores it stops a negative quantile from contracting more at larger
    # disagreement. Clip only (no dtype change), so output is bit-identical to
    # the signed rule whenever all scores are nonnegative.
    scores_tune = np.maximum(scores_tune, 0.0)
    best = (0.0, np.inf, None)
    scale_config = PREDAGN_PAPER_RUN_CONFIG["weighted_scale"]
    a0_grid = np.asarray(
        scale_config["a0_source_grid"], dtype=np.float64
    )
    a1_grid = np.asarray(
        scale_config["a1_source_grid"], dtype=np.float64
    )
    # Multiplying (a0, a1) by a common positive constant leaves every final
    # interval unchanged because the conformal quantile rescales inversely.
    # Search each unique gamma=a1/a0 once in the canonical a0=1 form.
    gamma_grid = sorted({float(a1 / a0) for a0 in a0_grid for a1 in a1_grid})
    for gamma in gamma_grid:
        scale = np.sqrt(1.0 + gamma * (s_dis_tune ** 2))
        scaled = scores_tune / np.maximum(scale, 1e-12)
        q_hat = conformal_quantile(scaled, alpha)
        width_proxy = float(q_hat * np.mean(scale))
        if np.isfinite(width_proxy) and width_proxy < best[1]:
            best = (gamma, width_proxy, float(q_hat))
    return {
        "a0": 1.0,
        "a1": best[0],
        "gamma": best[0],
        "width_proxy": best[1],
        "q_tune": best[2],
        "gamma_grid": gamma_grid,
        "canonicalization": "a0=1; equivalent common scale factors removed",
    }


DIFFICULTY_MODEL_CONFIG = PREDAGN_PAPER_RUN_CONFIG["difficulty_model"]


def _fit_difficulty_scale(
    X_tune: np.ndarray,
    residual_tune: np.ndarray,
    X_cal: np.ndarray,
    X_test: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Fit a positive XGBoost-learned scale using tune-fold data only.

    The final-calibration labels are never passed to this routine. Thus the
    fitted scale is part of D_pre and normalizing conformal scores by it leaves
    the usual split-conformal validity argument unchanged.
    """
    residual = np.maximum(np.asarray(residual_tune, dtype=np.float64), 0.0)
    floor = float(DIFFICULTY_MODEL_CONFIG["target_floor"])
    target = np.log(np.maximum(residual, floor))
    model = xgb.XGBRegressor(
        n_estimators=DIFFICULTY_MODEL_CONFIG["n_estimators"],
        max_depth=DIFFICULTY_MODEL_CONFIG["max_depth"],
        learning_rate=DIFFICULTY_MODEL_CONFIG["learning_rate"],
        subsample=DIFFICULTY_MODEL_CONFIG["subsample"],
        colsample_bytree=DIFFICULTY_MODEL_CONFIG["colsample_bytree"],
        objective=PAPER_XGB_POINT_OBJECTIVE,
        tree_method=DIFFICULTY_MODEL_CONFIG["tree_method"],
        device=DIFFICULTY_MODEL_CONFIG["device"],
        n_jobs=DIFFICULTY_MODEL_CONFIG["n_jobs"],
        random_state=seed,
    )
    model.fit(X_tune, target, verbose=False)

    def predict_scale(X: np.ndarray) -> np.ndarray:
        log_scale = np.asarray(model.predict(X), dtype=np.float64)
        return np.exp(np.clip(log_scale, -20.0, 20.0))

    scale_tune = predict_scale(X_tune)
    scale_cal = predict_scale(X_cal)
    scale_test = predict_scale(X_test)
    normalizer = float(np.median(scale_tune))
    if not np.isfinite(normalizer) or normalizer <= 0:
        raise ValueError("learned scale model produced a non-positive normalizer")
    scale_tune = (scale_tune / normalizer).astype(np.float32)
    scale_cal = (scale_cal / normalizer).astype(np.float32)
    scale_test = (scale_test / normalizer).astype(np.float32)
    for name, values in (
        ("tune", scale_tune),
        ("calibration", scale_cal),
        ("test", scale_test),
    ):
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError(f"learned scale model produced invalid {name} scales")
    info = {
        **DIFFICULTY_MODEL_CONFIG,
        "seed": int(seed),
        "fit_fold": "tune",
        "uses_final_calibration_labels": False,
        "validity": "scale fixed in D_pre before final calibration",
        "normalizer": normalizer,
        "scale_tune_mean": float(np.mean(scale_tune)),
        "scale_calib_mean": float(np.mean(scale_cal)),
        "scale_test_mean": float(np.mean(scale_test)),
        "scale_test_min": float(np.min(scale_test)),
        "scale_test_max": float(np.max(scale_test)),
    }
    return scale_tune, scale_cal, scale_test, info


def _difficulty_weighted_interval(
    scores_cal: np.ndarray,
    base_lo_test: np.ndarray,
    base_hi_test: np.ndarray,
    scale_cal: np.ndarray,
    scale_test: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Apply a pre-fitted positive learned scale to final split conformal."""
    scores = np.maximum(np.asarray(scores_cal), 0.0)
    if np.any(np.asarray(scale_cal) <= 0) or np.any(np.asarray(scale_test) <= 0):
        raise ValueError("learned scales must be strictly positive")
    q_hat = conformal_quantile(scores / np.asarray(scale_cal), alpha)
    return (
        np.asarray(base_lo_test) - q_hat * np.asarray(scale_test),
        np.asarray(base_hi_test) + q_hat * np.asarray(scale_test),
        float(q_hat),
    )


def _interval_metrics(
    y: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    alpha: float,
    f: np.ndarray | None = None,
) -> dict:
    nciw_value, info = compute_nciw(y, lo, hi, alpha=alpha, f=f)
    return {
        "picp": float(np.mean((y >= lo) & (y <= hi))),
        "mpiw": float(info["base_mpiw"]),
        "niw": float(info["base_niw"]),
        "nciw": float(nciw_value),
        "c_test_cal": float(info["c_test_cal"]),
        "crps": float(crps_uniform(y, lo, hi)),
    }


def _bin_edges(
    s_ref: np.ndarray,
    n_bins: int = PAPER_DISAGREEMENT_BINS,
) -> np.ndarray:
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    return np.quantile(s_ref, qs)


def _bin_diagnostics(
    y: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    s_dis_test: np.ndarray,
    s_dis_ref: np.ndarray,
    n_bins: int = PAPER_DISAGREEMENT_BINS,
) -> list[dict]:
    edges = _bin_edges(s_dis_ref, n_bins=n_bins)
    bins = np.digitize(s_dis_test, edges)
    labels = ["low", "mid", "high"] if n_bins == 3 else [str(i) for i in range(n_bins)]
    rows = []
    for b in range(n_bins):
        mask = bins == b
        if not np.any(mask):
            rows.append({"bin": labels[b], "n": 0, "picp": np.nan, "mpiw": np.nan, "crps": np.nan,
                         "disagreement_mean": np.nan})
            continue
        rows.append({
            "bin": labels[b],
            "n": int(mask.sum()),
            "picp": float(np.mean((y[mask] >= lo[mask]) & (y[mask] <= hi[mask]))),
            "mpiw": float(np.mean(hi[mask] - lo[mask])),
            "crps": float(crps_uniform(y[mask], lo[mask], hi[mask])),
            "disagreement_mean": float(np.mean(s_dis_test[mask])),
        })
    return rows


def _attach_bins(metrics: dict, y: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                 s_dis_test: np.ndarray, s_dis_ref: np.ndarray) -> dict:
    metrics["bins"] = _bin_diagnostics(y, lo, hi, s_dis_test, s_dis_ref)
    return metrics


def _point_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "mae": float(mean_absolute_error(y, pred)),
        "r2": float(r2_score(y, pred)),
    }


def _bin_scales(values: np.ndarray, edges: np.ndarray, scales: np.ndarray) -> np.ndarray:
    bins = np.digitize(values, edges)
    return scales[bins].astype(np.float32)


def _tune_monotone_bin_scale(
    scores_tune: np.ndarray,
    s_dis_tune: np.ndarray,
    alpha: float,
    base_width_tune: np.ndarray | None = None,
    n_bins: int = PAPER_DISAGREEMENT_BINS,
) -> dict:
    """Tune a monotone piecewise-constant disagreement scale on tune data only."""
    # Pure-expansion scores (paper eq. 5), as in _tune_alpha01.
    scores_tune = np.maximum(scores_tune, 0.0)
    edges = _bin_edges(s_dis_tune, n_bins=n_bins)
    ratios = PREDAGN_IMPLEMENTATION_POLICY["binned_scale_ratios"]
    candidates = [np.ones(n_bins, dtype=np.float32)]
    if n_bins == 3:
        for r in ratios[1:]:
            candidates.append(np.array([1.0, np.sqrt(r), r], dtype=np.float32))
    else:
        for r in ratios[1:]:
            candidates.append(np.geomspace(1.0, r, n_bins).astype(np.float32))

    if base_width_tune is None:
        base_width_tune = np.zeros_like(scores_tune, dtype=np.float32)

    best = None
    for raw_scales in candidates:
        scale_tune = _bin_scales(s_dis_tune, edges, raw_scales)
        q_hat = conformal_quantile(scores_tune / np.maximum(scale_tune, 1e-12), alpha)
        width_proxy = float(np.mean(base_width_tune + 2.0 * q_hat * scale_tune))
        if not np.isfinite(width_proxy):
            continue
        if best is None or width_proxy < best["width_proxy"]:
            best = {
                "edges": edges.tolist(),
                "scales": [float(x) for x in raw_scales],
                "q_tune": float(q_hat),
                "width_proxy": width_proxy,
                "method": "binned_weighted_dis",
            }
    if best is None:
        best = {
            "edges": edges.tolist(),
            "scales": [1.0] * n_bins,
            "q_tune": float(conformal_quantile(scores_tune, alpha)),
            "width_proxy": float("inf"),
            "method": "binned_weighted_dis",
        }
    return best


def _calibrate_point(
    y_tune: np.ndarray,
    pred_tune: np.ndarray,
    y_cal: np.ndarray,
    pred_cal: np.ndarray,
    pred_test: np.ndarray,
    y_test: np.ndarray,
    s_dis_tune: np.ndarray,
    s_dis_cal: np.ndarray,
    s_dis_test: np.ndarray,
    alpha: float,
    difficulty_features: tuple[np.ndarray, np.ndarray, np.ndarray],
    difficulty_seed: int,
) -> dict:
    e_tune = np.abs(y_tune - pred_tune)
    e_cal = np.abs(y_cal - pred_cal)
    out: dict[str, dict] = {}

    q = conformal_quantile(e_cal, alpha)
    lo = pred_test - q
    hi = pred_test + q
    out["marginal"] = _interval_metrics(y_test, lo, hi, alpha=alpha, f=pred_test)
    out["marginal"]["info"] = {"q_hat": q, "method": "split_cp_abs"}
    _attach_bins(out["marginal"], y_test, lo, hi, s_dis_test, s_dis_tune)

    pseudo_cal = np.stack([pred_cal, pred_cal, pred_cal], axis=1)
    pseudo_test = np.stack([pred_test, pred_test, pred_test], axis=1)
    lo, hi, info = disagreement_mondrian_cqr(
        y_cal, pseudo_cal, pseudo_test, s_dis_cal, s_dis_test,
        alpha=alpha,
        n_bins=PAPER_DISAGREEMENT_BINS,
        s_dis_ref=s_dis_tune,
    )
    out["mondrian_dis"] = _interval_metrics(y_test, lo, hi, alpha=alpha, f=pred_test)
    out["mondrian_dis"]["info"] = info
    _attach_bins(out["mondrian_dis"], y_test, lo, hi, s_dis_test, s_dis_tune)

    info = _tune_alpha01(e_tune, s_dis_tune, alpha)
    scale_cal = np.sqrt(info["a0"] + info["a1"] * (s_dis_cal ** 2))
    scale_test = np.sqrt(info["a0"] + info["a1"] * (s_dis_test ** 2))
    q = conformal_quantile(e_cal / np.maximum(scale_cal, 1e-12), alpha)
    lo = pred_test - q * scale_test
    hi = pred_test + q * scale_test
    out["weighted_dis"] = _interval_metrics(y_test, lo, hi, alpha=alpha, f=pred_test)
    info = dict(info)
    info.update({"q_hat": float(q), "method": "weighted_split_cp_abs"})
    out["weighted_dis"]["info"] = info
    _attach_bins(out["weighted_dis"], y_test, lo, hi, s_dis_test, s_dis_tune)

    _, difficulty_cal, difficulty_test, info = _fit_difficulty_scale(
        difficulty_features[0],
        e_tune,
        difficulty_features[1],
        difficulty_features[2],
        seed=difficulty_seed,
    )
    lo, hi, q = _difficulty_weighted_interval(
        e_cal,
        pred_test,
        pred_test,
        difficulty_cal,
        difficulty_test,
        alpha,
    )
    out["difficulty_weighted"] = _interval_metrics(
        y_test, lo, hi, alpha=alpha, f=pred_test
    )
    info = dict(info)
    info.update({"q_hat": q, "method": "difficulty_weighted_split_cp_abs"})
    out["difficulty_weighted"]["info"] = info
    _attach_bins(
        out["difficulty_weighted"], y_test, lo, hi, s_dis_test, s_dis_tune
    )

    info = _tune_monotone_bin_scale(e_tune, s_dis_tune, alpha, base_width_tune=None)
    edges = np.asarray(info["edges"], dtype=np.float64)
    scales = np.asarray(info["scales"], dtype=np.float32)
    scale_cal = _bin_scales(s_dis_cal, edges, scales)
    scale_test = _bin_scales(s_dis_test, edges, scales)
    q = conformal_quantile(e_cal / np.maximum(scale_cal, 1e-12), alpha)
    lo = pred_test - q * scale_test
    hi = pred_test + q * scale_test
    out["binned_weighted_dis"] = _interval_metrics(y_test, lo, hi, alpha=alpha, f=pred_test)
    info = dict(info)
    info.update({"q_hat": float(q)})
    out["binned_weighted_dis"]["info"] = info
    _attach_bins(out["binned_weighted_dis"], y_test, lo, hi, s_dis_test, s_dis_tune)
    return out


def _calibrate_quantile(
    y_tune: np.ndarray,
    preds_tune: np.ndarray,
    y_cal: np.ndarray,
    preds_cal: np.ndarray,
    preds_test: np.ndarray,
    y_test: np.ndarray,
    s_dis_tune: np.ndarray,
    s_dis_cal: np.ndarray,
    s_dis_test: np.ndarray,
    alpha: float,
    difficulty_features: tuple[np.ndarray, np.ndarray, np.ndarray],
    difficulty_seed: int,
) -> dict:
    out: dict[str, dict] = {}
    lo_c, mid_c, hi_c = preds_cal[:, 0], preds_cal[:, 1], preds_cal[:, 2]
    lo_t, mid_t, hi_t = preds_test[:, 0], preds_test[:, 1], preds_test[:, 2]
    lo_u, _, hi_u = preds_tune[:, 0], preds_tune[:, 1], preds_tune[:, 2]
    scores_tune = np.maximum(lo_u - y_tune, y_tune - hi_u)
    scores_cal = np.maximum(lo_c - y_cal, y_cal - hi_c)

    q = conformal_quantile(scores_cal, alpha)
    lo = lo_t - q
    hi = hi_t + q
    out["marginal"] = _interval_metrics(y_test, lo, hi, alpha=alpha, f=mid_t)
    out["marginal"]["info"] = {"q_hat": q, "method": "cqr"}
    _attach_bins(out["marginal"], y_test, lo, hi, s_dis_test, s_dis_tune)

    lo, hi, info = disagreement_mondrian_cqr(
        y_cal, preds_cal, preds_test, s_dis_cal, s_dis_test,
        alpha=alpha,
        n_bins=PAPER_DISAGREEMENT_BINS,
        s_dis_ref=s_dis_tune,
    )
    out["mondrian_dis"] = _interval_metrics(y_test, lo, hi, alpha=alpha, f=mid_t)
    out["mondrian_dis"]["info"] = info
    _attach_bins(out["mondrian_dis"], y_test, lo, hi, s_dis_test, s_dis_tune)

    info = _tune_alpha01(scores_tune, s_dis_tune, alpha)
    scale_cal = np.sqrt(info["a0"] + info["a1"] * (s_dis_cal ** 2))
    scale_test = np.sqrt(info["a0"] + info["a1"] * (s_dis_test ** 2))
    # pure-expansion score e^+ (paper eq. 5)
    q = conformal_quantile(np.maximum(scores_cal, 0.0) / np.maximum(scale_cal, 1e-12), alpha)
    lo = lo_t - q * scale_test
    hi = hi_t + q * scale_test
    out["weighted_dis"] = _interval_metrics(y_test, lo, hi, alpha=alpha, f=mid_t)
    info = dict(info)
    info.update({"q_hat": float(q), "method": "weighted_cqr"})
    out["weighted_dis"]["info"] = info
    _attach_bins(out["weighted_dis"], y_test, lo, hi, s_dis_test, s_dis_tune)

    _, difficulty_cal, difficulty_test, info = _fit_difficulty_scale(
        difficulty_features[0],
        np.maximum(scores_tune, 0.0),
        difficulty_features[1],
        difficulty_features[2],
        seed=difficulty_seed,
    )
    lo, hi, q = _difficulty_weighted_interval(
        scores_cal,
        lo_t,
        hi_t,
        difficulty_cal,
        difficulty_test,
        alpha,
    )
    out["difficulty_weighted"] = _interval_metrics(
        y_test, lo, hi, alpha=alpha, f=mid_t
    )
    info = dict(info)
    info.update({"q_hat": q, "method": "difficulty_weighted_cqr"})
    out["difficulty_weighted"]["info"] = info
    _attach_bins(
        out["difficulty_weighted"], y_test, lo, hi, s_dis_test, s_dis_tune
    )

    info = _tune_monotone_bin_scale(scores_tune, s_dis_tune, alpha, base_width_tune=hi_u - lo_u)
    edges = np.asarray(info["edges"], dtype=np.float64)
    scales = np.asarray(info["scales"], dtype=np.float32)
    scale_cal = _bin_scales(s_dis_cal, edges, scales)
    scale_test = _bin_scales(s_dis_test, edges, scales)
    # pure-expansion score e^+ (paper eq. 5)
    q = conformal_quantile(np.maximum(scores_cal, 0.0) / np.maximum(scale_cal, 1e-12), alpha)
    lo = lo_t - q * scale_test
    hi = hi_t + q * scale_test
    out["binned_weighted_dis"] = _interval_metrics(y_test, lo, hi, alpha=alpha, f=mid_t)
    info = dict(info)
    info.update({"q_hat": float(q), "method": "binned_weighted_cqr"})
    out["binned_weighted_dis"]["info"] = info
    _attach_bins(out["binned_weighted_dis"], y_test, lo, hi, s_dis_test, s_dis_tune)
    return out


def run_one(
    dataset: str,
    seed: int,
    alpha: float,
    calib_frac: float,
    tune_frac: float,
    n_estimators: int,
    q_n_estimators: int,
    max_depth: int,
    device: str,
    out_dir: Path,
    skip_existing: bool,
    campaign_tag: str | None = None,
) -> dict:
    suffix = alpha_result_tag(alpha)
    out_path = out_dir / f"predagn_{dataset}_{suffix}_seed{seed}.json"
    run_config = predagn_run_config(
        alpha=alpha,
        calib_frac=calib_frac,
        tune_frac=tune_frac,
        n_estimators=n_estimators,
        q_n_estimators=q_n_estimators,
        max_depth=max_depth,
    )
    if skip_existing and out_path.exists():
        existing = json.loads(out_path.read_text())
        config = existing.get("config", {})
        basic_match = (
            config.get("dataset") == dataset
            and int(config.get("seed", -1)) == seed
            and config.get("campaign_tag") == campaign_tag
            and config.get("device") == device
        )
        if not basic_match:
            raise RuntimeError(
                f"--skip-existing rejected {out_path}: basic config mismatch"
            )
        validate_campaign_payloads(
            [(out_path.name, existing)],
            ROOT,
            campaign_tag=campaign_tag,
            requested_device=device,
            run_config=run_config,
            producer_threads=thread_identity(),
        )
        return existing

    t0 = time.time()
    train, test = _load_dataset(dataset)
    y_train = train["y"].astype(np.float32)
    y_test = test["y"].astype(np.float32)
    fit_idx, tune_idx, cal_idx = _split_fit_tune_cal(
        len(y_train), calib_frac=calib_frac, tune_frac=tune_frac, seed=seed
    )
    blocks = _build_blocks_split_aware(dataset, train, test, fit_idx)

    X_fit_blocks = _block_dict(blocks, fit_idx)
    X_tune_blocks = _block_dict(blocks, tune_idx)
    X_cal_blocks = _block_dict(blocks, cal_idx)
    X_test_blocks = _test_block_dict(blocks)
    y_fit = y_train[fit_idx]
    y_tune = y_train[tune_idx]
    y_cal = y_train[cal_idx]

    X_fit = _concat_nonempty([X_fit_blocks[m] for m in MODALITIES])
    X_tune = _concat_nonempty([X_tune_blocks[m] for m in MODALITIES])
    X_cal = _concat_nonempty([X_cal_blocks[m] for m in MODALITIES])
    X_test = _concat_nonempty([X_test_blocks[m] for m in MODALITIES])

    mod_pred, mod_models = _fit_modality_points(
        X_fit_blocks, y_fit, X_tune_blocks, y_tune, X_cal_blocks, X_test_blocks,
        seed=seed, n_estimators=n_estimators, max_depth=max_depth, device=device,
    )
    s_tune, s_cal, s_test, dis_info = _normalize_disagreement(
        mod_pred["s_dis"]["tune"],
        mod_pred["s_dis"]["calib"],
        mod_pred["s_dis"]["test"],
    )

    results: dict[str, dict] = {}

    point_model = _fit_xgb_point(
        X_fit, y_fit, X_tune, y_tune, seed=seed,
        n_estimators=n_estimators, max_depth=max_depth, device=device,
    )
    pred_tune = point_model.predict(X_tune).astype(np.float32)
    pred_cal = point_model.predict(X_cal).astype(np.float32)
    pred_test = point_model.predict(X_test).astype(np.float32)
    results["xgb_point"] = {
        "point": _point_metrics(y_test, pred_test),
        "calibration": _calibrate_point(
            y_tune, pred_tune, y_cal, pred_cal, pred_test, y_test,
            s_tune, s_cal, s_test, alpha,
            (X_tune, X_cal, X_test),
            seed + PREDAGN_SEED_POLICY["difficulty_xgb_point"],
        ),
    }

    q_models = _fit_xgb_quantile(
        X_fit, y_fit, seed=seed, alpha=alpha,
        n_estimators=q_n_estimators, max_depth=max_depth, device=device,
    )
    q_tune = _predict_quantile(q_models, X_tune)
    q_cal = _predict_quantile(q_models, X_cal)
    q_test = _predict_quantile(q_models, X_test)
    results["xgb_quantile"] = {
        "point": _point_metrics(y_test, q_test[:, 1]),
        "calibration": _calibrate_quantile(
            y_tune, q_tune, y_cal, q_cal, q_test, y_test,
            s_tune, s_cal, s_test, alpha,
            (X_tune, X_cal, X_test),
            seed + PREDAGN_SEED_POLICY["difficulty_xgb_quantile"],
        ),
    }

    # Source-wise predictor: tune-only ridge stacker over modality point preds.
    stack_tune = mod_pred["per_modality"]["tune"].T
    stack_cal = mod_pred["per_modality"]["calib"].T
    stack_test = mod_pred["per_modality"]["test"].T
    stacker = Ridge(alpha=PREDAGN_PAPER_RUN_CONFIG["sourcewise_ridge_alpha"])
    stacker.fit(stack_tune, y_tune)
    sw_tune = stacker.predict(stack_tune).astype(np.float32)
    sw_cal = stacker.predict(stack_cal).astype(np.float32)
    sw_test = stacker.predict(stack_test).astype(np.float32)
    results["sourcewise_stack"] = {
        "point": _point_metrics(y_test, sw_test),
        "calibration": _calibrate_point(
            y_tune, sw_tune, y_cal, sw_cal, sw_test, y_test,
            s_tune, s_cal, s_test, alpha,
            (X_tune, X_cal, X_test),
            seed + PREDAGN_SEED_POLICY["difficulty_sourcewise_stack"],
        ),
        "stacker": {
            "coef": [float(x) for x in stacker.coef_.ravel()],
            "intercept": float(stacker.intercept_),
            "modalities": list(mod_models),
        },
    }

    # Source-wise quantile stacker: per-modality quantile predictors are combined
    # by convex inverse-RMSE weights selected on the tune fold. The convex form
    # keeps quantile order and avoids interval amplification from unconstrained
    # linear stacking.
    mod_q, _ = _fit_modality_quantiles(
        X_fit_blocks, y_fit, X_tune_blocks, X_cal_blocks, X_test_blocks,
        seed=seed, alpha=alpha, n_estimators=q_n_estimators,
        max_depth=max_depth, device=device,
    )
    med_tune = mod_q["tune"][:, :, 1]
    rmse_by_mod = np.sqrt(np.mean((med_tune - y_tune[None, :]) ** 2, axis=1))
    inv = 1.0 / np.maximum(rmse_by_mod, 1e-6)
    q_weights = (inv / np.sum(inv)).astype(np.float32)

    def combine_q(arr: np.ndarray) -> np.ndarray:
        # arr shape: (K, n, 3). Convex weights preserve quantile order.
        combined = np.tensordot(q_weights, arr, axes=(0, 0))
        return np.sort(combined.astype(np.float32), axis=1)

    swq_tune = combine_q(mod_q["tune"])
    swq_cal = combine_q(mod_q["calib"])
    swq_test = combine_q(mod_q["test"])
    results["sourcewise_quantile"] = {
        "point": _point_metrics(y_test, swq_test[:, 1]),
        "calibration": _calibrate_quantile(
            y_tune, swq_tune, y_cal, swq_cal, swq_test, y_test,
            s_tune, s_cal, s_test, alpha,
            (X_tune, X_cal, X_test),
            seed + PREDAGN_SEED_POLICY["difficulty_sourcewise_quantile"],
        ),
        "stacker": {
            "weights": [float(x) for x in q_weights.ravel()],
            "rmse_by_modality": [float(x) for x in rmse_by_mod.ravel()],
            "modalities": list(mod_models),
        },
    }

    out = {
        "config": {
            "dataset": dataset,
            "seed": seed,
            "alpha": alpha,
            "calib_frac": calib_frac,
            "tune_frac": tune_frac,
            "n_estimators": n_estimators,
            "q_n_estimators": q_n_estimators,
            "max_depth": max_depth,
            "device": device,
            "campaign_tag": campaign_tag,
            "run_config": run_config,
        },
        "shapes": {
            "n_fit": int(len(fit_idx)),
            "n_tune": int(len(tune_idx)),
            "n_calib": int(len(cal_idx)),
            "n_test": int(len(y_test)),
            "tab_dim": int(blocks["X_tab_tr"].shape[1]),
            "text_dim": int(blocks["X_text_tr"].shape[1]),
            "image_dim": int(blocks["X_image_tr"].shape[1]),
            "n_modalities": int(len(mod_models)),
        },
        "enc_info": blocks["enc_info"],
        "disagreement": dis_info | {
            "tune_mean": float(np.mean(s_tune)),
            "calib_mean": float(np.mean(s_cal)),
            "test_mean": float(np.mean(s_test)),
            "test_std": float(np.std(s_test)),
        },
        "results": results,
        "elapsed_seconds": float(time.time() - t0),
    }
    attach_run_provenance(
        out,
        ROOT,
        seed=seed,
        campaign_tag=campaign_tag,
        requested_device=device,
    )
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    os.replace(tmp, out_path)
    return out


EXPECTED_BASES = PREDAGN_BASE_PREDICTORS
EXPECTED_CALIBRATIONS = tuple(PREDAGN_PAPER_RUN_CONFIG["calibrations"])


def _read_rows(
    out_dir: Path,
    datasets: list[str],
    seeds: list[int],
    alpha: float,
    *,
    strict: bool = True,
    campaign_tag: str | None = None,
    requested_device: str = PAPER_CPU_DEVICE,
    run_config: dict | None = None,
) -> pd.DataFrame:
    suffix = alpha_result_tag(alpha)
    rows = []
    missing_paths = []
    invalid = []
    campaign_payloads = []
    for dataset in datasets:
        for seed in seeds:
            path = out_dir / f"predagn_{dataset}_{suffix}_seed{seed}.json"
            if not path.exists():
                missing_paths.append(path.name)
                continue
            payload = json.loads(path.read_text())
            campaign_payloads.append((path.name, payload))
            config = payload.get("config", {})
            if (
                config.get("dataset") != dataset
                or int(config.get("seed", -1)) != seed
                or abs(float(config.get("alpha", -1)) - alpha) > 1e-12
                or (
                    campaign_tag is not None
                    and config.get("campaign_tag") != campaign_tag
                )
            ):
                invalid.append(f"{path.name}: config mismatch")
            if strict:
                result_keys = set(payload.get("results", {}))
                expected_bases = set(EXPECTED_BASES)
                if result_keys != expected_bases:
                    invalid.append(
                        f"{path.name}: base grid differs; "
                        f"missing={sorted(expected_bases - result_keys)}, "
                        f"unexpected={sorted(result_keys - expected_bases)}"
                    )
                for base in EXPECTED_BASES:
                    calibrations = set(
                        payload.get("results", {})
                        .get(base, {})
                        .get("calibration", {})
                    )
                    expected_calibrations = set(EXPECTED_CALIBRATIONS)
                    if calibrations != expected_calibrations:
                        invalid.append(
                            f"{path.name}/{base}: calibration grid differs; "
                            f"missing={sorted(expected_calibrations - calibrations)}, "
                            f"unexpected={sorted(calibrations - expected_calibrations)}"
                        )
            for base, base_payload in payload["results"].items():
                point = base_payload["point"]
                for calib, metrics in base_payload["calibration"].items():
                    row = {
                        "dataset": dataset,
                        "seed": seed,
                        "base": base,
                        "calibration": calib,
                        "r2": point["r2"],
                        "rmse": point["rmse"],
                        "mae": point["mae"],
                    }
                    for key in ("picp", "mpiw", "niw", "nciw", "c_test_cal", "crps"):
                        row[key] = metrics[key]
                    if strict:
                        required_numeric = (
                            "r2",
                            "rmse",
                            "mae",
                            "picp",
                            "mpiw",
                            "niw",
                            "nciw",
                            "c_test_cal",
                            "crps",
                        )
                        try:
                            finite = all(
                                np.isfinite(float(row[key]))
                                for key in required_numeric
                            )
                        except (KeyError, TypeError, ValueError):
                            finite = False
                        if not finite:
                            invalid.append(
                                f"{path.name}/{base}/{calib}: "
                                "non-finite or non-numeric required metric"
                            )
                    info = metrics.get("info", {})
                    row["a0"] = info.get("a0")
                    row["a1"] = info.get("a1")
                    row["q_hat"] = info.get("q_hat")
                    rows.append(row)
    if strict and (missing_paths or invalid):
        details = []
        if missing_paths:
            details.append("missing files: " + ", ".join(missing_paths))
        if invalid:
            details.append("invalid files: " + "; ".join(invalid))
        raise ValueError("incomplete predictor-agnostic grid: " + " | ".join(details))
    if strict:
        validate_campaign_payloads(
            campaign_payloads,
            ROOT,
            campaign_tag=campaign_tag,
            requested_device=requested_device,
            run_config=run_config,
        )
    return pd.DataFrame(rows)


def _read_bin_rows(
    out_dir: Path,
    datasets: list[str],
    seeds: list[int],
    alpha: float,
) -> pd.DataFrame:
    suffix = alpha_result_tag(alpha)
    rows = []
    for dataset in datasets:
        for seed in seeds:
            path = out_dir / f"predagn_{dataset}_{suffix}_seed{seed}.json"
            if not path.exists():
                continue
            payload = json.loads(path.read_text())
            for base, base_payload in payload["results"].items():
                for calib, metrics in base_payload["calibration"].items():
                    for b in metrics.get("bins", []):
                        row = {
                            "dataset": dataset,
                            "seed": seed,
                            "base": base,
                            "calibration": calib,
                            "bin": b["bin"],
                        }
                        for key in ("n", "picp", "mpiw", "crps", "disagreement_mean"):
                            row[key] = b.get(key)
                        rows.append(row)
    return pd.DataFrame(rows)


def aggregate(
    out_dir: Path,
    datasets: list[str],
    seeds: list[int],
    alpha: float,
    *,
    strict: bool = True,
    expected_seed_count: int = len(PAPER_SEEDS),
    campaign_tag: str | None = None,
    calib_frac: float = PREDAGN_PAPER_RUN_CONFIG["split"]["calib_frac"],
    tune_frac: float = PREDAGN_PAPER_RUN_CONFIG["split"]["tune_frac"],
    n_estimators: int = PREDAGN_PAPER_RUN_CONFIG["point_xgb"]["n_estimators"],
    q_n_estimators: int = PREDAGN_PAPER_RUN_CONFIG[
        "quantile_xgb"
    ]["n_estimators"],
    max_depth: int = PREDAGN_PAPER_RUN_CONFIG["point_xgb"]["max_depth"],
    device: str = PAPER_CPU_DEVICE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if strict:
        expected_seeds(seeds, expected_seed_count=expected_seed_count)
        datasets, seeds = map(
            list, paper_campaign_scope(datasets, seeds)
        )
    run_config = predagn_run_config(
        alpha=alpha,
        calib_frac=calib_frac,
        tune_frac=tune_frac,
        n_estimators=n_estimators,
        q_n_estimators=q_n_estimators,
        max_depth=max_depth,
    )
    rows = _read_rows(
        out_dir,
        datasets,
        seeds,
        alpha,
        strict=strict,
        campaign_tag=campaign_tag,
        requested_device=device,
        run_config=run_config,
    )
    if strict and campaign_tag is not None:
        suffix = alpha_result_tag(alpha)
        first_payload = json.loads(
            (
                out_dir
                / f"predagn_{datasets[0]}_{suffix}_seed{seeds[0]}.json"
            ).read_text(encoding="utf-8")
        )
        write_campaign_manifest(
            out_dir,
            ROOT,
            campaign_tag=campaign_tag,
            datasets=datasets,
            seeds=seeds,
            methods=EXPECTED_BASES,
            requested_device=device,
            run_config=run_config,
            producer_threads=first_payload["provenance"]["threads"],
        )
    if rows.empty:
        raise FileNotFoundError(f"no predagn outputs found in {out_dir}")
    per_seed_path = out_dir / "predagn_ablation_per_seed.csv"
    rows.to_csv(per_seed_path, index=False)

    summary = summarize_predagn(rows)

    summary_path = out_dir / "predagn_ablation_summary.csv"
    summary.to_csv(summary_path, index=False)

    bin_rows = _read_bin_rows(out_dir, datasets, seeds, alpha)
    if not bin_rows.empty:
        bin_rows.to_csv(out_dir / "predagn_ablation_bins.csv", index=False)
        bin_summary = summarize_predagn_bins(bin_rows)
        bin_summary.to_csv(out_dir / "predagn_ablation_bins_summary.csv", index=False)
    return rows, summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(PAPER_SEEDS))
    ap.add_argument("--alpha", type=float, default=PREDAGN_PAPER_RUN_CONFIG["alpha"])
    ap.add_argument(
        "--calib-frac",
        type=float,
        default=PREDAGN_PAPER_RUN_CONFIG["split"]["calib_frac"],
    )
    ap.add_argument(
        "--tune-frac",
        type=float,
        default=PREDAGN_PAPER_RUN_CONFIG["split"]["tune_frac"],
    )
    ap.add_argument(
        "--n-estimators",
        type=int,
        default=PREDAGN_PAPER_RUN_CONFIG["point_xgb"]["n_estimators"],
    )
    ap.add_argument(
        "--q-n-estimators",
        type=int,
        default=PREDAGN_PAPER_RUN_CONFIG["quantile_xgb"]["n_estimators"],
    )
    ap.add_argument(
        "--max-depth",
        type=int,
        default=PREDAGN_PAPER_RUN_CONFIG["point_xgb"]["max_depth"],
    )
    ap.add_argument(
        "--device",
        default=os.environ.get("XGB_DEVICE", PAPER_CPU_DEVICE),
    )
    ap.add_argument("--out-dir", type=Path, default=RESULTS)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument(
        "--campaign-tag",
        default=None,
        help="immutable campaign namespace recorded in every JSON and manifest",
    )
    ap.add_argument(
        "--paper-run",
        action="store_true",
        help="require an explicit campaign tag and the complete paper seed grid",
    )
    ap.add_argument("--campaign-datasets", nargs="+", default=None)
    ap.add_argument("--campaign-seeds", nargs="+", type=int, default=None)
    ap.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="exploratory only: aggregate a partial grid",
    )
    ap.add_argument(
        "--expected-seed-count", type=int, default=len(PAPER_SEEDS)
    )
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    campaign_tag = validate_campaign_tag(
        args.campaign_tag, required=args.paper_run
    )
    run_config = predagn_run_config(
        alpha=args.alpha,
        calib_frac=args.calib_frac,
        tune_frac=args.tune_frac,
        n_estimators=args.n_estimators,
        q_n_estimators=args.q_n_estimators,
        max_depth=args.max_depth,
    )
    if args.paper_run:
        try:
            require_canonical_run_config("predagn", run_config)
        except ValueError as error:
            ap.error(str(error))
    if campaign_tag is not None:
        if args.paper_run:
            campaign_datasets, campaign_seeds = paper_campaign_scope(
                args.datasets,
                args.seeds,
                campaign_datasets=args.campaign_datasets,
                campaign_seeds=args.campaign_seeds,
            )
        else:
            campaign_datasets, campaign_seeds = args.datasets, args.seeds
        if not args.aggregate_only:
            write_campaign_manifest(
                args.out_dir,
                ROOT,
                campaign_tag=campaign_tag,
                datasets=campaign_datasets,
                seeds=campaign_seeds,
                methods=EXPECTED_BASES,
                requested_device=args.device,
                run_config=run_config,
                producer_threads=thread_identity(),
            )

    if not args.aggregate_only:
        for dataset in args.datasets:
            for seed in args.seeds:
                try:
                    print(f"\n=== predagn dataset={dataset} seed={seed} alpha={args.alpha} ===", flush=True)
                    out = run_one(
                        dataset=dataset,
                        seed=seed,
                        alpha=args.alpha,
                        calib_frac=args.calib_frac,
                        tune_frac=args.tune_frac,
                        n_estimators=args.n_estimators,
                        q_n_estimators=args.q_n_estimators,
                        max_depth=args.max_depth,
                        device=args.device,
                        out_dir=args.out_dir,
                        skip_existing=args.skip_existing,
                        campaign_tag=campaign_tag,
                    )
                    print(f"  done in {out['elapsed_seconds']:.1f}s", flush=True)
                except Exception:
                    traceback.print_exc()
                    raise

    if args.paper_run and (
        set(args.datasets) != set(campaign_datasets)
        or set(args.seeds) != set(campaign_seeds)
    ):
        print(
            "paper array worker completed its subset; defer aggregation until "
            "the full campaign JSON grid is consolidated"
        )
        return 0

    rows, summary = aggregate(
        args.out_dir,
        args.datasets,
        args.seeds,
        args.alpha,
        strict=not args.allow_incomplete,
        expected_seed_count=args.expected_seed_count,
        campaign_tag=campaign_tag,
        calib_frac=args.calib_frac,
        tune_frac=args.tune_frac,
        n_estimators=args.n_estimators,
        q_n_estimators=args.q_n_estimators,
        max_depth=args.max_depth,
        device=args.device,
    )
    print("\n=== per-seed rows ===")
    print(rows.head(12).to_string(index=False))
    print("\n=== summary ===")
    cols = [
        "dataset", "base", "calibration", "n_seeds", "picp_mean", "mpiw_mean",
        "nciw_mean", "crps_mean", "delta_mpiw_vs_marginal_mean",
        "delta_crps_vs_marginal_mean",
    ]
    show = [c for c in cols if c in summary.columns]
    print(summary[show].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
