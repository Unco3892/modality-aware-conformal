"""Conformal calibration utilities for the heterogeneous-modality study.

CQR (split conformal on quantile residuals) and a disagreement-Mondrian variant
that bins calibration points by the per-row standard deviation of per-modality
solo-point predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------


def validate_intervals(lo, hi) -> tuple[np.ndarray, np.ndarray]:
    """Return interval arrays and reject inverted endpoint intervals."""
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    if lo.shape != hi.shape:
        raise ValueError(f"lo and hi must have the same shape, got {lo.shape} and {hi.shape}")
    bad = lo > hi
    if np.any(bad):
        idx = int(np.flatnonzero(bad)[0])
        raise ValueError(f"inverted interval at index {idx}: lo={lo.flat[idx]}, hi={hi.flat[idx]}")
    return lo, hi


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample split-conformal quantile.

    Uses the kth order statistic with k=ceil((n+1)(1-alpha)). If k>n, the
    conservative split-conformal interval is unbounded.
    """
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    if n == 0:
        return float("inf")
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return float("inf")
    return float(np.partition(scores, k - 1)[k - 1])


def cqr_calibrate(
    y_calib: np.ndarray,
    preds_calib: np.ndarray,
    preds_test: np.ndarray,
    alpha: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Apply split-CQR to (lo, point, hi) predictions.

    Each row of preds_* is (lo, point, hi). q_hat is the (1-alpha)-quantile of
    nonconformity scores ``s_i = max(lo_i - y_i, y_i - hi_i)`` on calibration.
    """
    lo_v, _, hi_v = preds_calib[:, 0], preds_calib[:, 1], preds_calib[:, 2]
    lo_t, _, hi_t = preds_test[:, 0], preds_test[:, 1], preds_test[:, 2]
    scores = np.maximum(lo_v - y_calib, y_calib - hi_v)
    q_hat = conformal_quantile(scores, alpha)
    return lo_t - q_hat, hi_t + q_hat, {"q_hat": q_hat, "method": "cqr"}


def disagreement_score(per_modality_points: np.ndarray) -> np.ndarray:
    """per_modality_points: (K, n) -> per-row std across modalities, shape (n,)."""
    return per_modality_points.std(axis=0).astype(np.float32)


def disagreement_mondrian_cqr(
    y_calib: np.ndarray,
    preds_calib: np.ndarray,
    preds_test: np.ndarray,
    s_dis_calib: np.ndarray,
    s_dis_test: np.ndarray,
    alpha: float = 0.1,
    n_bins: int = 3,
    s_dis_ref: Optional[np.ndarray] = None,
    edges: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Mondrian CQR with bins defined by quantiles of the disagreement score.

    For formal finite-stratum coverage, pass ``s_dis_ref`` or explicit
    ``edges`` computed before final calibration. If neither is supplied, the
    function falls back to calibration-side disagreement as a legacy behavior.
    """
    if edges is None:
        ref = s_dis_calib if s_dis_ref is None else np.asarray(s_dis_ref)
        qs = np.linspace(0, 1, n_bins + 1)[1:-1]
        edges = np.quantile(ref, qs)
    else:
        edges = np.asarray(edges, dtype=np.float64)
    cal_bin = np.digitize(s_dis_calib, edges)
    test_bin = np.digitize(s_dis_test, edges)

    lo_v, _, hi_v = preds_calib[:, 0], preds_calib[:, 1], preds_calib[:, 2]
    lo_t, _, hi_t = preds_test[:, 0], preds_test[:, 1], preds_test[:, 2]
    cal_scores = np.maximum(lo_v - y_calib, y_calib - hi_v)

    out_lo = lo_t.copy()
    out_hi = hi_t.copy()
    info = {"q_hats": {}, "bin_sizes_calib": {}, "bin_sizes_test": {}, "edges": edges.tolist(), "method": "mondrian_dis"}
    for b in range(n_bins):
        mask_v = cal_bin == b
        mask_t = test_bin == b
        n_b = int(mask_v.sum())
        info["bin_sizes_calib"][int(b)] = n_b
        info["bin_sizes_test"][int(b)] = int(mask_t.sum())
        q_b = conformal_quantile(cal_scores[mask_v], alpha)
        info["q_hats"][int(b)] = q_b
        out_lo[mask_t] = lo_t[mask_t] - q_b
        out_hi[mask_t] = hi_t[mask_t] + q_b
    return out_lo, out_hi, info


# ---------------------------------------------------------------------------


def picp(y, lo, hi) -> float:
    lo, hi = validate_intervals(lo, hi)
    return float(np.mean((y >= lo) & (y <= hi)))


def mpiw(lo, hi) -> float:
    lo, hi = validate_intervals(lo, hi)
    return float(np.mean(hi - lo))


def crps_uniform(y, lo, hi) -> float:
    """CRPS for the uniform-on-[lo, hi] approximation of the predictive CDF."""
    y = np.asarray(y, dtype=np.float64)
    lo, hi = validate_intervals(lo, hi)
    if np.any(~np.isfinite(lo)) or np.any(~np.isfinite(hi)):
        return float("inf")
    d = hi - lo
    crps_zero = np.abs(y - lo)
    t_lo = ((lo + hi) / 2 - y) - d / 6
    t_hi = (y - (lo + hi) / 2) - d / 6
    t_in = np.zeros_like(d)
    np.divide(
        (y - lo) ** 2 + (hi - y) ** 2,
        2 * d,
        out=t_in,
        where=d > 0,
    )
    t_in -= d / 6
    out = np.where(d == 0, crps_zero,
                   np.where(y < lo, t_lo, np.where(y > hi, t_hi, t_in)))
    return float(np.mean(out))


def report(y, lo, hi) -> dict:
    return {
        "picp": picp(y, lo, hi),
        "mpiw": mpiw(lo, hi),
        "crps_uniform": crps_uniform(y, lo, hi),
    }
