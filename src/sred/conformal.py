"""Post-hoc conformal calibration over SEMF Monte Carlo samples.

Given calibration samples ``S_valid : (n_valid, R)`` and test samples
``S_test : (n_test, R)`` from SEMF.infer_semf(return_type='interval'), plus
true ``y_valid``, ``y_test``, this module produces three families of
prediction intervals at level 1 - alpha:

    1. ``cqr``      — split-conformalised quantile regression (baseline)
    2. ``density``  — density / negative-log-density conformal score (HPD-style)
                      via 1D Gaussian KDE over the sample distribution.
    3. ``mondrian`` — group-conditional split-CQR. Strata supplied as integer
                      labels per row; stratification by modality-availability
                      pattern is the intended use.

Each routine returns ``(lower, upper, info)`` where ``info`` includes the
conformity quantile q_hat and per-stratum sizes for Mondrian.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.stats import gaussian_kde


# ---------------------------------------------------------------------------
# split-CQR (reproduced for self-containment)


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    if n == 0:
        return float("inf")
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return float("inf")
    return float(np.partition(scores, k - 1)[k - 1])


def cqr(
    y_valid: np.ndarray,
    samples_valid: np.ndarray,
    samples_test: np.ndarray,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, dict]:
    lo_q = (alpha / 2) * 100
    hi_q = (1 - alpha / 2) * 100
    l_v = np.percentile(samples_valid, lo_q, axis=1)
    u_v = np.percentile(samples_valid, hi_q, axis=1)
    l_t = np.percentile(samples_test, lo_q, axis=1)
    u_t = np.percentile(samples_test, hi_q, axis=1)

    scores = np.maximum(l_v - y_valid, y_valid - u_v)
    q_hat = conformal_quantile(scores, alpha)

    return l_t - q_hat, u_t + q_hat, {"q_hat": q_hat, "method": "cqr"}


# ---------------------------------------------------------------------------
# density-conformal (negative log density score; HPD-style intervals)


def _kdes(samples: np.ndarray, bw: float | str = "scott", jitter: float = 1e-6):
    """Build one gaussian_kde per row of `samples`; jitter to avoid singular cov."""
    n, R = samples.shape
    out: list[gaussian_kde] = []
    for i in range(n):
        s = samples[i].copy()
        if s.std() < jitter:
            s = s + np.random.normal(0, jitter, size=s.shape)
        out.append(gaussian_kde(s, bw_method=bw))
    return out


def _hpd_interval(kde: gaussian_kde, log_density_threshold: float, grid: np.ndarray) -> tuple[float, float]:
    """Convex hull of {y on grid : log p(y) >= log_density_threshold}."""
    log_p = kde.logpdf(grid)
    mask = log_p >= log_density_threshold
    if not mask.any():
        # threshold too tight; fall back to full grid range
        return float(grid.min()), float(grid.max())
    sel = grid[mask]
    return float(sel.min()), float(sel.max())


@dataclass
class DensityConformalConfig:
    n_grid: int = 801
    bw: float | str = "silverman"
    grid_pad: float = 1.0   # fraction of samples range to extend grid each side


def density_conformal(
    y_valid: np.ndarray,
    samples_valid: np.ndarray,
    samples_test: np.ndarray,
    alpha: float = 0.05,
    cfg: DensityConformalConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    cfg = cfg or DensityConformalConfig()

    # 1. Calibration: score_i = -log p_hat(y_valid_i | x_valid_i)
    kdes_valid = _kdes(samples_valid, bw=cfg.bw)
    cal_scores = np.array(
        [
            -float(np.asarray(kdes_valid[i].logpdf(y_valid[i])).item())
            for i in range(len(y_valid))
        ]
    )
    q_hat = conformal_quantile(cal_scores, alpha)
    log_density_threshold = -q_hat  # i.e. accept y if p_hat(y) >= exp(-q_hat)

    # 2. Test: HPD level set at threshold (convex hull)
    kdes_test = _kdes(samples_test, bw=cfg.bw)
    pad = cfg.grid_pad
    los, his = [], []
    for i in range(samples_test.shape[0]):
        s = samples_test[i]
        smin, smax = float(s.min()), float(s.max())
        rng = max(smax - smin, 1e-3)
        grid = np.linspace(smin - pad * rng, smax + pad * rng, cfg.n_grid)
        lo_i, hi_i = _hpd_interval(kdes_test[i], log_density_threshold, grid)
        los.append(lo_i)
        his.append(hi_i)

    return np.array(los), np.array(his), {
        "q_hat": q_hat,
        "log_density_threshold": log_density_threshold,
        "method": "density",
    }


# ---------------------------------------------------------------------------
# Mondrian / group-conditional CQR


def mondrian_cqr(
    y_valid: np.ndarray,
    samples_valid: np.ndarray,
    samples_test: np.ndarray,
    valid_strata: np.ndarray,
    test_strata: np.ndarray,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Per-stratum split-CQR: q_hat computed on calibration rows in the same stratum."""
    lo_q = (alpha / 2) * 100
    hi_q = (1 - alpha / 2) * 100
    l_v = np.percentile(samples_valid, lo_q, axis=1)
    u_v = np.percentile(samples_valid, hi_q, axis=1)
    l_t = np.percentile(samples_test, lo_q, axis=1)
    u_t = np.percentile(samples_test, hi_q, axis=1)

    out_lo = l_t.copy()
    out_hi = u_t.copy()
    q_hats: dict[int, float] = {}
    sizes: dict[int, int] = {}

    strata = np.unique(np.concatenate([valid_strata, test_strata]))
    for s in strata:
        v_mask = valid_strata == s
        t_mask = test_strata == s
        scores = np.maximum(l_v[v_mask] - y_valid[v_mask], y_valid[v_mask] - u_v[v_mask])
        q_s = conformal_quantile(scores, alpha)
        q_hats[int(s)] = q_s
        sizes[int(s)] = int(v_mask.sum())
        out_lo[t_mask] = l_t[t_mask] - q_s
        out_hi[t_mask] = u_t[t_mask] + q_s

    return out_lo, out_hi, {"q_hats": q_hats, "valid_sizes": sizes, "method": "mondrian_cqr"}


# ---------------------------------------------------------------------------
# evaluation


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


def coverage_width(y_true, lo, hi):
    lo, hi = validate_intervals(lo, hi)
    cov = float(np.mean((y_true >= lo) & (y_true <= hi)))
    w = float(np.mean(hi - lo))
    return cov, w


def crps_uniform(y_true, lo, hi):
    y = np.asarray(y_true)
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


def evaluate(y, lo, hi):
    cov, w = coverage_width(y, lo, hi)
    return {
        "picp": cov,
        "mpiw": w,
        "crps_uniform": crps_uniform(y, lo, hi),
    }
