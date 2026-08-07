"""Normalized Calibrated Interval Width (NCIW).

Source: Unco3892/clear, src/clear/metrics.py (compute_nciw).
Paper and project page: https://unco3892.github.io/clear/

For two methods at slightly different actual coverages, raw MPIW comparisons
are unfair: an undercovering method gets unearned width credit. NCIW finds
the smallest scaling factor c_test_cal such that intervals scaled around the
center f exactly hit the target (1 - alpha) coverage on the test set, then
returns NIW (= MPIW / range(y_test)) at that calibrated width. If a supplied
center falls outside its interval, the interval midpoint is used for that row.
All methods end up reported at the same effective coverage.

Useful reported columns: PICP, MPIW, NIW, NCIW, c_test_cal.
"""

from __future__ import annotations

import numpy as np


def validate_intervals(lower, upper) -> tuple[np.ndarray, np.ndarray]:
    """Return flat interval arrays and reject inverted endpoint intervals."""
    lo = np.asarray(lower).flatten()
    hi = np.asarray(upper).flatten()
    if lo.shape != hi.shape:
        raise ValueError(f"lower and upper must have the same shape, got {lo.shape} and {hi.shape}")
    bad = lo > hi
    if np.any(bad):
        idx = int(np.flatnonzero(bad)[0])
        raise ValueError(f"inverted interval at index {idx}: lower={lo[idx]}, upper={hi[idx]}")
    return lo, hi


def picp(y_true, lower, upper) -> float:
    y = np.asarray(y_true)
    lo, hi = validate_intervals(lower, upper)
    return float(np.mean((y >= lo) & (y <= hi)))


def mpiw(lower, upper) -> float:
    lo, hi = validate_intervals(lower, upper)
    return float(np.mean(hi - lo))


def niw(y_true, lower, upper) -> float:
    width = mpiw(lower, upper)
    rng = float(np.max(y_true) - np.min(y_true))
    return width if rng == 0 else width / rng


def compute_nciw(y_true, lower, upper, alpha: float = 0.05, f=None) -> tuple[float, dict]:
    """Test-time calibrated normalized interval width.

    Scales intervals as:
        lo' = f - c * (f - lower)
        hi' = f + c * (upper - f)
    Finds the smallest c such that PICP >= 1 - alpha on the test set, then
    returns NIW evaluated at that c.

    Args:
        y_true: shape (n,)
        lower, upper: shape (n,)
        alpha: miscoverage level (e.g. 0.05 for 95% target)
        f: central estimate per row. If None, midpoint of (lower, upper).
           Values outside [lower, upper] are replaced by the midpoint.

    Returns:
        (nciw_value, info) where info always has the keys c_test_cal, base_niw,
        base_mpiw, y_range, and center_midpoint_fallbacks; the infinite-interval
        early return additionally carries empty sorted_c and PICPs arrays.
    """
    y_true = np.asarray(y_true).flatten()
    lower, upper = validate_intervals(lower, upper)
    if y_true.shape != lower.shape:
        raise ValueError(f"y_true and intervals must have the same shape, got {y_true.shape} and {lower.shape}")

    if np.any(np.isinf(lower)) or np.any(np.isinf(upper)):
        return float("inf"), {
            "c_test_cal": 0.0,
            "sorted_c": np.array([]),
            "PICPs": np.array([]),
            "base_niw": float("inf"),
            "base_mpiw": float("inf"),
            "y_range": float(np.max(y_true) - np.min(y_true)),
            "center_midpoint_fallbacks": 0,
        }

    midpoint = (lower + upper) / 2.0
    if f is None:
        f = midpoint
        fallback_count = 0
    else:
        f = np.asarray(f).flatten()
        if f.shape != lower.shape:
            raise ValueError(f"f and intervals must have the same shape, got {f.shape} and {lower.shape}")
        use_midpoint = (~np.isfinite(f)) | (f < lower) | (f > upper)
        fallback_count = int(np.sum(use_midpoint))
        if fallback_count:
            f = f.copy()
            f[use_midpoint] = midpoint[use_midpoint]

    l_base = f - lower
    u_base = upper - f

    c_candidates = np.zeros_like(y_true, dtype=float)
    for i in range(len(y_true)):
        if y_true[i] < f[i]:
            c_candidates[i] = (f[i] - y_true[i]) / l_base[i] if l_base[i] > 0 else float("inf")
        elif y_true[i] > f[i]:
            c_candidates[i] = (y_true[i] - f[i]) / u_base[i] if u_base[i] > 0 else float("inf")
        else:
            c_candidates[i] = 0.0

    sorted_c = np.sort(c_candidates)
    n = len(sorted_c)
    PICPs = np.arange(1, n + 1) / n
    target = 1.0 - alpha
    idx = int(np.searchsorted(PICPs, target))
    c_test_cal = float(sorted_c[idx]) if idx < n else float(sorted_c[-1])

    base_niw = niw(y_true, lower, upper)
    nciw_value = float("inf") if not np.isfinite(c_test_cal) else c_test_cal * base_niw

    return float(nciw_value), {
        "c_test_cal": c_test_cal,
        "base_niw": base_niw,
        "base_mpiw": mpiw(lower, upper),
        "y_range": float(np.max(y_true) - np.min(y_true)),
        "center_midpoint_fallbacks": fallback_count,
    }


# ---------------------------------------------------------------------------
# convenience: compute the full set of comparison metrics for a given method


def evaluate_with_nciw(y_true, lower, upper, alpha: float = 0.05, f=None) -> dict:
    """Return PICP, MPIW, NIW, NCIW, c_test_cal in one call."""
    nciw_v, info = compute_nciw(y_true, lower, upper, alpha=alpha, f=f)
    return {
        "picp": picp(y_true, lower, upper),
        "mpiw": info["base_mpiw"],
        "niw": info["base_niw"],
        "nciw": nciw_v,
        "c_test_cal": info["c_test_cal"],
        "y_range": info["y_range"],
    }


if __name__ == "__main__":
    # smoke test
    rng = np.random.default_rng(0)
    y = rng.normal(size=1000)
    half = 1.96
    lo, hi = y - half + 0.1 * rng.normal(size=1000), y + half + 0.1 * rng.normal(size=1000)
    print(evaluate_with_nciw(y, lo, hi, alpha=0.05))
