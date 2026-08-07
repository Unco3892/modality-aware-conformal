"""Score-level simulation: fixed versus tuned disagreement scales at small N.

The calibration layer sees only (score, disagreement) pairs, so the value of
tuning the scale parameter gamma can be studied without fitting any models:

  d ~ |N(0,1)|                             (disagreement covariate)
  e = sigma(d) * |eps|,  eps ~ N(0,1)      (absolute-residual score)
  sigma(d) = sqrt(1 + gamma_true * d^2)    (true heteroscedasticity profile)

Because the noise is Gaussian and d is one-dimensional, the expected test
coverage and width of any calibrated rule are deterministic integrals over the
half-normal law of d, evaluated here by dense trapezoidal quadrature. The only
randomness is the draw of the N labeled observations each strategy consumes.

Strategies, all spending the same labeled budget N:

  marginal      a == 1; all N observations calibrate the unscaled score.
  fixed_g1      gamma = 1 on d standardized by its population interquartile
                range, a pre-specified constant; all N observations calibrate.
  tuned_split   round(0.4*N) observations tune, the rest calibrate, a share
                close to the deployed 0.15/0.35 tune-to-calibration split.
                The tuner mirrors ``run_predagn_ablation``: the
                candidate set is every unique ratio gamma = a1/a0 of the
                paper's source grids, disagreement is standardized by the
                tuning-sample interquartile range with standard-deviation and
                unit fallbacks, the objective is the tune-sample conformal
                quantile of the scaled scores times the mean tune-sample
                scale, and strict improvement over the ascending candidate
                grid retains the smallest gamma on an exact tie.
  oracle_shape  a = sigma, the conditionally valid scale; all N calibrate.

The oracle is the conditionally valid optimum, not the marginal-width optimum:
under a marginal coverage constraint, flatter scales than sigma give narrower
expected width while under-covering the rare high-d tail. The deterministic
``width_optimal_scales`` analysis in the metadata quantifies this.

Outputs (results/exploratory/gamma_sim/):
  gamma_sim_per_rep.csv.gz   one row per (gamma_true, N, strategy, replication)
  gamma_sim_summary.csv      means and spreads over replications per cell
  gamma_sim_meta.json        configuration plus the deterministic analyses

Deterministic given the seed; rerun with ``python exp_gamma_sim.py``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import norm

ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))

from calibration import conformal_quantile  # noqa: E402
from experiment_config import (  # noqa: E402
    PREDAGN_IMPLEMENTATION_POLICY,
    PREDAGN_PAPER_RUN_CONFIG,
)

OUT_DIR = ROOT / "results" / "exploratory" / "gamma_sim"

ALPHA = 0.05
GAMMAS = (0.0, 1.0, 4.0, 16.0)
SIZES = (25, 31, 50, 100, 200, 400, 800, 1600, 3200)
N_REPS = 2000
TUNE_SHARE = 0.4
SEED = 0

# The paper's candidate set: every unique ratio gamma = a1/a0 of the source
# grids, searched once in the canonical a0=1 form (run_predagn_ablation).
_SCALE_CONFIG = PREDAGN_PAPER_RUN_CONFIG["weighted_scale"]
GAMMA_GRID = tuple(sorted({
    float(a1 / a0)
    for a0 in _SCALE_CONFIG["a0_source_grid"]
    for a1 in _SCALE_CONFIG["a1_source_grid"]
}))
_ROBUST_QS = PREDAGN_IMPLEMENTATION_POLICY["robust_scale_quantiles"]

# Population IQR of |N(0,1)|, the pre-specified standardization of fixed_g1.
C0 = float(norm.ppf(0.875) - norm.ppf(0.625))

# Dense grid and half-normal weights for expectations over d. The half-normal
# mass beyond d=10 is about 1.5e-23, far below quadrature error.
_D = np.linspace(0.0, 10.0, 8001)
_W = 2.0 * norm.pdf(_D)
_W /= np.trapz(_W, _D)

# Conditional-coverage probe in the rare high-disagreement tail.
D_PROBE = 3.0


def _expect(values: np.ndarray) -> float:
    return float(np.trapz(values * _W, _D))


def _tune_scale_estimate(d_tune: np.ndarray) -> tuple[float, str]:
    """Tuning-sample IQR with the fallbacks of ``_normalize_disagreement``."""
    q_lo, q_hi = np.quantile(d_tune, _ROBUST_QS)
    scale = float(q_hi - q_lo)
    if np.isfinite(scale) and scale > 1e-8:
        return scale, "iqr"
    scale = float(np.std(d_tune))
    if np.isfinite(scale) and scale > 1e-8:
        return scale, "std"
    return 1.0, "unit"


def tune_gamma(e_tune: np.ndarray, d_tune: np.ndarray) -> tuple[float, float]:
    """The paper's tuner on one tuning sample.

    Returns the selected gamma and the estimated standardization scale. Scores
    here are absolute residuals, so the pure-expansion clip is a no-op, and
    iterating the ascending grid with strict improvement retains the smallest
    gamma on an exact tie, as in ``_tune_alpha01``.
    """
    c_hat, _ = _tune_scale_estimate(d_tune)
    s2 = (d_tune / c_hat) ** 2
    best_gamma, best_proxy = 0.0, np.inf
    for gamma in GAMMA_GRID:
        scale = np.sqrt(1.0 + gamma * s2)
        q_hat = conformal_quantile(e_tune / scale, ALPHA)
        proxy = float(q_hat * np.mean(scale))
        if np.isfinite(proxy) and proxy < best_proxy:
            best_gamma, best_proxy = float(gamma), proxy
    return best_gamma, c_hat


def _metrics(q_hat: float, a_grid: np.ndarray, a_probe: float,
             gamma_true: float) -> tuple[float, float, float]:
    """Expected coverage, expected width, and coverage conditional on d=D_PROBE."""
    sigma_grid = np.sqrt(1.0 + gamma_true * _D ** 2)
    sigma_probe = float(np.sqrt(1.0 + gamma_true * D_PROBE ** 2))
    if not np.isfinite(q_hat):
        return 1.0, float("inf"), 1.0
    coverage = _expect(2.0 * norm.cdf(q_hat * a_grid / sigma_grid) - 1.0)
    width = 2.0 * q_hat * _expect(a_grid)
    cov_probe = float(2.0 * norm.cdf(q_hat * a_probe / sigma_probe) - 1.0)
    return coverage, width, cov_probe


def _scale_arrays(gamma: float, c: float) -> tuple[np.ndarray, float]:
    a_grid = np.sqrt(1.0 + gamma * (_D / c) ** 2)
    a_probe = float(np.sqrt(1.0 + gamma * (D_PROBE / c) ** 2))
    return a_grid, a_probe


def run() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for gamma_true in GAMMAS:
        sigma_grid = np.sqrt(1.0 + gamma_true * _D ** 2)
        sigma_probe = float(np.sqrt(1.0 + gamma_true * D_PROBE ** 2))
        for n_budget in SIZES:
            n_tune = int(round(TUNE_SHARE * n_budget))
            for rep in range(N_REPS):
                d = np.abs(rng.standard_normal(n_budget))
                e = np.sqrt(1.0 + gamma_true * d ** 2) * np.abs(
                    rng.standard_normal(n_budget))

                per_strategy = {}

                q = conformal_quantile(e, ALPHA)
                per_strategy["marginal"] = (
                    q, np.ones_like(_D), 1.0, None, None)

                a_grid, a_probe = _scale_arrays(1.0, C0)
                scale = np.sqrt(1.0 + (d / C0) ** 2)
                q = conformal_quantile(e / scale, ALPHA)
                per_strategy["fixed_g1"] = (q, a_grid, a_probe, 1.0, C0)

                gamma_hat, c_hat = tune_gamma(e[:n_tune], d[:n_tune])
                a_grid, a_probe = _scale_arrays(gamma_hat, c_hat)
                scale_cal = np.sqrt(1.0 + gamma_hat * (d[n_tune:] / c_hat) ** 2)
                q = conformal_quantile(e[n_tune:] / scale_cal, ALPHA)
                per_strategy["tuned_split"] = (
                    q, a_grid, a_probe, gamma_hat, c_hat)

                q = conformal_quantile(
                    e / np.sqrt(1.0 + gamma_true * d ** 2), ALPHA)
                per_strategy["oracle_shape"] = (
                    q, sigma_grid, sigma_probe, gamma_true, 1.0)

                for name, (q_hat, a_grid, a_probe, g, c) in per_strategy.items():
                    coverage, width, cov_probe = _metrics(
                        q_hat, a_grid, a_probe, gamma_true)
                    rows.append({
                        "gamma_true": gamma_true,
                        "n_budget": n_budget,
                        "strategy": name,
                        "rep": rep,
                        "n_cal": n_budget - n_tune
                        if name == "tuned_split" else n_budget,
                        "q_hat": q_hat if np.isfinite(q_hat) else np.inf,
                        "gamma_hat": g,
                        "c_hat": c,
                        "coverage": coverage,
                        "width": width if np.isfinite(width) else np.inf,
                        "cov_at_d3": cov_probe,
                    })
    return pd.DataFrame(rows)


def summarize(per_rep: pd.DataFrame) -> pd.DataFrame:
    out = []
    grouped = per_rep.groupby(["gamma_true", "n_budget", "strategy"], sort=True)
    for (gamma_true, n_budget, strategy), g in grouped:
        width = g["width"].to_numpy()
        finite = np.isfinite(width)
        record = {
            "gamma_true": gamma_true,
            "n_budget": n_budget,
            "strategy": strategy,
            "n_reps": len(g),
            "n_cal": int(g["n_cal"].iloc[0]),
            "p_infinite": float(1.0 - finite.mean()),
            "coverage_mean": float(g["coverage"].mean()),
            "coverage_sd": float(g["coverage"].std(ddof=1)),
            "width_mean": float(width[finite].mean()) if finite.any() else np.inf,
            "width_sd": float(width[finite].std(ddof=1)) if finite.sum() > 1 else np.nan,
            "width_q25": float(np.quantile(width[finite], 0.25)) if finite.any() else np.inf,
            "width_q75": float(np.quantile(width[finite], 0.75)) if finite.any() else np.inf,
            "cov_at_d3_mean": float(g["cov_at_d3"].mean()),
        }
        if strategy == "tuned_split":
            gh = g["gamma_hat"].to_numpy(dtype=float)
            record["gamma_hat_mean"] = float(np.mean(gh))
            record["gamma_hat_median"] = float(np.median(gh))
            record["share_gamma_zero"] = float(np.mean(gh == 0.0))
        out.append(record)
    summary = pd.DataFrame(out)
    oracle = summary[summary.strategy == "oracle_shape"].set_index(
        ["gamma_true", "n_budget"])["width_mean"]
    summary["width_vs_oracle"] = [
        row.width_mean / oracle.loc[(row.gamma_true, row.n_budget)]
        for row in summary.itertuples()
    ]
    return summary


def width_optimal_scales() -> list[dict]:
    """Population comparison of the conditionally valid and width-optimal scales.

    For the family a_g(d) = sqrt(1 + g d^2), find for each gamma_true the g
    minimizing population expected width subject to exact marginal coverage
    1 - ALPHA, and report it next to the oracle g = gamma_true together with
    the conditional coverage both scales attain at d = D_PROBE.
    """
    records = []
    for gamma_true in GAMMAS:
        sigma = np.sqrt(1.0 + gamma_true * _D ** 2)
        sigma_probe = float(np.sqrt(1.0 + gamma_true * D_PROBE ** 2))

        def population_q(g: float) -> float:
            a = np.sqrt(1.0 + g * _D ** 2)

            def gap(q: float) -> float:
                return _expect(2.0 * norm.cdf(q * a / sigma) - 1.0) - (1 - ALPHA)

            hi = 10.0
            while gap(hi) < 0:
                hi *= 2.0
            return brentq(gap, 1e-9, hi, xtol=1e-12)

        def population_width(g: float) -> float:
            a = np.sqrt(1.0 + g * _D ** 2)
            return 2.0 * population_q(g) * _expect(a)

        upper = max(4.0 * gamma_true, 4.0)
        result = minimize_scalar(
            population_width, bounds=(0.0, upper), method="bounded",
            options={"xatol": 1e-6})
        g_star = float(result.x)
        q_star = population_q(g_star)
        a_probe = float(np.sqrt(1.0 + g_star * D_PROBE ** 2))
        q_oracle = population_q(gamma_true)
        records.append({
            "gamma_true": gamma_true,
            "gamma_width_optimal": g_star,
            "width_optimal": float(result.fun),
            "width_oracle": population_width(gamma_true),
            "cov_at_d3_width_optimal": float(
                2.0 * norm.cdf(q_star * a_probe / sigma_probe) - 1.0),
            "cov_at_d3_oracle": float(
                2.0 * norm.cdf(q_oracle * sigma_probe / sigma_probe) - 1.0),
        })
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help="directory receiving the per-replication, summary, and meta files",
    )
    parser.add_argument(
        "--resummarize",
        action="store_true",
        help="rebuild the summary and metadata from the saved per-replication "
        "file instead of re-running the simulation",
    )
    args = parser.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.resummarize:
        with gzip.open(out_dir / "gamma_sim_per_rep.csv.gz", "rt") as fh:
            per_rep = pd.read_csv(fh)
    else:
        per_rep = run()
        with gzip.open(out_dir / "gamma_sim_per_rep.csv.gz", "wt", newline="") as fh:
            per_rep.to_csv(fh, index=False)

    summary = summarize(per_rep)
    summary.to_csv(out_dir / "gamma_sim_summary.csv", index=False)

    meta = {
        "alpha": ALPHA,
        "seed": SEED,
        "n_reps": N_REPS,
        "tune_share": TUNE_SHARE,
        "gammas": list(GAMMAS),
        "sizes": list(SIZES),
        "gamma_grid": list(GAMMA_GRID),
        "n_gamma_candidates": len(GAMMA_GRID),
        "robust_scale_quantiles": list(_ROBUST_QS),
        "population_iqr_half_normal": C0,
        "d_probe": D_PROBE,
        "quadrature": {"d_max": float(_D[-1]), "n_points": int(len(_D))},
        "width_optimal_scales": width_optimal_scales(),
    }
    (out_dir / "gamma_sim_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8", newline="\n")

    for gamma_true in GAMMAS:
        block = summary[summary.gamma_true == gamma_true].pivot_table(
            index="n_budget", columns="strategy", values="width_mean")
        print(f"\n=== gamma_true={gamma_true}: expected width ===")
        print(block.round(4).to_string())
    print("\nwrote", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
