"""Strict post-hoc calibration for saved SRED SEMF-derived samples.

The original SEMF and augmented-theta SRED runs used the validation split
for model selection. This script avoids reusing that validation split for
formal conformal calibration by splitting the saved held-out test predictions
into a final calibration subset and a final evaluation subset.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "sred"))

from conformal import (  # noqa: E402
    conformal_quantile,
    density_conformal,
    evaluate as eval_intervals,
    mondrian_cqr,
)
from multimodal_calibration.reproducibility import (  # noqa: E402
    artifact_identity,
    attach_run_provenance,
    atomic_write_json,
    validate_campaign_tag,
)
from multimodal_calibration.experiment_config import (  # noqa: E402
    PAPER_CPU_DEVICE,
    PAPER_SEEDS,
    SRED_REGION_COLUMNS,
    TESTSPLIT_PAPER_RUN_CONFIG,
    canonical_run_config,
    require_canonical_run_config,
)
from multimodal_calibration.result_aggregation import (  # noqa: E402
    summarize_testsplit,
)
from multimodal_calibration.result_grid import testsplit_expected_grid  # noqa: E402
from augmented_theta import (  # noqa: E402
    validate_augmented_companions,
    validate_semf_companions,
)

SRED = ROOT / "data" / "sred"
META = SRED / "metadata"
RESULTS = ROOT / "results" / "sred_semf"


def testsplit_run_config(
    alpha: float,
    cal_frac: float,
    include_density: bool,
) -> dict:
    config = canonical_run_config("testsplit")
    config["alpha"] = float(alpha)
    config["final_calibration_fraction"] = float(cal_frac)
    config["include_density"] = bool(include_density)
    return config


def expected_paper_grid() -> set[tuple[int, str, str, str, str]]:
    return testsplit_expected_grid(PAPER_SEEDS)


def _stem(base: str, tag: str) -> str:
    """Build an artifact filename stem, inserting `tag` only when set."""
    return f"{base}_{tag}" if tag else base


def split_cal_eval(n: int, seed: int, cal_frac: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(
        seed + TESTSPLIT_PAPER_RUN_CONFIG["split_seed_offset"]
    )
    idx = np.arange(n)
    rng.shuffle(idx)
    cut = int(round(cal_frac * n))
    cut = min(max(cut, 1), n - 1)
    return np.sort(idx[:cut]), np.sort(idx[cut:])


def interval_bounds(samples: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    lo_q = (alpha / 2.0) * 100.0
    hi_q = (1.0 - alpha / 2.0) * 100.0
    return np.percentile(samples, lo_q, axis=1), np.percentile(samples, hi_q, axis=1)


def cqr_qhat(y_cal: np.ndarray, samples_cal: np.ndarray, alpha: float) -> float:
    lo, hi = interval_bounds(samples_cal, alpha)
    scores = np.maximum(lo - y_cal, y_cal - hi)
    return conformal_quantile(scores, alpha)


def cqr_apply(samples_eval: np.ndarray, q_hat: float, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = interval_bounds(samples_eval, alpha)
    return lo - q_hat, hi + q_hat


def eval_row(
    *,
    family: str,
    seed: int,
    analysis: str,
    method: str,
    stratum,
    y_eval: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    n_cal: int,
) -> dict:
    metrics = eval_intervals(y_eval, lo, hi)
    return {
        "family": family,
        "seed": int(seed),
        "analysis": analysis,
        "method": method,
        "stratum": str(stratum),
        "n_cal": int(n_cal),
        "n_eval": int(len(y_eval)),
        **metrics,
    }


def mc_widths(samples: np.ndarray, alpha: float) -> np.ndarray:
    lo, hi = interval_bounds(samples, alpha)
    return hi - lo


def quantile_assign(cal_values: np.ndarray, eval_values: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray, list[float]]:
    edges = np.quantile(cal_values, np.linspace(0, 1, n_bins + 1))
    cal_bins = np.clip(np.digitize(cal_values, edges[1:-1]), 0, n_bins - 1)
    eval_bins = np.clip(np.digitize(eval_values, edges[1:-1]), 0, n_bins - 1)
    return cal_bins.astype(np.int64), eval_bins.astype(np.int64), edges.tolist()


def load_test_metadata(metadata_dir: Path = META) -> pd.DataFrame:
    return pd.read_csv(
        metadata_dir / "test_data_with_text.csv", encoding="latin-1"
    ).reset_index(drop=True)


def add_stratified_rows(
    rows: list[dict],
    *,
    family: str,
    seed: int,
    analysis: str,
    y_cal: np.ndarray,
    y_eval: np.ndarray,
    samples_cal: np.ndarray,
    samples_eval: np.ndarray,
    strata_cal: np.ndarray,
    strata_eval: np.ndarray,
    alpha: float,
) -> None:
    q_global = cqr_qhat(y_cal, samples_cal, alpha)
    lo_g, hi_g = cqr_apply(samples_eval, q_global, alpha)
    lo_m, hi_m, info = mondrian_cqr(y_cal, samples_cal, samples_eval, strata_cal, strata_eval, alpha=alpha)

    for s in sorted(np.unique(strata_eval)):
        mask = strata_eval == s
        n_cal_s = int(np.sum(strata_cal == s))
        rows.append(eval_row(
            family=family, seed=seed, analysis=analysis, method="global_cqr",
            stratum=s, y_eval=y_eval[mask], lo=lo_g[mask], hi=hi_g[mask], n_cal=len(y_cal),
        ))
        rows.append(eval_row(
            family=family, seed=seed, analysis=analysis, method="mondrian_cqr",
            stratum=s, y_eval=y_eval[mask], lo=lo_m[mask], hi=hi_m[mask], n_cal=n_cal_s,
        ))

    if any(v == 0 for v in info["valid_sizes"].values()):
        missing = {k: v for k, v in info["valid_sizes"].items() if v == 0}
        print(f"WARN seed={seed} family={family} analysis={analysis}: empty calibration strata {missing}")


def semf_rows(seed: int, alpha: float, cal_frac: float, include_density: bool,
              semf_tag: str = "", results_dir: Path = RESULTS,
              metadata_dir: Path = META) -> list[dict]:
    path = results_dir / f"{_stem('semf', semf_tag)}_seed{seed}_samples.npz"
    config = TESTSPLIT_PAPER_RUN_CONFIG
    regimes = config["semf_regimes"]
    with np.load(path, allow_pickle=False) as z:
        y = np.asarray(z["y_test"])
        test_samples = {
            regime: np.asarray(z[f"test_{regime}"])
            for regime in regimes
        }
    cal_idx, eval_idx = split_cal_eval(len(y), seed, cal_frac)
    y_cal, y_eval = y[cal_idx], y[eval_idx]
    family = "semf_testsplit"
    rows: list[dict] = []

    q_full = cqr_qhat(y_cal, test_samples["full"][cal_idx], alpha)
    for regime in regimes:
        samples_cal = test_samples[regime][cal_idx]
        samples_eval = test_samples[regime][eval_idx]

        lo_raw, hi_raw = interval_bounds(samples_eval, alpha)
        rows.append(eval_row(
            family=family, seed=seed, analysis="regime", method="raw",
            stratum=regime, y_eval=y_eval, lo=lo_raw, hi=hi_raw, n_cal=0,
        ))

        lo_g, hi_g = cqr_apply(samples_eval, q_full, alpha)
        rows.append(eval_row(
            family=family, seed=seed, analysis="regime", method="global_cqr",
            stratum=regime, y_eval=y_eval, lo=lo_g, hi=hi_g, n_cal=len(y_cal),
        ))

        q_regime = cqr_qhat(y_cal, samples_cal, alpha)
        lo_c, hi_c = cqr_apply(samples_eval, q_regime, alpha)
        rows.append(eval_row(
            family=family, seed=seed, analysis="regime", method="mask_matched_cqr",
            stratum=regime, y_eval=y_eval, lo=lo_c, hi=hi_c, n_cal=len(y_cal),
        ))

        if include_density:
            np.random.seed(seed)
            lo_d, hi_d, _ = density_conformal(y_cal, samples_cal, samples_eval, alpha=alpha)
            rows.append(eval_row(
                family=family, seed=seed, analysis="regime", method="density",
                stratum=regime, y_eval=y_eval, lo=lo_d, hi=hi_d, n_cal=len(y_cal),
            ))

    full_cal = test_samples["full"][cal_idx]
    full_eval = test_samples["full"][eval_idx]
    w_cal = mc_widths(full_cal, alpha)
    w_eval = mc_widths(full_eval, alpha)
    strata_cal, strata_eval, _ = quantile_assign(
        w_cal,
        w_eval,
        n_bins=config["uncertainty_bins"],
    )
    add_stratified_rows(
        rows, family=family, seed=seed, analysis="uncert",
        y_cal=y_cal, y_eval=y_eval, samples_cal=full_cal, samples_eval=full_eval,
        strata_cal=strata_cal, strata_eval=strata_eval, alpha=alpha,
    )

    meta = load_test_metadata(metadata_dir)
    region_columns = list(SRED_REGION_COLUMNS)
    region_calibration = meta.iloc[cal_idx][region_columns].to_numpy()
    region_evaluation = meta.iloc[eval_idx][region_columns].to_numpy()
    km = KMeans(
        n_clusters=config["region_clusters"],
        random_state=seed,
        n_init=config["kmeans_n_init"],
    ).fit(region_calibration)
    region_cal = km.predict(region_calibration).astype(np.int64)
    region_eval = km.predict(region_evaluation).astype(np.int64)
    add_stratified_rows(
        rows, family=family, seed=seed, analysis="region",
        y_cal=y_cal, y_eval=y_eval, samples_cal=full_cal, samples_eval=full_eval,
        strata_cal=region_cal, strata_eval=region_eval, alpha=alpha,
    )
    return rows


def aug_theta_rows(seed: int, alpha: float, cal_frac: float, include_density: bool,
                   aug_theta_tag: str = "", results_dir: Path = RESULTS) -> list[dict]:
    path = (
        results_dir
        / f"{_stem('aug_theta', aug_theta_tag)}_seed{seed}_aug_full_samples.npz"
    )
    with np.load(path, allow_pickle=False) as z:
        y = np.asarray(z["y_test"])
        samples = np.asarray(z["test_full"])
    cal_idx, eval_idx = split_cal_eval(len(y), seed, cal_frac)
    y_cal, y_eval = y[cal_idx], y[eval_idx]
    samples_cal = samples[cal_idx]
    samples_eval = samples[eval_idx]
    family = "aug_theta_testsplit"
    rows: list[dict] = []

    lo_raw, hi_raw = interval_bounds(samples_eval, alpha)
    rows.append(eval_row(
        family=family, seed=seed, analysis="regime", method="raw",
        stratum="full", y_eval=y_eval, lo=lo_raw, hi=hi_raw, n_cal=0,
    ))
    q = cqr_qhat(y_cal, samples_cal, alpha)
    lo_c, hi_c = cqr_apply(samples_eval, q, alpha)
    rows.append(eval_row(
        family=family, seed=seed, analysis="regime", method="cqr",
        stratum="full", y_eval=y_eval, lo=lo_c, hi=hi_c, n_cal=len(y_cal),
    ))
    if include_density:
        np.random.seed(seed)
        lo_d, hi_d, _ = density_conformal(y_cal, samples_cal, samples_eval, alpha=alpha)
        rows.append(eval_row(
            family=family, seed=seed, analysis="regime", method="density",
            stratum="full", y_eval=y_eval, lo=lo_d, hi=hi_d, n_cal=len(y_cal),
        ))

    w_cal = mc_widths(samples_cal, alpha)
    w_eval = mc_widths(samples_eval, alpha)
    strata_cal, strata_eval, _ = quantile_assign(
        w_cal,
        w_eval,
        n_bins=TESTSPLIT_PAPER_RUN_CONFIG["uncertainty_bins"],
    )
    add_stratified_rows(
        rows, family=family, seed=seed, analysis="uncert",
        y_cal=y_cal, y_eval=y_eval, samples_cal=samples_cal, samples_eval=samples_eval,
        strata_cal=strata_cal, strata_eval=strata_eval, alpha=alpha,
    )
    return rows


def fmt_pm(mean: float, std: float, decimals: int) -> str:
    if not np.isfinite(mean):
        return str(mean)
    if not np.isfinite(std):
        std = 0.0
    return f"{mean:.{decimals}f} +/- {std:.{decimals}f}"


def summarize(df: pd.DataFrame, analysis: str) -> pd.DataFrame:
    return summarize_testsplit(df, analysis)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(PAPER_SEEDS))
    ap.add_argument(
        "--alpha", type=float, default=TESTSPLIT_PAPER_RUN_CONFIG["alpha"]
    )
    ap.add_argument(
        "--cal-frac",
        type=float,
        default=TESTSPLIT_PAPER_RUN_CONFIG["final_calibration_fraction"],
    )
    ap.add_argument("--tag", default="testsplit")
    ap.add_argument(
        "--campaign-tag",
        default=None,
        help="reproduction identifier recorded in run provenance",
    )
    ap.add_argument("--paper-run", action="store_true")
    ap.add_argument("--semf-tag", default="",
                    help="tag of the SEMF samples to load; must match the --tag "
                         "used by run_full_experiment.py (default: untagged, "
                         "semf_seed<n>_samples.npz)")
    ap.add_argument("--aug-theta-tag", default="",
                    help="tag of the aug_theta samples to load; must match the "
                         "--tag used by augmented_theta.py (default: untagged, "
                         "aug_theta_seed<n>_aug_full_samples.npz)")
    ap.add_argument("--include-density", action="store_true")
    ap.add_argument("--results-dir", type=Path, default=RESULTS,
                    help="directory containing tagged SEMF and augmented-theta inputs")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    args.tag = validate_campaign_tag(args.tag, required=True)
    campaign_tag = validate_campaign_tag(
        args.campaign_tag, required=args.paper_run
    ) or args.tag
    run_config = testsplit_run_config(
        args.alpha, args.cal_frac, args.include_density
    )
    if args.paper_run:
        if tuple(sorted(set(args.seeds))) != PAPER_SEEDS:
            ap.error(
                "--paper-run requires canonical seeds "
                + " ".join(str(seed) for seed in PAPER_SEEDS)
            )
        if not args.semf_tag or not args.aug_theta_tag:
            ap.error("--paper-run requires explicit SEMF and augmented tags")
        if campaign_tag != args.semf_tag or campaign_tag != args.aug_theta_tag:
            ap.error("--paper-run requires matching campaign/input tags")
        if args.tag != f"testsplit_{campaign_tag}":
            ap.error(
                "--paper-run requires --tag testsplit_<campaign-tag>"
            )
        try:
            require_canonical_run_config("testsplit", run_config)
        except ValueError as error:
            ap.error(str(error))
        semf_companions = validate_semf_companions(
            args.results_dir, tuple(args.seeds), args.semf_tag
        )
        validate_augmented_companions(
            args.results_dir,
            tuple(args.seeds),
            args.aug_theta_tag,
            semf_payloads=semf_companions,
        )

    out_dir = args.out_dir or (args.results_dir / args.tag)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    input_artifacts: dict[str, dict[str, dict[str, object]]] = {}
    for seed in args.seeds:
        print(f"seed={seed}")
        semf_samples = (
            args.results_dir
            / f"{_stem('semf', args.semf_tag)}_seed{seed}_samples.npz"
        )
        augmented_samples = (
            args.results_dir
            / (
                f"{_stem('aug_theta', args.aug_theta_tag)}_seed{seed}"
                "_aug_full_samples.npz"
            )
        )
        input_artifacts[str(seed)] = {
            "semf_samples": artifact_identity(semf_samples),
            "augmented_samples": artifact_identity(augmented_samples),
        }
        rows.extend(semf_rows(seed, args.alpha, args.cal_frac, args.include_density,
                              semf_tag=args.semf_tag, results_dir=args.results_dir))
        rows.extend(aug_theta_rows(seed, args.alpha, args.cal_frac, args.include_density,
                                   aug_theta_tag=args.aug_theta_tag,
                                   results_dir=args.results_dir))

    df = pd.DataFrame(rows)
    if args.paper_run:
        actual_grid = set(
            zip(
                df.seed.astype(int),
                df.family,
                df.analysis,
                df.method,
                df.stratum.astype(str),
            )
        )
        expected_grid = expected_paper_grid()
        if actual_grid != expected_grid or len(df) != len(expected_grid):
            raise RuntimeError(
                "SRED test-split output grid differs; "
                f"missing={sorted(expected_grid - actual_grid)}, "
                f"unexpected={sorted(actual_grid - expected_grid)}"
            )
        numeric = df.select_dtypes(include=[np.number])
        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise RuntimeError("SRED test-split output contains non-finite values")
    per_seed = out_dir / "testsplit_per_seed.csv"
    df.to_csv(per_seed, index=False)

    outputs = {}
    for analysis, filename in [
        ("regime", "testsplit_intervals_by_regime.csv"),
        ("uncert", "testsplit_conditional_uncert.csv"),
        ("region", "testsplit_conditional_region.csv"),
    ]:
        summary = summarize(df, analysis)
        path = out_dir / filename
        summary.to_csv(path, index=False)
        outputs[analysis] = str(path)
        print(path)

    summary_payload = {
        "config": {
            "campaign_tag": campaign_tag,
            "semf_tag": args.semf_tag,
            "aug_theta_tag": args.aug_theta_tag,
            "output_tag": args.tag,
            "run_config": run_config,
        },
        "alpha": args.alpha,
        "cal_frac": args.cal_frac,
        "seeds": args.seeds,
        "include_density": args.include_density,
        "input_artifacts": input_artifacts,
        "outputs": outputs,
        "protocol": "saved held-out test predictions split into final calibration and final evaluation",
    }
    attach_run_provenance(
        summary_payload,
        ROOT,
        seed=None,
        campaign_tag=campaign_tag,
        requested_device=PAPER_CPU_DEVICE,
    )
    atomic_write_json(
        out_dir / "testsplit_summary.json",
        summary_payload,
    )
    print(per_seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
