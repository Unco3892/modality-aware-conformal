"""Pure deterministic builders for tracked paper result aggregations.

The experiment drivers and the offline verifier share these functions so a
rerun cannot silently change aggregation semantics on only one side of the
raw-to-summary contract.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

try:
    from .experiment_config import (
        PAPER_DATASETS,
        PAPER_SEEDS,
        PREDAGN_BASE_PREDICTORS,
        PREDAGN_HEADLINE_BASES,
    )
except ImportError:  # Direct-script imports used by experiment drivers.
    from experiment_config import (  # type: ignore[no-redef]
        PAPER_DATASETS,
        PAPER_SEEDS,
        PREDAGN_BASE_PREDICTORS,
        PREDAGN_HEADLINE_BASES,
    )

PREDAGN_METRICS = (
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
MISSING_METRICS = (
    "picp",
    "mpiw",
    "niw",
    "nciw",
    "crps",
    "c_test_cal",
    "q_global",
    "q_pooled_masked",
    "q_regime",
)
PAPER_PAIRED_BASES = PREDAGN_HEADLINE_BASES
AUX_METHODS_ORDER = (
    "solo_tab",
    "mlp_concat",
    "mm_attn_nn",
    "homo_xgb_concat",
    "hetero_gated_mixture",
    "solo_text",
    "solo_image",
    "homog_xgb_gated",
    "hetero_linear_stacker",
)
AUX_METHOD_LABEL = {
    "solo_tab": "Solo Tab (XGB-q)",
    "solo_text": "Solo Text (MLP-q)",
    "solo_image": "Solo Image (MLP-q)",
    "mlp_concat": "MLP on concat (NN)",
    "mm_attn_nn": "MM cross-attn NN",
    "homo_xgb_concat": "Homog. XGB on concat",
    "homog_xgb_gated": "Homog. XGB per-mod + gate",
    "hetero_linear_stacker": "Heterog. (linear stack)",
    "hetero_gated_mixture": "Heterog. (gated mix)",
}


def summarize_predagn(rows: pd.DataFrame) -> pd.DataFrame:
    """Reproduce ``predagn_ablation_summary.csv`` from per-seed rows."""
    keys = ["dataset", "base", "calibration"]
    grouped = rows.groupby(keys, as_index=False)
    mean = grouped[list(PREDAGN_METRICS)].mean()
    std = grouped[list(PREDAGN_METRICS)].std(ddof=1).fillna(0.0)
    n = grouped.size().rename(columns={"size": "n_seeds"})
    summary = mean.merge(n, on=keys)
    for column in PREDAGN_METRICS:
        summary[f"{column}_std"] = std[column].to_numpy()
        summary.rename(columns={column: f"{column}_mean"}, inplace=True)

    deltas: list[dict] = []
    for (dataset, base, seed), group in rows.groupby(
        ["dataset", "base", "seed"]
    ):
        marginal = group[group["calibration"] == "marginal"]
        if marginal.empty:
            continue
        baseline = marginal.iloc[0]
        for _, row in group.iterrows():
            if row["calibration"] == "marginal":
                continue
            deltas.append(
                {
                    "dataset": dataset,
                    "base": base,
                    "seed": seed,
                    "calibration": row["calibration"],
                    "delta_mpiw_vs_marginal": row["mpiw"] - baseline["mpiw"],
                    "delta_nciw_vs_marginal": row["nciw"] - baseline["nciw"],
                    "delta_crps_vs_marginal": row["crps"] - baseline["crps"],
                    "delta_picp_vs_marginal": row["picp"] - baseline["picp"],
                }
            )
    delta_rows = pd.DataFrame(deltas)
    if not delta_rows.empty:
        delta_group = delta_rows.groupby(keys, as_index=False)
        delta_summary = delta_group[
            [
                "delta_mpiw_vs_marginal",
                "delta_nciw_vs_marginal",
                "delta_crps_vs_marginal",
                "delta_picp_vs_marginal",
            ]
        ].agg(["mean", "std"])
        delta_summary.columns = [
            "_".join(column).rstrip("_")
            for column in delta_summary.columns
        ]
        delta_summary = delta_summary.reset_index(drop=True)
        summary = summary.merge(delta_summary, on=keys, how="left")
    return summary


def summarize_predagn_bins(bin_rows: pd.DataFrame) -> pd.DataFrame:
    """Reproduce ``predagn_ablation_bins_summary.csv``."""
    return (
        bin_rows.groupby(
            ["dataset", "base", "calibration", "bin"], as_index=False
        )
        .agg(
            n_mean=("n", "mean"),
            picp_mean=("picp", "mean"),
            picp_std=("picp", "std"),
            mpiw_mean=("mpiw", "mean"),
            mpiw_std=("mpiw", "std"),
            crps_mean=("crps", "mean"),
            disagreement_mean=("disagreement_mean", "mean"),
        )
    )


def summarize_missing(rows: pd.DataFrame) -> pd.DataFrame:
    """Reproduce ``missing_regime_summary.csv`` from per-seed rows."""
    keys = ["dataset", "regime", "method"]
    grouped = rows.groupby(keys, as_index=False)
    mean = grouped[list(MISSING_METRICS)].mean()
    std = grouped[list(MISSING_METRICS)].std(ddof=1).fillna(0.0)
    n = grouped.size().rename(columns={"size": "n_seeds"})
    summary = mean.merge(n, on=keys)
    for column in MISSING_METRICS:
        summary[f"{column}_std"] = std[column].to_numpy()
        summary.rename(columns={column: f"{column}_mean"}, inplace=True)

    deltas: list[dict] = []
    for (dataset, regime, seed), group in rows.groupby(
        ["dataset", "regime", "seed"]
    ):
        global_rows = group[group.method == "global_full_cal"]
        pooled_rows = group[group.method == "pooled_masked_cal"]
        matched_rows = group[group.method == "mask_matched_cal"]
        if global_rows.empty or matched_rows.empty:
            continue
        global_row = global_rows.iloc[0]
        matched_row = matched_rows.iloc[0]
        row = {
            "dataset": dataset,
            "regime": regime,
            "seed": seed,
            "coverage_gain": matched_row.picp - global_row.picp,
            "mpiw_change_pct": (
                100
                * (matched_row.mpiw - global_row.mpiw)
                / global_row.mpiw
                if global_row.mpiw
                else np.nan
            ),
            "crps_change_pct": (
                100
                * (matched_row.crps - global_row.crps)
                / global_row.crps
                if global_row.crps
                else np.nan
            ),
        }
        if not pooled_rows.empty:
            pooled_row = pooled_rows.iloc[0]
            row["coverage_gain_vs_pooled"] = (
                matched_row.picp - pooled_row.picp
            )
            row["mpiw_change_vs_pooled_pct"] = (
                100
                * (matched_row.mpiw - pooled_row.mpiw)
                / pooled_row.mpiw
                if pooled_row.mpiw
                else np.nan
            )
            row["crps_change_vs_pooled_pct"] = (
                100
                * (matched_row.crps - pooled_row.crps)
                / pooled_row.crps
                if pooled_row.crps
                else np.nan
            )
        deltas.append(row)
    delta_rows = pd.DataFrame(deltas)
    if not delta_rows.empty:
        aggregate_spec = {
            "coverage_gain_mean": ("coverage_gain", "mean"),
            "coverage_gain_std": ("coverage_gain", "std"),
            "mpiw_change_pct_mean": ("mpiw_change_pct", "mean"),
            "crps_change_pct_mean": ("crps_change_pct", "mean"),
        }
        if "coverage_gain_vs_pooled" in delta_rows.columns:
            aggregate_spec.update(
                {
                    "coverage_gain_vs_pooled_mean": (
                        "coverage_gain_vs_pooled",
                        "mean",
                    ),
                    "mpiw_change_vs_pooled_pct_mean": (
                        "mpiw_change_vs_pooled_pct",
                        "mean",
                    ),
                    "crps_change_vs_pooled_pct_mean": (
                        "crps_change_vs_pooled_pct",
                        "mean",
                    ),
                }
            )
        delta_summary = delta_rows.groupby(
            ["dataset", "regime"], as_index=False
        ).agg(**aggregate_spec)
        summary = summary.merge(
            delta_summary, on=["dataset", "regime"], how="left"
        )
    return summary


def _paired_rows(
    raw: pd.DataFrame,
    *,
    all_bases: bool,
    calibration: str,
    strict: bool,
) -> pd.DataFrame:
    selected = raw
    if not all_bases:
        selected = selected[selected.base.isin(PAPER_PAIRED_BASES)]
    keys = ["dataset", "base", "seed"]
    marginal = selected[selected.calibration == "marginal"].set_index(keys)
    calibrated = selected[selected.calibration == calibration].set_index(keys)
    pairs = marginal[["picp", "mpiw", "nciw", "crps"]].join(
        calibrated[["picp", "mpiw", "nciw", "crps"]],
        lsuffix="_marginal",
        rsuffix="_weighted",
        how="inner",
    )
    if pairs.isna().any().any():
        raise ValueError("unpaired or missing rows in per-seed file")
    base_count = (
        len(PREDAGN_BASE_PREDICTORS)
        if all_bases
        else len(PREDAGN_HEADLINE_BASES)
    )
    expected = len(PAPER_DATASETS) * len(PAPER_SEEDS) * base_count
    if strict and len(pairs) != expected:
        raise ValueError(
            f"expected {expected} complete marginal/{calibration} pairs, "
            f"found {len(pairs)}"
        )
    return pairs


def summarize_paired_lift(
    raw: pd.DataFrame,
    *,
    all_bases: bool = False,
    strict: bool = True,
) -> pd.DataFrame:
    """Reproduce ``paired_lift_summary.csv``."""
    selected = raw
    bases = (
        sorted(selected.base.unique())
        if all_bases
        else list(PAPER_PAIRED_BASES)
    )
    selected = selected[selected.base.isin(bases)]
    calibrations = sorted(set(selected.calibration) - {"marginal"})
    rows: list[dict] = []
    for calibration in calibrations:
        pairs = _paired_rows(
            raw,
            all_bases=all_bases,
            calibration=calibration,
            strict=strict,
        )
        for dataset, group in pairs.groupby(level=0):
            row: dict[str, object] = {
                "calibration": calibration,
                "dataset": dataset,
                "n": len(group),
                "marginal_picp": float(group.picp_marginal.mean()),
                "calib_picp": float(group.picp_weighted.mean()),
            }
            for metric in ("mpiw", "crps", "nciw"):
                marginal = group[f"{metric}_marginal"]
                calibrated = group[f"{metric}_weighted"]
                delta = calibrated - marginal
                improvement = (marginal - calibrated) / marginal * 100.0
                row[f"{metric}_delta_mean"] = float(delta.mean())
                row[f"{metric}_delta_sd"] = float(delta.std(ddof=1))
                row[f"{metric}_impr_mean"] = float(improvement.mean())
                row[f"{metric}_impr_sd"] = float(improvement.std(ddof=1))
                row[f"{metric}_strict_wins"] = int((delta < 0).sum())
                row[f"{metric}_ties"] = int((delta == 0).sum())
                row[f"{metric}_losses"] = int((delta > 0).sum())
                row[f"{metric}_wins"] = int((delta <= 0).sum())
            rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["calibration", "dataset"]
    ).reset_index(drop=True)


def summarize_paired_bins(
    raw: pd.DataFrame,
    *,
    all_bases: bool = False,
    strict: bool = True,
) -> pd.DataFrame:
    """Reproduce ``paired_bin_summary.csv``."""
    selected = raw
    if not all_bases:
        selected = selected[selected.base.isin(PAPER_PAIRED_BASES)]
    keys = ["dataset", "base", "seed", "bin"]
    metric_columns = ["picp", "mpiw", "crps"]
    marginal = (
        selected[selected.calibration == "marginal"]
        .set_index(keys)[metric_columns]
        .sort_index()
    )
    base_count = (
        len(PREDAGN_BASE_PREDICTORS)
        if all_bases
        else len(PREDAGN_HEADLINE_BASES)
    )
    expected_per_bin = (
        len(PAPER_DATASETS) * len(PAPER_SEEDS) * base_count
    )
    rows: list[dict] = []
    calibrations = [
        calibration
        for calibration in (
            "weighted_dis",
            "difficulty_weighted",
            "mondrian_dis",
        )
        if calibration in set(selected.calibration)
    ]
    for bin_name in sorted(selected.bin.unique()):
        marginal_bin = marginal[
            marginal.index.get_level_values("bin") == bin_name
        ]
        if strict and len(marginal_bin) != expected_per_bin:
            raise ValueError(
                f"bin {bin_name!r}: expected {expected_per_bin} marginal "
                f"cells, found {len(marginal_bin)}"
            )
        row: dict[str, object] = {
            "bin": bin_name,
            "n_pairs": len(marginal_bin),
            "marginal_picp": float(marginal_bin.picp.mean()),
            "marginal_mpiw": float(marginal_bin.mpiw.mean()),
        }
        for calibration in calibrations:
            calibrated = (
                selected[
                    (selected.calibration == calibration)
                    & (selected.bin == bin_name)
                ]
                .set_index(keys)[metric_columns]
                .sort_index()
            )
            pairs = marginal_bin.join(
                calibrated,
                lsuffix="_marginal",
                rsuffix="_calib",
                how="inner",
            )
            if strict and len(pairs) != len(marginal_bin):
                raise ValueError(
                    f"bin {bin_name!r}/{calibration}: incomplete paired grid"
                )
            prefix = {
                "weighted_dis": "weighted",
                "difficulty_weighted": "difficulty",
                "mondrian_dis": "mondrian",
            }[calibration]
            row[f"{prefix}_picp"] = float(pairs.picp_calib.mean())
            row[f"{prefix}_mpiw"] = float(pairs.mpiw_calib.mean())
            row[f"{prefix}_mpiw_impr"] = float(
                (
                    (pairs.mpiw_marginal - pairs.mpiw_calib)
                    / pairs.mpiw_marginal
                    * 100.0
                ).mean()
            )
            row[f"{prefix}_crps_impr"] = float(
                (
                    (pairs.crps_marginal - pairs.crps_calib)
                    / pairs.crps_marginal
                    * 100.0
                ).mean()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_auxiliary_payloads(
    hetero: Mapping[tuple[str, int], dict],
    neural: Mapping[tuple[str, str, int], dict],
    *,
    datasets: Sequence[str],
    seeds: Sequence[int],
    methods: Sequence[str],
    method_labels: Mapping[str, str],
) -> pd.DataFrame:
    """Reproduce the auxiliary aggregate CSV from worker payloads."""
    rows: list[dict] = []
    metric_names = ("r2", "rmse", "mae", "picp", "mpiw", "nciw", "niw")
    for dataset in datasets:
        for method in methods:
            values: dict[str, list[float]] = defaultdict(list)
            for seed in seeds:
                if method in {"mlp_concat", "mm_attn_nn"}:
                    payload = neural.get((dataset, method, int(seed)))
                    if payload is None:
                        continue
                    result = payload["result"]
                    extracted = {
                        "r2": result["point"]["r2"],
                        "rmse": result["point"]["rmse"],
                        "mae": result["point"]["mae"],
                        "picp": result["residual_cp"]["picp"],
                        "mpiw": result["residual_cp"]["mpiw"],
                        "nciw": result["residual_cp"]["nciw"],
                        "niw": result["residual_cp"]["niw"],
                    }
                else:
                    payload = hetero.get((dataset, int(seed)))
                    if payload is None:
                        continue
                    result = payload["results"].get(method)
                    if result is None:
                        continue
                    extracted = {
                        "r2": result["point"]["r2"],
                        "rmse": result["point"]["rmse"],
                        "mae": result["point"]["mae"],
                        "picp": result["cqr"]["picp"],
                        "mpiw": result["cqr"]["mpiw"],
                        "nciw": result["cqr"]["nciw"],
                        "niw": result["cqr"]["niw"],
                    }
                for name, value in extracted.items():
                    values[name].append(float(value))
            if not values:
                continue
            row: dict[str, object] = {
                "dataset": dataset,
                "method": method,
                "method_label": method_labels.get(method, method),
                "n_seeds": len(values["r2"]),
            }
            for name in metric_names:
                row[f"{name}_mean"] = float(np.mean(values[name]))
                row[f"{name}_std"] = float(
                    np.std(values[name], ddof=1)
                    if len(values[name]) > 1
                    else 0.0
                )
            rows.append(row)
    return pd.DataFrame(rows)


def render_auxiliary_tex(
    frame: pd.DataFrame,
    *,
    datasets: Sequence[str],
    methods: Sequence[str],
    method_labels: Mapping[str, str],
) -> str:
    """Reproduce the tracked full auxiliary TeX table."""
    lines = [
        "\\begin{tabular}{ll" + "r" * 5 + "}",
        "\\toprule",
        "Dataset & Method & $R^2$ & RMSE & PICP & MPIW & NCIW \\\\",
        "\\midrule",
    ]
    for dataset in datasets:
        subset = frame[frame.dataset == dataset]
        first = True
        for method in methods:
            selected = subset[subset.method == method]
            if selected.empty:
                continue
            row = selected.iloc[0]
            dataset_cell = dataset if first else ""
            first = False
            lines.append(
                f"{dataset_cell} & {method_labels.get(method, method)} "
                f"& {row['r2_mean']:.3f} $\\pm$ {row['r2_std']:.3f} "
                f"& {row['rmse_mean']:.3f} $\\pm$ {row['rmse_std']:.3f} "
                f"& {row['picp_mean']:.3f} $\\pm$ {row['picp_std']:.3f} "
                f"& {row['mpiw_mean']:.3f} $\\pm$ {row['mpiw_std']:.3f} "
                f"& {row['nciw_mean']:.3f} $\\pm$ {row['nciw_std']:.3f} \\\\"
            )
        lines.append("\\midrule")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def _format_pm(
    mean: pd.Series, std: pd.Series, decimals: int
) -> pd.Series:
    return mean.map(lambda value: f"{value:.{decimals}f}") + " +- " + std.map(
        lambda value: f"{value:.{decimals}f}"
    )


def predictor_mondrian_table(
    rows: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Reproduce the predictor-Mondrian table and JSON summary."""
    metrics = [
        "r2",
        "picp",
        "mpiw",
        "niw",
        "nciw",
        "c_test_cal",
        "crps_uniform",
    ]
    grouped = rows.groupby(["base", "calibration"])[metrics]
    mean = grouped.mean()
    std = grouped.std()
    table = pd.DataFrame(index=mean.index)
    table["R2"] = _format_pm(mean["r2"], std["r2"], 4)
    table["PICP"] = _format_pm(mean["picp"], std["picp"], 3)
    table["MPIW"] = _format_pm(mean["mpiw"], std["mpiw"], 3)
    table["NIW"] = _format_pm(mean["niw"], std["niw"], 4)
    table["NCIW"] = _format_pm(mean["nciw"], std["nciw"], 4)
    table["c_cal"] = _format_pm(
        mean["c_test_cal"], std["c_test_cal"], 3
    )
    table["CRPS"] = _format_pm(
        mean["crps_uniform"], std["crps_uniform"], 4
    )
    summary = {
        f"{base}|{calibration}": {
            "r2_mean": float(mean.loc[(base, calibration), "r2"]),
            "r2_std": float(std.loc[(base, calibration), "r2"]),
            "picp_mean": float(mean.loc[(base, calibration), "picp"]),
            "picp_std": float(std.loc[(base, calibration), "picp"]),
            "mpiw_mean": float(mean.loc[(base, calibration), "mpiw"]),
            "mpiw_std": float(std.loc[(base, calibration), "mpiw"]),
        }
        for base, calibration in mean.index
    }
    return table, summary


def _testsplit_pm(mean: float, std: float, decimals: int) -> str:
    if not np.isfinite(mean):
        return str(mean)
    if not np.isfinite(std):
        std = 0.0
    return f"{mean:.{decimals}f} +/- {std:.{decimals}f}"


def summarize_testsplit(rows: pd.DataFrame, analysis: str) -> pd.DataFrame:
    """Reproduce one of the three tracked test-split summary CSVs."""
    selected = rows[rows["analysis"] == analysis].copy()
    grouped = selected.groupby(["family", "method", "stratum"], sort=True)
    mean = grouped[
        ["picp", "mpiw", "crps_uniform", "n_eval", "n_cal"]
    ].mean()
    std = grouped[["picp", "mpiw", "crps_uniform"]].std().fillna(0.0)
    count = grouped.size().rename("n_seeds")
    output = mean.reset_index()
    output["PICP"] = [
        _testsplit_pm(m, s, 3)
        for m, s in zip(mean["picp"], std["picp"])
    ]
    output["MPIW"] = [
        _testsplit_pm(m, s, 3)
        for m, s in zip(mean["mpiw"], std["mpiw"])
    ]
    output["CRPS"] = [
        _testsplit_pm(m, s, 4)
        for m, s in zip(mean["crps_uniform"], std["crps_uniform"])
    ]
    output["n_seeds"] = count.to_numpy()
    output["n_eval_mean"] = output["n_eval"].round(1)
    output["n_cal_mean"] = output["n_cal"].round(1)
    return output[
        [
            "family",
            "method",
            "stratum",
            "PICP",
            "MPIW",
            "CRPS",
            "n_seeds",
            "n_cal_mean",
            "n_eval_mean",
        ]
    ]
