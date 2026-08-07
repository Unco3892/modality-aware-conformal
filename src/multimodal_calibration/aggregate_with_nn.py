"""Aggregate heterogeneous and neural-baseline results into one table.

The script reads the ``nnbase_<arch>_<dataset>_seed<n>.json`` files produced
by ``run_nn_baselines.py`` and emits a unified per-(dataset, method) summary
covering the source-wise methods and neural baselines.

Outputs:
  - results/multimodal_calibration/hetero_with_nn_aggregate.csv
  - results/multimodal_calibration/hetero_with_nn_aggregate.tex
"""

from __future__ import annotations

import os

import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np

from experiment_config import (
    AUXILIARY_PAPER_RUN_CONFIG,
    PAPER_NEURAL_ARCHITECTURES,
    PAPER_SEEDS,
    require_canonical_run_config,
)
from result_aggregation import (
    AUX_METHOD_LABEL,
    AUX_METHODS_ORDER,
    aggregate_auxiliary_payloads,
    render_auxiliary_tex,
)
from result_grid import (
    AUXILIARY_CAMPAIGN_METHODS,
    AUXILIARY_NN_METHODS,
    HETERO_EXPECTED_METHODS,
    auxiliary_campaign_scope,
    auxiliary_run_config,
    expected_seeds,
    require_dataset_seed_grid,
    require_expected_method_grid,
)
from reproducibility import (
    validate_campaign_payloads,
    write_campaign_manifest,
)

ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
RESULTS = ROOT / "results" / "multimodal_calibration"


# Ordered list of methods to display. New: mlp_concat + mm_attn_nn.
METHODS_ORDER = AUX_METHODS_ORDER
METHOD_LABEL = AUX_METHOD_LABEL

# how to extract metrics from a per-seed JSON, given a method id
def _row_for_method(payload_hetero: dict | None, payload_nn: dict | None,
                    method: str) -> dict | None:
    """Return dict with r2/rmse/mae/picp/mpiw/nciw or None if not available."""
    if method in AUXILIARY_NN_METHODS:
        if payload_nn is None:
            return None
        r = payload_nn["result"]
        calib = "residual_cp"
        if calib not in r:
            raise KeyError(f"NN result missing {calib}")
        return {
            "r2": r["point"]["r2"],
            "rmse": r["point"]["rmse"],
            "mae": r["point"]["mae"],
            "picp": r[calib]["picp"],
            "mpiw": r[calib]["mpiw"],
            "nciw": r[calib]["nciw"],
            "niw": r[calib]["niw"],
        }
    if payload_hetero is None:
        return None
    if method not in payload_hetero["results"]:
        return None
    r = payload_hetero["results"][method]
    return {
        "r2": r["point"]["r2"],
        "rmse": r["point"]["rmse"],
        "mae": r["point"]["mae"],
        "picp": r["cqr"]["picp"],
        "mpiw": r["cqr"]["mpiw"],
        "nciw": r["cqr"]["nciw"],
        "niw": r["cqr"]["niw"],
    }


def _parse_seed(stem_body: str) -> tuple[str, int] | None:
    if "_seed" not in stem_body:
        return None
    name, seed_str = stem_body.rsplit("_seed", 1)
    try:
        return name, int(seed_str)
    except ValueError:
        return None


def _load_hetero(
    tag: str | None = None,
    results_dir: Path = RESULTS,
) -> dict[str, dict[int, dict]]:
    """dataset -> seed -> hetero JSON."""
    out: dict[str, dict[int, dict]] = defaultdict(dict)
    for p in sorted(results_dir.glob("hetero_*_seed*.json")):
        stem = p.stem
        body = stem[len("hetero_"):]
        parsed = _parse_seed(body)
        if parsed is None:
            continue
        with open(p) as f:
            payload = json.load(f)
        cfg = payload.get("config", {})
        if tag is not None and cfg.get("output_tag") != tag:
            continue
        ds_from_path, seed = parsed
        ds = cfg.get("dataset") or (ds_from_path[:-len(tag) - 1] if tag and ds_from_path.endswith(f"_{tag}") else ds_from_path)
        out[ds][seed] = payload
    return out


def _load_nn(
    tag: str | None = None,
    results_dir: Path = RESULTS,
) -> dict[tuple[str, str], dict[int, dict]]:
    """(dataset, arch) -> seed -> nn JSON."""
    out: dict[tuple[str, str], dict[int, dict]] = defaultdict(dict)
    for p in sorted(results_dir.glob("nnbase_*_seed*.json")):
        stem = p.stem  # nnbase_<arch>_<dataset>_seed<N>
        body = stem[len("nnbase_"):]
        parsed = _parse_seed(body)
        if parsed is None:
            continue
        body, seed = parsed
        # body now is "<arch>_<dataset>" with arch in {mlp_concat, mm_attn_nn}
        if body.startswith("mlp_concat_"):
            arch = "mlp_concat"
            ds = body[len("mlp_concat_"):]
        elif body.startswith("mm_attn_nn_"):
            arch = "mm_attn_nn"
            ds = body[len("mm_attn_nn_"):]
        else:
            continue
        with open(p) as f:
            payload = json.load(f)
        cfg = payload.get("config", {})
        if tag is not None and cfg.get("output_tag") != tag:
            continue
        ds = cfg.get("dataset") or (ds[:-len(tag) - 1] if tag and ds.endswith(f"_{tag}") else ds)
        arch = cfg.get("arch", arch)
        out[(ds, arch)][seed] = payload
    return out


def aggregate_main(
    tag: str | None = None,
    *,
    results_dir: Path = RESULTS,
    datasets: list[str] | None = None,
    seeds: list[int] | None = None,
    archs: list[str] | None = None,
    alpha: float = AUXILIARY_PAPER_RUN_CONFIG["alpha"],
    tune_frac: float = AUXILIARY_PAPER_RUN_CONFIG["split"]["tune_frac"],
    strict: bool = True,
    expected_seed_count: int = len(PAPER_SEEDS),
):
    hetero = _load_hetero(tag=tag, results_dir=results_dir)
    nn = _load_nn(tag=tag, results_dir=results_dir)
    if not hetero and not nn:
        raise FileNotFoundError(f"no results found in {results_dir}")

    all_datasets = sorted(set(list(hetero.keys()) +
                              [k[0] for k in nn.keys()]))
    if strict:
        if tag is None:
            raise ValueError("strict paper aggregation requires --tag")
        required_seeds = expected_seeds(
            seeds, expected_seed_count=expected_seed_count
        )
        required_datasets = datasets or all_datasets
        required_datasets, required_seeds = auxiliary_campaign_scope(
            required_datasets,
            required_seeds,
        )
        require_dataset_seed_grid(
            hetero,
            required_datasets,
            required_seeds,
            label="heterogeneous",
        )
        required_archs = archs or list(PAPER_NEURAL_ARCHITECTURES)
        if set(required_archs) != AUXILIARY_NN_METHODS:
            raise ValueError(
                "strict auxiliary aggregation requires exactly architectures "
                f"{sorted(AUXILIARY_NN_METHODS)}, got {sorted(set(required_archs))}"
            )
        expected_nn = {
            (dataset, arch, seed)
            for dataset in required_datasets
            for arch in required_archs
            for seed in required_seeds
        }
        actual_nn = {
            (dataset, arch, seed)
            for (dataset, arch), seedmap in nn.items()
            for seed in seedmap
        }
        if actual_nn != expected_nn:
            raise ValueError(
                "neural grid differs; missing="
                f"{sorted(expected_nn - actual_nn)}, "
                f"unexpected={sorted(actual_nn - expected_nn)}"
            )
        unknown = sorted(set(required_datasets) - set(HETERO_EXPECTED_METHODS))
        if unknown:
            raise ValueError(
                "no explicit heterogeneous method contract for: "
                + ", ".join(unknown)
            )
        require_expected_method_grid(
            hetero,
            {
                dataset: HETERO_EXPECTED_METHODS[dataset]
                for dataset in required_datasets
            },
            required_seeds,
            label="heterogeneous",
        )
        payloads = [
            (
                f"hetero/{dataset}/seed{seed}",
                hetero[dataset][seed],
            )
            for dataset in required_datasets
            for seed in required_seeds
        ] + [
            (
                f"nn/{dataset}/{arch}/seed{seed}",
                nn[(dataset, arch)][seed],
            )
            for dataset in required_datasets
            for arch in required_archs
            for seed in required_seeds
        ]
        config_errors = []
        metric_errors = []
        for dataset in required_datasets:
            for seed in required_seeds:
                cfg = hetero[dataset][seed].get("config", {})
                if (
                    cfg.get("dataset") != dataset
                    or int(cfg.get("seed", -1)) != seed
                    or abs(float(cfg.get("alpha", -1)) - alpha) > 1e-12
                    or abs(float(cfg.get("tune_frac", -1)) - tune_frac) > 1e-12
                    or abs(
                        float(cfg.get("calib_frac", -1))
                        - AUXILIARY_PAPER_RUN_CONFIG["split"]["calib_frac"]
                    ) > 1e-12
                    or cfg.get("output_tag") != tag
                ):
                    config_errors.append(f"hetero/{dataset}/seed{seed}")
                for method in HETERO_EXPECTED_METHODS[dataset]:
                    label = f"hetero/{dataset}/seed{seed}/{method}"
                    try:
                        metric_row = _row_for_method(
                            hetero[dataset][seed], None, method
                        )
                        finite = (
                            metric_row is not None
                            and all(
                                np.isfinite(float(metric_row[key]))
                                for key in (
                                    "r2",
                                    "rmse",
                                    "mae",
                                    "picp",
                                    "mpiw",
                                    "niw",
                                    "nciw",
                                )
                            )
                        )
                    except (KeyError, TypeError, ValueError):
                        finite = False
                    if not finite:
                        metric_errors.append(label)
                for arch in required_archs:
                    cfg = nn[(dataset, arch)][seed].get("config", {})
                    if (
                        cfg.get("dataset") != dataset
                        or cfg.get("arch") != arch
                        or int(cfg.get("seed", -1)) != seed
                        or abs(float(cfg.get("alpha", -1)) - alpha) > 1e-12
                        or abs(float(cfg.get("tune_frac", -1)) - tune_frac) > 1e-12
                        or abs(
                            float(cfg.get("calib_frac", -1))
                            - AUXILIARY_PAPER_RUN_CONFIG["split"][
                                "calib_frac"
                            ]
                        ) > 1e-12
                        or cfg.get("output_tag") != tag
                    ):
                        config_errors.append(
                            f"nn/{dataset}/{arch}/seed{seed}"
                        )
                    label = f"nn/{dataset}/{arch}/seed{seed}"
                    try:
                        metric_row = _row_for_method(
                            None, nn[(dataset, arch)][seed], arch
                        )
                        finite = (
                            metric_row is not None
                            and all(
                                np.isfinite(float(metric_row[key]))
                                for key in (
                                    "r2",
                                    "rmse",
                                    "mae",
                                    "picp",
                                    "mpiw",
                                    "niw",
                                    "nciw",
                                )
                            )
                        )
                    except (KeyError, TypeError, ValueError):
                        finite = False
                    if not finite:
                        metric_errors.append(label)
        if config_errors:
            raise ValueError(
                "auxiliary payload configuration mismatch: "
                + ", ".join(config_errors)
            )
        if metric_errors:
            raise ValueError(
                "auxiliary payload contains non-finite or non-numeric "
                "required metrics: " + ", ".join(metric_errors)
            )
        run_config = auxiliary_run_config(alpha, tune_frac)
        require_canonical_run_config("aux", run_config)
        requested_device = validate_campaign_payloads(
            payloads,
            ROOT,
            campaign_tag=tag,
            run_config=run_config,
        )
        producer_threads = payloads[0][1]["provenance"]["threads"]
        write_campaign_manifest(
            results_dir,
            ROOT,
            campaign_tag=tag,
            datasets=required_datasets,
            seeds=required_seeds,
            methods=AUXILIARY_CAMPAIGN_METHODS,
            requested_device=requested_device,
            run_config=run_config,
            producer_threads=producer_threads,
        )
        all_datasets = list(required_datasets)

    hetero_cells = {
        (dataset, seed): payload
        for dataset, seedmap in hetero.items()
        for seed, payload in seedmap.items()
    }
    neural_cells = {
        (dataset, architecture, seed): payload
        for (dataset, architecture), seedmap in nn.items()
        for seed, payload in seedmap.items()
    }
    aggregate_seeds = sorted(
        {
            seed
            for dataset in all_datasets
            for seed in hetero.get(dataset, {})
        }
        | {
            seed
            for (dataset, _architecture), seedmap in nn.items()
            if dataset in all_datasets
            for seed in seedmap
        }
    )
    df = aggregate_auxiliary_payloads(
        hetero_cells,
        neural_cells,
        datasets=all_datasets,
        seeds=aggregate_seeds,
        methods=METHODS_ORDER,
        method_labels=METHOD_LABEL,
    )
    suffix = f"_{tag}" if tag else ""
    results_dir.mkdir(parents=True, exist_ok=True)
    out_csv = results_dir / f"hetero_with_nn_aggregate{suffix}.csv"
    df.to_csv(out_csv, index=False)
    print(f"wrote {out_csv} ({len(df)} rows)")

    # ---- TeX table: R2 / RMSE / PICP / MPIW / NCIW
    tex = render_auxiliary_tex(
        df,
        datasets=all_datasets,
        methods=METHODS_ORDER,
        method_labels=METHOD_LABEL,
    )
    out_tex = results_dir / f"hetero_with_nn_aggregate{suffix}.tex"
    out_tex.write_text(tex)
    print(f"wrote {out_tex}")

    # ---- Headline-table summary printed to stdout (R2 only)
    print("\n=== R2 summary (mean +/- std across seeds) ===")
    summary_methods = ["solo_tab", "mlp_concat", "mm_attn_nn",
                        "homo_xgb_concat", "hetero_gated_mixture"]
    print(f"\n{'dataset':12s}  " + "  ".join(f"{m:>22s}" for m in summary_methods))
    for ds in all_datasets:
        sub = df[df.dataset == ds]
        cells = []
        for m in summary_methods:
            r = sub[sub.method == m]
            if r.empty:
                cells.append(f"{'-':>22s}")
            else:
                rr = r.iloc[0]
                cells.append(f"{rr['r2_mean']:>10.3f}+/-{rr['r2_std']:.3f}        ".rstrip()[:22].rjust(22))
        print(f"{ds:12s}  " + "  ".join(cells))

    # ---- win/tie/loss analysis: hetero_gated vs each NN baseline
    print("\n=== heterogeneous gated mixture vs NN baselines (R2 delta) ===")
    for opp in ("mlp_concat", "mm_attn_nn", "homo_xgb_concat"):
        wins, ties, losses = [], [], []
        for ds in all_datasets:
            sub = df[df.dataset == ds]
            h = sub[sub.method == "hetero_gated_mixture"]
            o = sub[sub.method == opp]
            if h.empty or o.empty:
                continue
            d = float(h.iloc[0]["r2_mean"]) - float(o.iloc[0]["r2_mean"])
            tag = f"{ds} ({d:+.3f})"
            if d > 0.005:
                wins.append(tag)
            elif d < -0.005:
                losses.append(tag)
            else:
                ties.append(tag)
        print(f"  vs {opp}: WINS={wins} TIES={ties} LOSES={losses}")

    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None, help="only aggregate files with this output_tag")
    ap.add_argument("--results-dir", type=Path, default=RESULTS)
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(PAPER_SEEDS))
    ap.add_argument(
        "--archs", nargs="+", default=list(PAPER_NEURAL_ARCHITECTURES)
    )
    ap.add_argument(
        "--alpha", type=float, default=AUXILIARY_PAPER_RUN_CONFIG["alpha"]
    )
    ap.add_argument(
        "--tune-frac",
        type=float,
        default=AUXILIARY_PAPER_RUN_CONFIG["split"]["tune_frac"],
    )
    ap.add_argument(
        "--expected-seed-count", type=int, default=len(PAPER_SEEDS)
    )
    ap.add_argument("--allow-incomplete", action="store_true")
    args = ap.parse_args()
    aggregate_main(
        tag=args.tag,
        results_dir=args.results_dir,
        datasets=args.datasets,
        seeds=args.seeds,
        archs=args.archs,
        alpha=args.alpha,
        tune_frac=args.tune_frac,
        strict=not args.allow_incomplete,
        expected_seed_count=args.expected_seed_count,
    )
