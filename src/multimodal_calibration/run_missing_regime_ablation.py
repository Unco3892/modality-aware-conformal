"""Cross-dataset missing-modality calibration benchmark.

This experiment emphasizes the multimodal failure mode: a predictor trained on
full modality vectors can become miscalibrated when text or image sources are
missing at deployment. We compare:

  * global_full_cal: one residual quantile calibrated on full-modality examples.
  * pooled_masked_cal: one residual quantile pooled across calibration examples
    after applying all represented missing-modality masks.
  * mask_matched_cal: residual quantile calibrated after applying the same
    fixed missing-modality mask to the final calibration split. This is the
    constant-label special case of Mondrian calibration.

The base predictor is an XGB point model on concatenated modalities. The model
is deliberately kept fixed across regimes; only the calibration rule changes.
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

ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
EXP = Path(__file__).resolve().parent
SRED_EXP = ROOT / "src" / "sred"
RESULTS = ROOT / "results" / "multimodal_calibration"
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(SRED_EXP))

from calibration import conformal_quantile  # noqa: E402
from experiment_config import (  # noqa: E402
    MISSING_PAPER_RUN_CONFIG,
    PAPER_CPU_DEVICE,
    PAPER_SEEDS,
    alpha_result_tag,
    canonical_run_config,
    require_canonical_run_config,
)
from result_aggregation import summarize_missing  # noqa: E402
from run_predagn_ablation import (  # noqa: E402
    DEFAULT_DATASETS, MODALITIES, _block_dict, _build_blocks_split_aware,
    _concat_nonempty, _fit_xgb_point, _interval_metrics, _test_block_dict,
)
from run_hetero_mixture import _load_dataset, _split_fit_tune_cal  # noqa: E402
from reproducibility import (  # noqa: E402
    attach_run_provenance,
    thread_identity,
    validate_campaign_payloads,
    validate_campaign_tag,
    write_campaign_manifest,
)
from result_grid import (  # noqa: E402
    MISSING_EXPECTED_METHODS,
    MISSING_EXPECTED_REGIME_MISSING,
    expected_seeds,
    paper_campaign_scope,
)


def missing_run_config(
    *,
    alpha: float,
    calib_frac: float,
    tune_frac: float,
    n_estimators: int,
    max_depth: int,
) -> dict:
    config = canonical_run_config("missing")
    config["alpha"] = float(alpha)
    config["split"]["calib_frac"] = float(calib_frac)
    config["split"]["tune_frac"] = float(tune_frac)
    config["point_xgb"]["n_estimators"] = int(n_estimators)
    config["point_xgb"]["max_depth"] = int(max_depth)
    return config


EXPECTED_MISSING_METHODS = MISSING_EXPECTED_METHODS
EXPECTED_MISSING_SCHEMA_KEYS = frozenset(
    EXPECTED_MISSING_METHODS
    + ("q_global", "q_pooled_masked", "q_regime", "missing")
)
EXPECTED_REGIME_MISSING = MISSING_EXPECTED_REGIME_MISSING


def _mask_blocks(blocks: dict[str, np.ndarray], missing: set[str]) -> dict[str, np.ndarray]:
    out = {}
    for mod, x in blocks.items():
        if mod in missing:
            out[mod] = np.zeros_like(x, dtype=np.float32)
        else:
            out[mod] = x
    return out


def _regimes_for_shapes(shapes: dict[str, int]) -> dict[str, set[str]]:
    regimes: dict[str, set[str]] = {"full": set()}
    if shapes["text"] > 0:
        regimes["no_text"] = {"text"}
    if shapes["image"] > 0:
        regimes["no_image"] = {"image"}
    missing_all_non_tab = {m for m in ("text", "image") if shapes[m] > 0}
    if missing_all_non_tab:
        regimes["tab_only"] = missing_all_non_tab
    return regimes


def _eval_interval(y, pred, q, alpha):
    lo = pred - q
    hi = pred + q
    return _interval_metrics(y, lo, hi, alpha=alpha, f=pred)


def run_one(
    dataset: str,
    seed: int,
    alpha: float,
    calib_frac: float,
    tune_frac: float,
    n_estimators: int,
    max_depth: int,
    device: str,
    out_dir: Path,
    skip_existing: bool,
    campaign_tag: str | None = None,
) -> dict:
    suffix = alpha_result_tag(alpha)
    out_path = out_dir / f"missing_{dataset}_{suffix}_seed{seed}.json"
    run_config = missing_run_config(
        alpha=alpha,
        calib_frac=calib_frac,
        tune_frac=tune_frac,
        n_estimators=n_estimators,
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
    y_fit = y_train[fit_idx]
    y_tune = y_train[tune_idx]
    y_cal = y_train[cal_idx]

    X_fit_blocks = _block_dict(blocks, fit_idx)
    X_tune_blocks = _block_dict(blocks, tune_idx)
    X_cal_blocks = _block_dict(blocks, cal_idx)
    X_test_blocks = _test_block_dict(blocks)

    X_fit = _concat_nonempty([X_fit_blocks[m] for m in MODALITIES])
    X_tune = _concat_nonempty([X_tune_blocks[m] for m in MODALITIES])
    X_cal_full = _concat_nonempty([X_cal_blocks[m] for m in MODALITIES])

    model = _fit_xgb_point(
        X_fit, y_fit, X_tune, y_tune, seed=seed,
        n_estimators=n_estimators, max_depth=max_depth, device=device,
    )

    pred_cal_full = model.predict(X_cal_full).astype(np.float32)
    q_global = conformal_quantile(np.abs(y_cal - pred_cal_full), alpha)

    shapes = {
        "tab": int(blocks["X_tab_tr"].shape[1]),
        "text": int(blocks["X_text_tr"].shape[1]),
        "image": int(blocks["X_image_tr"].shape[1]),
    }
    regimes = _regimes_for_shapes(shapes)

    regime_cal = {}
    masked_residuals = []
    seen_mask_keys = set()
    for regime, missing in regimes.items():
        cal_masked = _mask_blocks(X_cal_blocks, missing)
        X_cal = _concat_nonempty([cal_masked[m] for m in MODALITIES])
        pred_cal = model.predict(X_cal).astype(np.float32)
        residuals = np.abs(y_cal - pred_cal)
        q_regime = conformal_quantile(residuals, alpha)
        regime_cal[regime] = {
            "missing": missing,
            "q_regime": float(q_regime),
            "residuals": residuals,
        }
        mask_key = tuple(sorted(missing))
        if regime != "full" and mask_key not in seen_mask_keys:
            masked_residuals.append(residuals)
            seen_mask_keys.add(mask_key)
    q_pooled_masked = (
        conformal_quantile(np.concatenate(masked_residuals), alpha)
        if masked_residuals else q_global
    )

    rows = {}
    for regime, cached in regime_cal.items():
        missing = cached["missing"]
        test_masked = _mask_blocks(X_test_blocks, missing)
        X_test = _concat_nonempty([test_masked[m] for m in MODALITIES])
        pred_test = model.predict(X_test).astype(np.float32)
        q_regime = cached["q_regime"]
        q_pooled = q_global if regime == "full" else q_pooled_masked
        rows[regime] = {
            "global_full_cal": _eval_interval(y_test, pred_test, q_global, alpha),
            "pooled_masked_cal": _eval_interval(y_test, pred_test, q_pooled, alpha),
            "mask_matched_cal": _eval_interval(y_test, pred_test, q_regime, alpha),
            "q_global": float(q_global),
            "q_pooled_masked": float(q_pooled_masked),
            "q_regime": float(q_regime),
            "missing": sorted(missing),
        }

    out = {
        "config": {
            "dataset": dataset,
            "seed": seed,
            "alpha": alpha,
            "calib_frac": calib_frac,
            "tune_frac": tune_frac,
            "n_estimators": n_estimators,
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
            "tab_dim": shapes["tab"],
            "text_dim": shapes["text"],
            "image_dim": shapes["image"],
        },
        "regimes": rows,
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
            path = out_dir / f"missing_{dataset}_{suffix}_seed{seed}.json"
            if not path.exists():
                missing_paths.append(path.name)
                continue
            payload = json.loads(path.read_text())
            campaign_payloads.append((path.name, payload))
            cfg = payload.get("config", {})
            if (
                cfg.get("dataset") != dataset
                or int(cfg.get("seed", -1)) != seed
                or abs(float(cfg.get("alpha", -1)) - alpha) > 1e-12
                or (
                    campaign_tag is not None
                    and cfg.get("campaign_tag") != campaign_tag
                )
            ):
                invalid.append(f"{path.name}: config mismatch")
            if strict:
                expected_regimes = EXPECTED_REGIME_MISSING.get(dataset)
                actual_regimes = set(payload.get("regimes", {}))
                if expected_regimes is None:
                    invalid.append(
                        f"{path.name}: no strict regime contract for {dataset}"
                    )
                elif actual_regimes != set(expected_regimes):
                    invalid.append(
                        f"{path.name}: regime grid differs; "
                        f"missing={sorted(set(expected_regimes) - actual_regimes)}, "
                        f"unexpected={sorted(actual_regimes - set(expected_regimes))}"
                    )
            for regime, r in payload["regimes"].items():
                if strict:
                    actual_keys = set(r)
                    if actual_keys != EXPECTED_MISSING_SCHEMA_KEYS:
                        invalid.append(
                            f"{path.name}/{regime}: result schema differs; "
                            f"missing={sorted(EXPECTED_MISSING_SCHEMA_KEYS - actual_keys)}, "
                            f"unexpected={sorted(actual_keys - EXPECTED_MISSING_SCHEMA_KEYS)}"
                        )
                    expected_missing = (
                        EXPECTED_REGIME_MISSING
                        .get(dataset, {})
                        .get(regime)
                    )
                    if (
                        expected_missing is not None
                        and tuple(sorted(r.get("missing", ())))
                        != tuple(sorted(expected_missing))
                    ):
                        invalid.append(
                            f"{path.name}/{regime}: missing-modality label differs"
                        )
                for method in EXPECTED_MISSING_METHODS:
                    source_method = method
                    if method == "mask_matched_cal" and method not in r:
                        # Read old raw JSON only to permit a label-only
                        # exploratory re-aggregation; every strict/new artifact
                        # must use the canonical name.
                        if not strict:
                            source_method = "regime_mondrian"
                    if source_method not in r:
                        invalid.append(f"{path.name}/{regime}: missing {method}")
                        continue
                    metrics = r[source_method]
                    row = {"dataset": dataset, "seed": seed, "regime": regime, "method": method}
                    for key in (
                        "picp", "mpiw", "niw", "nciw", "crps", "c_test_cal"
                    ):
                        row[key] = metrics[key]
                    row["q_global"] = r["q_global"]
                    row["q_pooled_masked"] = r.get("q_pooled_masked", r["q_global"])
                    row["q_regime"] = r["q_regime"]
                    if strict:
                        required_numeric = (
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
                        try:
                            finite = all(
                                np.isfinite(float(row[key]))
                                for key in required_numeric
                            )
                        except (KeyError, TypeError, ValueError):
                            finite = False
                        if not finite:
                            invalid.append(
                                f"{path.name}/{regime}/{method}: "
                                "non-finite or non-numeric required metric"
                            )
                    rows.append(row)
    if strict and (missing_paths or invalid):
        details = []
        if missing_paths:
            details.append("missing files: " + ", ".join(missing_paths))
        if invalid:
            details.append("invalid files: " + "; ".join(invalid))
        raise ValueError("incomplete missing-modality grid: " + " | ".join(details))
    if strict:
        validate_campaign_payloads(
            campaign_payloads,
            ROOT,
            campaign_tag=campaign_tag,
            requested_device=requested_device,
            run_config=run_config,
        )
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
    calib_frac: float = MISSING_PAPER_RUN_CONFIG["split"]["calib_frac"],
    tune_frac: float = MISSING_PAPER_RUN_CONFIG["split"]["tune_frac"],
    n_estimators: int = MISSING_PAPER_RUN_CONFIG["point_xgb"]["n_estimators"],
    max_depth: int = MISSING_PAPER_RUN_CONFIG["point_xgb"]["max_depth"],
    device: str = PAPER_CPU_DEVICE,
):
    if strict:
        expected_seeds(seeds, expected_seed_count=expected_seed_count)
        datasets, seeds = map(
            list, paper_campaign_scope(datasets, seeds)
        )
    run_config = missing_run_config(
        alpha=alpha,
        calib_frac=calib_frac,
        tune_frac=tune_frac,
        n_estimators=n_estimators,
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
                / f"missing_{datasets[0]}_{suffix}_seed{seeds[0]}.json"
            ).read_text(encoding="utf-8")
        )
        write_campaign_manifest(
            out_dir,
            ROOT,
            campaign_tag=campaign_tag,
            datasets=datasets,
            seeds=seeds,
            methods=(
                "global_full_cal",
                "pooled_masked_cal",
                "mask_matched_cal",
            ),
            requested_device=device,
            run_config=run_config,
            producer_threads=first_payload["provenance"]["threads"],
        )
    if rows.empty:
        raise FileNotFoundError(f"no missing-modality outputs found in {out_dir}")
    rows.to_csv(out_dir / "missing_regime_per_seed.csv", index=False)
    summary = summarize_missing(rows)
    summary.to_csv(out_dir / "missing_regime_summary.csv", index=False)
    return rows, summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(PAPER_SEEDS))
    ap.add_argument("--alpha", type=float, default=MISSING_PAPER_RUN_CONFIG["alpha"])
    ap.add_argument(
        "--calib-frac",
        type=float,
        default=MISSING_PAPER_RUN_CONFIG["split"]["calib_frac"],
    )
    ap.add_argument(
        "--tune-frac",
        type=float,
        default=MISSING_PAPER_RUN_CONFIG["split"]["tune_frac"],
    )
    ap.add_argument(
        "--n-estimators",
        type=int,
        default=MISSING_PAPER_RUN_CONFIG["point_xgb"]["n_estimators"],
    )
    ap.add_argument(
        "--max-depth",
        type=int,
        default=MISSING_PAPER_RUN_CONFIG["point_xgb"]["max_depth"],
    )
    ap.add_argument(
        "--device",
        default=os.environ.get("XGB_DEVICE", PAPER_CPU_DEVICE),
    )
    ap.add_argument("--out-dir", type=Path, default=RESULTS)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--campaign-tag", default=None)
    ap.add_argument(
        "--paper-run",
        action="store_true",
        help="require an explicit campaign tag and complete paper seed grid",
    )
    ap.add_argument("--campaign-datasets", nargs="+", default=None)
    ap.add_argument("--campaign-seeds", nargs="+", type=int, default=None)
    ap.add_argument("--allow-incomplete", action="store_true")
    ap.add_argument(
        "--expected-seed-count", type=int, default=len(PAPER_SEEDS)
    )
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    campaign_tag = validate_campaign_tag(
        args.campaign_tag, required=args.paper_run
    )
    run_config = missing_run_config(
        alpha=args.alpha,
        calib_frac=args.calib_frac,
        tune_frac=args.tune_frac,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
    )
    if args.paper_run:
        try:
            require_canonical_run_config("missing", run_config)
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
                methods=(
                    "global_full_cal",
                    "pooled_masked_cal",
                    "mask_matched_cal",
                ),
                requested_device=args.device,
                run_config=run_config,
                producer_threads=thread_identity(),
            )

    if not args.aggregate_only:
        for dataset in args.datasets:
            for seed in args.seeds:
                print(f"\n=== missing dataset={dataset} seed={seed} alpha={args.alpha} ===", flush=True)
                try:
                    out = run_one(
                        dataset=dataset,
                        seed=seed,
                        alpha=args.alpha,
                        calib_frac=args.calib_frac,
                        tune_frac=args.tune_frac,
                        n_estimators=args.n_estimators,
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
        max_depth=args.max_depth,
        device=args.device,
    )
    print("\n=== summary ===")
    print(summary[[
        "dataset", "regime", "method", "n_seeds", "picp_mean", "mpiw_mean",
        "nciw_mean", "coverage_gain_mean", "mpiw_change_pct_mean"
    ]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
