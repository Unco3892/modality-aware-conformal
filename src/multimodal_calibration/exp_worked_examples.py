"""Extract real SRED test listings that illustrate the weighted mechanism.

Reruns the SRED branch of the cross-dataset predictor-agnostic pipeline for
seed 0 with the same splits, models, and weighted-scale selection as
run_predagn_ablation.py, then saves per-example quantities for two
illustrative test listings. Example A has low disagreement, so the weighted
interval tightens relative to marginal CQR while still covering. Example B has
high disagreement, so the weighted interval widens. Both are selected using
test labels and are therefore illustrations, not evaluation results.

Output: results/exploratory/worked_examples/worked_examples.json with source
predictions, disagreement, marginal and weighted CQR intervals, and the true
rent, in log units and in CHF (exponentiated).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP))

from calibration import conformal_quantile  # noqa: E402
from experiment_config import (  # noqa: E402
    WORKED_EXAMPLE_COUNT,
    WORKED_IMPLEMENTATION_POLICY,
    WORKED_PAPER_RUN_CONFIG,
    canonical_run_config,
    require_canonical_run_config,
)
from run_hetero_mixture import _load_dataset, _split_fit_tune_cal  # noqa: E402
from run_predagn_ablation import (  # noqa: E402
    MODALITIES, _block_dict, _build_blocks_split_aware, _concat_nonempty,
    _fit_modality_points, _fit_xgb_quantile, _normalize_disagreement,
    _predict_quantile, _test_block_dict, _tune_alpha01,
)
from reproducibility import (  # noqa: E402
    attach_run_provenance,
    validate_campaign_tag,
)

OUT_DIR = ROOT / "results" / "exploratory" / "worked_examples"
SEED = WORKED_PAPER_RUN_CONFIG["seed"]
ALPHA = WORKED_PAPER_RUN_CONFIG["parent_predagn"]["alpha"]


def worked_examples_run_config() -> dict:
    return canonical_run_config("worked")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-tag", default=None)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--paper-run", action="store_true")
    args = parser.parse_args()
    campaign_tag = validate_campaign_tag(
        args.campaign_tag, required=args.paper_run
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_config = worked_examples_run_config()
    implementation_fields = {
        "base": WORKED_PAPER_RUN_CONFIG["base"],
        "calibrations": tuple(WORKED_PAPER_RUN_CONFIG["calibrations"]),
        "selection_uses_test_labels": WORKED_PAPER_RUN_CONFIG[
            "selection_uses_test_labels"
        ],
        "example_count": WORKED_EXAMPLE_COUNT,
    }
    if implementation_fields != WORKED_IMPLEMENTATION_POLICY:
        raise ValueError(
            "worked-example run config differs from the implemented contract"
        )
    if args.paper_run:
        try:
            require_canonical_run_config("worked", run_config)
        except ValueError as error:
            parser.error(str(error))
    dataset = WORKED_PAPER_RUN_CONFIG["dataset"]
    parent = WORKED_PAPER_RUN_CONFIG["parent_predagn"]
    train, test = _load_dataset(dataset)
    y_train = train["y"].astype(np.float32)
    y_test = test["y"].astype(np.float32)
    fit_idx, tune_idx, cal_idx = _split_fit_tune_cal(
        len(y_train),
        calib_frac=parent["split"]["calib_frac"],
        tune_frac=parent["split"]["tune_frac"],
        seed=SEED,
    )
    blocks = _build_blocks_split_aware(dataset, train, test, fit_idx)
    Xb = {"fit": _block_dict(blocks, fit_idx), "tune": _block_dict(blocks, tune_idx),
          "cal": _block_dict(blocks, cal_idx), "test": _test_block_dict(blocks)}
    y = {"fit": y_train[fit_idx], "tune": y_train[tune_idx],
         "cal": y_train[cal_idx], "test": y_test}
    Xc = {k: _concat_nonempty([Xb[k][m] for m in MODALITIES]) for k in Xb}

    mod_pred, mod_models = _fit_modality_points(
        Xb["fit"], y["fit"], Xb["tune"], y["tune"], Xb["cal"], Xb["test"],
        seed=SEED,
        n_estimators=parent["point_xgb"]["n_estimators"],
        max_depth=parent["point_xgb"]["max_depth"],
        device=WORKED_PAPER_RUN_CONFIG["device"],
    )
    s_tune, s_cal, s_test, dis_info = _normalize_disagreement(
        mod_pred["s_dis"]["tune"], mod_pred["s_dis"]["calib"], mod_pred["s_dis"]["test"])

    q_models = _fit_xgb_quantile(
        Xc["fit"],
        y["fit"],
        seed=SEED,
        alpha=ALPHA,
        n_estimators=parent["quantile_xgb"]["n_estimators"],
        max_depth=parent["quantile_xgb"]["max_depth"],
        device=WORKED_PAPER_RUN_CONFIG["device"],
    )
    qp = {k: _predict_quantile(q_models, Xc[k]) for k in ("tune", "cal", "test")}
    scores_tune = np.maximum(qp["tune"][:, 0] - y["tune"], y["tune"] - qp["tune"][:, 2])
    scores_cal = np.maximum(qp["cal"][:, 0] - y["cal"], y["cal"] - qp["cal"][:, 2])

    q_marg = conformal_quantile(scores_cal, ALPHA)
    lo_m = qp["test"][:, 0] - q_marg
    hi_m = qp["test"][:, 2] + q_marg

    info = _tune_alpha01(scores_tune, s_tune, ALPHA)
    scale_cal = np.sqrt(info["a0"] + info["a1"] * s_cal ** 2)
    scale_test = np.sqrt(info["a0"] + info["a1"] * s_test ** 2)
    # pure-expansion score e^+ (paper eq. 5); archived q_w was nonnegative
    q_w = conformal_quantile(np.maximum(scores_cal, 0.0) / np.maximum(scale_cal, 1e-12), ALPHA)
    lo_w = qp["test"][:, 0] - q_w * scale_test
    hi_w = qp["test"][:, 2] + q_w * scale_test

    width_m = hi_m - lo_m
    width_w = hi_w - lo_w
    cover_m = (y["test"] >= lo_m) & (y["test"] <= hi_m)
    cover_w = (y["test"] >= lo_w) & (y["test"] <= hi_w)

    # Example A: lowest-disagreement test listing where both intervals cover
    # and the weighted one is strictly narrower.
    ok_a = cover_m & cover_w & (width_w < width_m)
    idx_a = int(np.flatnonzero(ok_a)[np.argmin(s_test[ok_a])])
    # Example B: highest-disagreement listing where the weighted interval covers
    # and marginal does not, if one exists, otherwise the highest-d covered one.
    ok_b = cover_w & ~cover_m
    if ok_b.any():
        idx_b = int(np.flatnonzero(ok_b)[np.argmax(s_test[ok_b])])
        b_kind = "weighted covers, marginal misses"
    else:
        ok_b = cover_w & (width_w > width_m)
        idx_b = int(np.flatnonzero(ok_b)[np.argmax(s_test[ok_b])])
        b_kind = "highest disagreement, weighted widens"

    mods = list(mod_models)
    per_mod_test = mod_pred["per_modality"]["test"]

    def pack(idx: int, name: str) -> dict:
        marginal_chf = [float(np.exp(lo_m[idx])), float(np.exp(hi_m[idx]))]
        weighted_chf = [float(np.exp(lo_w[idx])), float(np.exp(hi_w[idx]))]
        return {
            "name": name,
            "test_index": idx,
            "listing_id": int(test["id"][idx]),
            "sources_log": {m: float(per_mod_test[j, idx]) for j, m in enumerate(mods)},
            "sources_chf": {m: float(np.exp(per_mod_test[j, idx])) for j, m in enumerate(mods)},
            "d_standardized": float(s_test[idx]),
            "y_log": float(y["test"][idx]),
            "y_chf": float(np.exp(y["test"][idx])),
            "marginal_log": [float(lo_m[idx]), float(hi_m[idx])],
            "marginal_chf": marginal_chf,
            "weighted_log": [float(lo_w[idx]), float(hi_w[idx])],
            "weighted_chf": weighted_chf,
            "width_change_log_pct": float(
                100.0 * (width_w[idx] / width_m[idx] - 1.0)),
            "width_change_chf_pct": float(
                100.0
                * ((weighted_chf[1] - weighted_chf[0])
                   / (marginal_chf[1] - marginal_chf[0]) - 1.0)),
            "covered_marginal": bool(cover_m[idx]),
            "covered_weighted": bool(cover_w[idx]),
        }

    out = {
        "config": {
            "campaign_tag": campaign_tag,
            "run_config": run_config,
        },
        "seed": SEED,
        "alpha": ALPHA,
        "base": WORKED_PAPER_RUN_CONFIG["base"],
        "selection_note": (
            "post-hoc illustrations selected using test outcomes; "
            "A minimizes disagreement among cases covered by both methods "
            "where weighting narrows; B: " + b_kind),
        "scale_info": {k: info[k] for k in ("a0", "a1")},
        "disagreement_standardization": {
            "kind": dis_info["scale_kind"],
            "scale": dis_info["scale"],
        },
        "q_marginal": float(q_marg), "q_weighted": float(q_w),
        "examples": [
            pack(idx_a, "A (sources agree)"),
            pack(idx_b, "B (sources conflict)"),
        ],
    }
    if len(out["examples"]) != WORKED_EXAMPLE_COUNT:
        raise ValueError(
            "worked-example output count differs from the canonical contract"
        )
    attach_run_provenance(
        out,
        ROOT,
        seed=SEED,
        campaign_tag=campaign_tag,
        requested_device=WORKED_PAPER_RUN_CONFIG["device"],
    )
    output = args.out_dir / "worked_examples.json"
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, output)
    print(json.dumps(out["examples"], indent=1)[:1200])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
