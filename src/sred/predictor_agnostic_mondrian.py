"""Predictor-agnostic Mondrian / disagreement calibration sweep.

Wraps three base predictors with multiple calibration schemes and reports
per-row + per-disagreement-bin coverage and width. Demonstrates that the
multi-modal disagreement-aware Mondrian / locally-adaptive scheme is *base-
predictor-agnostic*: it improves conditional validity (and sometimes width)
on top of vanilla split-CP, CQR, and our aug_theta+CQR alike.

Base predictors (for each configured paper seed):
  1. XGB-point + |y − ŷ|     (vanilla split-CP)
  2. XGB-quantile + CQR        (locally-adaptive split-CP)
  3. aug_theta + CQR           (our SEMF-derived predictor + CQR)

Calibration schemes per base:
  A. Marginal (single q̂)
  B. Disagreement-Mondrian (configured quantile bins of s_dis)
  C. Disagreement-weighted (locally-adaptive: rescale residuals by sqrt(α₀ + α₁·s_dis²))

Note: weighted paths use the pure-expansion score max(e, 0) of paper eq. 5,
matching the cross-dataset pipeline (switched 2026-08-04 for the from-scratch
regeneration; the 2026-05 packaged summaries had used signed scores).

Outputs:
  results/sred_semf/predagn_mondrian_per_seed.csv
  results/sred_semf/predagn_mondrian_agg.json
  results/sred_semf/predagn_mondrian_table.csv
"""

from __future__ import annotations

import os
import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2])).resolve()
SRED = ROOT / "data" / "sred"
META = SRED / "metadata"
CACHE = SRED / "embeddings"
RESULTS = ROOT / "results" / "sred_semf"

sys.path.insert(0, str(ROOT / "src" / "sred"))
# The SEMF package, needed only so the saved pickles resolve their classes.
sys.path.insert(0, str(ROOT / "src"))
from conformal import evaluate as eval_intervals  # noqa: E402
from multimodal_calibration.reproducibility import (  # noqa: E402
    artifact_identity,
    attach_run_provenance,
    atomic_write_json,
    validate_campaign_tag,
)
from multimodal_calibration.experiment_config import (  # noqa: E402
    PAPER_CPU_DEVICE,
    PAPER_DISAGREEMENT_BINS,
    PAPER_SEEDS,
    PAPER_XGB_EVAL_METRIC,
    PAPER_XGB_TREE_METHOD,
    PREDMOND_IMPLEMENTATION_POLICY,
    PREDMOND_PAPER_RUN_CONFIG,
    SEMF_PAPER_RUN_CONFIG,
    SRED_TABULAR_COLUMNS,
    canonical_run_config,
    require_canonical_run_config,
)
from multimodal_calibration.result_aggregation import (  # noqa: E402
    predictor_mondrian_table,
)
from nciw import evaluate_with_nciw  # noqa: E402
from augmented_theta import (  # noqa: E402
    validate_augmented_companions,
    validate_semf_companions,
)


def predmond_run_config(alpha: float) -> dict:
    config = canonical_run_config("predictor_mondrian")
    config["alpha"] = float(alpha)
    return config


class _PicklePlaceholder:
    """Attribute container for old SEMF classes needed only for saved arrays."""

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.__dict__["_state"] = state


class _AttributeOnlyUnpickler(pickle.Unpickler):
    """Load old SEMF artifacts without importing the full training package."""

    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (AttributeError, ImportError, ModuleNotFoundError):
            if module == "models" or module.startswith("semf"):
                return _PicklePlaceholder
            raise


def load_semf_state(path: Path):
    with open(path, "rb") as f:
        return _AttributeOnlyUnpickler(f).load()


def _stem(base: str, tag: str) -> str:
    """Build an artifact filename stem, inserting `tag` only when set."""
    return f"{base}_{tag}" if tag else base


def eval_full(
    y,
    lo,
    hi,
    f=None,
    alpha=PREDMOND_PAPER_RUN_CONFIG["alpha"],
):
    """Coverage + raw width + range-normalized width + calibrated width."""
    base = eval_intervals(y, lo, hi)
    nc = evaluate_with_nciw(y, lo, hi, alpha=alpha, f=f)
    base.update({"niw": nc["niw"], "nciw": nc["nciw"], "c_test_cal": nc["c_test_cal"]})
    return base


# ---------------------------------------------------------------------------
# data


def load_meta():
    tr = pd.read_csv(META / "train_data_with_text.csv", encoding="latin-1")
    te = pd.read_csv(META / "test_data_with_text.csv", encoding="latin-1")
    return tr, te


def load_cached(split, kind, slug):
    return np.load(CACHE / f"{split}_{kind}_{slug}.npy")


def pca_block(tr, te, n, seed):
    n = min(n, tr.shape[1])
    pca = PCA(n_components=n, random_state=seed)
    return pca.fit_transform(tr).astype(np.float32), pca.transform(te).astype(np.float32)


def canonical_block_slices() -> dict[str, tuple[int, int]]:
    """Return SRED feature blocks from the shared tab/PCA specification."""
    features = SEMF_PAPER_RUN_CONFIG["features"]
    start = 0
    blocks: dict[str, tuple[int, int]] = {}
    tab_count = len(SRED_TABULAR_COLUMNS)
    blocks["tab"] = (start, start + tab_count)
    start += tab_count
    for field in PREDMOND_IMPLEMENTATION_POLICY["text_fields"]:
        width = features["text_pca"]
        blocks[f"txt_{field}"] = (start, start + width)
        start += width
    for image_type in features["image_types"]:
        width = features["image_pca"]
        blocks[f"img_{image_type}"] = (start, start + width)
        start += width
    return blocks


def build_features(tr_meta, te_meta, seed: int):
    """Build the standard 36-feature matrix (tab + 4 PCA blocks)."""
    features = SEMF_PAPER_RUN_CONFIG["features"]
    tab_cols = list(SRED_TABULAR_COLUMNS)
    text_slug = features["text_slug"]
    img_slug = features["image_slug"]
    feats_tr = [tr_meta[tab_cols].astype(np.float32).to_numpy()]
    feats_te = [te_meta[tab_cols].astype(np.float32).to_numpy()]
    tab_count = len(tab_cols)
    block_slices = {"tab": (0, tab_count)}
    j = tab_count
    text_fields = PREDMOND_IMPLEMENTATION_POLICY["text_fields"]
    text_pca = features["text_pca"]
    for f in text_fields:
        ptr, pte = pca_block(load_cached("train", f, text_slug),
                             load_cached("test", f, text_slug), text_pca, seed)
        feats_tr.append(ptr); feats_te.append(pte)
        block_slices[f"txt_{f}"] = (j, j + text_pca); j += text_pca
    image_pca = features["image_pca"]
    for k in features["image_types"]:
        ptr, pte = pca_block(load_cached("train", k, img_slug),
                             load_cached("test", k, img_slug), image_pca, seed)
        feats_tr.append(ptr); feats_te.append(pte)
        block_slices[f"img_{k}"] = (j, j + image_pca); j += image_pca
    X_tr = np.concatenate(feats_tr, axis=1).astype(np.float32)
    X_te = np.concatenate(feats_te, axis=1).astype(np.float32)
    y_tr = np.log(tr_meta["price"].to_numpy(dtype=np.float64))
    y_te = np.log(te_meta["price"].to_numpy(dtype=np.float64))
    return X_tr, X_te, y_tr, y_te, block_slices


def split_train_valid(
    n: int,
    seed: int,
    valid_size: float = SEMF_PAPER_RUN_CONFIG["data_valid_fraction"],
):
    rng = np.random.RandomState(seed)
    idx = np.arange(n); rng.shuffle(idx)
    cut = int((1 - valid_size) * n)
    return idx[:cut], idx[cut:]


def split_tune_cal(
    n: int,
    seed: int,
    tune_frac: float = PREDMOND_PAPER_RUN_CONFIG[
        "validation_tune_fraction"
    ],
):
    rng = np.random.RandomState(
        seed + PREDMOND_PAPER_RUN_CONFIG["split_seed_offset"]
    )
    idx = np.arange(n); rng.shuffle(idx)
    cut = int(round(tune_frac * n))
    minimum = PREDMOND_IMPLEMENTATION_POLICY[
        "minimum_tune_or_calibration_size"
    ]
    cut = min(max(cut, minimum), n - minimum)
    return idx[:cut], idx[cut:]


# ---------------------------------------------------------------------------
# base predictors


def fit_xgb_point(X_tr, y_tr, X_va, y_va, seed):
    config = PREDMOND_PAPER_RUN_CONFIG["xgb_point"]
    m = xgb.XGBRegressor(
        **config,
        eval_metric=PAPER_XGB_EVAL_METRIC,
        tree_method=PAPER_XGB_TREE_METHOD,
        random_state=seed,
    )
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return m


def fit_xgb_quantile(X_tr, y_tr, X_va, y_va, alpha, seed):
    common = dict(PREDMOND_PAPER_RUN_CONFIG["xgb_quantile"])
    common.update(
        tree_method=PAPER_XGB_TREE_METHOD,
        random_state=seed,
    )
    lo = xgb.XGBRegressor(objective="reg:quantileerror",
                          quantile_alpha=alpha / 2, **common)
    hi = xgb.XGBRegressor(objective="reg:quantileerror",
                          quantile_alpha=1 - alpha / 2, **common)
    lo.fit(X_tr, y_tr); hi.fit(X_tr, y_tr)
    return lo, hi


# ---------------------------------------------------------------------------
# calibration


def cal_split_abs(scores, n, alpha):
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    if n == 0:
        return float("inf")
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return float("inf")
    return float(np.partition(scores, k - 1)[k - 1])


def calibrate_marginal(scores, alpha):
    return cal_split_abs(scores, len(scores), alpha)


def calibrate_mondrian(scores, strata, n_strata, alpha):
    qs = {}
    for s in range(n_strata):
        mask = strata == s
        qs[s] = cal_split_abs(scores[mask], int(mask.sum()), alpha)
    return qs


def quantile_bin(
    values,
    ref_values,
    n_bins=PAPER_DISAGREEMENT_BINS,
):
    edges = np.quantile(ref_values, np.linspace(0, 1, n_bins + 1))
    return np.clip(np.digitize(values, edges[1:-1]), 0, n_bins - 1)


def tune_alpha01(residuals_va, sigma_va, alpha):
    """Grid search (α₀, α₁) minimising q̂ × MPIW proxy under residual rescaling.

    Score: max(e_i, 0) / sqrt(α₀ + α₁ sigma_i²), the pure-expansion score of
    paper eq. 5 (a no-op for absolute residuals). q̂ at level (1-α). We pick
    the (α₀, α₁) that gives the smallest mean predicted half-width on
    validation.
    """
    best = (1.0, 0.0, np.inf, None)
    grid = PREDMOND_PAPER_RUN_CONFIG["weighted_scale_grid"]
    for a0 in grid["a0"]:
        for a1 in grid["a1"]:
            scale = np.sqrt(a0 + a1 * sigma_va ** 2)
            scores = np.maximum(residuals_va, 0.0) / scale
            q = cal_split_abs(scores, len(scores), alpha)
            mean_w = float(np.mean(q * scale))
            if mean_w < best[2]:
                best = (a0, a1, mean_w, q)
    return best  # (a0, a1, mean_w_va, q_hat)


# ---------------------------------------------------------------------------
# per-modality solo predictors → disagreement signal


def fit_modality_solo(X_tr, y_tr, X_va, X_te, block_slices, seed):
    config = PREDMOND_PAPER_RUN_CONFIG["modality_solo_xgb"]
    preds = {}  # modality -> (yhat_va, yhat_te)
    for name, (s, e) in block_slices.items():
        X_tr_m = X_tr[:, s:e]
        X_va_m = X_va[:, s:e]
        X_te_m = X_te[:, s:e]
        m = xgb.XGBRegressor(
            **config,
            eval_metric=PAPER_XGB_EVAL_METRIC,
            tree_method=PAPER_XGB_TREE_METHOD,
            random_state=seed,
        )
        es_cut = int(
            (
                1
                - PREDMOND_IMPLEMENTATION_POLICY[
                    "modality_solo_validation_fraction"
                ]
            )
            * len(X_tr_m)
        )
        m.fit(X_tr_m[:es_cut], y_tr[:es_cut],
              eval_set=[(X_tr_m[es_cut:], y_tr[es_cut:])], verbose=False)
        preds[name] = (m.predict(X_va_m).astype(np.float32),
                       m.predict(X_te_m).astype(np.float32))
    return preds


def disagreement(preds: dict[str, tuple[np.ndarray, np.ndarray]]):
    va = np.std(np.stack([v for v, _ in preds.values()], axis=0), axis=0)
    te = np.std(np.stack([t for _, t in preds.values()], axis=0), axis=0)
    return va.astype(np.float32), te.astype(np.float32)


# ---------------------------------------------------------------------------
# evaluators per base predictor


def run_xgb_point(X_train, y_train, X_tune, y_tune, X_cal, y_cal, X_test, y_test,
                  s_dis_tune, s_dis_cal, s_dis_te, alpha, seed) -> dict:
    """Vanilla XGB-point + abs residual; with marginal / mondrian / weighted."""
    es_cut = int(
        (
            1
            - PREDMOND_IMPLEMENTATION_POLICY[
                "modality_solo_validation_fraction"
            ]
        )
        * len(X_train)
    )
    model = fit_xgb_point(X_train[:es_cut], y_train[:es_cut],
                          X_train[es_cut:], y_train[es_cut:], seed)
    yhat_tune = model.predict(X_tune)
    yhat_cal = model.predict(X_cal)
    yhat_te = model.predict(X_test)
    e_cal = np.abs(y_cal - yhat_cal)

    out = {"point": {
        "rmse": float(np.sqrt(mean_squared_error(y_test, yhat_te))),
        "mae": float(mean_absolute_error(y_test, yhat_te)),
        "r2": float(r2_score(y_test, yhat_te)),
    }}

    # A. marginal
    q = calibrate_marginal(e_cal, alpha)
    lo, hi = yhat_te - q, yhat_te + q
    out["marginal"] = eval_full(y_test, lo, hi, f=yhat_te, alpha=alpha); out["marginal"]["q"] = q

    # B. disagreement-Mondrian
    bins_cal = quantile_bin(
        s_dis_cal, s_dis_tune, n_bins=PAPER_DISAGREEMENT_BINS
    )
    bins_te = quantile_bin(
        s_dis_te, s_dis_tune, n_bins=PAPER_DISAGREEMENT_BINS
    )
    qs = calibrate_mondrian(
        e_cal, bins_cal, PAPER_DISAGREEMENT_BINS, alpha
    )
    lo = yhat_te - np.array([qs[b] for b in bins_te])
    hi = yhat_te + np.array([qs[b] for b in bins_te])
    out["mondrian_dis"] = eval_full(y_test, lo, hi, f=yhat_te, alpha=alpha)
    out["mondrian_dis"]["qs"] = {int(k): v for k, v in qs.items()}

    # C. disagreement-weighted
    # Absolute residuals are the point-base score (eq. 5's clip is then a
    # no-op); the CQR bases instead pass signed scores that the tuner clips.
    a0, a1, _, _ = tune_alpha01(np.abs(y_tune - yhat_tune), s_dis_tune, alpha)
    scale_cal = np.sqrt(a0 + a1 * s_dis_cal ** 2)
    scale_te = np.sqrt(a0 + a1 * s_dis_te ** 2)
    scores = e_cal / scale_cal
    q = calibrate_marginal(scores, alpha)
    lo, hi = yhat_te - q * scale_te, yhat_te + q * scale_te
    out["weighted_dis"] = eval_full(y_test, lo, hi, f=yhat_te, alpha=alpha)
    out["weighted_dis"]["a0"] = a0; out["weighted_dis"]["a1"] = a1; out["weighted_dis"]["q"] = q
    return out


def run_xgb_quantile(X_train, y_train, X_tune, y_tune, X_cal, y_cal, X_test, y_test,
                     s_dis_tune, s_dis_cal, s_dis_te, alpha, seed) -> dict:
    lo_m, hi_m = fit_xgb_quantile(X_train, y_train, X_tune, y_tune, alpha, seed)
    lo_tune, hi_tune = lo_m.predict(X_tune), hi_m.predict(X_tune)
    lo_cal, hi_cal = lo_m.predict(X_cal), hi_m.predict(X_cal)
    lo_te, hi_te = lo_m.predict(X_test),  hi_m.predict(X_test)
    s_tune = np.maximum(lo_tune - y_tune, y_tune - hi_tune)
    s_cal = np.maximum(lo_cal - y_cal, y_cal - hi_cal)
    out = {"point": {
        "rmse": float(np.sqrt(mean_squared_error(y_test, (lo_te + hi_te) / 2))),
        "mae": float(mean_absolute_error(y_test, (lo_te + hi_te) / 2)),
        "r2": float(r2_score(y_test, (lo_te + hi_te) / 2)),
    }}
    f_te = (lo_te + hi_te) / 2.0
    # A. marginal CQR
    q = calibrate_marginal(s_cal, alpha)
    out["marginal"] = eval_full(y_test, lo_te - q, hi_te + q, f=f_te, alpha=alpha)
    out["marginal"]["q"] = q

    # B. disagreement-Mondrian on CQR score
    bins_cal = quantile_bin(
        s_dis_cal, s_dis_tune, n_bins=PAPER_DISAGREEMENT_BINS
    )
    bins_te = quantile_bin(
        s_dis_te, s_dis_tune, n_bins=PAPER_DISAGREEMENT_BINS
    )
    qs = calibrate_mondrian(
        s_cal, bins_cal, PAPER_DISAGREEMENT_BINS, alpha
    )
    qb = np.array([qs[b] for b in bins_te])
    out["mondrian_dis"] = eval_full(y_test, lo_te - qb, hi_te + qb, f=f_te, alpha=alpha)
    out["mondrian_dis"]["qs"] = {int(k): v for k, v in qs.items()}

    # C. disagreement-weighted on CQR score
    a0, a1, _, _ = tune_alpha01(s_tune, s_dis_tune, alpha)
    scale_cal = np.sqrt(a0 + a1 * s_dis_cal ** 2)
    scale_te = np.sqrt(a0 + a1 * s_dis_te ** 2)
    scores = np.maximum(s_cal, 0.0) / scale_cal  # pure-expansion score (paper eq. 5)
    q = calibrate_marginal(scores, alpha)
    out["weighted_dis"] = eval_full(y_test, lo_te - q * scale_te, hi_te + q * scale_te, f=f_te, alpha=alpha)
    out["weighted_dis"]["a0"] = a0; out["weighted_dis"]["a1"] = a1; out["weighted_dis"]["q"] = q
    return out


def run_aug_theta(seed, alpha, tune_idx, cal_idx, s_dis_tune, s_dis_cal, s_dis_te,
                  aug_theta_tag: str = "", y_valid_expected=None,
                  results_dir: Path = RESULTS) -> dict:
    """Use the saved aug_theta MC samples; apply marginal / mondrian / weighted."""
    path = (
        results_dir
        / f"{_stem('aug_theta', aug_theta_tag)}_seed{seed}_aug_full_samples.npz"
    )
    with np.load(path, allow_pickle=False) as npz:
        y_valid = np.asarray(npz["y_valid"])
        y_te = np.asarray(npz["y_test"])
        sv_all = np.asarray(npz["valid_full"])
        st = np.asarray(npz["test_full"])
    if y_valid_expected is not None:
        exp = np.asarray(y_valid_expected, dtype=np.float64)
        if len(y_valid) != len(exp) or not np.allclose(np.asarray(y_valid, dtype=np.float64), exp):
            raise ValueError(
                f"seed {seed}: aug_theta npz validation labels do not match the "
                "SEMF pickle splits; check --semf-tag/--aug-theta-tag pairing."
            )
    y_tune = y_valid[tune_idx]; y_cal = y_valid[cal_idx]
    sv_tune = sv_all[tune_idx]; sv_cal = sv_all[cal_idx]
    lo_q = (alpha / 2) * 100; hi_q = (1 - alpha / 2) * 100
    lo_tune = np.percentile(sv_tune, lo_q, axis=1); hi_tune = np.percentile(sv_tune, hi_q, axis=1)
    lo_cal = np.percentile(sv_cal, lo_q, axis=1); hi_cal = np.percentile(sv_cal, hi_q, axis=1)
    lo_te = np.percentile(st, lo_q, axis=1); hi_te = np.percentile(st, hi_q, axis=1)
    yhat_te = st.mean(axis=1)
    s_tune = np.maximum(lo_tune - y_tune, y_tune - hi_tune)
    s_cal = np.maximum(lo_cal - y_cal, y_cal - hi_cal)
    out = {"point": {
        "rmse": float(np.sqrt(mean_squared_error(y_te, yhat_te))),
        "mae": float(mean_absolute_error(y_te, yhat_te)),
        "r2": float(r2_score(y_te, yhat_te)),
    }}
    # marginal CQR
    q = calibrate_marginal(s_cal, alpha)
    out["marginal"] = eval_full(y_te, lo_te - q, hi_te + q, f=yhat_te, alpha=alpha); out["marginal"]["q"] = q
    # mondrian by disagreement bins
    bins_cal = quantile_bin(
        s_dis_cal, s_dis_tune, n_bins=PAPER_DISAGREEMENT_BINS
    )
    bins_te = quantile_bin(
        s_dis_te, s_dis_tune, n_bins=PAPER_DISAGREEMENT_BINS
    )
    qs = calibrate_mondrian(
        s_cal, bins_cal, PAPER_DISAGREEMENT_BINS, alpha
    )
    qb = np.array([qs[b] for b in bins_te])
    out["mondrian_dis"] = eval_full(y_te, lo_te - qb, hi_te + qb, f=yhat_te, alpha=alpha)
    out["mondrian_dis"]["qs"] = {int(k): v for k, v in qs.items()}
    # weighted
    a0, a1, _, _ = tune_alpha01(s_tune, s_dis_tune, alpha)
    scale_cal = np.sqrt(a0 + a1 * s_dis_cal ** 2)
    scale_te = np.sqrt(a0 + a1 * s_dis_te ** 2)
    scores = np.maximum(s_cal, 0.0) / scale_cal  # pure-expansion score (paper eq. 5)
    q = calibrate_marginal(scores, alpha)
    out["weighted_dis"] = eval_full(y_te, lo_te - q * scale_te, hi_te + q * scale_te, f=yhat_te, alpha=alpha)
    out["weighted_dis"]["a0"] = a0; out["weighted_dis"]["a1"] = a1; out["weighted_dis"]["q"] = q
    return out


# ---------------------------------------------------------------------------
# main


def main(
    seeds=PAPER_SEEDS,
    alpha=PREDMOND_PAPER_RUN_CONFIG["alpha"],
    semf_tag: str = "",
    aug_theta_tag: str = "",
    results_dir: Path = RESULTS,
    out_dir: Path | None = None,
    campaign_tag: str | None = None,
    paper_run: bool = False,
):
    """Use the SEMF pickle's stored splits for perfect alignment with saved aug_theta npz."""
    results_dir.mkdir(parents=True, exist_ok=True)
    output_dir = out_dir or results_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    campaign_tag = validate_campaign_tag(
        campaign_tag, required=paper_run
    )
    run_config = predmond_run_config(alpha)
    if paper_run:
        if tuple(sorted(set(seeds))) != PAPER_SEEDS:
            raise ValueError("paper run requires the canonical seed grid")
        if not semf_tag or not aug_theta_tag:
            raise ValueError("paper run requires explicit SEMF and augmented tags")
        if campaign_tag != semf_tag or campaign_tag != aug_theta_tag:
            raise ValueError("paper run requires matching campaign/input tags")
        require_canonical_run_config("predictor_mondrian", run_config)
        semf_companions = validate_semf_companions(
            results_dir, tuple(seeds), semf_tag
        )
        validate_augmented_companions(
            results_dir,
            tuple(seeds),
            aug_theta_tag,
            semf_payloads=semf_companions,
        )
    per_seed_rows = []
    input_artifacts: dict[str, dict[str, dict[str, object]]] = {}
    for seed in seeds:
        print(f"\n=== seed {seed} ===")
        # Load SEMF pickle to get standardized splits matching the aug_theta npz exactly.
        semf_path = results_dir / f"{_stem('semf', semf_tag)}_seed{seed}.pkl"
        if not semf_path.exists():
            raise FileNotFoundError(
                f"Missing saved SEMF state {semf_path}. This analysis requires "
                "the untracked SEMF prediction artifacts used to produce the summaries."
            )
        aug_samples_path = (
            results_dir
            / f"{_stem('aug_theta', aug_theta_tag)}_seed{seed}_aug_full_samples.npz"
        )
        input_artifacts[str(seed)] = {
            "semf_checkpoint": artifact_identity(semf_path),
            "augmented_samples": artifact_identity(aug_samples_path),
        }
        semf = load_semf_state(semf_path)
        Xt = np.asarray(semf.x_train, dtype=np.float32)
        Xv = np.asarray(semf.x_valid, dtype=np.float32)
        Xte = np.asarray(semf.x_test, dtype=np.float32)
        yt = np.asarray(semf.y_train, dtype=np.float32).squeeze()
        yv = np.asarray(semf.y_valid, dtype=np.float32).squeeze()
        yte = np.asarray(semf.y_test, dtype=np.float32).squeeze()
        tune_idx, cal_idx = split_tune_cal(len(yv), seed)
        block_slices = canonical_block_slices()
        expected_width = max(end for _start, end in block_slices.values())
        if Xt.shape[1] != expected_width:
            raise ValueError(
                "saved SEMF feature width differs from canonical blocks: "
                f"{Xt.shape[1]} != {expected_width}"
            )

        # disagreement signal
        solo = fit_modality_solo(Xt, yt, Xv, Xte, block_slices, seed)
        s_dis_va, s_dis_te = disagreement(solo)
        s_dis_tune = s_dis_va[tune_idx]
        s_dis_cal = s_dis_va[cal_idx]
        print(f"  s_dis tune: mean={s_dis_tune.mean():.3f} max={s_dis_tune.max():.3f}")
        print(f"  split: n_fit={len(yt)} n_tune={len(tune_idx)} n_cal={len(cal_idx)} n_test={len(yte)}")

        # Canonical base-predictor by calibration grid.
        r_pt = run_xgb_point(
            Xt, yt, Xv[tune_idx], yv[tune_idx], Xv[cal_idx], yv[cal_idx], Xte, yte,
            s_dis_tune, s_dis_cal, s_dis_te, alpha, seed,
        )
        r_qt = run_xgb_quantile(
            Xt, yt, Xv[tune_idx], yv[tune_idx], Xv[cal_idx], yv[cal_idx], Xte, yte,
            s_dis_tune, s_dis_cal, s_dis_te, alpha, seed,
        )
        r_at = run_aug_theta(seed, alpha, tune_idx, cal_idx, s_dis_tune, s_dis_cal, s_dis_te,
                             aug_theta_tag=aug_theta_tag, y_valid_expected=yv,
                             results_dir=results_dir)

        for base, r in [("xgb_point", r_pt), ("xgb_quantile", r_qt), ("aug_theta", r_at)]:
            for calib in ("marginal", "mondrian_dis", "weighted_dis"):
                row = dict(seed=seed, base=base, calibration=calib)
                row.update(r["point"])
                row.update({k: v for k, v in r[calib].items()
                            if k in ("picp", "mpiw", "crps_uniform", "niw", "nciw", "c_test_cal")})
                # Persist the selected scale and final quantiles so the
                # non-negativity of weighted quantiles is auditable from the
                # artifact itself (review finding, 2026-08-04).
                for k in ("q", "a0", "a1"):
                    if k in r[calib]:
                        row[k] = r[calib][k]
                if "qs" in r[calib]:
                    for b, v in r[calib]["qs"].items():
                        row[f"q_bin{b}"] = v
                per_seed_rows.append(row)
        print(f"  done.")

    df = pd.DataFrame(per_seed_rows)
    if paper_run:
        expected_grid = {
            (seed, base, calibration)
            for seed in PAPER_SEEDS
            for base in PREDMOND_PAPER_RUN_CONFIG["bases"]
            for calibration in PREDMOND_PAPER_RUN_CONFIG["calibrations"]
        }
        actual_grid = set(
            zip(df.seed.astype(int), df.base, df.calibration)
        )
        if actual_grid != expected_grid or len(df) != len(expected_grid):
            raise RuntimeError(
                "SRED predictor-agnostic output grid differs; "
                f"missing={sorted(expected_grid - actual_grid)}, "
                f"unexpected={sorted(actual_grid - expected_grid)}"
            )
        required_numeric = [
            "r2", "rmse", "mae", "picp", "mpiw", "niw",
            "nciw", "c_test_cal", "crps_uniform",
        ]
        if not np.isfinite(
            df[required_numeric].to_numpy(dtype=float)
        ).all():
            raise RuntimeError(
                "SRED predictor-agnostic output contains non-finite metrics"
            )
    print("\n=== per-seed rows ===")
    print(df.head(10).to_string())

    out, summary = predictor_mondrian_table(df)

    print(
        "\n=== predictor-agnostic Mondrian table "
        f"({len(PAPER_SEEDS)} seeds) ==="
    )
    print(out.to_string())

    csv_path = output_dir / "predagn_mondrian_table.csv"
    out.to_csv(csv_path)
    df.to_csv(output_dir / "predagn_mondrian_per_seed.csv", index=False)
    payload = {
        "config": {
            "campaign_tag": campaign_tag,
            "semf_tag": semf_tag,
            "aug_theta_tag": aug_theta_tag,
            "run_config": run_config,
        },
        "alpha": alpha,
        "input_artifacts": input_artifacts,
        "rows": (
            df.astype(object)
            .where(pd.notna(df), None)
            .to_dict(orient="records")
        ),
        "summary": summary,
    }
    attach_run_provenance(
        payload,
        ROOT,
        seed=None,
        campaign_tag=campaign_tag,
        requested_device=PAPER_CPU_DEVICE,
    )
    atomic_write_json(
        output_dir / "predagn_mondrian_agg.json",
        payload,
    )
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=list(PAPER_SEEDS))
    parser.add_argument(
        "--alpha", type=float, default=PREDMOND_PAPER_RUN_CONFIG["alpha"]
    )
    parser.add_argument("--semf-tag", default="",
                        help="tag of the SEMF pickle to load; must match the --tag "
                             "used by run_full_experiment.py (default: untagged, "
                             "semf_seed<n>.pkl)")
    parser.add_argument("--aug-theta-tag", default="",
                        help="tag of the aug_theta samples to load; must match the "
                             "--tag used by augmented_theta.py (default: untagged, "
                             "aug_theta_seed<n>_aug_full_samples.npz)")
    parser.add_argument("--results-dir", type=Path, default=RESULTS)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--campaign-tag",
        default=None,
        help="reproduction identifier recorded in run provenance",
    )
    parser.add_argument("--paper-run", action="store_true")
    args = parser.parse_args()
    main(seeds=tuple(args.seeds), alpha=args.alpha,
         semf_tag=args.semf_tag, aug_theta_tag=args.aug_theta_tag,
         results_dir=args.results_dir, out_dir=args.out_dir,
         campaign_tag=args.campaign_tag, paper_run=args.paper_run)
