"""Neural baselines for the multi-modal calibration study.

Adds two strong NN reference baselines so reviewers cannot reduce the result to
"you only beat XGBoost-on-concat":

1. **MLP-on-concat (homogeneous neural)**: tuned MLP regressor on the same
   concatenated tab + frozen text emb + frozen image emb input that the
   homog_xgb_concat baseline uses.
2. **Multi-modal NN with cross-attention**: per-modality encoders + 2-layer
   transformer-style cross-attention fusion + small head.

Both are trained with point-MSE on standardized y, then a marginal residual
split-conformal layer is fit on validation residuals to provide intervals
comparable to the existing hetero_gated CQR widths.

Each run emits a JSON whose filename includes the architecture, dataset,
alpha-derived tag, and seed.

Usage:

    python src/multimodal_calibration/run_nn_baselines.py \\
        --datasets sred mercari pawpularity imdb_wiki \\
        --archs mlp_concat mm_attn_nn \\
        --seeds 0 1 2 3 4 --alpha 0.05
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
EXP = Path(__file__).resolve().parent
SRED_EXP = ROOT / "src" / "sred"
RESULTS = ROOT / "results" / "multimodal_calibration"
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(SRED_EXP))

from nciw import compute_nciw  # noqa: E402
from calibration import conformal_quantile  # noqa: E402
from experiment_config import (  # noqa: E402
    AUXILIARY_NEURAL_EVAL_BATCH_SIZE,
    AUXILIARY_NEURAL_TRAINING_POLICY,
    AUXILIARY_NEURAL_TINY_POLICY,
    AUXILIARY_NEURAL_WARMUP_FRACTION,
    AUXILIARY_PAPER_RUN_CONFIG,
    PAPER_DATASETS,
    PAPER_NEURAL_ARCHITECTURES,
    PAPER_SEEDS,
    require_canonical_run_config,
)
from reproducibility import (  # noqa: E402
    attach_run_provenance,
    thread_identity,
    validate_campaign_payloads,
    validate_campaign_tag,
    write_campaign_manifest,
)
from result_grid import (  # noqa: E402
    AUXILIARY_CAMPAIGN_METHODS,
    auxiliary_campaign_scope,
    auxiliary_run_config,
)

# Reuse data assembly + preprocessing from run_hetero_mixture so the inputs
# are byte-identical to the existing baseline runs.
from run_hetero_mixture import (  # noqa: E402
    _alpha_tag,
    _build_blocks,
    _load_dataset,
    _split_fit_tune_cal,
    _standardize_y,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Architectures
# ---------------------------------------------------------------------------


class MLPConcatRegressor(nn.Module):
    """3-hidden-layer MLP regressor on concatenated tab+text+image features.

    LayerNorm -> Linear -> GELU -> Dropout per block. Single regression head.
    """

    def __init__(
        self,
        in_dim: int,
        hidden: tuple[int, int, int] = tuple(
            AUXILIARY_PAPER_RUN_CONFIG["neural"]["mlp_concat"]["hidden"]
        ),
        dropout: float = AUXILIARY_PAPER_RUN_CONFIG["neural"]["dropout"],
    ):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [
                nn.LayerNorm(prev),
                nn.Linear(prev, h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return self.body(x).squeeze(-1)


class MultiModalAttnRegressor(nn.Module):
    """Per-modality encoders + cross-attention fusion + MLP head.

    Each modality is mapped to a 64-dim token, then 2 layers of standard
    transformer-style multi-head attention (4 heads) are applied. Tokens are
    mean-pooled and passed through a [64, 1] head.
    """

    def __init__(
        self,
        d_tab: int = 0,
        d_text: int = 0,
        d_image: int = 0,
        d_token: int = AUXILIARY_PAPER_RUN_CONFIG["neural"]["mm_attn_nn"][
            "d_token"
        ],
        n_heads: int = AUXILIARY_PAPER_RUN_CONFIG["neural"]["mm_attn_nn"][
            "n_heads"
        ],
        n_layers: int = AUXILIARY_PAPER_RUN_CONFIG["neural"]["mm_attn_nn"][
            "n_layers"
        ],
        dropout: float = AUXILIARY_PAPER_RUN_CONFIG["neural"]["dropout"],
    ):
        super().__init__()
        self.d_token = d_token

        def encoder(in_dim):
            if in_dim <= 0:
                return None
            return nn.Sequential(
                nn.Linear(in_dim, d_token),
                nn.LayerNorm(d_token),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        self.enc_tab = encoder(d_tab)
        self.enc_text = encoder(d_text)
        self.enc_image = encoder(d_image)

        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(d_token, num_heads=n_heads, dropout=dropout,
                                  batch_first=True)
            for _ in range(n_layers)
        ])
        self.norm_layers = nn.ModuleList([nn.LayerNorm(d_token) for _ in range(n_layers)])
        self.ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(
                    d_token,
                    d_token
                    * AUXILIARY_NEURAL_TRAINING_POLICY[
                        "attention_feedforward_expansion"
                    ],
                ),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(
                    d_token
                    * AUXILIARY_NEURAL_TRAINING_POLICY[
                        "attention_feedforward_expansion"
                    ],
                    d_token,
                ),
            )
            for _ in range(n_layers)
        ])
        self.norm_ff = nn.ModuleList([nn.LayerNorm(d_token) for _ in range(n_layers)])

        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.Linear(d_token, d_token),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_token, 1),
        )

    def _encode_tokens(self, x_tab, x_text, x_image):
        toks = []
        if self.enc_tab is not None and x_tab is not None:
            toks.append(self.enc_tab(x_tab))
        if self.enc_text is not None and x_text is not None:
            toks.append(self.enc_text(x_text))
        if self.enc_image is not None and x_image is not None:
            toks.append(self.enc_image(x_image))
        if not toks:
            raise RuntimeError("MultiModalAttnRegressor: no modalities!")
        # (B, T, d_token)
        return torch.stack(toks, dim=1)

    def forward(self, x_tab=None, x_text=None, x_image=None):
        h = self._encode_tokens(x_tab, x_text, x_image)
        for attn, n1, ff, n2 in zip(self.attn_layers, self.norm_layers,
                                     self.ff_layers, self.norm_ff):
            # Pre-norm transformer block
            h_n = n1(h)
            attn_out, _ = attn(h_n, h_n, h_n, need_weights=False)
            h = h + attn_out
            h_n = n2(h)
            h = h + ff(h_n)
        pooled = h.mean(dim=1)  # (B, d_token)
        return self.head(pooled).squeeze(-1)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------


def _cosine_warmup_lr(
    opt,
    total_steps: int,
    warmup_frac: float = AUXILIARY_NEURAL_WARMUP_FRACTION,
):
    warmup_steps = max(1, int(warmup_frac * total_steps))
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)


def _train_pointwise(
    model: nn.Module,
    train_batches,
    val_batches,
    *,
    lr: float,
    epochs: int,
    weight_decay: float,
    patience: int,
    verbose: bool = False,
):
    """Training loop with early stopping on validation MSE.

    train_batches / val_batches are iterables of dicts with keys:
      'x' or ('x_tab', 'x_text', 'x_image'), and 'y'.
    """
    optimizer_class = getattr(
        torch.optim,
        AUXILIARY_NEURAL_TRAINING_POLICY["optimizer"],
    )
    opt = optimizer_class(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    n_train_steps = epochs * sum(1 for _ in train_batches[0])  # see comment
    # train_batches is a list/sequence; we count steps lazily inside loop
    sched = None  # initialised after we know steps

    best_val = float("inf")
    best_state = None
    no_improve = 0
    history = []

    for epoch in range(epochs):
        model.train()
        running = 0.0
        n_obs = 0
        for batch in train_batches[epoch] if isinstance(train_batches, list) and len(train_batches) == epochs else train_batches:
            opt.zero_grad(set_to_none=True)
            y = batch["y"]
            if "x" in batch:
                pred = model(batch["x"])
            else:
                pred = model(
                    x_tab=batch.get("x_tab"),
                    x_text=batch.get("x_text"),
                    x_image=batch.get("x_image"),
                )
            loss = F.mse_loss(pred, y)
            loss.backward()
            opt.step()
            if sched is not None:
                sched.step()
            running += float(loss) * len(y)
            n_obs += len(y)
        train_mse = running / max(1, n_obs)

        model.eval()
        v_run = 0.0
        v_n = 0
        with torch.no_grad():
            for batch in val_batches:
                y = batch["y"]
                if "x" in batch:
                    pred = model(batch["x"])
                else:
                    pred = model(
                        x_tab=batch.get("x_tab"),
                        x_text=batch.get("x_text"),
                        x_image=batch.get("x_image"),
                    )
                loss = F.mse_loss(pred, y, reduction="sum")
                v_run += float(loss)
                v_n += len(y)
        val_mse = v_run / max(1, v_n)
        history.append((epoch, train_mse, val_mse))

        if (
            val_mse
            + AUXILIARY_NEURAL_TRAINING_POLICY[
                "early_stopping_min_delta"
            ]
            < best_val
        ):
            best_val = val_mse
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_val_mse": best_val, "epochs_trained": epoch + 1, "history": history}


# ---------------------------------------------------------------------------
# Batch builders
# ---------------------------------------------------------------------------


def _make_concat_batches(X, y, batch_size: int, shuffle: bool, seed: int = 0):
    n = len(y)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n) if shuffle else np.arange(n)
    out = []
    for s in range(0, n, batch_size):
        b = idx[s:s + batch_size]
        out.append({
            "x": torch.from_numpy(X[b].astype(np.float32)).to(DEVICE),
            "y": torch.from_numpy(y[b].astype(np.float32)).to(DEVICE),
        })
    return out


def _make_mm_batches(X_tab, X_text, X_image, y, batch_size: int,
                     shuffle: bool, seed: int = 0):
    n = len(y)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n) if shuffle else np.arange(n)
    out = []
    for s in range(0, n, batch_size):
        b = idx[s:s + batch_size]
        item = {"y": torch.from_numpy(y[b].astype(np.float32)).to(DEVICE)}
        if X_tab is not None and X_tab.shape[1] > 0:
            item["x_tab"] = torch.from_numpy(X_tab[b].astype(np.float32)).to(DEVICE)
        if X_text is not None and X_text.shape[1] > 0:
            item["x_text"] = torch.from_numpy(X_text[b].astype(np.float32)).to(DEVICE)
        if X_image is not None and X_image.shape[1] > 0:
            item["x_image"] = torch.from_numpy(X_image[b].astype(np.float32)).to(DEVICE)
        out.append(item)
    return out


# ---------------------------------------------------------------------------
# Small training driver
# ---------------------------------------------------------------------------


def train_with_early_stop(
    model: nn.Module,
    *,
    train_batches_fn,
    val_batches,
    lr: float,
    epochs: int,
    weight_decay: float,
    patience: int,
    seed: int,
    use_cosine: bool = (
        AUXILIARY_NEURAL_TRAINING_POLICY["scheduler"] == "cosine_warmup"
    ),
):
    """Generic training loop: build train batches per-epoch (re-shuffle), eval each epoch."""
    optimizer_class = getattr(
        torch.optim,
        AUXILIARY_NEURAL_TRAINING_POLICY["optimizer"],
    )
    opt = optimizer_class(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    # estimate total steps for cosine schedule
    sample_batches = train_batches_fn(seed)
    steps_per_epoch = len(sample_batches)
    total_steps = epochs * steps_per_epoch
    if use_cosine:
        sched = _cosine_warmup_lr(
            opt,
            total_steps,
            warmup_frac=AUXILIARY_NEURAL_WARMUP_FRACTION,
        )
    else:
        sched = None

    best_val = float("inf")
    best_state = None
    no_improve = 0
    history = []
    epochs_trained = 0

    for epoch in range(epochs):
        model.train()
        running = 0.0
        n_obs = 0
        train_batches = train_batches_fn(
            seed
            + epoch
            * AUXILIARY_NEURAL_TRAINING_POLICY[
                "epoch_shuffle_seed_stride"
            ]
        )
        for batch in train_batches:
            opt.zero_grad(set_to_none=True)
            y = batch["y"]
            if "x" in batch:
                pred = model(batch["x"])
            else:
                pred = model(
                    x_tab=batch.get("x_tab"),
                    x_text=batch.get("x_text"),
                    x_image=batch.get("x_image"),
                )
            loss = F.mse_loss(pred, y)
            loss.backward()
            opt.step()
            if sched is not None:
                sched.step()
            running += float(loss) * len(y)
            n_obs += len(y)
        train_mse = running / max(1, n_obs)

        model.eval()
        v_run = 0.0
        v_n = 0
        with torch.no_grad():
            for batch in val_batches:
                y = batch["y"]
                if "x" in batch:
                    pred = model(batch["x"])
                else:
                    pred = model(
                        x_tab=batch.get("x_tab"),
                        x_text=batch.get("x_text"),
                        x_image=batch.get("x_image"),
                    )
                loss = F.mse_loss(pred, y, reduction="sum")
                v_run += float(loss)
                v_n += len(y)
        val_mse = v_run / max(1, v_n)
        history.append((epoch, train_mse, val_mse))
        epochs_trained = epoch + 1

        if (
            val_mse
            + AUXILIARY_NEURAL_TRAINING_POLICY[
                "early_stopping_min_delta"
            ]
            < best_val
        ):
            best_val = val_mse
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {"best_val_mse": best_val, "epochs_trained": epochs_trained, "history": history}


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def _predict_concat(
    model: MLPConcatRegressor,
    X: np.ndarray,
    batch_size: int = AUXILIARY_NEURAL_EVAL_BATCH_SIZE,
) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[s:s + batch_size].astype(np.float32)).to(DEVICE)
            out.append(model(xb).cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def _predict_mm(
    model: MultiModalAttnRegressor,
    X_tab: Optional[np.ndarray],
    X_text: Optional[np.ndarray],
    X_image: Optional[np.ndarray],
    batch_size: int = AUXILIARY_NEURAL_EVAL_BATCH_SIZE,
) -> np.ndarray:
    model.eval()
    n = max(
        X_tab.shape[0] if X_tab is not None else 0,
        X_text.shape[0] if X_text is not None else 0,
        X_image.shape[0] if X_image is not None else 0,
    )
    out = []
    with torch.no_grad():
        for s in range(0, n, batch_size):
            kw = {}
            if X_tab is not None and X_tab.shape[1] > 0:
                kw["x_tab"] = torch.from_numpy(X_tab[s:s + batch_size].astype(np.float32)).to(DEVICE)
            if X_text is not None and X_text.shape[1] > 0:
                kw["x_text"] = torch.from_numpy(X_text[s:s + batch_size].astype(np.float32)).to(DEVICE)
            if X_image is not None and X_image.shape[1] > 0:
                kw["x_image"] = torch.from_numpy(X_image[s:s + batch_size].astype(np.float32)).to(DEVICE)
            out.append(model(**kw).cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Marginal CQR using validation residuals (point-only model -> intervals)
# ---------------------------------------------------------------------------


def _marginal_residual_calibration(
    y_calib: np.ndarray,
    point_calib: np.ndarray,
    point_test: np.ndarray,
    alpha: float = 0.1,
):
    """Symmetric-residual conformal calibration on a point predictor.

    Score: |y - point|. q_hat is the finite-sample split-conformal quantile.
    Returns (lo_test, hi_test, info).
    """
    res = np.abs(y_calib - point_calib)
    n = len(res)
    q_hat = conformal_quantile(res, alpha)
    lo = point_test - q_hat
    hi = point_test + q_hat
    return lo, hi, {"method": "marginal_residual", "q_hat": q_hat, "n_calib": n}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _point_metrics(y, point) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, point))),
        "mae": float(mean_absolute_error(y, point)),
        "r2": float(r2_score(y, point)),
    }


def _interval_metrics(y, lo, hi, alpha: float) -> dict:
    nciw_v, info = compute_nciw(y, lo, hi, alpha=alpha)
    return {
        "picp": float(np.mean((y >= lo) & (y <= hi))),
        "mpiw": float(info["base_mpiw"]),
        "niw": float(info["base_niw"]),
        "nciw": float(nciw_v),
        "c_test_cal": float(info["c_test_cal"]),
        "y_range": float(info["y_range"]),
    }


# ---------------------------------------------------------------------------
# Per-architecture training entry points
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    epochs: int = AUXILIARY_PAPER_RUN_CONFIG["neural"]["epochs"]
    patience: int = AUXILIARY_PAPER_RUN_CONFIG["neural"]["patience"]
    batch_size: int = AUXILIARY_PAPER_RUN_CONFIG["neural"]["batch_size"]
    lr: float = AUXILIARY_PAPER_RUN_CONFIG["neural"]["mlp_concat"][
        "learning_rate"
    ]
    weight_decay: float = AUXILIARY_PAPER_RUN_CONFIG["neural"]["weight_decay"]
    dropout: float = AUXILIARY_PAPER_RUN_CONFIG["neural"]["dropout"]

    def for_tiny(self) -> "TrainConfig":
        # Tiny datasets (e.g. AutoMPG, n=263): clamp epochs and bump dropout.
        return TrainConfig(
            epochs=AUXILIARY_NEURAL_TINY_POLICY["epochs"],
            patience=AUXILIARY_NEURAL_TINY_POLICY["patience"],
            batch_size=min(
                AUXILIARY_NEURAL_TINY_POLICY["maximum_batch_size"],
                self.batch_size,
            ),
            lr=self.lr,
            weight_decay=self.weight_decay,
            dropout=AUXILIARY_NEURAL_TINY_POLICY["dropout"],
        )


def run_mlp_concat(blocks: dict, fit_idx, calib_idx, y_mu, y_sd, alpha: float,
                    seed: int) -> dict:
    """Train MLP-on-concat, calibrate on val residuals, return result payload."""
    Xtab_tr = blocks["X_tab_tr"]; Xtab_te = blocks["X_tab_te"]
    Xtxt_tr = blocks["X_text_tr"]; Xtxt_te = blocks["X_text_te"]
    Ximg_tr = blocks["X_image_tr"]; Ximg_te = blocks["X_image_te"]
    y_tr_raw = blocks["y_tr"]; y_te_raw = blocks["y_te"]
    n_tr = len(y_tr_raw)

    parts_tr = []; parts_te = []
    for X_tr, X_te in [(Xtab_tr, Xtab_te), (Xtxt_tr, Xtxt_te), (Ximg_tr, Ximg_te)]:
        if X_tr.shape[1] > 0:
            parts_tr.append(X_tr); parts_te.append(X_te)
    Xall_tr = np.concatenate(parts_tr, axis=1).astype(np.float32)
    Xall_te = np.concatenate(parts_te, axis=1).astype(np.float32)

    Xall_fit = Xall_tr[fit_idx]; Xall_calib = Xall_tr[calib_idx]
    y_fit = (y_tr_raw[fit_idx] - y_mu) / y_sd
    y_calib_std = (y_tr_raw[calib_idx] - y_mu) / y_sd

    cfg = (
        TrainConfig().for_tiny()
        if n_tr < AUXILIARY_NEURAL_TINY_POLICY["train_threshold"]
        else TrainConfig()
    )
    architecture = AUXILIARY_PAPER_RUN_CONFIG["neural"]["mlp_concat"]

    torch.manual_seed(seed); np.random.seed(seed)
    model = MLPConcatRegressor(
        in_dim=Xall_fit.shape[1],
        hidden=tuple(architecture["hidden"]),
        dropout=cfg.dropout,
    ).to(DEVICE)

    # split fit into train/internal-val (90/10) for early stopping
    rng = np.random.default_rng(seed)
    n_fit = len(y_fit)
    perm = rng.permutation(n_fit)
    cut = int(
        (
            1
            - AUXILIARY_PAPER_RUN_CONFIG["neural"][
                "internal_validation_fraction"
            ]
        )
        * n_fit
    )
    tr_idx, va_idx = perm[:cut], perm[cut:]
    Xtr, ytr = Xall_fit[tr_idx], y_fit[tr_idx]
    Xva, yva = Xall_fit[va_idx], y_fit[va_idx]

    def make_train_batches(s):
        return _make_concat_batches(Xtr, ytr, batch_size=cfg.batch_size,
                                     shuffle=True, seed=s)
    val_batches = _make_concat_batches(
        Xva,
        yva,
        batch_size=AUXILIARY_NEURAL_EVAL_BATCH_SIZE,
        shuffle=False,
    )

    t0 = time.time()
    train_info = train_with_early_stop(
        model,
        train_batches_fn=make_train_batches,
        val_batches=val_batches,
        lr=cfg.lr,
        epochs=cfg.epochs,
        weight_decay=cfg.weight_decay,
        patience=cfg.patience,
        seed=seed,
    )
    train_seconds = time.time() - t0

    # predictions in standardized y
    point_calib_std = _predict_concat(model, Xall_calib)
    point_test_std = _predict_concat(model, Xall_te)

    # un-standardize for reporting
    point_calib = point_calib_std * y_sd + y_mu
    point_test = point_test_std * y_sd + y_mu
    y_calib_orig = y_tr_raw[calib_idx]
    y_te = y_te_raw

    lo_te, hi_te, cal_info = _marginal_residual_calibration(
        y_calib_orig, point_calib, point_test, alpha=alpha
    )

    res = {
        "label": "mlp_concat",
        "point": _point_metrics(y_te, point_test),
        "residual_cp": _interval_metrics(y_te, lo_te, hi_te, alpha),
        "calibration_info": cal_info,
        "train_info": {
            "best_val_mse": float(train_info["best_val_mse"]),
            "epochs_trained": int(train_info["epochs_trained"]),
            "train_seconds": float(train_seconds),
            "in_dim": int(Xall_fit.shape[1]),
            "config": {
                "epochs": cfg.epochs, "batch_size": cfg.batch_size,
                "lr": cfg.lr, "dropout": cfg.dropout,
                "weight_decay": cfg.weight_decay, "patience": cfg.patience,
                "hidden": list(architecture["hidden"]),
            },
        },
    }
    return res


def run_mm_attn(blocks: dict, fit_idx, calib_idx, y_mu, y_sd, alpha: float,
                seed: int) -> dict:
    """Train multi-modal NN with cross-attention; same calibration scheme as MLP."""
    Xtab_tr = blocks["X_tab_tr"]; Xtab_te = blocks["X_tab_te"]
    Xtxt_tr = blocks["X_text_tr"]; Xtxt_te = blocks["X_text_te"]
    Ximg_tr = blocks["X_image_tr"]; Ximg_te = blocks["X_image_te"]
    y_tr_raw = blocks["y_tr"]; y_te_raw = blocks["y_te"]
    n_tr = len(y_tr_raw)

    Xtab_fit = Xtab_tr[fit_idx]; Xtab_calib = Xtab_tr[calib_idx]
    Xtxt_fit = Xtxt_tr[fit_idx]; Xtxt_calib = Xtxt_tr[calib_idx]
    Ximg_fit = Ximg_tr[fit_idx]; Ximg_calib = Ximg_tr[calib_idx]
    y_fit = (y_tr_raw[fit_idx] - y_mu) / y_sd

    architecture = AUXILIARY_PAPER_RUN_CONFIG["neural"]["mm_attn_nn"]
    base_config = TrainConfig(lr=architecture["learning_rate"])
    cfg = (
        base_config.for_tiny()
        if n_tr < AUXILIARY_NEURAL_TINY_POLICY["train_threshold"]
        else base_config
    )

    torch.manual_seed(seed); np.random.seed(seed)
    model = MultiModalAttnRegressor(
        d_tab=Xtab_fit.shape[1],
        d_text=Xtxt_fit.shape[1],
        d_image=Ximg_fit.shape[1],
        d_token=architecture["d_token"],
        n_heads=architecture["n_heads"],
        n_layers=architecture["n_layers"],
        dropout=cfg.dropout,
    ).to(DEVICE)

    rng = np.random.default_rng(seed)
    n_fit = len(y_fit)
    perm = rng.permutation(n_fit)
    cut = int(
        (
            1
            - AUXILIARY_PAPER_RUN_CONFIG["neural"][
                "internal_validation_fraction"
            ]
        )
        * n_fit
    )
    tr_idx, va_idx = perm[:cut], perm[cut:]

    def slice_block(M, idx):
        return M[idx] if M.shape[1] > 0 else M[idx]

    Xtab_tr_in = slice_block(Xtab_fit, tr_idx)
    Xtxt_tr_in = slice_block(Xtxt_fit, tr_idx)
    Ximg_tr_in = slice_block(Ximg_fit, tr_idx)
    ytr = y_fit[tr_idx]

    Xtab_va = slice_block(Xtab_fit, va_idx)
    Xtxt_va = slice_block(Xtxt_fit, va_idx)
    Ximg_va = slice_block(Ximg_fit, va_idx)
    yva = y_fit[va_idx]

    def make_train_batches(s):
        return _make_mm_batches(
            Xtab_tr_in, Xtxt_tr_in, Ximg_tr_in, ytr,
            batch_size=cfg.batch_size, shuffle=True, seed=s,
        )
    val_batches = _make_mm_batches(
        Xtab_va, Xtxt_va, Ximg_va, yva,
        batch_size=AUXILIARY_NEURAL_EVAL_BATCH_SIZE, shuffle=False,
    )

    t0 = time.time()
    train_info = train_with_early_stop(
        model,
        train_batches_fn=make_train_batches,
        val_batches=val_batches,
        lr=cfg.lr,
        epochs=cfg.epochs,
        weight_decay=cfg.weight_decay,
        patience=cfg.patience,
        seed=seed,
    )
    train_seconds = time.time() - t0

    point_calib_std = _predict_mm(model, Xtab_calib, Xtxt_calib, Ximg_calib)
    point_test_std = _predict_mm(model, Xtab_te, Xtxt_te, Ximg_te)

    point_calib = point_calib_std * y_sd + y_mu
    point_test = point_test_std * y_sd + y_mu
    y_calib_orig = y_tr_raw[calib_idx]
    y_te = y_te_raw

    lo_te, hi_te, cal_info = _marginal_residual_calibration(
        y_calib_orig, point_calib, point_test, alpha=alpha
    )

    res = {
        "label": "mm_attn_nn",
        "point": _point_metrics(y_te, point_test),
        "residual_cp": _interval_metrics(y_te, lo_te, hi_te, alpha),
        "calibration_info": cal_info,
        "train_info": {
            "best_val_mse": float(train_info["best_val_mse"]),
            "epochs_trained": int(train_info["epochs_trained"]),
            "train_seconds": float(train_seconds),
            "modalities": {
                "tab": int(Xtab_fit.shape[1]),
                "text": int(Xtxt_fit.shape[1]),
                "image": int(Ximg_fit.shape[1]),
            },
            "config": {
                "epochs": cfg.epochs, "batch_size": cfg.batch_size,
                "lr": cfg.lr, "dropout": cfg.dropout,
                "weight_decay": cfg.weight_decay, "patience": cfg.patience,
                "d_token": architecture["d_token"],
                "n_heads": architecture["n_heads"],
                "n_layers": architecture["n_layers"],
            },
        },
    }
    return res


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


ARCH_FUNCS = {
    "mlp_concat": run_mlp_concat,
    "mm_attn_nn": run_mm_attn,
}


def run_one(
    name: str,
    arch: str,
    seed: int,
    alpha: float,
    calib_frac: float = AUXILIARY_PAPER_RUN_CONFIG["split"]["calib_frac"],
    tune_frac: float = AUXILIARY_PAPER_RUN_CONFIG["split"]["tune_frac"],
    verbose: bool = True,
) -> dict:
    if verbose:
        print(f"\n=== {arch} | {name} | seed={seed} | alpha={alpha} ===")
    train, test = _load_dataset(name)
    n_tr = len(train["y"])
    fit_idx, tune_idx, calib_idx = _split_fit_tune_cal(n_tr, calib_frac, tune_frac, seed)
    model_idx = np.concatenate([fit_idx, tune_idx])
    blocks = _build_blocks(name, train, test, fit_idx=model_idx)
    y_mu, y_sd = _standardize_y(blocks["y_tr"][model_idx])

    if verbose:
        print(f"  shapes: tab={blocks['X_tab_tr'].shape[1]} "
              f"text={blocks['X_text_tr'].shape[1]} "
              f"image={blocks['X_image_tr'].shape[1]}")
        print(f"  n_fit={len(model_idx)} n_calib={len(calib_idx)} n_test={len(blocks['y_te'])}")

    fn = ARCH_FUNCS[arch]
    res = fn(blocks, model_idx, calib_idx, y_mu, y_sd, alpha, seed)
    if verbose:
        print(f"  R2={res['point']['r2']:.4f}  RMSE={res['point']['rmse']:.4f}  "
              f"PICP={res['residual_cp']['picp']:.3f}  MPIW={res['residual_cp']['mpiw']:.4f}  "
              f"NCIW={res['residual_cp']['nciw']:.4f}  ({res['train_info']['train_seconds']:.1f}s)")
    return {
        "config": {
            "dataset": name, "arch": arch, "seed": seed, "alpha": alpha,
            "calib_frac": calib_frac, "tune_frac": tune_frac,
            "split_protocol": "fit_plus_tune_for_model_calibration_test",
            "n_train_total": int(n_tr), "n_fit": int(len(model_idx)),
            "n_tune_fold": int(len(tune_idx)),
            "n_calib": int(len(calib_idx)), "n_test": int(len(blocks["y_te"])),
            "tab_dim": int(blocks["X_tab_tr"].shape[1]),
            "text_dim": int(blocks["X_text_tr"].shape[1]),
            "image_dim": int(blocks["X_image_tr"].shape[1]),
            "encoders": blocks["enc_info"],
        },
        "result": res,
        "y_stats": {"train_mean": float(y_mu), "train_std": float(y_sd)},
    }


def _nn_path(
    name: str,
    arch: str,
    seed: int,
    tag: str | None = None,
    results_dir: Path = RESULTS,
) -> Path:
    mid = f"{arch}_{name}_{tag}" if tag else f"{arch}_{name}"
    return results_dir / f"nnbase_{mid}_seed{seed}.json"


def _save(
    payload: dict,
    name: str,
    arch: str,
    seed: int,
    tag: str | None = None,
    results_dir: Path = RESULTS,
    run_config: dict | None = None,
):
    if tag:
        payload.setdefault("config", {})["output_tag"] = tag
    payload.setdefault("config", {})["run_config"] = run_config or {}
    attach_run_provenance(
        payload,
        ROOT,
        seed=seed,
        campaign_tag=tag,
        requested_device="cuda" if torch.cuda.is_available() else "cpu",
    )
    p = _nn_path(name, arch, seed, tag, results_dir)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, p)
    print(f"  -> {p}")
    return p


def _valid_existing(
    path: Path,
    name: str,
    arch: str,
    seed: int,
    alpha: float,
    tag: str,
    run_config: dict,
) -> bool:
    try:
        payload = json.loads(path.read_text())
        cfg = payload.get("config", {})
        valid = (
            cfg.get("dataset") == name
            and cfg.get("arch") == arch
            and int(cfg.get("seed")) == seed
            and abs(float(cfg.get("alpha")) - alpha) < 1e-12
            and cfg.get("output_tag") == tag
            and "residual_cp" in payload.get("result", {})
        )
        if not valid:
            return False
        validate_campaign_payloads(
            [(path.name, payload)],
            ROOT,
            campaign_tag=tag,
            requested_device="cuda" if torch.cuda.is_available() else "cpu",
            run_config=run_config,
            producer_threads=thread_identity(),
        )
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(PAPER_DATASETS))
    ap.add_argument("--archs", nargs="+", choices=list(ARCH_FUNCS.keys()),
                    default=list(PAPER_NEURAL_ARCHITECTURES))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(PAPER_SEEDS))
    ap.add_argument(
        "--alpha", type=float, default=AUXILIARY_PAPER_RUN_CONFIG["alpha"]
    )
    ap.add_argument(
        "--tune-frac",
        type=float,
        default=AUXILIARY_PAPER_RUN_CONFIG["split"]["tune_frac"],
    )
    ap.add_argument("--tag", default=None,
                    help="result filename tag; defaults to an alpha-derived tag")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=RESULTS)
    ap.add_argument(
        "--paper-run",
        action="store_true",
        help="require an explicit campaign tag and write a source manifest",
    )
    ap.add_argument(
        "--campaign-datasets",
        nargs="+",
        default=None,
        help="full paper campaign scope when this worker runs a subset",
    )
    ap.add_argument(
        "--campaign-seeds",
        nargs="+",
        type=int,
        default=None,
        help="full paper campaign scope when this worker runs a subset",
    )
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.paper_run and args.tag is None:
        ap.error("--paper-run requires an explicit --tag")
    tag = args.tag if args.tag is not None else _alpha_tag(args.alpha)
    validate_campaign_tag(tag, required=args.paper_run)
    run_config = auxiliary_run_config(args.alpha, args.tune_frac)
    if args.paper_run:
        try:
            require_canonical_run_config("aux", run_config)
        except ValueError as error:
            ap.error(str(error))
        campaign_datasets, campaign_seeds = auxiliary_campaign_scope(
            args.datasets,
            args.seeds,
            campaign_datasets=args.campaign_datasets,
            campaign_seeds=args.campaign_seeds,
        )
        write_campaign_manifest(
            args.out_dir,
            ROOT,
            campaign_tag=tag,
            datasets=campaign_datasets,
            seeds=campaign_seeds,
            methods=AUXILIARY_CAMPAIGN_METHODS,
            requested_device="cuda" if torch.cuda.is_available() else "cpu",
            run_config=run_config,
            producer_threads=thread_identity(),
        )
    summary = []
    failures = 0
    for name in args.datasets:
        for arch in args.archs:
            for seed in args.seeds:
                out_path = _nn_path(
                    name, arch, seed, tag, args.out_dir
                )
                if args.skip_existing and out_path.exists():
                    if _valid_existing(
                        out_path,
                        name,
                        arch,
                        seed,
                        args.alpha,
                        tag,
                        run_config,
                    ):
                        print(f"[skip] {out_path.name} exists and matches config")
                        continue
                    if args.paper_run:
                        raise RuntimeError(
                            f"--skip-existing rejected stale/incompatible {out_path}"
                        )
                try:
                    p = run_one(name, arch, seed, args.alpha, tune_frac=args.tune_frac)
                    _save(
                        p, name, arch, seed, tag=tag,
                        results_dir=args.out_dir,
                        run_config=run_config,
                    )
                    summary.append({
                        "dataset": name, "arch": arch, "seed": seed, "ok": True,
                        "r2": p["result"]["point"]["r2"],
                        "rmse": p["result"]["point"]["rmse"],
                        "picp": p["result"]["residual_cp"]["picp"],
                        "nciw": p["result"]["residual_cp"]["nciw"],
                        "train_seconds": p["result"]["train_info"]["train_seconds"],
                    })
                except Exception as e:
                    traceback.print_exc()
                    failures += 1
                    summary.append({"dataset": name, "arch": arch, "seed": seed,
                                    "ok": False, "err": repr(e)})
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    sys.exit(main() or 0)
