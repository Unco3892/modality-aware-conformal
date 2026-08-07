"""Heterogeneous per-modality probabilistic predictors.

Each predictor satisfies a tiny common interface:

    fit(X_fit, y_fit, X_val=None, y_val=None) -> self
    predict(X) -> ndarray of shape (n, 3) with columns (lo, point, hi)

`lo` and `hi` are the alpha/2 and 1-alpha/2 quantiles produced by the model;
`point` is the median (alpha=0.5) where available, otherwise the mean.

Model classes here:

* ``XGBQuantileRegressor`` — three XGBoost models (lo, mid, hi), tabular.
* ``MLPQuantileRegressor`` — single GPU MLP head trained with the multi-output
  pinball loss, on frozen embeddings (text or image).
* ``LinearQuantileRegressor`` — sklearn QuantileRegressor; cheap fallback.
* ``TabPFNQuantileRegressor`` — TabPFNRegressor sampling-based quantile head;
  capped to a 1024-row context. Falls back to a dummy when the package is
  unusable (e.g. license/network issues at import time).
"""

from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.linear_model import QuantileRegressor

try:
    from .experiment_config import AUXILIARY_PREDICTOR_POLICY
except ImportError:  # Direct-script imports used by experiment drivers.
    from experiment_config import (  # type: ignore[no-redef]
        AUXILIARY_PREDICTOR_POLICY,
    )

# Silence sklearn QuantileRegressor convergence warnings (we don't tune solvers).
warnings.filterwarnings("ignore", category=UserWarning)


# ---------------------------------------------------------------------------
# common shape helpers


def _stack(lo, mid, hi) -> np.ndarray:
    out = np.stack([lo, mid, hi], axis=1).astype(np.float32)
    # Sort to enforce lo <= mid <= hi to guard against quantile crossing.
    out = np.sort(out, axis=1)
    return out


# ---------------------------------------------------------------------------
# XGBoost three-quantile predictor


@dataclass
class XGBQuantileRegressor:
    """Tabular three-quantile predictor based on XGBoost.

    Fits an independent XGBRegressor (objective='reg:quantileerror') per
    quantile alpha/2, 0.5, 1-alpha/2.
    """

    alpha: float = 0.1   # 90% interval by default; CQR will recalibrate
    n_estimators: int = 600
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = AUXILIARY_PREDICTOR_POLICY["xgb_subsample"]
    colsample_bytree: float = AUXILIARY_PREDICTOR_POLICY[
        "xgb_colsample_bytree"
    ]
    seed: int = 0
    use_gpu: bool = True

    def fit(self, X, y, X_val=None, y_val=None):
        qs = (self.alpha / 2, 0.5, 1 - self.alpha / 2)
        self.models_: list[xgb.XGBRegressor] = []
        device = "cuda" if (self.use_gpu and self._cuda_ok()) else "cpu"
        for q in qs:
            m = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                objective="reg:quantileerror",
                quantile_alpha=q,
                tree_method=AUXILIARY_PREDICTOR_POLICY[
                    "xgb_tree_method"
                ],
                device=device,
                random_state=self.seed,
            )
            m.fit(X, y, verbose=False)
            self.models_.append(m)
        return self

    def predict(self, X) -> np.ndarray:
        out = [m.predict(X) for m in self.models_]
        return _stack(out[0], out[1], out[2])

    @staticmethod
    def _cuda_ok() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False


# ---------------------------------------------------------------------------
# MLP head with pinball loss


def pinball_loss(pred: torch.Tensor, target: torch.Tensor, taus: torch.Tensor) -> torch.Tensor:
    """pred: (B, Q), target: (B,), taus: (Q,) ; returns scalar mean loss."""
    diff = target.unsqueeze(1) - pred
    return torch.maximum(taus * diff, (taus - 1) * diff).mean()


class _MLPQuantileNet(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, n_quantiles: int = 3, dropout: float = 0.2):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, n_quantiles),
        )

    def forward(self, x):
        return self.body(x)


@dataclass
class MLPQuantileRegressor:
    """Frozen-embedding MLP head trained with the multi-quantile pinball loss."""

    alpha: float = 0.1
    hidden: int = 256
    epochs: int = 60
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-4
    dropout: float = 0.2
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    val_frac: float = AUXILIARY_PREDICTOR_POLICY[
        "mlp_internal_validation_fraction"
    ]
    name: str = "mlp"

    def fit(self, X, y, X_val=None, y_val=None):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        in_dim = X.shape[1]
        self.taus_ = torch.tensor(
            [self.alpha / 2, 0.5, 1 - self.alpha / 2],
            dtype=torch.float32, device=self.device,
        )
        self.net_ = _MLPQuantileNet(in_dim, self.hidden, dropout=self.dropout).to(self.device)
        optimizer_class = getattr(
            torch.optim,
            AUXILIARY_PREDICTOR_POLICY["optimizer"],
        )
        opt = optimizer_class(
            self.net_.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        if X_val is None:
            n = len(X)
            cut = int((1 - self.val_frac) * n)
            rng = np.random.default_rng(self.seed)
            perm = rng.permutation(n)
            tr_idx, va_idx = perm[:cut], perm[cut:]
            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_va, y_va = X[va_idx], y[va_idx]
        else:
            X_tr, y_tr, X_va, y_va = X, y, X_val, y_val

        x_tr_t = torch.from_numpy(X_tr.astype(np.float32)).to(self.device)
        y_tr_t = torch.from_numpy(y_tr.astype(np.float32)).to(self.device)
        x_va_t = torch.from_numpy(X_va.astype(np.float32)).to(self.device)
        y_va_t = torch.from_numpy(y_va.astype(np.float32)).to(self.device)

        n_tr = len(x_tr_t)
        best_val = float("inf")
        best_state = None
        no_improve = 0
        patience = AUXILIARY_PREDICTOR_POLICY[
            "mlp_early_stopping_patience"
        ]

        for epoch in range(self.epochs):
            self.net_.train()
            perm = torch.randperm(n_tr, device=self.device)
            running = 0.0
            for s in range(0, n_tr, self.batch_size):
                idx = perm[s:s + self.batch_size]
                xb, yb = x_tr_t[idx], y_tr_t[idx]
                opt.zero_grad(set_to_none=True)
                pred = self.net_(xb)
                loss = pinball_loss(pred, yb, self.taus_)
                loss.backward()
                opt.step()
                running += float(loss) * len(xb)
            running /= n_tr

            self.net_.eval()
            with torch.no_grad():
                v_pred = self.net_(x_va_t)
                v_loss = float(pinball_loss(v_pred, y_va_t, self.taus_))
            if (
                v_loss
                + AUXILIARY_PREDICTOR_POLICY[
                    "mlp_early_stopping_min_delta"
                ]
                < best_val
            ):
                best_val = v_loss
                best_state = {k: v.detach().clone() for k, v in self.net_.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        if best_state is not None:
            self.net_.load_state_dict(best_state)
        self.best_val_ = best_val
        return self

    def predict(self, X) -> np.ndarray:
        self.net_.eval()
        with torch.no_grad():
            x = torch.from_numpy(X.astype(np.float32)).to(self.device)
            preds = self.net_(x).cpu().numpy()
        # ensure monotone q's
        preds.sort(axis=1)
        return preds.astype(np.float32)


# ---------------------------------------------------------------------------
# Linear quantile regressor (cheap baseline / fallback)


@dataclass
class LinearQuantileRegressor:
    alpha: float = 0.1
    quantile_alpha: float = 0.0
    seed: int = 0

    def fit(self, X, y, X_val=None, y_val=None):
        qs = (self.alpha / 2, 0.5, 1 - self.alpha / 2)
        self.models_ = []
        for q in qs:
            m = QuantileRegressor(quantile=q, alpha=self.quantile_alpha, solver="highs")
            m.fit(X, y)
            self.models_.append(m)
        return self

    def predict(self, X) -> np.ndarray:
        out = [m.predict(X) for m in self.models_]
        return _stack(out[0], out[1], out[2])


# ---------------------------------------------------------------------------
# LightGBM quantile (for true model-class heterogeneity)


@dataclass
class LGBMQuantileRegressor:
    """LightGBM regressor with objective=quantile; one model per quantile."""

    alpha: float = 0.1
    n_estimators: int = 600
    learning_rate: float = 0.05
    num_leaves: int = 63
    min_data_in_leaf: int = 50
    seed: int = 0

    def fit(self, X, y, X_val=None, y_val=None):
        import lightgbm as lgb
        qs = (self.alpha / 2, 0.5, 1 - self.alpha / 2)
        self.models_ = []
        for q in qs:
            m = lgb.LGBMRegressor(
                objective="quantile",
                alpha=q,
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                min_data_in_leaf=self.min_data_in_leaf,
                random_state=self.seed,
                verbosity=-1,
            )
            m.fit(X, y)
            self.models_.append(m)
        return self

    def predict(self, X) -> np.ndarray:
        out = [m.predict(X) for m in self.models_]
        return _stack(out[0], out[1], out[2])


# ---------------------------------------------------------------------------
# Random forest quantile via sklearn-quantile (sklearn doesn't ship it; use
# a simple reduction: train a regressor + use residual quantiles per-tree).
# Skip if too brittle.


# ---------------------------------------------------------------------------
# TabPFN quantile regressor (sampling-based)


@dataclass
class TabPFNQuantileRegressor:
    """Use TabPFNRegressor.predict to get quantile/output samples.

    TabPFN-v2 has a 10k-row context limit. We subsample the fit set to
    ``max_context`` rows for in-context learning. If the model fails to load
    (license/network), ``fit`` raises a RuntimeError that the caller can catch.
    """

    alpha: float = 0.1
    max_context: int = 1024
    seed: int = 0

    def fit(self, X, y, X_val=None, y_val=None):
        from tabpfn import TabPFNRegressor

        rng = np.random.default_rng(self.seed)
        n = len(X)
        if n > self.max_context:
            idx = rng.choice(n, size=self.max_context, replace=False)
            X = X[idx]
            y = y[idx]
        os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", str(os.path.expanduser("~/.tabpfn")))
        self.model_ = TabPFNRegressor(random_state=self.seed)
        try:
            self.model_.fit(X, y)
        except Exception as e:
            raise RuntimeError(f"TabPFN fit failed: {e!r}")
        return self

    def predict(self, X) -> np.ndarray:
        out = self.model_.predict(
            X,
            output_type="quantiles",
            quantiles=[self.alpha / 2, 0.5, 1 - self.alpha / 2],
        )
        # tabpfn returns a list of arrays, one per quantile.
        if isinstance(out, list):
            arr = np.stack(out, axis=1).astype(np.float32)
        else:
            arr = np.asarray(out, dtype=np.float32)
        arr.sort(axis=1)
        return arr


# ---------------------------------------------------------------------------
# Predictor registry — hetero by construction.


PER_MODALITY_DEFAULTS = {
    "tab": "xgb_quantile",     # gradient boosting on 4 features
    "text": "mlp_pinball",     # GPU MLP on 384-dim sentence-transformer emb
    "image": "mlp_pinball",    # GPU MLP on 384-dim DINOv2 emb
}


def make_predictor(kind: str, alpha: float, seed: int) -> object:
    if kind == "xgb_quantile":
        return XGBQuantileRegressor(alpha=alpha, seed=seed)
    if kind == "mlp_pinball":
        return MLPQuantileRegressor(alpha=alpha, seed=seed)
    if kind == "linear_quantile":
        return LinearQuantileRegressor(alpha=alpha, seed=seed)
    if kind == "lgbm_quantile":
        return LGBMQuantileRegressor(alpha=alpha, seed=seed)
    if kind == "tabpfn":
        return TabPFNQuantileRegressor(alpha=alpha, seed=seed)
    raise ValueError(f"unknown predictor kind: {kind}")


# ---------------------------------------------------------------------------


def fit_modality_block(
    name: str, kind: str, X_fit, y_fit, X_calib, X_test, alpha: float, seed: int
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Returns (preds_calib, preds_test, info) where preds are (n, 3) (lo, pt, hi)."""
    t0 = time.time()
    p = make_predictor(kind, alpha=alpha, seed=seed)
    p.fit(X_fit, y_fit)
    fit_s = time.time() - t0
    p_calib = p.predict(X_calib)
    p_test = p.predict(X_test)
    return p_calib, p_test, {
        "modality": name,
        "kind": kind,
        "fit_seconds": fit_s,
        "in_dim": int(X_fit.shape[1]),
    }
