"""Composition layer: merge per-modality (lo, point, hi) into a single triple.

Two strategies:

* ``LinearStacker`` — scalar weights over modalities fitted on the tune fold
  to minimise pinball loss on the merged quantiles. Same weights are used
  for lo, point, hi (i.e. one set of mixture weights per row, shared across
  quantiles).

* ``GatedMixture`` — small MLP that maps the concat of all modality features
  to per-row mixture weights via softmax, trained end-to-end with the
  multi-quantile pinball loss. Per-row, per-modality weights.

Both stackers consume calibration matrices ``preds_calib`` of shape
``(K, n_calib, 3)`` and yield calibrated predictions for any new ``preds`` of
shape ``(K, n, 3)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

try:
    from .experiment_config import (
        AUXILIARY_COMPOSITION_POLICY,
        AUXILIARY_GATE_BATCH_POLICY,
        AUXILIARY_PAPER_RUN_CONFIG,
    )
except ImportError:  # Direct-script imports used by experiment drivers.
    from experiment_config import (  # type: ignore[no-redef]
        AUXILIARY_COMPOSITION_POLICY,
        AUXILIARY_GATE_BATCH_POLICY,
        AUXILIARY_PAPER_RUN_CONFIG,
    )


# ---------------------------------------------------------------------------
# helpers


def stack_preds(preds_per_modality: dict[str, np.ndarray]) -> np.ndarray:
    """Stack a dict of (n, 3) arrays into a (K, n, 3) tensor with deterministic key order."""
    arrs = [preds_per_modality[k] for k in preds_per_modality]
    return np.stack(arrs, axis=0)  # (K, n, 3)


def pinball_np(pred: np.ndarray, target: np.ndarray, taus: np.ndarray) -> float:
    """pred: (n, Q), target: (n,), taus: (Q,) ; mean pinball loss across rows and quantiles."""
    diff = target[:, None] - pred
    loss = np.maximum(taus * diff, (taus - 1) * diff)
    return float(loss.mean())


# ---------------------------------------------------------------------------
# linear stacker


@dataclass
class LinearStacker:
    """Per-modality mixture weights tuned by Nelder-Mead on calibration pinball loss."""

    alpha: float = 0.1
    seed: int = 0

    def fit(self, preds_calib: np.ndarray, y_calib: np.ndarray):
        from scipy.optimize import minimize

        K = preds_calib.shape[0]
        taus = np.array([self.alpha / 2, 0.5, 1 - self.alpha / 2])

        def obj(theta):
            # softmax over K weights so sum to 1 and stay positive
            w = np.exp(theta - theta.max())
            w = w / w.sum()
            pred = (w[:, None, None] * preds_calib).sum(axis=0)  # (n, 3)
            pred = np.sort(pred, axis=1)
            return pinball_np(pred, y_calib, taus)

        x0 = np.zeros(K)
        res = minimize(
            obj,
            x0,
            method=AUXILIARY_COMPOSITION_POLICY["linear_optimizer"],
            options={
                "maxiter": AUXILIARY_COMPOSITION_POLICY[
                    "linear_max_iterations"
                ],
                "xatol": AUXILIARY_COMPOSITION_POLICY[
                    "linear_x_tolerance"
                ],
            },
        )
        w = np.exp(res.x - res.x.max())
        self.weights_ = (w / w.sum()).astype(np.float32)
        self.opt_loss_ = float(res.fun)
        return self

    def predict(self, preds: np.ndarray) -> np.ndarray:
        out = (self.weights_[:, None, None] * preds).sum(axis=0)
        out = np.sort(out, axis=1)
        return out


# ---------------------------------------------------------------------------
# gated mixture (per-row weights from a small MLP)


class _GatingNet(nn.Module):
    def __init__(
        self,
        in_dim: int,
        K: int,
        hidden: int = AUXILIARY_PAPER_RUN_CONFIG["heterogeneous"][
            "gated_mixture"
        ]["hidden"],
    ):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(AUXILIARY_COMPOSITION_POLICY["gate_dropout"]),
            nn.Linear(hidden, K),
        )

    def forward(self, x):
        logits = self.body(x)
        return torch.softmax(logits, dim=-1)


@dataclass
class GatedMixture:
    """Per-row gate over the modality predictions; trained on the tune fold."""

    alpha: float = 0.1
    epochs: int = AUXILIARY_PAPER_RUN_CONFIG["heterogeneous"][
        "gated_mixture"
    ]["epochs_standard"]
    batch_size: int = AUXILIARY_GATE_BATCH_POLICY["maximum"]
    lr: float = AUXILIARY_PAPER_RUN_CONFIG["heterogeneous"][
        "gated_mixture"
    ]["learning_rate"]
    weight_decay: float = AUXILIARY_COMPOSITION_POLICY["gate_weight_decay"]
    seed: int = 0
    hidden: int = AUXILIARY_PAPER_RUN_CONFIG["heterogeneous"][
        "gated_mixture"
    ]["hidden"]
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def fit(
        self,
        preds_calib: np.ndarray,
        y_calib: np.ndarray,
        gate_features_calib: np.ndarray,
    ):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        K = preds_calib.shape[0]
        in_dim = gate_features_calib.shape[1]
        self.K_ = K
        self.gate_ = _GatingNet(in_dim, K, hidden=self.hidden).to(self.device)
        self.taus_ = torch.tensor(
            [self.alpha / 2, 0.5, 1 - self.alpha / 2],
            dtype=torch.float32, device=self.device,
        )
        optimizer_class = getattr(
            torch.optim,
            AUXILIARY_COMPOSITION_POLICY["gate_optimizer"],
        )
        opt = optimizer_class(
            self.gate_.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        n = len(y_calib)
        rng = np.random.default_rng(self.seed)
        cut = int(
            (
                1
                - AUXILIARY_COMPOSITION_POLICY[
                    "gate_internal_validation_fraction"
                ]
            )
            * n
        )
        perm = rng.permutation(n)
        tr_idx, va_idx = perm[:cut], perm[cut:]

        # tensors (K, n, 3) and gate inputs (n, F)
        preds_t = torch.from_numpy(preds_calib.astype(np.float32)).to(self.device)
        y_t = torch.from_numpy(y_calib.astype(np.float32)).to(self.device)
        gf_t = torch.from_numpy(gate_features_calib.astype(np.float32)).to(self.device)

        best_val = float("inf")
        best_state = None
        no_improve = 0
        patience = AUXILIARY_COMPOSITION_POLICY[
            "gate_early_stopping_patience"
        ]

        tr_idx_t = torch.from_numpy(tr_idx).to(self.device)
        va_idx_t = torch.from_numpy(va_idx).to(self.device)

        for epoch in range(self.epochs):
            self.gate_.train()
            sub = tr_idx_t[torch.randperm(len(tr_idx_t), device=self.device)]
            running = 0.0
            for s in range(0, len(sub), self.batch_size):
                b = sub[s:s + self.batch_size]
                w = self.gate_(gf_t[b])  # (B, K)
                # mix: sum_k w_k * preds[k, b, :] -> (B, 3)
                p = preds_t[:, b, :]  # (K, B, 3)
                # einsum-friendly: (B, K) * (K, B, 3) -> (B, 3)
                mix = (w.permute(1, 0).unsqueeze(-1) * p).sum(dim=0)
                # diff: y - mix , pinball with shared taus
                diff = y_t[b].unsqueeze(1) - mix
                loss = torch.maximum(self.taus_ * diff, (self.taus_ - 1) * diff).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                running += float(loss) * len(b)
            running /= max(len(sub), 1)

            self.gate_.eval()
            with torch.no_grad():
                w_v = self.gate_(gf_t[va_idx_t])
                p_v = preds_t[:, va_idx_t, :]
                mix_v = (w_v.permute(1, 0).unsqueeze(-1) * p_v).sum(dim=0)
                diff = y_t[va_idx_t].unsqueeze(1) - mix_v
                v_loss = float(torch.maximum(self.taus_ * diff, (self.taus_ - 1) * diff).mean())
            if (
                v_loss
                + AUXILIARY_COMPOSITION_POLICY[
                    "gate_early_stopping_min_delta"
                ]
                < best_val
            ):
                best_val = v_loss
                best_state = {k: v.detach().clone() for k, v in self.gate_.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        if best_state is not None:
            self.gate_.load_state_dict(best_state)
        self.best_val_ = best_val
        return self

    def predict(self, preds: np.ndarray, gate_features: np.ndarray) -> np.ndarray:
        self.gate_.eval()
        with torch.no_grad():
            preds_t = torch.from_numpy(preds.astype(np.float32)).to(self.device)
            gf_t = torch.from_numpy(gate_features.astype(np.float32)).to(self.device)
            w = self.gate_(gf_t)  # (n, K)
            mix = (w.permute(1, 0).unsqueeze(-1) * preds_t).sum(dim=0)  # (n, 3)
        out = mix.cpu().numpy()
        out.sort(axis=1)
        return out

    def get_gate_weights(self, gate_features: np.ndarray) -> np.ndarray:
        self.gate_.eval()
        with torch.no_grad():
            gf_t = torch.from_numpy(gate_features.astype(np.float32)).to(self.device)
            w = self.gate_(gf_t).cpu().numpy()
        return w
