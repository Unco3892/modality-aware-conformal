"""Data loaders for the multi-modal calibration study.

Loads SRED metadata + cached pretrained embeddings. Splits the train set into
``fit`` and ``calib`` subsets for the conformal split. All paths are absolute
so this module works from any cwd.
"""

from __future__ import annotations

import os

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
SRED = ROOT / "data" / "sred"
META = SRED / "metadata"
EMB = SRED / "embeddings"

TAB_COLS = ["living_space", "rooms", "lat", "lon"]


@dataclass
class ModalitySplit:
    """One modality's features for fit / calib / test."""

    name: str
    X_fit: np.ndarray
    X_calib: np.ndarray
    X_test: np.ndarray


@dataclass
class SREDPack:
    """Everything one experiment seed needs."""

    y_fit: np.ndarray
    y_calib: np.ndarray
    y_test: np.ndarray
    modalities: dict[str, ModalitySplit]
    fit_idx: np.ndarray
    calib_idx: np.ndarray

    @property
    def names(self) -> list[str]:
        return list(self.modalities.keys())

    def concat(self, split: str) -> np.ndarray:
        parts = []
        for m in self.modalities.values():
            parts.append(getattr(m, f"X_{split}"))
        return np.concatenate(parts, axis=1).astype(np.float32)


# ---------------------------------------------------------------------------


def load_metadata() -> tuple[pd.DataFrame, pd.DataFrame]:
    tr = pd.read_csv(META / "train_data_with_text.csv", encoding="latin-1")
    te = pd.read_csv(META / "test_data_with_text.csv", encoding="latin-1")
    for df in (tr, te):
        df["listing_id"] = df["listing_id"].astype("Int64").astype(str)
    return tr, te


def load_embeddings(split: str) -> dict[str, np.ndarray]:
    """Return per-field embedding matrices as a dict keyed by short name.

    Concatenates {header, ad_description} into a single ``text`` block (mean of
    the two fields rather than concatenation, to keep the per-modality block
    small and dense), and {montage_organized, satellite} into ``image`` (also
    mean-pooled). Using mean keeps each modality at 384 dims so the per-row
    quantile heads stay small.
    """
    text_h = np.load(EMB / f"{split}_header_paraphrase-multilingual-MiniLM-L12-v2.npy")
    text_d = np.load(EMB / f"{split}_ad_description_paraphrase-multilingual-MiniLM-L12-v2.npy")
    text = (text_h + text_d) * 0.5  # average two fields, both already L2-normalized

    img_m = np.load(EMB / f"{split}_montage_organized_dinov2-vits14.npy")
    img_s = np.load(EMB / f"{split}_satellite_dinov2-vits14.npy")
    image = (img_m + img_s) * 0.5

    return {"text": text.astype(np.float32), "image": image.astype(np.float32)}


def make_pack(seed: int, calib_frac: float = 0.15) -> SREDPack:
    """Load SRED + embeddings; deterministic train/calib split for conformal."""
    tr, te = load_metadata()
    rng = np.random.default_rng(seed)
    n = len(tr)
    perm = rng.permutation(n)
    n_calib = int(round(calib_frac * n))
    calib_idx = perm[:n_calib]
    fit_idx = perm[n_calib:]

    tab_tr = tr[TAB_COLS].to_numpy(dtype=np.float32)
    tab_te = te[TAB_COLS].to_numpy(dtype=np.float32)

    # standardise tabular based on fit set only.
    mean = tab_tr[fit_idx].mean(axis=0)
    std = tab_tr[fit_idx].std(axis=0) + 1e-6
    tab_tr = (tab_tr - mean) / std
    tab_te = (tab_te - mean) / std

    emb_tr = load_embeddings("train")
    emb_te = load_embeddings("test")

    modalities: dict[str, ModalitySplit] = {}
    modalities["tab"] = ModalitySplit(
        name="tab",
        X_fit=tab_tr[fit_idx],
        X_calib=tab_tr[calib_idx],
        X_test=tab_te,
    )
    for k in ("text", "image"):
        Xtr = emb_tr[k]
        Xte = emb_te[k]
        modalities[k] = ModalitySplit(
            name=k,
            X_fit=Xtr[fit_idx],
            X_calib=Xtr[calib_idx],
            X_test=Xte,
        )

    y_tr = np.log(tr["price"].to_numpy(dtype=np.float32))
    y_te = np.log(te["price"].to_numpy(dtype=np.float32))
    # standardise log-price using fit set only -- cleaner numerics for MLP heads.
    y_mean = float(y_tr[fit_idx].mean())
    y_std = float(y_tr[fit_idx].std() + 1e-6)
    y_tr_std = (y_tr - y_mean) / y_std
    y_te_std = (y_te - y_mean) / y_std

    pack = SREDPack(
        y_fit=y_tr_std[fit_idx],
        y_calib=y_tr_std[calib_idx],
        y_test=y_te_std,
        modalities=modalities,
        fit_idx=fit_idx,
        calib_idx=calib_idx,
    )
    pack.y_scale = (y_mean, y_std)  # type: ignore[attr-defined]
    return pack
