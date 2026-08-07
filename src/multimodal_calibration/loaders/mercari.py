"""Mercari Price Suggestion Challenge (Kaggle).

Target: ``log(price+1)`` (continuous, RMSLE evaluation in the original challenge).
Modalities:
  - tabular: ``item_condition_id``, ``shipping``, ``category_name`` (categorical),
    ``brand_name`` (categorical with many levels)
  - text: ``name`` (item title) and ``item_description`` (longer)
  - **no image modality** — Mercari challenge is tab + text only, but text is
    very informative for price (this is the original Kaggle competition's whole
    point, with text features carrying ~70 % of signal).

This loader exists primarily to balance the modality coverage table for the
cross-dataset feasibility study. It would slot in alongside SRED (text + image + tab),
DVM-CAR (image + tab), and IMDB-WIKI (image + minor tab + name) to give us a
*text-dominant* example.

Acquisition::

    kaggle competitions download -c mercari-price-suggestion-challenge \\
        -p data/mercari/raw

Requires Kaggle credentials, normally in ``~/.kaggle/kaggle.json``. If
credentials are missing, ``load`` raises ``RuntimeError`` so the driver script
can log + skip.

Sub-sampling: 50k rows by default, seeded.
"""

from __future__ import annotations

import os

from pathlib import Path

import numpy as np
import pandas as pd
try:
    from ..experiment_config import DATASET_LOADER_POLICY
except ImportError:
    from experiment_config import DATASET_LOADER_POLICY

ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[3]))
DATA = ROOT / "data" / "mercari"
RAW = DATA / "raw"
CACHE = DATA / "embeddings"

MERCARI_POLICY = DATASET_LOADER_POLICY["mercari"]
N_SAMPLES = MERCARI_POLICY["maximum_labeled_rows"]


def _have_data() -> bool:
    return (RAW / "train.tsv").exists() or (RAW / "train.tsv.7z").exists() or \
        any(p.suffix == ".tsv" for p in RAW.glob("*"))


def load(
    split: str = "train",
    n_rows: int = N_SAMPLES,
    test_frac: float = MERCARI_POLICY["test_fraction"],
    seed: int = MERCARI_POLICY["split_seed"],
) -> dict:
    """Mercari: only `train.tsv` has labels (`test.tsv` is the Kaggle holdout
    without `price`). Subsample N_SAMPLES from the labeled set, then split
    80/20 into train/test internally so we have ground-truth prices in both."""
    if not _have_data():
        raise RuntimeError(
            "mercari data not present. Acquire via kagglehub then place under data/mercari/raw/."
        )
    df = pd.read_csv(RAW / "train.tsv", sep="\t")
    if n_rows and len(df) > n_rows:
        df = df.sample(n=n_rows, random_state=seed).reset_index(drop=True)
    rng = np.random.RandomState(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    cut = int((1 - test_frac) * len(idx))
    if split == "train":
        df = df.iloc[idx[:cut]].reset_index(drop=True)
    elif split == "test":
        df = df.iloc[idx[cut:]].reset_index(drop=True)
    else:
        raise ValueError(f"unknown split: {split}")

    tab = df[["item_condition_id", "category_name", "brand_name", "shipping"]].copy()
    text = {
        "name": df["name"].fillna("").astype(str).tolist(),
        "item_description": df["item_description"].fillna("").astype(str).tolist(),
    }
    image = None
    y = np.log(df["price"].to_numpy(dtype=np.float32) + 1.0)
    return {
        "y": y,
        "tab": tab,
        "text": text,
        "image": image,
        "id": df["train_id"].to_numpy(),
    }
