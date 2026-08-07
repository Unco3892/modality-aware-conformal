"""PetFinder Pawpularity (Kaggle competition, true continuous regression).

Target: ``Pawpularity`` (continuous score 0..100). Modalities: tabular (12 photo
metadata tags — Subject Focus, Eyes, Face, Near, Action, Accessory, Group,
Collage, Human, Occlusion, Info, Blur), image (one photo per row, ~10k rows).

Requires access to the Kaggle Pawpularity competition files. If the raw files
are missing, ``load`` raises ``RuntimeError``.
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
DATA = ROOT / "data" / "pawpularity"
RAW = DATA / "raw"
CACHE = DATA / "embeddings"


def _have_data() -> bool:
    return (RAW / "train.csv").exists()


PAWPULARITY_POLICY = DATASET_LOADER_POLICY["pawpularity"]


def load(
    split: str = "train",
    test_frac: float = PAWPULARITY_POLICY["test_fraction"],
    seed: int = PAWPULARITY_POLICY["split_seed"],
) -> dict:
    """Pawpularity competition: only `train.csv` has labels (`test.csv` is the
    Kaggle holdout, unlabeled, ~8 rows). We therefore split the labeled
    training set 80/20 internally so we have a usable test set with ground-truth
    Pawpularity scores.
    """
    if not _have_data():
        raise RuntimeError(
            "pawpularity data not present. Acquire via Kaggle:\n"
            "  kagglehub.competition_download('petfinder-pawpularity-score') "
            "and place under data/pawpularity/raw/."
        )
    df = pd.read_csv(RAW / "train.csv")
    rng = np.random.RandomState(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    cut = int((1 - test_frac) * len(idx))
    if split == "train":
        df = df.iloc[idx[:cut]].reset_index(drop=True)
        img_split = "train"
    elif split == "test":
        df = df.iloc[idx[cut:]].reset_index(drop=True)
        img_split = "train"  # images for the test split also live under train/
    else:
        raise ValueError(f"unknown split: {split}")

    tab_cols = [
        "Subject Focus", "Eyes", "Face", "Near", "Action", "Accessory",
        "Group", "Collage", "Human", "Occlusion", "Info", "Blur",
    ]
    tab = df[tab_cols].copy()
    image = {"photo": [(RAW / img_split / f"{i}.jpg") for i in df["Id"]]}
    text = None
    y = df["Pawpularity"].to_numpy(dtype=np.float32)

    return {
        "y": y,
        "tab": tab,
        "text": text,
        "image": image,
        "id": df["Id"].to_numpy(),
    }
