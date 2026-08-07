"""IMDB-WIKI face age regression — WIKI subset only.

Target: ``age`` (continuous, integer years; computed as ``photo_taken - dob_year``).
Modalities:
  - tabular (gender, photo_taken year, face_score)
  - text (name)
  - image (face crop, single per row)

Source: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki_crop.tar
File: ``wiki_crop/wiki.mat`` + JPGs in ``wiki_crop/<bucket>/``.

Filtering: drop rows with NaN/missing dob/year, low face score (<1.0), gender NaN,
multi-face images (second_face_score not NaN), or impossible ages (<1 or >100).
This typically leaves ~50-60k clean rows.

Splits: deterministic 90/10 train/test on hashed ``full_path``.
"""

from __future__ import annotations

import os

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
try:
    from ..experiment_config import DATASET_LOADER_POLICY
except ImportError:
    from experiment_config import DATASET_LOADER_POLICY
from scipy.io import loadmat

ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[3]))
DATA = ROOT / "data" / "imdb_wiki"
WIKI = DATA / "wiki_crop"
META_MAT = WIKI / "wiki.mat"
PROCESSED = DATA / "processed.parquet"
CACHE = DATA / "embeddings"


def _matlab_serial_to_year(serial: int) -> int | None:
    """MATLAB serial date number -> Gregorian year. Day 1 = 0000-01-01."""
    if serial is None or pd.isna(serial) or serial <= 0:
        return None
    try:
        # MATLAB serial 1 == 0000-01-01 in proleptic Gregorian; Python's datetime
        # cannot represent year 0, so we use 1970 epoch for stability.
        # Days since 1970-01-01: serial - 719529.
        dt = datetime(1970, 1, 1) + timedelta(days=int(serial) - 719529)
        return dt.year
    except (OverflowError, ValueError):
        return None


def _build_processed() -> pd.DataFrame:
    if PROCESSED.exists():
        return pd.read_parquet(PROCESSED)
    if not META_MAT.exists():
        raise FileNotFoundError(
            f"WIKI metadata not present at {META_MAT}. "
            "Acquire wiki_crop.tar from https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/"
        )
    m = loadmat(META_MAT)["wiki"][0, 0]
    n = m["dob"].size
    rows = []
    for i in range(n):
        full_path = m["full_path"][0, i][0]
        dob_serial = int(m["dob"][0, i])
        photo_year = int(m["photo_taken"][0, i])
        gender = float(m["gender"][0, i])
        face_score = float(m["face_score"][0, i])
        second_face = float(m["second_face_score"][0, i])
        name_arr = m["name"][0, i]
        name = name_arr[0] if name_arr.size else ""
        rows.append({
            "full_path": full_path,
            "dob_serial": dob_serial,
            "photo_taken": photo_year,
            "gender": gender,
            "face_score": face_score,
            "second_face_score": second_face,
            "name": name,
        })
    df = pd.DataFrame(rows)
    df["dob_year"] = df["dob_serial"].apply(_matlab_serial_to_year)
    df["age"] = df["photo_taken"] - df["dob_year"]
    df["abs_path"] = df["full_path"].apply(lambda p: str((WIKI / p).as_posix()))
    df["exists"] = df["abs_path"].apply(lambda p: Path(p).exists())

    policy = DATASET_LOADER_POLICY["imdb_wiki"]
    keep = (
        df["dob_year"].notna()
        & df["face_score"].replace(-np.inf, np.nan).notna()
        & (df["face_score"] >= policy["minimum_face_score"])
        & df["second_face_score"].isna()
        & df["gender"].notna()
        & df["age"].between(
            policy["minimum_age"], policy["maximum_age"]
        )
        & df["exists"]
    )
    out = df.loc[keep].reset_index(drop=True)
    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(PROCESSED, index=False)
    return out


def _split_assignments(df: pd.DataFrame, seed: int = 0) -> np.ndarray:
    """Hash full_path -> 90% train, 10% test."""
    h = pd.util.hash_pandas_object(df["full_path"], index=False).to_numpy()
    policy = DATASET_LOADER_POLICY["imdb_wiki"]
    bucket = (h % policy["hash_bucket_count"]).astype(int)
    return np.where(
        bucket < policy["train_bucket_count"], "train", "test"
    )


def load(split: str = "train") -> dict:
    df = _build_processed()
    assn = _split_assignments(df)
    mask = (assn == split) if split in ("train", "test") else np.ones(len(df), dtype=bool)
    sub = df.loc[mask].reset_index(drop=True)

    tab = sub[["gender", "photo_taken", "face_score"]].copy()
    text = {"name": sub["name"].fillna("").astype(str).tolist()}
    image = {"face": [Path(p) for p in sub["abs_path"]]}
    y = sub["age"].to_numpy(dtype=np.float32)
    return {
        "y": y,
        "tab": tab,
        "text": text,
        "image": image,
        "id": sub["full_path"].to_numpy(),
    }
