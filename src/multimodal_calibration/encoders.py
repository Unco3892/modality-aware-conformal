"""Frozen encoder utilities shared across dataset loaders.

Mirrors the SRED baseline's encoders (DINOv2-B/14 for image, multilingual-e5-large
for text), with a thin caching layer keyed on (split, modality, encoder_slug).

Pulls ``ImageEncoder`` and ``embed_text`` from the SRED experiment to avoid code
duplication. SRED experiment dir is added to ``sys.path`` lazily.
"""

from __future__ import annotations

import os

import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch

ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
SRED_EXP = ROOT / "src" / "sred"
if str(SRED_EXP) not in sys.path:
    sys.path.insert(0, str(SRED_EXP))

# Reuse the encoders / helpers from the SRED baseline so we stay drop-in compatible.
from baseline import (  # noqa: E402
    DINOV2_REVISION,
    TEXT_MODEL_REVISIONS,
    ImageEncoder,
    cached,
    embed_text,
)


# ---------------------------------------------------------------------------
# defaults
TEXT_MODEL_DEFAULT = "intfloat/multilingual-e5-large"   # AAAI-grade text encoder
TEXT_MODEL_FAST = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # smaller fallback

IMAGE_ARCH_DEFAULT = "dinov2_vitb14"   # B/14 = 768d, decent compute/quality trade-off
IMAGE_ARCH_FAST = "dinov2_vits14"      # S/14 = 384d, faster on big datasets

PAPER_CACHE_SLUGS: dict[str, dict[str, str | None]] = {
    "sred": {
        "text": "multilingual-e5-large",
        "image": "dinov2-vitb14",
    },
    "mercari": {
        "text": "paraphrase-multilingual-MiniLM-L12-v2",
        "image": None,
    },
    "pawpularity": {
        "text": None,
        "image": "dinov2-vits14",
    },
    "imdb_wiki": {
        "text": "paraphrase-multilingual-MiniLM-L12-v2",
        "image": "dinov2-vits14",
    },
}
TEXT_MODEL_BY_SLUG = {
    "multilingual-e5-large": "intfloat/multilingual-e5-large",
    "paraphrase-multilingual-MiniLM-L12-v2":
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
}


def paper_cache_slugs(name: str) -> tuple[str | None, str | None]:
    """Return the audited cache slugs for one cross-dataset pipeline."""
    if name not in PAPER_CACHE_SLUGS:
        raise KeyError(f"no paper cache contract for dataset {name!r}")
    contract = PAPER_CACHE_SLUGS[name]
    return contract["text"], contract["image"]


def slugify_text(model_id: str) -> str:
    return model_id.split("/")[-1].replace(".", "-")


def slugify_image(arch: str) -> str:
    return arch.replace("_", "-")


# ---------------------------------------------------------------------------
# cached embedding helpers


def cache_text(
    cache_dir: Path,
    split: str,
    field: str,
    texts: Sequence[str],
    model_id: str,
    device: str,
    revision: str | None = None,
) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    slug = slugify_text(model_id)
    path = cache_dir / f"{split}_{field}_{slug}.npy"
    if path.exists():
        print(f"  load text cache {path.name} ({path.stat().st_size / 1e6:.1f} MB)")
        return np.load(path)
    print(f"  compute text embeddings {path.name} (n={len(texts)})")
    revision = revision or TEXT_MODEL_REVISIONS.get(model_id)
    arr = embed_text(texts, model_id, device, revision)
    np.save(path, arr)
    return arr


def cache_images(
    cache_dir: Path,
    split: str,
    field: str,
    paths: Sequence[Path],
    arch: str,
    device: str,
    encoder: ImageEncoder | None = None,
    revision: str = DINOV2_REVISION,
) -> tuple[np.ndarray, ImageEncoder | None]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    slug = slugify_image(arch)
    cache_path = cache_dir / f"{split}_{field}_{slug}.npy"
    if cache_path.exists():
        print(f"  load image cache {cache_path.name} ({cache_path.stat().st_size / 1e6:.1f} MB)")
        return np.load(cache_path), encoder
    print(f"  compute image embeddings {cache_path.name} (n={len(paths)})")
    if encoder is None:
        encoder = ImageEncoder(device=device, arch=arch, revision=revision)
    t0 = time.time()
    arr = encoder.encode_paths(list(paths))
    print(f"    encoded in {time.time() - t0:.1f}s  shape={arr.shape}")
    np.save(cache_path, arr)
    return arr, encoder
