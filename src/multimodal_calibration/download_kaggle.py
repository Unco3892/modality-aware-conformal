"""Download Kaggle competition datasets via the new API token (kagglehub).

Reads `KAGGLE_API_TOKEN` from .env or environment, authenticates with
kagglehub, and downloads each requested competition into
`data/<dataset_slug>/raw/`.

Usage:
    python src/multimodal_calibration/download_kaggle.py \
        --datasets petfinder-pawpularity-score mercari-price-suggestion-challenge avito-demand-prediction
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
DATA = ROOT / "data"


def load_env():
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())


def download_competition(slug: str, out_dir: Path) -> Path:
    import kagglehub

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[{slug}] starting download -> {out_dir}")
    # kagglehub returns a cached path; copy/symlink into our project layout.
    cache_path = kagglehub.competition_download(slug)
    print(f"[{slug}] kagglehub cache: {cache_path}")

    # mirror cache contents into out_dir for project consistency
    import shutil
    cache_path = Path(cache_path)
    if cache_path.is_dir():
        for item in cache_path.iterdir():
            target = out_dir / item.name
            if target.exists():
                continue
            if item.is_dir():
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)
    else:
        target = out_dir / cache_path.name
        if not target.exists():
            shutil.copy2(cache_path, target)

    n = sum(1 for _ in out_dir.rglob("*") if _.is_file())
    print(f"[{slug}] mirrored {n} files into {out_dir}")
    return out_dir


# Kaggle competition slugs do not match the short dataset names the loaders use,
# so map them explicitly. Unmapped slugs keep the original underscore fallback.
SLUG_TO_DATASET = {
    "mercari-price-suggestion-challenge": "mercari",
    "petfinder-pawpularity-score": "pawpularity",
}


def dataset_dir(slug: str) -> Path:
    """Raw-input directory for a competition slug, matching what the loaders read."""
    return DATA / SLUG_TO_DATASET.get(slug, slug.replace("-", "_")) / "raw"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=[
        "petfinder-pawpularity-score",
    ], help="Kaggle competition slugs")
    args = ap.parse_args()

    load_env()
    token = os.environ.get("KAGGLE_API_TOKEN")
    if not token:
        print("ERROR: KAGGLE_API_TOKEN not set (looked in env and .env file).")
        return 1

    # kagglehub picks up KAGGLE_API_TOKEN automatically.
    print(f"using KAGGLE_API_TOKEN (length={len(token)})")

    failures = []
    for slug in args.datasets:
        out_dir = dataset_dir(slug)
        try:
            download_competition(slug, out_dir)
        except Exception as e:
            print(f"[{slug}] FAILED: {type(e).__name__}: {e}")
            failures.append((slug, str(e)))

    if failures:
        print("\n=== summary: failures ===")
        for slug, err in failures:
            print(f"  - {slug}: {err}")
        return 1
    print("\nAll downloads complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
