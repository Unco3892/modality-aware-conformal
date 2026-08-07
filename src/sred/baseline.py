"""SRED multi-modal baseline with frozen pretrained encoders.

Computes (and caches to disk) text embeddings (multilingual sentence-transformer)
and image embeddings (DINOv2 ViT-S/14) for the Swiss Real Estate Dataset, then
trains XGBoost regressors on log(price) with four feature sets:
  - tab        : living_space, rooms, lat, lon
  - tab+text   : tab + header_emb + ad_description_emb
  - tab+img    : tab + montage_organized_emb + satellite_emb
  - all        : tab + text + image embeddings

Reports MAE / RMSE / R^2 (on log-price) for each.

Run from the repository root with the multi_cyclone env's python.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from PIL import Image
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
SRED = ROOT / "data" / "sred"
META = SRED / "metadata"
IMG = SRED / "processed_images"
CACHE = SRED / "embeddings"
RESULTS = ROOT / "results" / "sred_baseline"



# ---------------------------------------------------------------------------
# data loading


def load_sred() -> tuple[pd.DataFrame, pd.DataFrame]:
    tr = pd.read_csv(META / "train_data_with_text.csv", encoding="latin-1")
    te = pd.read_csv(META / "test_data_with_text.csv", encoding="latin-1")
    for df in (tr, te):
        df["listing_id"] = df["listing_id"].astype("Int64").astype(str)
        df["header"] = df["header"].fillna("").astype(str)
        df["ad_description"] = df["ad_description"].fillna("").astype(str)
    return tr, te


def image_path(split: str, kind: str, listing_id: str) -> Path:
    return IMG / split / kind / f"{listing_id}.jpeg"


# ---------------------------------------------------------------------------
# text embeddings


TEXT_MODEL_REVISIONS = {
    "intfloat/multilingual-e5-large": "3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2":
        "e8f8c211226b894fcb81acc59f3b34ba3efd5f42",
}
DINOV2_REVISION = "7b187bd4df8efce2cbcbbb67bd01532c19bf4c9c"


def embed_text(
    texts: Sequence[str],
    model_id: str,
    device: str,
    revision: str | None = None,
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    # Models that require an input prefix — see https://huggingface.co/intfloat/multilingual-e5-large
    # We treat real-estate ad text as "passage" (long-form documents).
    prefix = ""
    if "multilingual-e5" in model_id or "/e5-" in model_id:
        prefix = "passage: "
    elif "bge-m3" in model_id.lower():
        prefix = ""  # bge-m3 does not need a prefix for dense retrieval

    inputs = [f"{prefix}{t}" for t in texts] if prefix else list(texts)

    # Pick a batch size that fits the bigger models on a 4090 (24GB).
    batch_size = 64 if any(s in model_id for s in ("e5-large", "bge-m3", "mpnet-base-v2")) else 128

    revision = revision or TEXT_MODEL_REVISIONS.get(model_id)
    if revision is None:
        raise ValueError(
            f"no pinned revision is registered for {model_id!r}; pass an explicit revision"
        )
    m = SentenceTransformer(model_id, device=device, revision=revision)
    embs = m.encode(
        inputs,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embs.astype(np.float32)


# ---------------------------------------------------------------------------
# image embeddings (DINOv2 ViT-S/14, frozen, mean-pooled CLS)


DINOV2_DIMS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}


class ImageEncoder:
    def __init__(
        self,
        device: str,
        arch: str = "dinov2_vits14",
        revision: str = DINOV2_REVISION,
    ):
        from torchvision import transforms

        self.device = device
        self.arch = arch
        self.revision = revision
        self.model = torch.hub.load(
            f"facebookresearch/dinov2:{revision}",
            arch,
            trust_repo=True,
        )
        self.model.eval().to(device)
        # DINOv2 default preprocessing (224x224, ImageNet stats).
        self.tx = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.dim = DINOV2_DIMS[arch]

    @torch.no_grad()
    def encode_paths(self, paths: Sequence[Path], batch_size: int = 64) -> np.ndarray:
        out = np.zeros((len(paths), self.dim), dtype=np.float32)
        buf: list[torch.Tensor] = []
        idxs: list[int] = []

        def flush():
            if not buf:
                return
            x = torch.stack(buf).to(self.device)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=self.device == "cuda"):
                z = self.model(x)
            z = torch.nn.functional.normalize(z, dim=-1).float().cpu().numpy()
            for i, j in enumerate(idxs):
                out[j] = z[i]
            buf.clear()
            idxs.clear()

        t0 = time.time()
        for j, p in enumerate(paths):
            try:
                img = Image.open(p).convert("RGB")
            except FileNotFoundError:
                # Leave row as zeros; will be flagged downstream.
                continue
            buf.append(self.tx(img))
            idxs.append(j)
            if len(buf) >= batch_size:
                flush()
            if (j + 1) % 1000 == 0:
                rate = (j + 1) / max(time.time() - t0, 1e-6)
                print(f"  encoded {j+1}/{len(paths)} ({rate:.1f}/s)")
        flush()
        return out


# ---------------------------------------------------------------------------
# caching


def cached(path: Path, fn, *args, **kwargs) -> np.ndarray:
    if path.exists():
        print(f"  load cache {path.name}")
        return np.load(path)
    print(f"  compute {path.name}")
    arr = fn(*args, **kwargs)
    np.save(path, arr)
    return arr


# ---------------------------------------------------------------------------
# XGBoost training + eval


def fit_xgb(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, y_te: np.ndarray, seed: int = 0) -> dict:
    model = xgb.XGBRegressor(
        n_estimators=2000,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        early_stopping_rounds=50,
        eval_metric="rmse",
        tree_method="hist",
        device="cuda",
        random_state=seed,
    )
    # 90/10 split inside train for early stopping.
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X_tr))
    cut = int(0.9 * len(idx))
    Xt, yt = X_tr[idx[:cut]], y_tr[idx[:cut]]
    Xv, yv = X_tr[idx[cut:]], y_tr[idx[cut:]]
    t0 = time.time()
    model.fit(Xt, yt, eval_set=[(Xv, yv)], verbose=False)
    fit_s = time.time() - t0
    pred = model.predict(X_te)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_te, pred))),
        "mae": float(mean_absolute_error(y_te, pred)),
        "r2": float(r2_score(y_te, pred)),
        "fit_seconds": fit_s,
        "n_features": X_tr.shape[1],
    }


def assemble(parts: Iterable[np.ndarray]) -> np.ndarray:
    return np.concatenate(list(parts), axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-model", default="intfloat/multilingual-e5-large")
    ap.add_argument(
        "--text-revision",
        default=None,
        help="immutable Hugging Face revision; known paper encoders are pinned automatically",
    )
    ap.add_argument("--image-arch", default="dinov2_vitb14",
                    choices=["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"])
    ap.add_argument(
        "--image-revision",
        default=DINOV2_REVISION,
        help="immutable facebookresearch/dinov2 Git revision",
    )
    ap.add_argument(
        "--approximate-fast",
        action="store_true",
        help="explicitly use the smaller approximate reference recipe",
    )
    ap.add_argument("--image-types", nargs="+", default=["montage_organized", "satellite"],
                    choices=["montage_organized", "montage_random", "satellite", "cat"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--variants", nargs="+",
                    default=["tab", "tab_text", "tab_img", "all"])
    ap.add_argument("--results-tag", default=None,
                    help="If set, append to results filename: baseline_<tag>_seed{seed}.json")
    ap.add_argument("--results-dir", default=None,
                    help="Directory to write results into (default: results/sred_baseline)")
    args = ap.parse_args()
    if args.approximate_fast:
        args.text_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        args.image_arch = "dinov2_vits14"
    text_revision = args.text_revision or TEXT_MODEL_REVISIONS.get(args.text_model)
    if text_revision is None:
        ap.error("--text-revision is required for an unregistered text model")
    CACHE.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}", "gpu=", torch.cuda.get_device_name(0) if device == "cuda" else "")

    print("loading SRED ...")
    tr, te = load_sred()
    print(f"  train={len(tr)} test={len(te)}")

    tab_cols = ["living_space", "rooms", "lat", "lon"]
    X_tab_tr = tr[tab_cols].to_numpy(dtype=np.float32)
    X_tab_te = te[tab_cols].to_numpy(dtype=np.float32)
    y_tr = np.log(tr["price"].to_numpy(dtype=np.float32))
    y_te = np.log(te["price"].to_numpy(dtype=np.float32))

    # ---- text embeddings (cached) ----------------------------------------
    text_slug = args.text_model.split("/")[-1].replace(".", "-")
    text_parts_tr, text_parts_te = [], []
    for col in ["header", "ad_description"]:
        for split, df, parts in [("train", tr, text_parts_tr), ("test", te, text_parts_te)]:
            cache = CACHE / f"{split}_{col}_{text_slug}.npy"
            arr = cached(
                cache,
                embed_text,
                df[col].tolist(),
                args.text_model,
                device,
                text_revision,
            )
            parts.append(arr)
    X_text_tr = assemble(text_parts_tr)
    X_text_te = assemble(text_parts_te)
    print(f"  text emb dim per split = {X_text_tr.shape[1]}")

    # ---- image embeddings (cached) ---------------------------------------
    img_parts_tr, img_parts_te = [], []
    encoder = None
    img_slug = args.image_arch.replace("_", "-")
    for kind in args.image_types:
        for split, df, parts in [("train", tr, img_parts_tr), ("test", te, img_parts_te)]:
            cache = CACHE / f"{split}_{kind}_{img_slug}.npy"
            if cache.exists():
                print(f"  load cache {cache.name}")
                parts.append(np.load(cache))
                continue
            if encoder is None:
                encoder = ImageEncoder(
                    device=device,
                    arch=args.image_arch,
                    revision=args.image_revision,
                )
            paths = [image_path(split, kind, lid) for lid in df["listing_id"]]
            arr = encoder.encode_paths(paths)
            np.save(cache, arr)
            parts.append(arr)
    X_img_tr = assemble(img_parts_tr) if img_parts_tr else np.zeros((len(tr), 0), dtype=np.float32)
    X_img_te = assemble(img_parts_te) if img_parts_te else np.zeros((len(te), 0), dtype=np.float32)
    print(f"  image emb dim per split = {X_img_tr.shape[1]}")

    # ---- variants --------------------------------------------------------
    feature_sets = {
        "tab": (X_tab_tr, X_tab_te),
        "tab_text": (assemble([X_tab_tr, X_text_tr]), assemble([X_tab_te, X_text_te])),
        "tab_img": (assemble([X_tab_tr, X_img_tr]), assemble([X_tab_te, X_img_te])),
        "all": (
            assemble([X_tab_tr, X_text_tr, X_img_tr]),
            assemble([X_tab_te, X_text_te, X_img_te]),
        ),
    }

    out: dict[str, dict] = {}
    for name in args.variants:
        Xtr, Xte = feature_sets[name]
        print(f"\n[{name}] features={Xtr.shape[1]}")
        out[name] = fit_xgb(Xtr, y_tr, Xte, y_te, seed=args.seed)
        print(f"  RMSE={out[name]['rmse']:.4f}  MAE={out[name]['mae']:.4f}  R2={out[name]['r2']:.4f}  ({out[name]['fit_seconds']:.1f}s)")

    results_dir = Path(args.results_dir) if args.results_dir else RESULTS
    results_dir.mkdir(parents=True, exist_ok=True)
    if args.results_tag:
        out_path = results_dir / f"baseline_{args.results_tag}_seed{args.seed}.json"
    else:
        out_path = results_dir / f"baseline_seed{args.seed}.json"
    payload = {
        "config": {
            "image_arch": args.image_arch,
            "text_model": args.text_model,
            "text_revision": text_revision,
            "image_revision": args.image_revision,
            "cache_rebuild_mode": (
                "approximate_fast"
                if args.approximate_fast
                else "pinned_reference_rebuild_hash_verification_required"
            ),
            "image_types": list(args.image_types),
            "seed": args.seed,
        },
        "results": out,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    sys.exit(main() or 0)
