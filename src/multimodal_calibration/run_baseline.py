"""Unified baseline driver for the AAAI heterogeneous-modality study.

Usage::

    python src/multimodal_calibration/run_baseline.py --dataset imdb_wiki

Steps for each dataset:
  1. Call ``datasets.<name>.load("train")`` and ``load("test")``.
  2. Cache frozen-encoder embeddings (DINOv2 image, multilingual-e5-large text)
     to ``data/<name>/embeddings/{split}_{field}_{slug}.npy``.
  3. Build modality feature blocks (tab via one-hot/standardize, text/image via
     cached embeddings).
  4. Run a single-seed XGBoost regression baseline on each modality subset:
     ``tab``, ``tab_text``, ``tab_img``, ``all``.
  5. Write JSON to ``results/multimodal_calibration/<name>_baseline.json``.

Skips gracefully if a dataset's ``load`` raises (e.g. missing Kaggle creds).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[2]))
EXP = ROOT / "src" / "multimodal_calibration"
RESULTS = ROOT / "results" / "multimodal_calibration"
sys.path.insert(0, str(EXP))
sys.path.insert(0, str(ROOT / "src" / "sred"))

from loaders import REGISTRY  # noqa: E402
from encoders import (  # noqa: E402
    DINOV2_REVISION,
    TEXT_MODEL_BY_SLUG,
    TEXT_MODEL_REVISIONS,
    cache_images,
    cache_text,
    paper_cache_slugs,
)


def _onehot_tab(tab_train: pd.DataFrame | None, tab_test: pd.DataFrame | None
                ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Standardize numeric columns; one-hot encode categoricals (cap top-50 levels)."""
    if tab_train is None or len(tab_train) == 0:
        return (np.zeros((0 if tab_test is None else len(tab_test), 0), dtype=np.float32),
                np.zeros((0 if tab_test is None else len(tab_test), 0), dtype=np.float32),
                [])

    cat_cols, num_cols = [], []
    for c in tab_train.columns:
        if pd.api.types.is_numeric_dtype(tab_train[c]):
            num_cols.append(c)
        else:
            cat_cols.append(c)

    parts_tr, parts_te, names = [], [], []

    if num_cols:
        Xn_tr = tab_train[num_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        Xn_te = tab_test[num_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        # impute with column mean from training set
        col_mean = np.nanmean(Xn_tr, axis=0)
        col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
        for j, c in enumerate(num_cols):
            mask_tr = np.isnan(Xn_tr[:, j])
            mask_te = np.isnan(Xn_te[:, j])
            Xn_tr[mask_tr, j] = col_mean[j]
            Xn_te[mask_te, j] = col_mean[j]
        # standardize
        mu = Xn_tr.mean(axis=0)
        sd = Xn_tr.std(axis=0) + 1e-6
        Xn_tr = (Xn_tr - mu) / sd
        Xn_te = (Xn_te - mu) / sd
        parts_tr.append(Xn_tr); parts_te.append(Xn_te)
        names += list(num_cols)

    for c in cat_cols:
        vc = tab_train[c].astype(str).value_counts()
        levels = vc.head(50).index.tolist()
        for lv in levels:
            parts_tr.append((tab_train[c].astype(str).to_numpy() == lv).astype(np.float32)[:, None])
            parts_te.append((tab_test[c].astype(str).to_numpy() == lv).astype(np.float32)[:, None])
            names.append(f"{c}={lv}")

    Xtr = np.concatenate(parts_tr, axis=1) if parts_tr else np.zeros((len(tab_train), 0), dtype=np.float32)
    Xte = np.concatenate(parts_te, axis=1) if parts_te else np.zeros((len(tab_test), 0), dtype=np.float32)
    return Xtr.astype(np.float32), Xte.astype(np.float32), names


def _embed_modalities(
    name: str,
    train: dict,
    test: dict,
    text_model: str,
    image_arch: str,
    device: str,
    cache_dir: Path,
    n_image_max: int | None = None,
    text_revision: str | None = None,
    image_revision: str = DINOV2_REVISION,
):
    """Return dict of per-modality (Xtr, Xte) arrays. Image/text cached on disk."""
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    Xtab_tr, Xtab_te, _ = _onehot_tab(train.get("tab"), test.get("tab"))
    out["tab"] = (Xtab_tr, Xtab_te)
    print(f"  tab dim: {Xtab_tr.shape[1]}")

    if train.get("text"):
        text_parts_tr, text_parts_te = [], []
        for field in train["text"].keys():
            arr_tr = cache_text(
                cache_dir, "train", field, train["text"][field],
                text_model, device, text_revision,
            )
            arr_te = cache_text(
                cache_dir, "test", field, test["text"][field],
                text_model, device, text_revision,
            )
            text_parts_tr.append(arr_tr); text_parts_te.append(arr_te)
        Xtxt_tr = np.concatenate(text_parts_tr, axis=1).astype(np.float32)
        Xtxt_te = np.concatenate(text_parts_te, axis=1).astype(np.float32)
        out["text"] = (Xtxt_tr, Xtxt_te)
        print(f"  text dim: {Xtxt_tr.shape[1]}")
    else:
        out["text"] = (np.zeros((len(train["y"]), 0), dtype=np.float32),
                       np.zeros((len(test["y"]), 0), dtype=np.float32))

    if train.get("image"):
        img_parts_tr, img_parts_te = [], []
        encoder = None
        for field in train["image"].keys():
            paths_tr = train["image"][field]
            paths_te = test["image"][field]
            if n_image_max:
                # truncate paths but keep alignment by zero-padding the rest later
                pass
            arr_tr, encoder = cache_images(
                cache_dir, "train", field, paths_tr, image_arch, device,
                encoder, image_revision,
            )
            arr_te, encoder = cache_images(
                cache_dir, "test", field, paths_te, image_arch, device,
                encoder, image_revision,
            )
            img_parts_tr.append(arr_tr); img_parts_te.append(arr_te)
        Ximg_tr = np.concatenate(img_parts_tr, axis=1).astype(np.float32)
        Ximg_te = np.concatenate(img_parts_te, axis=1).astype(np.float32)
        out["image"] = (Ximg_tr, Ximg_te)
        print(f"  image dim: {Ximg_tr.shape[1]}")
    else:
        out["image"] = (np.zeros((len(train["y"]), 0), dtype=np.float32),
                        np.zeros((len(test["y"]), 0), dtype=np.float32))

    return out


def _fit_xgb(X_tr, y_tr, X_te, y_te, seed=0, device_str="cuda"):
    model = xgb.XGBRegressor(
        n_estimators=1500, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        early_stopping_rounds=50, eval_metric="rmse",
        tree_method="hist", device=device_str, random_state=seed,
    )
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
        "n_features": int(X_tr.shape[1]),
    }


def run_one(
    name: str,
    text_model: str,
    image_arch: str,
    seed: int = 0,
    log_target: bool = False,
    text_revision: str | None = None,
    image_revision: str = DINOV2_REVISION,
) -> dict:
    print(f"\n=== dataset={name} text={text_model} image={image_arch} ===")
    if name not in REGISTRY:
        raise KeyError(f"unknown dataset {name}; registry={list(REGISTRY)}")

    cache_dir = ROOT / "data" / name / "embeddings"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    t0 = time.time()
    train = REGISTRY[name]("train")
    test = REGISTRY[name]("test")
    print(f"  load: train n={len(train['y'])} test n={len(test['y'])} ({time.time()-t0:.1f}s)")

    y_tr = train["y"]
    y_te = test["y"]
    if log_target:
        y_tr = np.log(np.maximum(y_tr, 1e-6))
        y_te = np.log(np.maximum(y_te, 1e-6))
    print(f"  y range: train [{y_tr.min():.3f}, {y_tr.max():.3f}] mean={y_tr.mean():.3f} std={y_tr.std():.3f}")

    mods = _embed_modalities(
        name, train, test, text_model, image_arch, device, cache_dir,
        text_revision=text_revision, image_revision=image_revision,
    )
    Xtab_tr, Xtab_te = mods["tab"]
    Xtxt_tr, Xtxt_te = mods["text"]
    Ximg_tr, Ximg_te = mods["image"]

    # XGBoost on GPU has been flaky on Windows; default to cpu.
    device_str = os.environ.get("XGB_DEVICE", "cuda")

    variants = {}
    if Xtab_tr.shape[1] > 0:
        variants["tab"] = (Xtab_tr, Xtab_te)
    if Xtab_tr.shape[1] > 0 and Xtxt_tr.shape[1] > 0:
        variants["tab_text"] = (np.concatenate([Xtab_tr, Xtxt_tr], axis=1),
                                np.concatenate([Xtab_te, Xtxt_te], axis=1))
    if Xtab_tr.shape[1] > 0 and Ximg_tr.shape[1] > 0:
        variants["tab_img"] = (np.concatenate([Xtab_tr, Ximg_tr], axis=1),
                               np.concatenate([Xtab_te, Ximg_te], axis=1))
    parts_tr = [Xtab_tr, Xtxt_tr, Ximg_tr]
    parts_te = [Xtab_te, Xtxt_te, Ximg_te]
    if any(p.shape[1] > 0 for p in parts_tr):
        variants["all"] = (np.concatenate(parts_tr, axis=1), np.concatenate(parts_te, axis=1))

    out: dict = {
        "config": {
            "dataset": name, "text_model": text_model, "image_arch": image_arch,
            "text_revision": text_revision, "image_revision": image_revision,
            "seed": seed, "log_target": log_target,
            "device_xgb": device_str,
        },
        "shapes": {
            "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
            "tab_dim": int(Xtab_tr.shape[1]), "text_dim": int(Xtxt_tr.shape[1]),
            "image_dim": int(Ximg_tr.shape[1]),
        },
        "y_stats": {
            "train_mean": float(y_tr.mean()), "train_std": float(y_tr.std()),
            "test_mean": float(y_te.mean()), "test_std": float(y_te.std()),
        },
        "variants": {},
    }
    for vname, (Xtr, Xte) in variants.items():
        print(f"  fit {vname}: features={Xtr.shape[1]}")
        try:
            out["variants"][vname] = _fit_xgb(Xtr, y_tr, Xte, y_te, seed=seed, device_str=device_str)
        except Exception as e:
            print(f"    XGBoost failed on {device_str}: {e!r}; falling back to CPU")
            out["variants"][vname] = _fit_xgb(Xtr, y_tr, Xte, y_te, seed=seed, device_str="cpu")
        m = out["variants"][vname]
        print(f"    {vname} R2={m['r2']:.4f} RMSE={m['rmse']:.4f} MAE={m['mae']:.4f} ({m['fit_seconds']:.1f}s)")

    out_path = RESULTS / f"{name}_baseline.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  wrote {out_path}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(REGISTRY) + ["all"])
    ap.add_argument("--text-model", default=None,
                    help="HF text encoder id; defaults to each dataset's paper contract")
    ap.add_argument("--text-revision", default=None,
                    help="immutable HF revision; paper encoders are pinned automatically")
    ap.add_argument("--image-arch", default=None,
                    choices=list({"dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"}))
    ap.add_argument("--image-revision", default=DINOV2_REVISION)
    ap.add_argument(
        "--approximate-fast",
        action="store_true",
        help="explicitly rebuild every dataset with the smaller approximate recipe",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-target", action="store_true",
                    help="apply np.log(y) before fitting (use for prices)")
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)

    targets = list(REGISTRY) if args.dataset == "all" else [args.dataset]
    summary = {}
    for n in targets:
        text_slug, image_slug = paper_cache_slugs(n)
        text_model = args.text_model or (
            TEXT_MODEL_BY_SLUG[text_slug] if text_slug is not None
            else "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        image_arch = args.image_arch or (
            image_slug.replace("-", "_") if image_slug is not None
            else "dinov2_vits14"
        )
        if args.approximate_fast:
            text_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            image_arch = "dinov2_vits14"
        text_revision = args.text_revision or TEXT_MODEL_REVISIONS.get(text_model)
        if text_revision is None:
            ap.error("--text-revision is required for an unregistered text model")
        try:
            r = run_one(
                n, text_model, image_arch, seed=args.seed,
                log_target=args.log_target, text_revision=text_revision,
                image_revision=args.image_revision,
            )
            summary[n] = {"ok": True, "best_R2": max((v["r2"] for v in r["variants"].values()), default=None)}
        except Exception as e:
            traceback.print_exc()
            summary[n] = {"ok": False, "err": repr(e)}
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    sys.exit(main() or 0)
