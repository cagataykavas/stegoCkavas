#!/usr/bin/env python3
"""
Qualitative report generator:
- Finds a saved model under <work-dir>/saved_models
- Scores images from the prepared split folders (train/val/test)
- Exports:
    * confusion_matrix.png
    * top_correct.md / top_errors.md
    * (optional) occlusion heatmaps for a few worst cases

This is intentionally robust to your saved-model format:
- If joblib.load(...) returns a dict, we try common keys (binary, multiclass, model, clf, estimator, etc.)
"""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import joblib
import cv2
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------
# Imports from your project
# -----------------------------
def _import_extract_features(project_root: Path):
    # Ensure "stego_ai" is importable when running from repo root
    sys.path.insert(0, str(project_root))
    try:
        from stego_ai.models import extract_features  # type: ignore
        return extract_features
    except Exception as e:
        raise RuntimeError(
            "Could not import stego_ai.models.extract_features. "
            "Run this script from the project root (where stego_ai/ exists). "
            f"Original import error: {e}"
        )


# -----------------------------
# Dataset discovery
# -----------------------------
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _list_images(d: Path) -> List[Path]:
    if not d.exists():
        return []
    out: List[Path] = []
    for p in d.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            out.append(p)
    return out


def find_prepared_dataset_root(work_dir: Path, split: str = "test") -> Path:
    """
    Find a folder that contains:
        <root>/<split>/cover/...
    Typical: <work_dir>/classification/<split>/cover/...
             <work_dir>/classification_binary/<split>/cover/...
    We search heuristically to avoid hardcoding your pipeline layout.
    """
    candidates = [
        work_dir / "classification",
        work_dir / "classification_binary",
        work_dir / "prepared",
        work_dir,
    ]
    # quick direct checks
    for c in candidates:
        if (c / split / "cover").exists():
            return c

    # deeper search (bounded)
    for p in work_dir.rglob("cover"):
        if p.is_dir() and p.parent.name == split:
            return p.parent.parent  # root = .../<split>/<cover>
    raise RuntimeError(
        f"Could not find prepared dataset folder under {work_dir}. "
        f"Expected something like <root>/{split}/cover/..."
    )


def load_binary_split(root: Path, split: str = "test") -> Tuple[List[Path], np.ndarray, List[str]]:
    cover_dir = root / split / "cover"
    stego_dir = root / split / "stego"
    if not cover_dir.exists():
        raise RuntimeError(f"Missing cover dir: {cover_dir}")
    if not stego_dir.exists():
        # some pipelines use 'stego_all' or similar; fall back to any 'stego*' dir
        stego_candidates = [d for d in (root / split).iterdir() if d.is_dir() and d.name.startswith("stego")]
        if not stego_candidates:
            raise RuntimeError(f"Missing stego dir under: {root/split} (expected 'stego' or 'stego_*')")
        # merge all stego_* as stego label
        paths = _list_images(cover_dir)
        ys = [0] * len(paths)
        for sd in stego_candidates:
            sp = _list_images(sd)
            paths.extend(sp)
            ys.extend([1] * len(sp))
        return paths, np.array(ys, dtype=np.int64), ["cover", "stego"]

    paths = _list_images(cover_dir) + _list_images(stego_dir)
    y = np.array([0] * len(_list_images(cover_dir)) + [1] * len(_list_images(stego_dir)), dtype=np.int64)
    return paths, y, ["cover", "stego"]


# -----------------------------
# Model loading (robust)
# -----------------------------
def _looks_like_estimator(obj: Any) -> bool:
    return hasattr(obj, "predict") or hasattr(obj, "predict_proba") or hasattr(obj, "decision_function")


def _unwrap_model(obj: Any, model_name: str) -> Any:
    """
    Handle various save formats:
    - estimator directly
    - dict with keys: model, clf, estimator
    - dict with "binary": {"rf": estimator, ...}
    """
    if _looks_like_estimator(obj):
        return obj

    if isinstance(obj, dict):
        # common wrappers
        for k in ["model", "clf", "estimator", "classifier"]:
            if k in obj and _looks_like_estimator(obj[k]):
                return obj[k]

        # nested dicts
        for k in ["binary", "binary_models", "models", "classifiers"]:
            if k in obj and isinstance(obj[k], dict):
                sub = obj[k]
                if model_name in sub:
                    return _unwrap_model(sub[model_name], model_name)
        # direct by name
        if model_name in obj:
            return _unwrap_model(obj[model_name], model_name)

        # last resort: first estimator-like value
        for v in obj.values():
            if _looks_like_estimator(v):
                return v

    raise RuntimeError(
        f"Loaded model object has unexpected type/structure: {type(obj)}. "
        "Open the saved .joblib and check its keys."
    )


def find_model_joblib(saved_models_dir: Path, model_name: str) -> Path:
    joblibs = sorted(saved_models_dir.glob("*.joblib"))
    if not joblibs:
        raise RuntimeError(f"No .joblib files found under {saved_models_dir}")

    # Prefer file names that mention binary + model name
    needles = [
        f"binary_{model_name}",
        f"{model_name}_binary",
        model_name,
    ]
    for nd in needles:
        for p in joblibs:
            if nd.lower() in p.name.lower():
                return p

    # If only one, use it.
    if len(joblibs) == 1:
        return joblibs[0]

    # otherwise, try a generic models.joblib
    for p in joblibs:
        if "models" in p.name.lower():
            return p

    raise RuntimeError(
        "Could not decide which joblib to use. Candidates:\n" +
        "\n".join(f" - {p.name}" for p in joblibs)
    )


# -----------------------------
# Prediction utilities
# -----------------------------
def predict_proba_binary(clf: Any, X: np.ndarray) -> np.ndarray:
    """
    Returns P(stego=1) for each row.
    """
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X)
        proba = np.asarray(proba)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
        if proba.ndim == 1:
            return proba
    if hasattr(clf, "decision_function"):
        s = np.asarray(clf.decision_function(X)).reshape(-1)
        # logistic squash
        return 1.0 / (1.0 + np.exp(-s))
    # last resort: hard prediction
    pred = np.asarray(clf.predict(X)).reshape(-1)
    return pred.astype(np.float32)


def confusion_matrix_binary(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    cm = np.zeros((2, 2), dtype=np.int64)
    for yt, yp in zip(y_true, y_pred):
        cm[int(yt), int(yp)] += 1
    return cm


def save_confusion_matrix(cm: np.ndarray, class_names: List[str], out_path: Path) -> None:
    fig = plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.xticks([0, 1], class_names)
    plt.yticks([0, 1], class_names)
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.ylabel("True")
    plt.xlabel("Pred")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# -----------------------------
# Occlusion heatmap (optional)
# -----------------------------
def occlusion_heatmap(
    image_path: Path,
    extract_features,
    clf: Any,
    method: str,
    win: int = 32,
    stride: int = 16,
) -> np.ndarray:
    """
    Perturbation-based localization:
    - Replace a patch with the local mean
    - Recompute features for the whole image
    - Measure probability drop for stego class
    Produces a heatmap (higher = patch is important for predicting stego).
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    h, w, _ = img.shape

    # baseline
    x0 = extract_features(str(image_path), method=method)
    p0 = float(predict_proba_binary(clf, np.asarray([x0], dtype=np.float32))[0])

    heat = np.zeros((h, w), dtype=np.float32)
    counts = np.zeros((h, w), dtype=np.float32)

    # work copy
    for y in range(0, max(1, h - win + 1), stride):
        for x in range(0, max(1, w - win + 1), stride):
            patch = img[y:y+win, x:x+win]
            if patch.size == 0:
                continue
            mean_val = patch.mean(axis=(0, 1), keepdims=True).astype(img.dtype)
            tmp = img.copy()
            tmp[y:y+win, x:x+win] = mean_val

            # save temp to avoid rewriting extract_features to accept arrays
            # we write into memory by using cv2.imencode -> decode? simplest: write a temp file
            # but for speed, use a named temp path in the same folder
            tmp_path = image_path.parent / ("__tmp_occl.png")
            cv2.imwrite(str(tmp_path), tmp)

            x1 = extract_features(str(tmp_path), method=method)
            p1 = float(predict_proba_binary(clf, np.asarray([x1], dtype=np.float32))[0])

            delta = max(0.0, p0 - p1)  # importance = probability drop
            heat[y:y+win, x:x+win] += delta
            counts[y:y+win, x:x+win] += 1.0

    # cleanup temp
    try:
        (image_path.parent / "__tmp_occl.png").unlink(missing_ok=True)  # type: ignore[attr-defined]
    except Exception:
        pass

    heat = heat / np.maximum(counts, 1.0)
    # normalize 0..1
    if heat.max() > 0:
        heat = heat / heat.max()
    return heat


def save_heatmap_overlay(image_path: Path, heat: np.ndarray, out_path: Path) -> None:
    img = Image.open(image_path).convert("RGB")
    arr = np.asarray(img)
    h, w = arr.shape[:2]

    # resize heat to image
    heat_r = cv2.resize(heat, (w, h), interpolation=cv2.INTER_LINEAR)

    fig = plt.figure(figsize=(6, 6))
    plt.imshow(arr)
    plt.imshow(heat_r, alpha=0.45)
    plt.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# -----------------------------
# Report writing
# -----------------------------
def write_markdown_cards(items: List[Dict[str, Any]], out_md: Path) -> None:
    with open(out_md, "w", encoding="utf-8") as f:
        for it in items:
            f.write(f"### {it['title']}\n\n")
            f.write(f"- Path: `{it['path']}`\n")
            f.write(f"- True: **{it['true']}**, Pred: **{it['pred']}**, P(stego): **{it['p_stego']:.4f}**\n\n")
            if it.get("thumb_rel"):
                f.write(f"![img]({it['thumb_rel']})\n\n")
            if it.get("heat_rel"):
                f.write(f"**Occlusion heatmap:**\n\n")
                f.write(f"![heat]({it['heat_rel']})\n\n")
            f.write("---\n\n")


def make_thumbnail(src: Path, dst: Path, max_side: int = 512) -> None:
    img = Image.open(src).convert("RGB")
    w, h = img.size
    scale = max(w, h) / float(max_side)
    if scale > 1.0:
        img = img.resize((int(w / scale), int(h / scale)), Image.LANCZOS)
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--model", required=True, help="Model name, e.g. rf / xgb / lgb / svm / logreg")
    ap.add_argument("--feature-method", default="residual_cooc")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-per-class", type=int, default=12)
    ap.add_argument("--make-heatmaps", action="store_true", help="Generate occlusion heatmaps for worst cases (slow)")
    args = ap.parse_args()

    work_dir = Path(args.work_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Project root = parent of scripts/
    project_root = Path(__file__).resolve().parents[1]
    extract_features = _import_extract_features(project_root)

    # load model
    saved_models = work_dir / "saved_models"
    if not saved_models.exists():
        raise RuntimeError(f"Missing saved_models folder under: {work_dir}")

    joblib_path = find_model_joblib(saved_models, args.model)
    obj = joblib.load(joblib_path)
    clf = _unwrap_model(obj, args.model)

    # load dataset
    ds_root = find_prepared_dataset_root(work_dir, split=args.split)
    paths, y_true, class_names = load_binary_split(ds_root, split=args.split)

    # extract X
    X = []
    valid_paths: List[Path] = []
    valid_y: List[int] = []
    for pth, y in zip(paths, y_true):
        try:
            feat = extract_features(str(pth), method=args.feature_method)
            X.append(feat)
            valid_paths.append(pth)
            valid_y.append(int(y))
        except Exception:
            # skip unreadable images
            continue
    X = np.asarray(X, dtype=np.float32)
    y_true2 = np.asarray(valid_y, dtype=np.int64)

    p_stego = predict_proba_binary(clf, X)
    y_pred = (p_stego >= 0.5).astype(np.int64)

    cm = confusion_matrix_binary(y_true2, y_pred)
    save_confusion_matrix(cm, class_names, out_dir / "confusion_matrix.png")

    # build ranked lists
    items: List[Dict[str, Any]] = []
    for pth, yt, yp, ps in zip(valid_paths, y_true2, y_pred, p_stego):
        correct = int(yt) == int(yp)
        items.append({
            "path": str(pth),
            "true": class_names[int(yt)],
            "pred": class_names[int(yp)],
            "p_stego": float(ps),
            "correct": correct,
            "conf": float(ps if yp == 1 else (1.0 - ps)),
            "title": ("OK" if correct else "ERROR") + f" | conf={float(ps if yp==1 else 1-ps):.3f}",
            "_pth": pth,
        })

    correct_items = sorted([it for it in items if it["correct"]], key=lambda d: d["conf"], reverse=True)
    error_items = sorted([it for it in items if not it["correct"]], key=lambda d: d["conf"], reverse=True)

    # make thumbnails + heatmaps
    def enrich(it: Dict[str, Any], idx: int, make_heat: bool) -> Dict[str, Any]:
        pth = it["_pth"]
        thumb = out_dir / "thumbs" / f"{idx:03d}.png"
        make_thumbnail(pth, thumb)
        it["thumb_rel"] = os.path.relpath(thumb, out_dir).replace("\\", "/")
        if make_heat:
            heat = occlusion_heatmap(pth, extract_features, clf, method=args.feature_method, win=32, stride=16)
            heat_path = out_dir / "heatmaps" / f"{idx:03d}.png"
            heat_path.parent.mkdir(parents=True, exist_ok=True)
            save_heatmap_overlay(pth, heat, heat_path)
            it["heat_rel"] = os.path.relpath(heat_path, out_dir).replace("\\", "/")
        return it

    top_ok = [enrich(dict(it), i, False) for i, it in enumerate(correct_items[: args.max_per_class])]
    top_err = [enrich(dict(it), i, args.make_heatmaps) for i, it in enumerate(error_items[: args.max_per_class])]

    write_markdown_cards(top_ok, out_dir / "top_correct.md")
    write_markdown_cards(top_err, out_dir / "top_errors.md")

    print("Saved:")
    print(" -", out_dir / "confusion_matrix.png")
    print(" -", out_dir / "top_correct.md")
    print(" -", out_dir / "top_errors.md")
    print(" - thumbs/ (and heatmaps/ if enabled)")


if __name__ == "__main__":
    main()
