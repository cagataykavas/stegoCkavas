"""
scripts/make_report_figures.py

Creates "teacher-friendly" figures:
- examples of TP / TN / FP / FN
- difference heatmaps (cover vs stego) for the same stem
- residual (high-pass) heatmaps

It uses the saved binary model from:
  work_dir/saved_models/models_binary.joblib

Example:
  python scripts\make_report_figures.py --work-dir ..\runs\bpp04_rescooc --model xgb --out-dir ..\runs\bpp04_rescooc\report_figs
"""

import argparse
import os
from pathlib import Path

import joblib
import numpy as np
import cv2
import matplotlib.pyplot as plt

from stego_ai import models as md


def _stem(p: str) -> str:
    return Path(p).stem


def _read_bgr(p: str) -> np.ndarray:
    img = cv2.imread(p, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(p)
    return img


def _residual_map(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # simple high-pass: Laplacian magnitude
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    lap = np.abs(lap)
    # normalize for display
    lap = lap / (lap.max() + 1e-8)
    return lap


def _diff_map(a_bgr: np.ndarray, b_bgr: np.ndarray) -> np.ndarray:
    # amplified absolute difference for visibility
    d = cv2.absdiff(a_bgr, b_bgr).astype(np.float32)
    d = d.mean(axis=2)  # grayscale diff
    d = d / (d.max() + 1e-8)
    return d


def _save_triptych(out_path: str, img: np.ndarray, res: np.ndarray, diff: np.ndarray, title: str) -> None:
    plt.figure(figsize=(10, 3.4))
    plt.suptitle(title)

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(res)
    plt.title("Residual (high-pass)")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(diff)
    plt.title("Diff (cover vs stego)")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--model", default="xgb", help="rf/xgb/lgb/svm/logreg")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--feature-method", default=None, help="override; else uses metrics.json args")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--num", type=int, default=4, help="examples per category (FP/FN)")
    args = ap.parse_args()

    work = Path(args.work_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models_path = work / "saved_models" / "models_binary.joblib"
    if not models_path.exists():
        raise FileNotFoundError(f"Missing saved model: {models_path}. Re-run pipeline with --save-models.")

    models_bin = joblib.load(models_path)
    if args.model not in models_bin:
        raise ValueError(f"Model '{args.model}' not found in saved models: {list(models_bin.keys())}")

    # Infer feature method from metrics.json unless user overrides
    method = args.feature_method
    metrics_path = work / "saved_models" / "metrics.json"
    if method is None and metrics_path.exists():
        import json
        met = json.loads(metrics_path.read_text(encoding="utf-8"))
        method = (met.get("args", {}) or {}).get("feature_method", "residual_cooc")
    if method is None:
        method = "residual_cooc"

    # Load test split
    split_dir = work / "classification_binary" / args.split
    cover_dir = split_dir / "cover"
    stego_dir = split_dir / "stego"
    cover_files = sorted([str(p) for p in cover_dir.glob("*.png")])
    stego_files = sorted([str(p) for p in stego_dir.glob("*.png")])

    cover_by_stem = {_stem(p): p for p in cover_files}
    stego_by_stem = {_stem(p): p for p in stego_files}
    stems = sorted(set(cover_by_stem.keys()) & set(stego_by_stem.keys()))
    if not stems:
        raise RuntimeError("No paired cover/stego stems found in the split folder.")

    # Build feature matrix and labels for BOTH classes
    X = []
    y = []
    paths = []
    for st in stems:
        # cover sample
        cpath = cover_by_stem[st]
        X.append(md.extract_features(cpath, method=method, size=(64, 64), dct_size=8))
        y.append(0)
        paths.append(cpath)
        # stego sample
        spath = stego_by_stem[st]
        X.append(md.extract_features(spath, method=method, size=(64, 64), dct_size=8))
        y.append(1)
        paths.append(spath)

    X = np.vstack([x.reshape(1, -1) for x in X]).astype(np.float32)
    y = np.array(y, dtype=np.int64)

    model = models_bin[args.model]
    yhat = model.predict(X)

    # Identify cases
    fp = [paths[i] for i in range(len(paths)) if y[i] == 0 and yhat[i] == 1]
    fn = [paths[i] for i in range(len(paths)) if y[i] == 1 and yhat[i] == 0]
    tp = [paths[i] for i in range(len(paths)) if y[i] == 1 and yhat[i] == 1]
    tn = [paths[i] for i in range(len(paths)) if y[i] == 0 and yhat[i] == 0]

    def dump_examples(tag: str, ex_paths: list, n: int):
        for k, p in enumerate(ex_paths[:n]):
            st = _stem(p)
            img = _read_bgr(p)
            res = _residual_map(img)

            # Diff map: always between paired cover/stego of the same stem
            cimg = _read_bgr(cover_by_stem[st])
            simg = _read_bgr(stego_by_stem[st])
            diff = _diff_map(cimg, simg)

            title = f"{tag} | stem={st} | true={'cover' if y[paths.index(p)]==0 else 'stego'} pred={'cover' if yhat[paths.index(p)]==0 else 'stego'}"
            out = out_dir / f"{tag}_{k+1:02d}_{st}.png"
            _save_triptych(str(out), img, res, diff, title)

    dump_examples("FP", fp, args.num)
    dump_examples("FN", fn, args.num)
    dump_examples("TP", tp, args.num)
    dump_examples("TN", tn, args.num)

    print("Saved figures to:", out_dir)


if __name__ == "__main__":
    main()
