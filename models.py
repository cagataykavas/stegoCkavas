"""
stego_ai.models
---------------
Feature extraction + model training/evaluation.

This module intentionally focuses on classical ML (RF/XGB/LGB/SVM/LogReg)
and handcrafted features. For CNN-based SOTA (SRNet, YeNet, etc.), you'd
build a separate deep-learning pipeline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Optional deps
try:
    from xgboost import XGBClassifier  # type: ignore
except Exception:
    XGBClassifier = None  # type: ignore

try:
    from lightgbm import LGBMClassifier  # type: ignore
except Exception:
    LGBMClassifier = None  # type: ignore


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pgm"}


def _list_images(folder: str) -> List[str]:
    files: List[str] = []
    for root, _, names in os.walk(folder):
        for n in names:
            ext = os.path.splitext(n)[1].lower()
            if ext in IMAGE_EXTS:
                files.append(os.path.join(root, n))
    files.sort()
    return files


def extract_features(
    image_path: str,
    method: str = "raw",
    size: Tuple[int, int] = (64, 64),
    dct_size: int = 8,
) -> np.ndarray:
    """
    Feature extractors:

    - raw: resize to `size`, flatten RGB [0,1]
    - dct: block-wise mean abs DCT coefficients (b×b) on grayscale (no resizing unless needed)
    - residual_hist: histogram of horizontal & vertical pixel differences, clipped
    - residual_cooc: co-occurrence matrices of residuals (SPAM-like, compact)
      (default params chosen so output dim = 162)
    """
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    # RAW (content-heavy; mostly a baseline)
    if method == "raw":
        if size is not None:
            img_bgr = cv2.resize(img_bgr, size, interpolation=cv2.INTER_AREA)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return img_rgb.reshape(-1).astype(np.float32)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # DCT summary
    if method == "dct":
        b = int(dct_size) if dct_size else 8
        h, w = gray.shape
        h2 = (h // b) * b
        w2 = (w // b) * b
        g = (gray[:h2, :w2] / 255.0).astype(np.float32)

        coeff_sum = np.zeros((b, b), dtype=np.float64)
        n_blocks = 0
        for y0 in range(0, h2, b):
            for x0 in range(0, w2, b):
                block = g[y0:y0+b, x0:x0+b]
                d = cv2.dct(block)
                coeff_sum += np.abs(d)
                n_blocks += 1

        feat = coeff_sum / max(n_blocks, 1)
        return feat.flatten().astype(np.float32)

    # Residual histogram (tiny, fast)
    if method == "residual_hist":
        g = gray.astype(np.int16)
        dh = g[:, 1:] - g[:, :-1]
        dv = g[1:, :] - g[:-1, :]

        T = 4
        dh = np.clip(dh, -T, T)
        dv = np.clip(dv, -T, T)

        bins = np.arange(-T, T + 2)  # inclusive edges
        hist_h, _ = np.histogram(dh, bins=bins)
        hist_v, _ = np.histogram(dv, bins=bins)

        hist_h = hist_h.astype(np.float32) / (dh.size + 1e-8)
        hist_v = hist_v.astype(np.float32) / (dv.size + 1e-8)
        return np.concatenate([hist_h, hist_v]).astype(np.float32)

    # Residual co-occurrence (compact SPAM-ish)
    if method == "residual_cooc":
        # Choose T=4 -> levels=9 -> 9x9=81 per direction; use H and V => 162 features
        g = gray.astype(np.int16)
        # residuals
        rh = g[:, 1:] - g[:, :-1]      # H residuals
        rv = g[1:, :] - g[:-1, :]      # V residuals

        T = 4
        levels = 2 * T + 1

        def cooc_2d(r: np.ndarray, axis: int) -> np.ndarray:
            # axis=1 => along columns, axis=0 => along rows
            r = np.clip(r, -T, T) + T  # shift to [0..2T]
            if axis == 1:
                a = r[:, :-1].ravel()
                b = r[:, 1:].ravel()
            else:
                a = r[:-1, :].ravel()
                b = r[1:, :].ravel()

            idx = a * levels + b
            counts = np.bincount(idx, minlength=levels * levels).astype(np.float32)
            counts /= (counts.sum() + 1e-8)
            return counts.reshape(levels, levels)

        C_h = cooc_2d(rh, axis=1)
        C_v = cooc_2d(rv, axis=0)
        feat = np.concatenate([C_h.flatten(), C_v.flatten()]).astype(np.float32)
        return feat

    raise ValueError(f"Unknown feature method: {method}")


def load_classification_data(
    dataset_dir: str,
    feature_method: str,
    feature_size: int = 64,
    dct_size: int = 8,
    class_order: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads a single split folder (train/val/test).

    Expected structure:
      dataset_dir/
        cover/
        stego/                  (binary)
        stego_lsb/ stego_pvd/... (multiclass)

    class_order controls which subfolders (classes) we load and their label ids.
    If None, we infer by listing subdirectories (sorted).
    """
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"dataset_dir not found: {dataset_dir}")

    if class_order is None:
        class_order = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])

    X_list: List[np.ndarray] = []
    y_list: List[int] = []

    size = (feature_size, feature_size)

    for yi, cname in enumerate(class_order):
        cdir = os.path.join(dataset_dir, cname)
        if not os.path.isdir(cdir):
            # allow missing class dirs (e.g., if you skip some algorithms)
            continue
        files = _list_images(cdir)
        for fp in files:
            feat = extract_features(fp, method=feature_method, size=size, dct_size=dct_size)
            X_list.append(feat)
            y_list.append(yi)

    if not X_list:
        raise RuntimeError(f"No samples found under {dataset_dir} with classes={class_order}")

    X = np.vstack([x.reshape(1, -1) for x in X_list]).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)

    return X, y


def _build_estimator(name: str, seed: int, num_classes: int):
    name = name.lower().strip()

    if name == "rf":
        return RandomForestClassifier(
            n_estimators=400,
            random_state=seed,
            n_jobs=-1,
        )

    if name == "logreg":
        # scaling is important for linear models
        return LogisticRegression(
            max_iter=2000,
            random_state=seed,
            n_jobs=-1,
        )

    if name == "svm":
        # works well on small/medium feature dims
        return SVC(
            kernel="rbf",
            C=5.0,
            gamma="scale",
        )

    if name == "xgb":
        if XGBClassifier is None:
            raise ImportError("xgboost not installed")
        # Note: for xgboost>=2, eval_metric is required for warnings-free training.
        return XGBClassifier(
            n_estimators=600,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed,
            n_jobs=-1,
            objective="multi:softprob" if num_classes > 2 else "binary:logistic",
            eval_metric="mlogloss" if num_classes > 2 else "logloss",
        )

    if name == "lgb":
        if LGBMClassifier is None:
            raise ImportError("lightgbm not installed")
        return LGBMClassifier(
            n_estimators=800,
            learning_rate=0.05,
            num_leaves=63,
            random_state=seed,
            n_jobs=-1,
            objective="multiclass" if num_classes > 2 else "binary",
        )

    raise ValueError(f"Unknown model name: {name}")


def train_classifiers(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_names: List[str],
    seed: int = 1337,
    pca_components: Optional[int] = None,
    num_classes: int = 2,
) -> Dict[str, Pipeline]:
    """
    Train selected models. Returns dict {name: fitted sklearn Pipeline}.
    Each pipeline contains optional StandardScaler, optional PCA, and classifier.
    """
    trained: Dict[str, Pipeline] = {}

    for name in model_names:
        name_l = name.lower().strip()

        # Build steps
        steps = []

        # Scale for SVM / LogReg; keep trees unscaled
        needs_scale = name_l in {"svm", "logreg"}
        if needs_scale:
            steps.append(("scaler", StandardScaler(with_mean=True, with_std=True)))

        if pca_components is not None and pca_components > 0:
            steps.append(("pca", PCA(n_components=pca_components, random_state=seed)))

        try:
            clf = _build_estimator(name_l, seed=seed, num_classes=num_classes)
        except Exception as e:
            print(f"[warn] Skipping model '{name_l}': {e}")
            continue

        steps.append(("clf", clf))
        pipe = Pipeline(steps)
        try:
            pipe.fit(X_train, y_train)
            trained[name_l] = pipe
        except Exception as e:
            print(f"[warn] Model '{name_l}' failed during fit: {e}")
            continue

    if not trained:
        raise RuntimeError("No models were successfully trained.")

    return trained


def _metrics_dict(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]) -> dict:
    acc = float(accuracy_score(y_true, y_pred))

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), zero_division=0
    )
    f1_macro = float(np.mean(f1))

    out = {
        "accuracy": acc,
        "f1_macro": f1_macro,
    }

    for i, cname in enumerate(class_names):
        out[f"precision_{cname}"] = float(prec[i])
        out[f"recall_{cname}"] = float(rec[i])
        out[f"f1_{cname}"] = float(f1[i])

    # store confusion matrix for your report plots
    out["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names)))).tolist()
    return out


def evaluate_classifiers(
    models: Dict[str, Pipeline],
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
    header: str = "",
) -> Dict[str, dict]:
    """
    Evaluate models and print a clean summary. Returns dict of metrics per model.
    """
    if header:
        print("\n" + header + "\n")

    results: Dict[str, dict] = {}
    for name, model in models.items():
        try:
            y_pred = model.predict(X_test)
        except Exception as e:
            print(f"[warn] Model '{name}' failed during predict: {e}")
            continue

        met = _metrics_dict(y_test, y_pred, class_names)
        results[name] = met

        # Print friendly summary
        print(f"Model: {name}")
        print(f"  Accuracy: {met['accuracy']:.4f}")
        for cname in class_names:
            print(f"  {cname.capitalize()} precision: {met[f'precision_{cname}']:.4f}")
            print(f"  {cname.capitalize()} recall:    {met[f'recall_{cname}']:.4f}")
            print(f"  {cname.capitalize()} F1:        {met[f'f1_{cname}']:.4f}")
        print(f"  Macro F1: {met['f1_macro']:.4f}\n")

    return results
