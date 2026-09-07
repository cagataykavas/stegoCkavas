"""
stego_ai.pipeline
----------------
Command-line pipeline to:

1) (Optional) normalize cover images
2) generate stego images (LSB / PVD / DCT / DWT) for a given payload (bpp)
3) prepare binary and multi-class classification datasets
4) train and evaluate classical ML classifiers on handcrafted features

Designed for university-course "steganography + steganalysis" projects.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import joblib

from stego_ai import dataset_preparation as dp
from stego_ai import models as md


def _ensure_dir(p: str) -> str:
    Path(p).mkdir(parents=True, exist_ok=True)
    return p


def _save_bundle(work_dir: str, bundle: dict, models_bin=None, models_multi=None) -> None:
    """Save metrics.json and (optionally) fitted models under work_dir/saved_models/."""
    out_dir = Path(work_dir) / "saved_models"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2)

    # Save fitted models for later analysis/plots (FP/FN etc.)
    if models_bin:
        joblib.dump(models_bin, out_dir / "models_binary.joblib")
    if models_multi:
        joblib.dump(models_multi, out_dir / "models_multiclass.joblib")


def run_pipeline(args: argparse.Namespace) -> dict:
    work_dir = _ensure_dir(args.work_dir)
    stego_root = _ensure_dir(os.path.join(work_dir, "stego"))
    cls_bin_dir = _ensure_dir(os.path.join(work_dir, "classification_binary"))
    cls_multi_dir = _ensure_dir(os.path.join(work_dir, "classification_multiclass"))

    cover_dir = args.cover_dir

    # 0) Normalize covers (optional but recommended for BOSS/BOWS etc).
    if args.normalize_covers:
        norm_dir = _ensure_dir(os.path.join(work_dir, "covers_normalized"))
        cover_dir = dp.normalize_covers(
            cover_dir=cover_dir,
            out_dir=norm_dir,
            force=args.force_prepare,  # reuse "prepare" force for normalization too
        )

    # 1) Generate stego images
    if args.skip_generate and not args.force_generate:
        stego_dirs: Dict[str, str] = {alg: os.path.join(stego_root, f"stego_{alg}") for alg in args.algorithms}
    else:
        stego_dirs = dp.generate_stego_datasets(
            cover_dir=cover_dir,
            output_root=stego_root,
            algorithms=args.algorithms,
            bpp=args.bpp,
            seed=args.seed,
            force=args.force_generate,
        )

    # 2) Prepare classification datasets
    if not (args.skip_prepare and not args.force_prepare):
        dp.prepare_classification_sets(
            cover_dir=cover_dir,
            stego_dirs=stego_dirs,
            output_dir=cls_bin_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
            multiclass=False,
            max_per_split=args.max_per_split,
            force=args.force_prepare,
        )

        class_order = dp.prepare_classification_sets(
            cover_dir=cover_dir,
            stego_dirs=stego_dirs,
            output_dir=cls_multi_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.seed,
            multiclass=True,
            max_per_split=args.max_per_split,
            force=args.force_prepare,
        ) or (["cover"] + [f"stego_{a}" for a in args.algorithms])
    else:
        class_order = ["cover"] + [f"stego_{a}" for a in args.algorithms]

    # 3) Train + evaluate binary models
    X_train, y_train = md.load_classification_data(
        dataset_dir=os.path.join(cls_bin_dir, "train"),
        feature_method=args.feature_method,
        feature_size=args.feature_size,
        dct_size=args.dct_size,
        class_order=["cover", "stego"],
    )
    X_test, y_test = md.load_classification_data(
        dataset_dir=os.path.join(cls_bin_dir, "test"),
        feature_method=args.feature_method,
        feature_size=args.feature_size,
        dct_size=args.dct_size,
        class_order=["cover", "stego"],
    )

    models_bin = md.train_classifiers(
        X_train=X_train,
        y_train=y_train,
        model_names=args.models,
        seed=args.seed,
        pca_components=args.pca_components,
        num_classes=2,
    )
    results_bin = md.evaluate_classifiers(
        models=models_bin,
        X_test=X_test,
        y_test=y_test,
        class_names=["cover", "stego"],
        header="Binary classification results (cover vs stego):",
    )

    # 4) Multiclass (optional)
    results_multi = None
    models_multi = None
    if args.skip_multiclass:
        print("[skip] Multiclass disabled (--skip-multiclass).")
    else:
        X_train_m, y_train_m = md.load_classification_data(
            dataset_dir=os.path.join(cls_multi_dir, "train"),
            feature_method=args.feature_method,
            feature_size=args.feature_size,
            dct_size=args.dct_size,
            class_order=class_order,
        )
        X_test_m, y_test_m = md.load_classification_data(
            dataset_dir=os.path.join(cls_multi_dir, "test"),
            feature_method=args.feature_method,
            feature_size=args.feature_size,
            dct_size=args.dct_size,
            class_order=class_order,
        )

        models_multi = md.train_classifiers(
            X_train=X_train_m,
            y_train=y_train_m,
            model_names=args.models,
            seed=args.seed,
            pca_components=args.pca_components,
            num_classes=len(class_order),
        )
        results_multi = md.evaluate_classifiers(
            models=models_multi,
            X_test=X_test_m,
            y_test=y_test_m,
            class_names=class_order,
            header="Multi-class classification results (cover vs specific algorithms):",
        )

    bundle = {
        "args": {
            "cover_dir": args.cover_dir,
            "work_dir": args.work_dir,
            "bpp": args.bpp,
            "algorithms": args.algorithms,
            "feature_method": args.feature_method,
            "models": args.models,
            "max_per_split": args.max_per_split,
            "pca_components": args.pca_components,
            "normalize_covers": bool(args.normalize_covers),
        },
        "binary": results_bin,
        "multiclass": results_multi,
    }

    if args.save_models:
        _save_bundle(work_dir, bundle, models_bin=models_bin, models_multi=models_multi)

    return bundle


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Steganography + steganalysis pipeline.")

    p.add_argument("--cover-dir", required=True, help="Folder containing cover images.")
    p.add_argument("--work-dir", required=True, help="Working folder for outputs.")

    p.add_argument("--bpp", type=float, default=0.4, help="Payload in bits-per-pixel (bpp).")
    p.add_argument("--algorithms", nargs="+", default=["lsb", "pvd", "dct", "dwt"], help="Stego algorithms to generate.")
    p.add_argument("--seed", type=int, default=1337, help="Random seed.")

    # Dataset split
    p.add_argument("--train-ratio", type=float, default=0.725, help="Train split ratio.")
    p.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio.")
    p.add_argument("--max-per-split", type=int, default=None, help="Max images per split (paired stems).")

    # Feature extraction
    p.add_argument(
        "--feature-method",
        type=str,
        default="raw",
        choices=["raw", "dct", "residual_hist", "residual_cooc"],
        help="Feature extraction method.",
    )
    p.add_argument(
        "--feature-size",
        type=int,
        default=64,
        help="For raw/dct: resize images to feature_size×feature_size. Ignored for residual methods.",
    )
    p.add_argument("--dct-size", type=int, default=8, help="DCT block size (typically 8).")

    # Models
    p.add_argument(
        "--models",
        nargs="+",
        default=["rf", "logreg", "svm"],
        help="Models to train: rf logreg svm (xgb/lgb require optional dependencies).",
    )
    p.add_argument("--pca-components", type=int, default=None, help="Optional PCA components.")

    # Flags
    p.add_argument("--normalize-covers", dest="normalize_covers", action="store_true",
                   help="Normalize covers to clean PNGs in work_dir before processing.")
    p.add_argument("--skip-generate", action="store_true", help="Skip stego generation if already present.")
    p.add_argument("--skip-prepare", action="store_true", help="Skip dataset preparation if already present.")
    p.add_argument("--force-generate", action="store_true", help="Force regenerate stego images.")
    p.add_argument("--force-prepare", action="store_true", help="Force rebuild prepared datasets.")
    p.add_argument("--skip-multiclass", action="store_true", help="Skip multi-class training/evaluation.")
    p.add_argument("--save-models", action="store_true", help="Save metrics.json and fitted models (joblib).")

    return p


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_pipeline(args)


if __name__ == "__main__":
    main()
