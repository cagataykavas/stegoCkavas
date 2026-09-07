"""
stego_ai.dataset_preparation
----------------------------
Utilities to:

- list and normalize cover images
- generate stego images using selected algorithms
- create train/val/test folder structures for binary and multiclass classification

Folder conventions produced:

work_dir/
  stego/
    stego_lsb/
    stego_pvd/
    stego_dct/
    stego_dwt/
  classification_binary/
    train/{cover,stego}/
    val/{cover,stego}/
    test/{cover,stego}/
  classification_multiclass/
    train/{cover,stego_lsb,stego_pvd,...}/
    ...
"""

from __future__ import annotations

import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2

from stego_ai import stego_algorithms as sa

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pgm"}


def list_image_files(folder: str) -> List[str]:
    files: List[str] = []
    for root, _, names in os.walk(folder):
        for n in names:
            ext = os.path.splitext(n)[1].lower()
            if ext in IMAGE_EXTS:
                files.append(os.path.join(root, n))
    files.sort()
    return files


def _stem(p: str) -> str:
    return Path(p).stem


def normalize_covers(cover_dir: str, out_dir: str, force: bool = False) -> str:
    """
    Convert all covers to clean PNG files (8-bit, BGR), stripping metadata and ensuring
    consistent decoding (helps when original dataset has .pgm or mixed formats).

    Output filenames keep the same stem as input (e.g., IMG_001.pgm -> IMG_001.png).
    """
    os.makedirs(out_dir, exist_ok=True)

    existing = list_image_files(out_dir)
    if existing and not force:
        return out_dir

    # Clear output to avoid stale files when force=True
    if force:
        for p in existing:
            try:
                os.remove(p)
            except OSError:
                pass

    cover_files = list_image_files(cover_dir)
    if not cover_files:
        raise FileNotFoundError(f"No images found in cover_dir='{cover_dir}'")

    for src in cover_files:
        img = cv2.imread(src, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[warn] normalize_covers: failed to read '{src}', skipping.")
            continue

        dst = os.path.join(out_dir, f"{Path(src).stem}.png")
        ok = cv2.imwrite(dst, img)
        if not ok:
            print(f"[warn] normalize_covers: failed to write '{dst}'")

    return out_dir


def _write_png_from_src(src_path: str, dst_path: str) -> None:
    """Read any supported image and write a clean PNG to dst_path."""
    img = cv2.imread(src_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {src_path}")
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    ok = cv2.imwrite(dst_path, img)
    if not ok:
        raise IOError(f"Failed to write PNG: {dst_path}")


def generate_stego_datasets(
    cover_dir: str,
    output_root: str,
    algorithms: List[str],
    bpp: float,
    seed: int = 1337,
    force: bool = False,
) -> Dict[str, str]:
    """
    Generate stego images for each algorithm into:
      output_root/stego_{alg}/

    Returns:
      dict: {alg: stego_dir}
    """
    os.makedirs(output_root, exist_ok=True)
    rng = random.Random(seed)

    cover_files = list_image_files(cover_dir)
    if not cover_files:
        raise FileNotFoundError(f"No cover images in '{cover_dir}'")

    stego_dirs: Dict[str, str] = {}
    for alg in algorithms:
        out_dir = os.path.join(output_root, f"stego_{alg}")
        os.makedirs(out_dir, exist_ok=True)
        stego_dirs[alg] = out_dir

        # If already generated and not forcing, keep as-is.
        if (not force) and any(Path(out_dir).glob("*.png")):
            continue

        # Clear old outputs when forcing.
        if force:
            for p in Path(out_dir).glob("*"):
                if p.is_file():
                    try:
                        p.unlink()
                    except OSError:
                        pass

        for src in cover_files:
            dst = os.path.join(out_dir, f"{Path(src).stem}.png")
            sa.embed_image(
                cover_path=src,
                out_path=dst,
                algorithm=alg,
                bpp=bpp,
                rng=rng,
            )

    return stego_dirs


def _make_splits(items: List[str], train_ratio: float, val_ratio: float, seed: int) -> Dict[str, List[str]]:
    """
    Split list into train/val/test; returns dict with keys train/val/test.
    """
    if not (0.0 < train_ratio < 1.0) or not (0.0 <= val_ratio < 1.0):
        raise ValueError("train_ratio and val_ratio must be in (0,1)")

    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0")

    rng = random.Random(seed)
    items = list(items)
    rng.shuffle(items)

    n = len(items)
    n_train = int(round(train_ratio * n))
    n_val = int(round(val_ratio * n))
    # ensure at least 1 test if possible
    n_train = max(min(n_train, n - 1), 1) if n >= 2 else n_train
    n_val = max(min(n_val, n - n_train), 0)

    train = items[:n_train]
    val = items[n_train:n_train + n_val]
    test = items[n_train + n_val:]
    return {"train": train, "val": val, "test": test}


def prepare_classification_sets(
    cover_dir: str,
    stego_dirs: Dict[str, str],
    output_dir: str,
    train_ratio: float,
    val_ratio: float,
    seed: int = 1337,
    multiclass: bool = False,
    max_per_split: Optional[int] = None,
    force: bool = False,
) -> Optional[List[str]]:
    """
    Prepare datasets in folder format.

    For multiclass=False (binary):
      output_dir/{train,val,test}/{cover,stego}/

    For multiclass=True:
      output_dir/{train,val,test}/{cover,stego_lsb,stego_pvd,...}/

    Returns:
      - If multiclass=True: list of class names in order (cover + stego_{alg})
      - Else: None
    """
    os.makedirs(output_dir, exist_ok=True)

    # Reuse if exists and not forcing.
    if (not force) and any(Path(output_dir).glob("train")):
        if multiclass:
            return ["cover"] + [f"stego_{a}" for a in stego_dirs.keys()]
        return None

    # Wipe old output when forcing.
    if force:
        for p in Path(output_dir).glob("*"):
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)

    cover_files = list_image_files(cover_dir)
    if not cover_files:
        raise FileNotFoundError(f"No cover images in '{cover_dir}'")

    # Use stems to match regardless of extension.
    cover_by_stem = {_stem(p): p for p in cover_files}

    # For each algorithm, build mapping stem -> stego_path
    stego_by_alg_stem: Dict[str, Dict[str, str]] = {}
    for alg, sdir in stego_dirs.items():
        sf = list_image_files(sdir)
        stego_by_alg_stem[alg] = {_stem(p): p for p in sf}

    # Determine which stems are valid for all selected algorithms.
    valid_stems = set(cover_by_stem.keys())
    for alg in stego_dirs.keys():
        valid_stems &= set(stego_by_alg_stem[alg].keys())

    valid_stems = sorted(valid_stems)
    if not valid_stems:
        raise RuntimeError(
            "No matching cover/stego pairs found. "
            "Likely extension mismatch or stego generation failed."
        )

    # Split by stems.
    splits = _make_splits(valid_stems, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)

    # Optional cap per split
    if max_per_split is not None:
        rng = random.Random(seed + 999)
        for k in ["train", "val", "test"]:
            lst = splits[k]
            if len(lst) > max_per_split:
                rng.shuffle(lst)
                splits[k] = lst[:max_per_split]

    if not multiclass:
        # Binary: create cover + a single stego folder, where the stego is sampled from
        # the provided algorithms (randomly per-image) to make "stego" a mix.
        rng = random.Random(seed + 123)
        algs = list(stego_dirs.keys())

        for split_name, stems in splits.items():
            cover_out = Path(output_dir) / split_name / "cover"
            stego_out = Path(output_dir) / split_name / "stego"
            cover_out.mkdir(parents=True, exist_ok=True)
            stego_out.mkdir(parents=True, exist_ok=True)

            for st in stems:
                cover_src = cover_by_stem[st]
                cover_dst = cover_out / f"{st}.png"
                _write_png_from_src(cover_src, str(cover_dst))

                # pick algorithm for this sample
                alg = rng.choice(algs)
                stego_src = stego_by_alg_stem[alg][st]
                stego_dst = stego_out / f"{st}.png"
                _write_png_from_src(stego_src, str(stego_dst))

        return None

    # Multiclass: cover + each algorithm as its own folder.
    class_order = ["cover"] + [f"stego_{a}" for a in stego_dirs.keys()]

    for split_name, stems in splits.items():
        cover_out = Path(output_dir) / split_name / "cover"
        cover_out.mkdir(parents=True, exist_ok=True)

        stego_out_dirs = {alg: (Path(output_dir) / split_name / f"stego_{alg}") for alg in stego_dirs.keys()}
        for d in stego_out_dirs.values():
            d.mkdir(parents=True, exist_ok=True)

        for st in stems:
            cover_src = cover_by_stem[st]
            _write_png_from_src(cover_src, str(cover_out / f"{st}.png"))

            for alg in stego_dirs.keys():
                stego_src = stego_by_alg_stem[alg][st]
                _write_png_from_src(stego_src, str(stego_out_dirs[alg] / f"{st}.png"))

    return class_order
