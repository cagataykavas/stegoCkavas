#!/usr/bin/env python3
"""
visualize_residual_map.py
-------------------------
Generates residual/noise heatmap overlays to include in your report.

This is NOT a model explanation (like Grad-CAM). It's a classic steganalysis visualization:
we remove image content with a high-pass filter and visualize the residual energy.

Why it helps in reports:
- shows "where the high-frequency noise lives"
- makes it easier to discuss why resizing/blur hurts detection
- gives qualitative comparisons across BPP values

Outputs:
  output_dir/
    <name>_orig.png
    <name>_residual.png
    <name>_overlay.png
"""
import argparse
import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pgm"}


def collect_images(input_dir: Path, max_images: int) -> List[Path]:
    paths: List[Path] = []
    for p in input_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            paths.append(p)
            if len(paths) >= max_images:
                break
    return paths


def residual_map(gray: np.ndarray) -> np.ndarray:
    """
    Simple high-pass residual magnitude:
    - apply 3x3 Laplacian-like kernel (steganalyzer friendly)
    - take absolute value and normalize
    """
    # A common high-pass kernel family used in steganalysis residuals
    k = np.array([[0, -1, 0],
                  [-1, 4, -1],
                  [0, -1, 0]], dtype=np.float32)
    r = cv2.filter2D(gray.astype(np.float32), -1, k)
    r = np.abs(r)
    # robust normalize using percentile
    hi = np.percentile(r, 99.5)
    r = np.clip(r / (hi + 1e-6), 0, 1)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-images", type=int, default=40)
    args = ap.parse_args()

    inp = Path(args.input_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    imgs = collect_images(inp, args.max_images)
    if not imgs:
        raise FileNotFoundError(f"No images found under {inp}")

    for p in imgs:
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        r = residual_map(gray)  # 0..1
        r_u8 = (r * 255).astype(np.uint8)
        r_color = cv2.applyColorMap(r_u8, cv2.COLORMAP_JET)  # OpenCV default

        overlay = cv2.addWeighted(bgr, 0.60, r_color, 0.40, 0)

        stem = p.stem
        # write
        cv2.imwrite(str(out / f"{stem}_orig.png"), bgr)
        cv2.imwrite(str(out / f"{stem}_residual.png"), r_color)
        cv2.imwrite(str(out / f"{stem}_overlay.png"), overlay)

    print(f"Saved {len(imgs)} residual visualizations to {out}")


if __name__ == "__main__":
    main()
