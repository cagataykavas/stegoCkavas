#!/usr/bin/env python3
"""Generate input-gradient saliency overlays for ``StegoNetLite`` checkpoints."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch

from stego_ai.train_stego_cnn import StegoNetLite


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def list_samples(dataset_dir: Path, split: str, classes: List[str]) -> List[Tuple[Path, int]]:
    samples: List[Tuple[Path, int]] = []
    for class_index, class_name in enumerate(classes):
        folder = dataset_dir / split / class_name
        if not folder.is_dir():
            continue
        samples.extend(
            (path, class_index)
            for path in sorted(folder.iterdir())
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    return samples


def load_tensor(path: Path, size: int, device: str) -> Tuple[np.ndarray, torch.Tensor]:
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"Could not read {path}")
    resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized.astype(np.float32) / 255.0)[None, None].to(device)
    return gray, tensor


def explain(model: StegoNetLite, tensor: torch.Tensor) -> Tuple[int, float, np.ndarray]:
    value = tensor.clone().detach().requires_grad_(True)
    logits = model(value)
    probabilities = torch.softmax(logits, dim=1)
    confidence, prediction = probabilities.max(dim=1)
    model.zero_grad(set_to_none=True)
    logits[0, prediction.item()].backward()
    saliency = np.abs(value.grad.detach().cpu().numpy()[0, 0])
    saliency -= saliency.min()
    saliency /= saliency.max() + 1e-8
    return int(prediction.item()), float(confidence.item()), saliency


def overlay(gray: np.ndarray, saliency: np.ndarray) -> np.ndarray:
    resized = cv2.resize(saliency, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.applyColorMap(np.uint8(np.clip(resized, 0.0, 1.0) * 255), cv2.COLORMAP_JET)
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(base, 0.58, heatmap, 0.42, 0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-per-class", type=int, default=12)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    classes = list(checkpoint["classes"])
    patch_size = int(checkpoint.get("args", {}).get("patch", 256))
    model = StegoNetLite(n_classes=len(classes)).to(args.device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    samples = list_samples(args.dataset_dir, args.split, classes)
    if not samples:
        raise FileNotFoundError(f"No images found under {args.dataset_dir / args.split}")

    selected: List[Tuple[Path, int]] = []
    for class_index in range(len(classes)):
        candidates = [sample for sample in samples if sample[1] == class_index]
        random.shuffle(candidates)
        selected.extend(candidates[: args.max_per_class])

    records: List[Dict[str, object]] = []
    for index, (path, true_index) in enumerate(selected):
        gray, tensor = load_tensor(path, patch_size, args.device)
        predicted_index, confidence, saliency = explain(model, tensor)
        output_name = (
            f"{index:03d}_true-{classes[true_index]}_pred-{classes[predicted_index]}_{path.stem}.png"
        )
        cv2.imwrite(str(args.out_dir / output_name), overlay(gray, saliency))
        records.append(
            {
                "source": str(path),
                "true": classes[true_index],
                "predicted": classes[predicted_index],
                "confidence": confidence,
                "correct": predicted_index == true_index,
                "overlay": output_name,
            }
        )

    manifest = {
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset_dir),
        "split": args.split,
        "classes": classes,
        "records": records,
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Saved {len(records)} saliency overlays to {args.out_dir}")


if __name__ == "__main__":
    main()

