#!/usr/bin/env python3
"""
Generate saliency/Grad-CAM style heatmaps for a trained CNN (scripts/train_cnn.py output).

This is the "teachers love it" part:
- Show true/pred, confidence
- Overlay a heatmap indicating which pixels drove the decision

We implement *input-gradient saliency* (robust + minimal deps) and an optional Grad-CAM.

Usage:
  python scripts/heatmaps_cnn.py --dataset-dir ..\\runs\\final_bpp0p4\\classification_binary --checkpoint ..\\runs\\final_bpp0p4\\classification_binary\\cnn_binary\\cnn_best.pt --out-dir ..\\runs\\final_bpp0p4\\cnn_heatmaps --max-per-class 12
"""
import argparse, os, json, random
from typing import List, Tuple, Dict

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix

# Keep this identical to train_cnn.py
class SmallStegCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = torch.flatten(x, 1)
        x = self.drop(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

def _list_images(folder: str) -> List[str]:
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")
    if not os.path.isdir(folder):
        return []
    out = []
    for n in os.listdir(folder):
        if n.lower().endswith(exts):
            out.append(os.path.join(folder, n))
    out.sort()
    return out

def _load_gray(path: str, img_size: int, make_residual: bool) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to read: {path}")
    if img.shape[0] != img_size or img.shape[1] != img_size:
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    if make_residual:
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        img = img - blur
    return img  # HxW float32

def _overlay_heatmap(gray: np.ndarray, heat: np.ndarray) -> np.ndarray:
    # gray: HxW in [0,1] or residual range; heat: HxW in [0,1]
    base = gray.copy()
    base = base - base.min()
    base = base / (base.max() + 1e-8)
    base_u8 = (base * 255).astype(np.uint8)
    base_rgb = cv2.cvtColor(base_u8, cv2.COLOR_GRAY2BGR)

    heat_u8 = (np.clip(heat, 0, 1) * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    out = cv2.addWeighted(base_rgb, 0.55, heat_color, 0.45, 0)
    return out

@torch.no_grad()
def _predict(model: nn.Module, x: torch.Tensor) -> Tuple[int, float, np.ndarray]:
    logits = model(x)
    prob = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred = int(prob.argmax())
    conf = float(prob[pred])
    return pred, conf, prob

def _saliency(model: nn.Module, x: torch.Tensor, target_class: int) -> np.ndarray:
    # input-gradient saliency
    model.eval()
    x = x.clone().detach().requires_grad_(True)
    logits = model(x)
    score = logits[0, target_class]
    model.zero_grad(set_to_none=True)
    score.backward()
    grad = x.grad.detach().cpu().numpy()[0, 0]  # HxW
    sal = np.abs(grad)
    sal = sal - sal.min()
    sal = sal / (sal.max() + 1e-8)
    return sal

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-per-class", type=int, default=12)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    class_names = ckpt["class_names"]
    cfg = ckpt.get("config", {})
    img_size = int(cfg.get("img_size", 256))
    make_residual = bool(cfg.get("make_residual", True))

    model = SmallStegCNN(num_classes=len(class_names)).to(args.device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # gather samples
    samples: List[Tuple[str, int]] = []
    for ci, cname in enumerate(class_names):
        folder = os.path.join(args.dataset_dir, args.split, cname)
        for p in _list_images(folder):
            samples.append((p, ci))
    if len(samples) == 0:
        raise RuntimeError(f"No samples found under {os.path.join(args.dataset_dir, args.split)}")

    # run predictions
    results = []
    for path, true in samples:
        gray = _load_gray(path, img_size, make_residual)
        x = torch.from_numpy(gray[None, None]).float().to(args.device)
        pred, conf, prob = _predict(model, x)
        results.append({"path": path, "true": true, "pred": pred, "conf": conf})

    # pick top correct and top errors per class (by confidence)
    by_class_correct = {i: [] for i in range(len(class_names))}
    by_class_error = {i: [] for i in range(len(class_names))}
    for r in results:
        if r["pred"] == r["true"]:
            by_class_correct[r["true"]].append(r)
        else:
            by_class_error[r["true"]].append(r)

    for i in range(len(class_names)):
        by_class_correct[i].sort(key=lambda d: -d["conf"])
        by_class_error[i].sort(key=lambda d: -d["conf"])

    def _save_group(group: List[dict], tag: str, true_class: int):
        take = group[:args.max_per_class]
        for k, r in enumerate(take):
            pth = r["path"]
            gray = _load_gray(pth, img_size, make_residual)
            x = torch.from_numpy(gray[None, None]).float().to(args.device)

            # use predicted class as "why it chose that"
            sal = _saliency(model, x, target_class=r["pred"])
            overlay = _overlay_heatmap(gray, sal)

            out_name = f"{tag}_true-{class_names[true_class]}_pred-{class_names[r['pred']]}_{k:02d}.png"
            out_path = os.path.join(args.out_dir, out_name)
            cv2.imwrite(out_path, overlay)

    for ci in range(len(class_names)):
        _save_group(by_class_correct[ci], "correct", ci)
        _save_group(by_class_error[ci], "error", ci)

    # save a small index JSON
    index = {
        "dataset_dir": os.path.abspath(args.dataset_dir),
        "split": args.split,
        "checkpoint": os.path.abspath(args.checkpoint),
        "out_dir": os.path.abspath(args.out_dir),
        "class_names": class_names,
        "img_size": img_size,
        "make_residual": make_residual,
    }
    with open(os.path.join(args.out_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    print("Saved heatmaps to:", args.out_dir)

if __name__ == "__main__":
    main()
