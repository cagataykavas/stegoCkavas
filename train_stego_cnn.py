import os
import json
import time
import math
import argparse
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from sklearn.metrics import f1_score, confusion_matrix
except Exception:
    f1_score = None
    confusion_matrix = None


# -------------------------
# Utils
# -------------------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def resolve_split_root(dataset_dir: str) -> str:
    """
    Your pipeline should create:
      dataset_dir/train/<class>/*.png
      dataset_dir/val/<class>/*.png
      dataset_dir/test/<class>/*.png

    If user points one level too high, try to find the correct folder.
    """
    dataset_dir = os.path.abspath(dataset_dir)
    if os.path.isdir(os.path.join(dataset_dir, "train")) and os.path.isdir(os.path.join(dataset_dir, "val")):
        return dataset_dir

    # search up to depth 2
    candidates = []
    for root, dirs, _files in os.walk(dataset_dir):
        # limit depth
        rel = os.path.relpath(root, dataset_dir)
        if rel.count(os.sep) > 2:
            continue
        if "train" in dirs and "val" in dirs:
            candidates.append(root)

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        # pick the shortest path
        candidates.sort(key=lambda p: len(p))
        return candidates[0]

    raise RuntimeError(
        f"Could not find train/val folders under: {dataset_dir}\n"
        f"Expected: <dataset_dir>/train, <dataset_dir>/val, <dataset_dir>/test"
    )


def list_classes(split_root: str, split: str) -> List[str]:
    p = os.path.join(split_root, split)
    if not os.path.isdir(p):
        return []
    classes = [d for d in os.listdir(p) if os.path.isdir(os.path.join(p, d))]
    classes.sort()
    return classes


def pil_load_grayscale(path: str) -> np.ndarray:
    # Keep original pixels; do NOT resize here.
    im = Image.open(path).convert("L")
    arr = np.array(im, dtype=np.float32) / 255.0
    return arr


def random_crop(arr: np.ndarray, crop: int) -> np.ndarray:
    h, w = arr.shape
    if crop <= 0:
        return arr
    if h < crop or w < crop:
        # If somehow smaller, center-pad (rare in your 256x256 setup)
        pad_h = max(0, crop - h)
        pad_w = max(0, crop - w)
        arr = np.pad(arr, ((pad_h//2, pad_h - pad_h//2), (pad_w//2, pad_w - pad_w//2)), mode="reflect")
        h, w = arr.shape
    y = np.random.randint(0, h - crop + 1)
    x = np.random.randint(0, w - crop + 1)
    return arr[y:y+crop, x:x+crop]


# -------------------------
# Dataset
# -------------------------
class FolderStegoDataset(Dataset):
    def __init__(self, split_root: str, split: str, class_to_idx: Dict[str, int], patch: int = 256):
        self.split_root = split_root
        self.split = split
        self.class_to_idx = class_to_idx
        self.patch = patch

        self.samples: List[Tuple[str, int]] = []
        for cls, idx in class_to_idx.items():
            cls_dir = os.path.join(split_root, split, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fn in os.listdir(cls_dir):
                if fn.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                    self.samples.append((os.path.join(cls_dir, fn), idx))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found under: {os.path.join(split_root, split)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, y = self.samples[i]
        x = pil_load_grayscale(path)
        x = random_crop(x, self.patch)
        # shape -> (1,H,W)
        x = torch.from_numpy(x).unsqueeze(0)
        return x, torch.tensor(y, dtype=torch.long), path


# -------------------------
# Steganalysis CNN (HPF + TLU + small conv net)
# -------------------------
class Truncation(nn.Module):
    def __init__(self, t: float = 3.0):
        super().__init__()
        self.t = t

    def forward(self, x):
        return torch.clamp(x, -self.t, self.t)


def make_hpf_kernels() -> torch.Tensor:
    """
    Simple fixed high-pass filters (not full SRM-30, but enough to stop the 0.5/0.2 collapse).
    Shape: (out_ch, in_ch, kH, kW)
    """
    k1 = np.array([[0, 0, 0, 0, 0],
                   [0, -1, 2, -1, 0],
                   [0, 2, -4, 2, 0],
                   [0, -1, 2, -1, 0],
                   [0, 0, 0, 0, 0]], dtype=np.float32)

    k2 = np.array([[-1, 2, -2, 2, -1],
                   [2, -6, 8, -6, 2],
                   [-2, 8, -12, 8, -2],
                   [2, -6, 8, -6, 2],
                   [-1, 2, -2, 2, -1]], dtype=np.float32)

    k3 = np.array([[0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, -1, 2, -1, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0]], dtype=np.float32)

    k4 = np.array([[0, 0, -1, 0, 0],
                   [0, 0, 2, 0, 0],
                   [-1, 2, -4, 2, -1],
                   [0, 0, 2, 0, 0],
                   [0, 0, -1, 0, 0]], dtype=np.float32)

    k5 = np.array([[-1, 0, 2, 0, -1],
                   [0, 0, 0, 0, 0],
                   [2, 0, -4, 0, 2],
                   [0, 0, 0, 0, 0],
                   [-1, 0, 2, 0, -1]], dtype=np.float32)

    kernels = np.stack([k1, k2, k3, k4, k5], axis=0)  # (5,5,5)
    kernels = kernels[:, None, :, :]  # (5,1,5,5)
    return torch.from_numpy(kernels)


class StegoNetLite(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.hpf = nn.Conv2d(1, 5, kernel_size=5, padding=2, bias=False)
        with torch.no_grad():
            self.hpf.weight.copy_(make_hpf_kernels())
        for p in self.hpf.parameters():
            p.requires_grad = False

        self.tlu = Truncation(3.0)

        # small conv tower
        self.conv1 = nn.Sequential(
            nn.Conv2d(5, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        x = self.hpf(x)
        x = self.tlu(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.head(x)
        return x


# -------------------------
# Train/Eval
# -------------------------
@torch.no_grad()
def evaluate(model, loader, device, n_classes: int):
    model.eval()
    ys = []
    ps = []
    total = 0
    correct = 0
    loss_sum = 0.0
    ce = nn.CrossEntropyLoss()

    for x, y, _paths in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = ce(logits, y)
        loss_sum += float(loss.item()) * x.size(0)

        pred = torch.argmax(logits, dim=1)
        correct += int((pred == y).sum().item())
        total += x.size(0)

        ys.append(y.cpu().numpy())
        ps.append(pred.cpu().numpy())

    y_true = np.concatenate(ys) if ys else np.array([], dtype=np.int64)
    y_pred = np.concatenate(ps) if ps else np.array([], dtype=np.int64)

    acc = correct / max(total, 1)
    if f1_score is not None and len(y_true) > 0:
        f1m = float(f1_score(y_true, y_pred, average="macro"))
    else:
        f1m = float("nan")

    return {
        "loss": loss_sum / max(total, 1),
        "acc": acc,
        "f1_macro": f1m,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True, help="Folder containing train/val/test splits.")
    ap.add_argument("--task", choices=["binary", "multiclass"], default="binary")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patch", type=int, default=256, help="Random crop size. Use 256 for BOSS256; try 128 to speed up.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=0)  # Windows: keep 0 unless you know it's safe
    ap.add_argument("--out-dir", default=None, help="Where to save outputs. Default: <dataset_dir>/stegocnn_<task>")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    set_seed(args.seed)

    split_root = resolve_split_root(args.dataset_dir)
    train_classes = list_classes(split_root, "train")
    v"al"_classes = list_classes(split_root, "val")

    if train_classes != val_classes:
        print("[warn] train/val class folders differ. Using train classes as canonical.")

    classes = train_classes
    if args.task == "binary":
        # try to enforce common names
        # expected: cover and stego
        if len(classes) != 2:
            print(f"[warn] binary task but found {len(classes)} classes: {classes}")
    else:
        if len(classes) < 3:
            print(f"[warn] multiclass task but found {len(classes)} classes: {classes}")

    class_to_idx = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)

    out_dir = args.out_dir or os.path.join(split_root, f"stegocnn_{args.task}")
    os.makedirs(out_dir, exist_ok=True)

    print("Split root:", split_root)
    print("Classes:", classes)
    print("Out dir:", os.path.abspath(out_dir))

    ds_train = FolderStegoDataset(split_root, "train", class_to_idx, patch=args.patch)
    ds_val = FolderStegoDataset(split_root, "val", class_to_idx, patch=args.patch)

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=False)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=False)

    device = torch.device(args.device)
    model = StegoNetLite(n_classes=n_classes).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best = {"acc": -1.0, "f1_macro": -1.0}
    best_path = os.path.join(out_dir, "stegocnn_best.pt")

    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        seen = 0

        for x, y, _paths in dl_train:
            x = x.to(device)
            y = y.to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()

            loss_sum += float(loss.item()) * x.size(0)
            seen += x.size(0)

        train_loss = loss_sum / max(seen, 1)
        val = evaluate(model, dl_val, device, n_classes)

        print(f"Epoch {ep:02d}/{args.epochs} | train_loss={train_loss:.4f} | "
              f"val_loss={val['loss']:.4f} | val_acc={val['acc']:.4f} | val_f1m={val['f1_macro']:.4f} | "
              f"{(time.time()-t0):.1f}s")

        # choose best by macro-f1 then acc
        if (not math.isnan(val["f1_macro"]) and val["f1_macro"] > best["f1_macro"] + 1e-6) or \
           (math.isnan(val["f1_macro"]) and val["acc"] > best["acc"] + 1e-6):
            best = {"acc": val["acc"], "f1_macro": val["f1_macro"]}
            torch.save({
                "model_state": model.state_dict(),
                "classes": classes,
                "args": vars(args),
            }, best_path)

    # final eval on test if exists
    metrics = {
        "best_val": best,
        "classes": classes,
        "split_root": split_root,
        "out_dir": os.path.abspath(out_dir),
    }

    test_dir = os.path.join(split_root, "test")
    if os.path.isdir(test_dir):
        ds_test = FolderStegoDataset(split_root, "test", class_to_idx, patch=args.patch)
        dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=False)

        # load best
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        test = evaluate(model, dl_test, device, n_classes)
        metrics["test"] = {"loss": test["loss"], "acc": test["acc"], "f1_macro": test["f1_macro"]}

        # confusion matrix
        if confusion_matrix is not None and len(test["y_true"]) > 0:
            cm = confusion_matrix(test["y_true"], test["y_pred"], labels=list(range(n_classes)))
            np.save(os.path.join(out_dir, "confusion_matrix.npy"), cm)

        print("Saved best:", best_path)
        print("Test acc:", metrics["test"]["acc"], "Test macroF1:", metrics["test"]["f1_macro"])
    else:
        print("[warn] No test/ folder found; skipping test evaluation.")

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Saved metrics:", os.path.join(out_dir, "metrics.json"))


if __name__ == "__main__":
    main()
