#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local evaluation on a *labeled* folder-structured dataset.

Expected layout (your screenshot):
  <DATA_DIR>/
    disease/   (positive class, label=1)
      xxx.jpg/png/...
    normal/    (negative class, label=0)
      yyy.jpg/png/...

It loads one or multiple checkpoints and reports:
- best threshold (by accuracy, like your training logs "val_acc@bestT")
- accuracy / F1 / confusion matrix at best threshold
- optional per-image probability dump

NOTE: Using this set to pick models is "label peeking" for Kaggle final evaluation.
Use it only to reduce trial-and-error due to submission limits.
"""
import argparse
import os
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

try:
    import timm
except Exception as e:
    raise SystemExit("timm is required. pip install timm") from e

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except Exception as e:
    raise SystemExit("albumentations is required. pip install albumentations") from e


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def build_eval_transform(img_size: int):
    # Keep it conservative: resize -> center crop -> normalize -> tensor
    # You can adjust to match your submit/inference pipeline exactly.
    return A.Compose([
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=0, value=0),
        A.CenterCrop(height=img_size, width=img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


class FolderBinaryDataset(Dataset):
    def __init__(self, root: str, img_size: int):
        self.root = Path(root)
        self.transform = build_eval_transform(img_size)
        self.samples: List[Tuple[Path, int]] = []

        # Positive = disease, Negative = normal (per your naming)
        for sub, lab in [("normal", 0), ("disease", 1)]:
            p = self.root / sub
            if not p.exists():
                continue
            for fp in sorted(p.rglob("*")):
                if fp.is_file() and fp.suffix.lower() in IMG_EXTS:
                    self.samples.append((fp, lab))

        if len(self.samples) == 0:
            raise ValueError(
                f"No images found under {root}. Expected subfolders: normal/ disease/."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img_np = np.array(img)
        x = self.transform(image=img_np)["image"]  # (C,H,W) float tensor
        return x, torch.tensor(y, dtype=torch.float32), str(path.name)


def load_checkpoint_state(ckpt_path: str) -> Dict:
    obj = torch.load(ckpt_path, map_location="cpu")
    # Common patterns:
    # 1) {"state_dict": ..., ...}
    # 2) {"model": ...}
    # 3) raw state_dict
    if isinstance(obj, dict):
        for k in ["state_dict", "model", "model_state_dict", "net"]:
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
    if isinstance(obj, dict):
        return obj
    raise ValueError(f"Unrecognized checkpoint format: {ckpt_path}")


def create_model(model_name: str, num_classes: int = 1, img_size: int = 224, pretrained: bool = False):
    # For Swin models, timm uses fixed img_size unless you pass it at create_model time.
    # We pass img_size to avoid "Input height doesn't match model" assertions.
    kwargs = {}
    # many timm models accept img_size; safe to pass and ignore if unsupported
    kwargs["img_size"] = img_size

    m = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        **kwargs
    )
    return m


@torch.no_grad()
def infer_probs(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    tta_hflip: bool = True,
    use_amp: bool = True,
):
    model.eval()
    all_probs = []
    all_labels = []
    all_names = []
    scaler = torch.cuda.amp.autocast if (use_amp and "cuda" in device) else torch.cpu.amp.autocast

    for x, y, name in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with scaler():
            logits = model(x).view(-1)  # (B,)
            prob = torch.sigmoid(logits)

            if tta_hflip:
                x2 = torch.flip(x, dims=[3])
                logits2 = model(x2).view(-1)
                prob2 = torch.sigmoid(logits2)
                prob = (prob + prob2) / 2.0

        all_probs.append(prob.detach().float().cpu().numpy())
        all_labels.append(y.detach().float().cpu().numpy())
        all_names.extend(list(name))

    probs = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return probs, labels, all_names


def best_threshold_by_accuracy(probs: np.ndarray, labels: np.ndarray, steps: int = 2001):
    # steps=2001 -> resolution 0.0005
    ts = np.linspace(0.0, 1.0, steps)
    best = {"thr": 0.5, "acc": -1.0, "tp": 0, "tn": 0, "fp": 0, "fn": 0}
    for t in ts:
        pred = (probs >= t).astype(np.int32)
        y = labels.astype(np.int32)
        tp = int(((pred == 1) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        acc = (tp + tn) / max(1, (tp + tn + fp + fn))
        if acc > best["acc"]:
            best = {"thr": float(t), "acc": float(acc), "tp": tp, "tn": tn, "fp": fp, "fn": fn}
    return best


def f1_from_counts(tp, fp, fn):
    denom = (2 * tp + fp + fn)
    return (2 * tp) / denom if denom > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True, help="Path to labeled test folder (contains disease/ normal/)")
    ap.add_argument("--model_name", type=str, required=True, help="timm model name, e.g. swin_base_patch4_window7_224")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tta", action="store_true", help="Enable 2x TTA (hflip)")
    ap.add_argument("--no_amp", action="store_true", help="Disable AMP")
    ap.add_argument("--ckpts", type=str, nargs="+", required=True, help="One or more checkpoint paths")
    ap.add_argument("--out_json", type=str, default="", help="Optional: save summary json")
    ap.add_argument("--out_csv", type=str, default="", help="Optional: save per-image probs csv for the BEST checkpoint")
    args = ap.parse_args()

    ds = FolderBinaryDataset(args.data_dir, args.img_size)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=("cuda" in args.device)
    )

    results = []
    for ckpt in args.ckpts:
        ckpt = str(ckpt)
        print(f"\n=== Evaluating: {ckpt}")
        model = create_model(args.model_name, num_classes=1, img_size=args.img_size, pretrained=False)
        state = load_checkpoint_state(ckpt)

        # allow a bit of key mismatch (heads, etc.)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
        model.to(args.device)

        probs, labels, names = infer_probs(
            model, loader, args.device,
            tta_hflip=args.tta,
            use_amp=(not args.no_amp),
        )
        best = best_threshold_by_accuracy(probs, labels, steps=2001)
        f1 = f1_from_counts(best["tp"], best["fp"], best["fn"])
        print(f"[best] thr={best['thr']:.4f} acc={best['acc']:.4f} f1={f1:.4f} "
              f"CM=[[tn fp][fn tp]]=[[{best['tn']} {best['fp']}][{best['fn']} {best['tp']}]]")

        results.append({
            "ckpt": ckpt,
            "model_name": args.model_name,
            "img_size": args.img_size,
            "tta": bool(args.tta),
            "best_thr": best["thr"],
            "acc": best["acc"],
            "f1": f1,
            "tn": best["tn"], "fp": best["fp"], "fn": best["fn"], "tp": best["tp"],
            "n": int(labels.shape[0]),
        })

    # sort by acc desc then f1
    results = sorted(results, key=lambda d: (d["acc"], d["f1"]), reverse=True)
    print("\n===== Summary (sorted) =====")
    for r in results:
        print(f"{r['acc']:.4f} | f1={r['f1']:.4f} | thr={r['best_thr']:.4f} | {Path(r['ckpt']).name}")

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            import json
            json.dump({"results": results}, f, indent=2, ensure_ascii=False)
        print(f"\nSaved: {args.out_json}")

    # Optionally dump per-image probabilities for the BEST ckpt (top-1)
    if args.out_csv and len(results) > 0:
        best_ckpt = results[0]["ckpt"]
        model = create_model(args.model_name, num_classes=1, img_size=args.img_size, pretrained=False)
        state = load_checkpoint_state(best_ckpt)
        model.load_state_dict(state, strict=False)
        model.to(args.device)
        probs, labels, names = infer_probs(model, loader, args.device, tta_hflip=args.tta, use_amp=(not args.no_amp))
        thr = results[0]["best_thr"]
        pred = (probs >= thr).astype(np.int32)

        import csv
        Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["filename", "label", "prob", "pred"])
            for n, y, p, pr in zip(names, labels.astype(int), probs, pred):
                w.writerow([n, int(y), float(p), int(pr)])
        print(f"Saved: {args.out_csv}")


if __name__ == "__main__":
    main()
