#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local evaluation on a labeled folder-structured dataset.

Goal: faithfully mirror the submit/inference pipeline while adding labels
and threshold search. This file DOES NOT affect submit pipeline automatically.
"""
import argparse
import importlib
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except Exception as e:
    raise SystemExit("albumentations is required. pip install albumentations") from e


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

from infer_utils import build_model, smart_load_state_dict, get_ckpt_meta

# 将 config 延迟到解析完 CLI 后再导入（需要读取 CURRENT_STAGE/IMAGE_SIZE 环境变量）
config = None


class FolderBinaryDataset(Dataset):
    """Labeled dataset for local eval, reusing project's preprocessing & normalization.

    - Label mapping matches training: parent == 'disease' -> 1, else 0.
    - Ben Graham preprocessing is applied, then val/submit normalization.
    """

    def __init__(self, root: str, img_size: int):
        self.root = Path(root)
        self.img_size = img_size

        # Reuse project's val/test transforms for normalization only
        # (we apply Ben Graham preprocessing beforehand for consistency).
        from dataset import get_transforms
        _, val_tf, _ = get_transforms()
        self.val_tf = val_tf

        self.samples: List[Tuple[Path, int]] = []
        for fp in sorted(self.root.rglob("*")):
            if fp.is_file() and fp.suffix.lower() in IMG_EXTS:
                parent = fp.parent.name.lower()
                lab = 1 if parent == "disease" else 0
                self.samples.append((fp, lab))

        if len(self.samples) == 0:
            raise ValueError(
                f"No images found under {root}. Expected subfolders containing images."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        img = Image.open(path).convert("RGB")

        # Apply project Ben Graham preprocessing for consistency
        from dataset import ben_graham_preprocessing
        img = ben_graham_preprocessing(img, target_size=self.img_size)
        img_np = np.array(img)

        x = self.val_tf(image=img_np)["image"]
        return x, torch.tensor(y, dtype=torch.float32), str(path.name)


def create_model_for_eval(ckpt_path: str, model_name: str, img_size: int, device: str = None) -> nn.Module:
    global config
    dev = device or config.DEVICE
    m = build_model(model_name=model_name, img_size=img_size, device=dev)
    smart_load_state_dict(m, ckpt_path, strict=False)
    return m.to(dev).eval()


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


@torch.no_grad()
def infer_probs_ensemble(
    models: List[nn.Module],
    loader: DataLoader,
    device: str,
    tta_hflip: bool = True,
    use_amp: bool = True,
):
    for m in models:
        m.eval()
    all_probs = []
    all_labels = []
    all_names = []
    scaler = torch.cuda.amp.autocast if (use_amp and "cuda" in device) else torch.cpu.amp.autocast

    for x, y, name in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with scaler():
            views = [x]
            if tta_hflip:
                views.append(torch.flip(x, dims=[3]))

            prob_sum = torch.zeros(x.size(0), device=device, dtype=torch.float32)
            denom = float(len(models) * len(views))
            for v in views:
                for m in models:
                    logits = m(v).view(-1)
                    prob_sum += torch.sigmoid(logits)
            prob = prob_sum / max(1.0, denom)

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
    ap.add_argument("--stage", type=int, default=int(os.environ.get("CURRENT_STAGE", "2")), help="Stage id to align config/env (1 or 2)")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tta", action="store_true", help="Enable 2x TTA (hflip)")
    ap.add_argument("--no_amp", action="store_true", help="Disable AMP")
    ap.add_argument("--ckpts", type=str, nargs="+", required=True, help="One or more checkpoint paths")
    ap.add_argument("--out_json", type=str, default="", help="Optional: save summary json")
    ap.add_argument("--out_csv", type=str, default="", help="Optional: save per-image probs csv for the BEST checkpoint")
    args = ap.parse_args()

    # 先同步环境变量，再导入 config（其初始化依赖 env）
    os.environ["CURRENT_STAGE"] = str(args.stage)
    os.environ["IMAGE_SIZE"] = str(args.img_size)

    global config
    import config as cfg
    importlib.reload(cfg)
    config = cfg

    # 读取首个 ckpt 的 meta，用于自动对齐评测配置，避免模型/尺寸不匹配
    auto_model = args.model_name
    auto_size = args.img_size
    if args.ckpts and len(args.ckpts) > 0:
        meta = get_ckpt_meta(args.ckpts[0])
        m_model = meta.get("model_name")
        m_size = meta.get("image_size")
        if m_model and m_model != auto_model:
            print(f"[meta] override model_name: {auto_model} -> {m_model} (from ckpt)")
            auto_model = m_model
        if isinstance(m_size, int) and m_size != auto_size:
            print(f"[meta] override image_size: {auto_size} -> {m_size} (from ckpt)")
            auto_size = int(m_size)

    # 将提交使用的关键配置对齐到本地评测：模型名/输入尺寸/设备（优先使用 ckpt 元数据）
    config.MODEL_NAME = auto_model
    config.IMAGE_SIZE = auto_size
    config.DEVICE = args.device
    config.CURRENT_STAGE = int(os.environ.get("CURRENT_STAGE", config.CURRENT_STAGE))

    # 记录当前评测使用的阶段/分辨率/模型，避免误用 Stage1 配置评测 Stage2 权重
    print(f"[config] CURRENT_STAGE={config.CURRENT_STAGE} IMAGE_SIZE={config.IMAGE_SIZE} MODEL_NAME={config.MODEL_NAME} DEVICE={config.DEVICE}")

    # 评测的数据预处理应与模型分辨率一致（防止 Swin attn mask 尺寸不匹配）
    args.img_size = auto_size

    ds = FolderBinaryDataset(args.data_dir, args.img_size)
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=("cuda" in args.device)
    )

    results = []

    # 单模型逐个评测
    for ckpt in args.ckpts:
        ckpt = str(ckpt)
        print(f"\n=== Evaluating: {ckpt}")
        model = create_model_for_eval(
            ckpt, model_name=auto_model, img_size=auto_size, device=args.device
        )

        probs, labels, names = infer_probs(
            model, loader, args.device,
            tta_hflip=args.tta,
            use_amp=(not args.no_amp),
        )
        pos = labels == 1
        neg = labels == 0
        pos_mean = float(probs[pos].mean()) if pos.any() else float("nan")
        neg_mean = float(probs[neg].mean()) if neg.any() else float("nan")
        print(f"[data] pos={int(pos.sum())} neg={int(neg.sum())} pos_prob_mean={pos_mean:.4f} neg_prob_mean={neg_mean:.4f}")

        best = best_threshold_by_accuracy(probs, labels, steps=2001)
        f1 = f1_from_counts(best["tp"], best["fp"], best["fn"])
        print(f"[best] thr={best['thr']:.4f} acc={best['acc']:.4f} f1={f1:.4f} "
              f"CM=[[tn fp][fn tp]]=[[{best['tn']} {best['fp']}][{best['fn']} {best['tp']}]]")
                # ====== DEBUG: 固定阈值下的指标（用来对比 best_thr 是否“虚高/不稳定”）======
        for t in [0.5, 0.2, 0.9]:
            pred_t = (probs >= t).astype(np.int32)
            y = labels.astype(np.int32)
            tp = int(((pred_t == 1) & (y == 1)).sum())
            tn = int(((pred_t == 0) & (y == 0)).sum())
            fp = int(((pred_t == 1) & (y == 0)).sum())
            fn = int(((pred_t == 0) & (y == 1)).sum())
            acc_t = (tp + tn) / max(1, (tp + tn + fp + fn))
            f1_t = f1_from_counts(tp, fp, fn)
            print(f"[fixed] thr={t:.3f} acc={acc_t:.4f} f1={f1_t:.4f} "
                  f"CM=[[tn fp][fn tp]]=[[{tn} {fp}][{fn} {tp}]]")


        results.append({
            "ckpt": ckpt,
            # 用 auto_*（可能被 ckpt meta 覆盖）避免记录错误参数
            "model_name": auto_model,
            "img_size": auto_size,
            "tta": bool(args.tta),
            "best_thr": best["thr"],
            "acc": best["acc"],
            "f1": f1,
            "tn": best["tn"], "fp": best["fp"], "fn": best["fn"], "tp": best["tp"],
            "n": int(labels.shape[0]),
        })

    # 多模型融合评测（若选择了多个 ckpt）
    if len(args.ckpts) > 1:
        print(f"\n=== Evaluating Ensemble ({len(args.ckpts)} ckpts)")

        # 修复：补齐参数
        ens_models = [
            create_model_for_eval(p, model_name=auto_model, img_size=auto_size, device=args.device)
            for p in args.ckpts
        ]

        probs, labels, names = infer_probs_ensemble(
            ens_models, loader, args.device, tta_hflip=args.tta, use_amp=(not args.no_amp)
        )

        pos = labels == 1
        neg = labels == 0
        pos_mean = float(probs[pos].mean()) if pos.any() else float("nan")
        neg_mean = float(probs[neg].mean()) if neg.any() else float("nan")
        print(f"[data] pos={int(pos.sum())} neg={int(neg.sum())} pos_prob_mean={pos_mean:.4f} neg_prob_mean={neg_mean:.4f}")

        best = best_threshold_by_accuracy(probs, labels, steps=2001)
        f1 = f1_from_counts(best["tp"], best["fp"], best["fn"])
        print(f"[best][Ensemble] thr={best['thr']:.4f} acc={best['acc']:.4f} f1={f1:.4f} "
              f"CM=[[tn fp][fn tp]]=[[{best['tn']} {best['fp']}][{best['fn']} {best['tp']}]]")

        results.append({
            "ckpt": f"ensemble({len(args.ckpts)})",
            "model_name": auto_model,
            "img_size": auto_size,
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

    # Default JSON output if single ckpt and not provided
    out_json = args.out_json
    if not out_json and len(args.ckpts) == 1:
        out_json = str(Path(config.OUTPUT_DIR) / f"local_eval_{Path(args.ckpts[0]).stem}.json")
    if out_json:
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            import json
            json.dump({"results": results}, f, indent=2, ensure_ascii=False)
        print(f"\nSaved: {out_json}")

    # Optionally dump per-image probabilities for the BEST *single* checkpoint
    if args.out_csv and len(results) > 0:
        # 如果 top1 是 ensemble(x)，就回退到最好的单模型 ckpt
        best_ckpt = None
        for r in results:
            p = r.get("ckpt", "")
            if isinstance(p, str) and os.path.isfile(p):
                best_ckpt = p
                break
        if best_ckpt is None:
            raise RuntimeError("No valid checkpoint file found to dump CSV.")

        model = create_model_for_eval(
            best_ckpt, model_name=auto_model, img_size=auto_size, device=args.device
        )
        probs, labels, names = infer_probs(
            model, loader, args.device, tta_hflip=args.tta, use_amp=(not args.no_amp)
        )

        # thr 用对应 ckpt 的 best_thr（不要拿 ensemble 的阈值）
        thr = None
        for r in results:
            if r.get("ckpt") == best_ckpt:
                thr = r.get("best_thr")
                break
        if thr is None:
            thr = 0.5

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
