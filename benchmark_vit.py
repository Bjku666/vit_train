"""benchmark_vit.py

在有标签数据集上评测权重（默认使用 config.LABELED_TEST_DIR）。
支持：
- 单划分：评测单个权重或多 epoch 扫描
- KFold：传入多个权重并对概率求均值
- TTA：原图 + 水平翻转（2x）
"""

import argparse
import glob
import json
import os
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import MedicalDataset, get_transforms
from model import get_model


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def search_best_threshold(probs: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    best_thr, best_acc = 0.5, -1.0
    for thr in np.linspace(0.2, 0.8, 121):
        pred = (probs >= thr).astype(np.int64)
        acc = (pred == labels).mean()
        if acc > best_acc:
            best_acc, best_thr = acc, float(thr)
    return best_thr, float(best_acc)


def confusion(labels: np.ndarray, probs: np.ndarray, thr: float):
    pred = (probs >= thr).astype(np.int64)
    tn = int(((pred == 0) & (labels == 0)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    tp = int(((pred == 1) & (labels == 1)).sum())
    return [[tn, fp], [fn, tp]]


def load_model_from_ckpt(ckpt_path: str) -> torch.nn.Module:
    # 按当前配置构建模型（确保环境中 MODEL_NAME/IMAGE_SIZE 一致）
    model = get_model().to(config.DEVICE)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt.get("model", ckpt))
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


@torch.no_grad()
def predict_probs(models: List[torch.nn.Module], loader: DataLoader, tta: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    all_probs = []
    all_labels = []
    for images, labels in tqdm(loader, desc="Infer", leave=False):
        images = images.to(config.DEVICE, non_blocking=True)
        labels_np = labels.numpy().astype(np.int64)

        prob_sum = torch.zeros(images.size(0), device=config.DEVICE, dtype=torch.float32)

        views = [images]
        if tta:
            views.append(torch.flip(images, dims=[3]))  # hflip

        denom = len(models) * len(views)

        for v in views:
            for m in models:
                logits = m(v).float()
                prob_sum += torch.sigmoid(logits)

        probs = (prob_sum / denom).detach().cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels_np)

    return np.concatenate(all_probs), np.concatenate(all_labels)


def resolve_ckpts(ckpt_input: str, pattern: str, select: str, avg_last_k: int) -> List[str]:
    # 若输入是文件，直接使用
    if os.path.isfile(ckpt_input):
        return [ckpt_input]

    # 若输入是目录，则在目录内 glob
    if os.path.isdir(ckpt_input):
        ckpts = sorted(glob.glob(os.path.join(ckpt_input, pattern)))
    else:
        # 直接按 glob 模式处理
        ckpts = sorted(glob.glob(ckpt_input))

    if len(ckpts) == 0:
        raise FileNotFoundError(f"No checkpoints found for: {ckpt_input} (pattern={pattern})")

    # 选择策略
    s = (select or "").lower().strip()
    if s.startswith("epoch:"):
        e = int(s.split("epoch:")[1])
        hit = [p for p in ckpts if f"epoch{e:03d}" in os.path.basename(p)]
        if not hit:
            raise FileNotFoundError(f"epoch:{e} not found in ckpts")
        return hit

    if s in ["best", "best.pth"]:
        hit = [p for p in ckpts if p.endswith("_best.pth") or "best.pth" in os.path.basename(p)]
        if hit:
            return hit[:1]
        # 回退：使用最后一个
        return [ckpts[-1]]

    if s == "last":
        return [ckpts[-1]]

    if avg_last_k > 1:
        return ckpts[-avg_last_k:]

    # 默认：全部使用（集成）
    return ckpts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="", help="Checkpoint path/dir/glob. Default: current run dir.")
    parser.add_argument("--pattern", type=str, default="*_stage2_*.pth", help="Glob pattern within a dir.")
    parser.add_argument("--select", type=str, default="best", help="best | last | epoch:NNN")
    parser.add_argument("--avg_last_k", type=int, default=0, help="If >1, average last K epoch checkpoints (same arch).")
    parser.add_argument("--data_dir", type=str, default="", help="Labeled eval dir. Default: config.LABELED_TEST_DIR")
    parser.add_argument("--batch_size", type=int, default=0, help="Override batch size for benchmark.")
    parser.add_argument("--no_tta", action="store_true")
    args = parser.parse_args()

    eval_dir = args.data_dir or config.LABELED_TEST_DIR
    _, val_tf, _ = get_transforms()
    ds = MedicalDataset(root_dir=eval_dir, mode="val", transform=val_tf)
    bs = args.batch_size or max(1, config.BATCH_SIZE)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)

    ckpt_input = args.ckpt or config.CURRENT_RUN_MODELS_DIR
    ckpt_paths = resolve_ckpts(ckpt_input, args.pattern, args.select, args.avg_last_k)

    models = [load_model_from_ckpt(p) for p in ckpt_paths]
    probs, labels = predict_probs(models, dl, tta=(not args.no_tta))

    best_thr, best_acc = search_best_threshold(probs, labels)
    cm = confusion(labels, probs, best_thr)

    result = {
        "ckpts": ckpt_paths,
        "best_threshold": best_thr,
        "accuracy": best_acc,
        "confusion_matrix": cm,
        "tta": not args.no_tta,
        "avg_last_k": args.avg_last_k,
        "select": args.select,
        "pattern": args.pattern,
        "model_name": config.MODEL_NAME,
        "image_size": config.IMAGE_SIZE,
        "stage": config.CURRENT_STAGE,
    }

    print("===== Benchmark =====")
    print(f"ckpts: {len(ckpt_paths)}")
    print(f"best_thr: {best_thr:.3f}")
    print(f"acc: {best_acc:.4f}")
    print(f"cm: {cm}")

    # save json
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(config.OUTPUT_DIR, f"benchmark_result_{config.RUN_ID}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
