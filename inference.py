"""inference.py

为无标签测试集生成提交 CSV。
- 支持按 epoch/best/last 选择权重
- 支持对最后 K 个 epoch 做均值（简易 SWA 集成）
- 使用 sigmoid（二分类 logit）与阈值输出
"""

import argparse
import glob
import json
import os
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import MedicalDataset, get_transforms
from model import get_model
from collections import OrderedDict


def _filter_state_dict(model: torch.nn.Module, state: dict) -> OrderedDict:
    """仅保留与当前模型形状一致的权重，避免尺寸不匹配报错（如 Swin 位置偏置表）。"""
    msd = model.state_dict()
    out = OrderedDict()
    for k, v in state.items():
        if k in msd and msd[k].shape == v.shape:
            out[k] = v
    return out


def resolve_ckpts(ckpt_input: str, pattern: str, select: str, avg_last_k: int) -> List[str]:
    if os.path.isfile(ckpt_input):
        return [ckpt_input]
    if os.path.isdir(ckpt_input):
        ckpts = sorted(glob.glob(os.path.join(ckpt_input, pattern)))
    else:
        ckpts = sorted(glob.glob(ckpt_input))
    if len(ckpts) == 0:
        raise FileNotFoundError(f"No checkpoints found for: {ckpt_input} (pattern={pattern})")

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
        return [ckpts[-1]]

    if s == "last":
        return [ckpts[-1]]

    if avg_last_k > 1:
        return ckpts[-avg_last_k:]

    return ckpts


def load_model_from_ckpt(ckpt_path: str) -> torch.nn.Module:
    model = get_model().to(config.DEVICE)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("state_dict", ckpt.get("model", ckpt))
    sd = _filter_state_dict(model, sd)
    msg = model.load_state_dict(sd, strict=False)
    missing = getattr(msg, "missing_keys", [])
    unexpected = getattr(msg, "unexpected_keys", [])
    if missing or unexpected:
        print(f"[Load] filtered state_dict: missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()
    return model


@torch.no_grad()
def predict_probs(models: List[torch.nn.Module], loader: DataLoader, tta: bool = True):
    filenames = []
    probs_out = []
    for batch in tqdm(loader, desc="推理", leave=False):
        images, names = batch
        images = images.to(config.DEVICE, non_blocking=True)

        prob_sum = torch.zeros(images.size(0), device=config.DEVICE, dtype=torch.float32)
        views = [images]
        if tta:
            views.append(torch.flip(images, dims=[3]))  # 水平翻转
        denom = len(models) * len(views)

        for v in views:
            for m in models:
                logits = m(v).float()
                prob_sum += torch.sigmoid(logits)

        probs = (prob_sum / denom).detach().cpu().numpy()
        probs_out.append(probs)
        filenames.extend(list(names))
    return filenames, np.concatenate(probs_out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="", help="权重路径/目录/通配（默认：当前运行目录）")
    parser.add_argument("--pattern", type=str, default="*_stage2_*.pth", help="在目录内匹配的通配模式")
    parser.add_argument("--select", type=str, default="best", help="best | last | epoch:NNN")
    parser.add_argument("--avg_last_k", type=int, default=0, help="对最后 K 个 epoch 权重做均值")
    parser.add_argument("--threshold", type=float, default=None, help="手动覆盖阈值")
    parser.add_argument("--threshold_json", type=str, default="", help="包含 best_threshold 的 JSON 文件")
    parser.add_argument("--no_tta", action="store_true", help="关闭 TTA")
    parser.add_argument("--output_csv", type=str, default="", help="输出 CSV 路径")
    args = parser.parse_args()

    thr = 0.5
    if args.threshold is not None:
        thr = float(args.threshold)
    elif args.threshold_json:
        with open(args.threshold_json, "r", encoding="utf-8") as f:
            j = json.load(f)
        thr = float(j.get("best_threshold", j.get("threshold", 0.5)))

    ckpt_input = args.ckpt or config.CURRENT_RUN_MODELS_DIR
    ckpts = resolve_ckpts(ckpt_input, args.pattern, args.select, args.avg_last_k)
    models = [load_model_from_ckpt(p) for p in ckpts]

    _, _, test_tf = get_transforms()
    ds = MedicalDataset(root_dir=config.UNLABELED_TEST_DIR, mode="test", transform=test_tf)
    dl = DataLoader(ds, batch_size=max(1, config.BATCH_SIZE), shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)

    names, probs = predict_probs(models, dl, tta=(not args.no_tta))
    preds = (probs >= thr).astype(np.int64)

    # 竞赛要求首列为 id，这里直接使用文件名（含扩展名）。
    df = pd.DataFrame({"id": names, "label": preds})
    out_csv = args.output_csv or os.path.join(config.OUTPUT_DIR, f"submission_{config.RUN_ID}.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"已保存: {out_csv}")
    print(f"权重数: {len(ckpts)} 阈值={thr:.3f} TTA={not args.no_tta}")


if __name__ == "__main__":
    main()
