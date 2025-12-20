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
from infer_utils import get_ckpt_meta


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


def load_model_from_ckpt(ckpt_path: str, model_name: str, img_size: int) -> torch.nn.Module:
    model = get_model(model_name=model_name, img_size=img_size).to(config.DEVICE)
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
    parser.add_argument("--ckpts", type=str, nargs="+", default=None,
                        help="显式指定多个 ckpt 路径（用于集成）。提供该参数时将忽略 --ckpt/--pattern/--select")
    parser.add_argument("--pattern", type=str, default="*_stage2_*.pth", help="在目录内匹配的通配模式")
    parser.add_argument("--select", type=str, default="best", help="best | last | epoch:NNN")
    parser.add_argument("--avg_last_k", type=int, default=0, help="对最后 K 个 epoch 权重做均值")
    parser.add_argument("--threshold", type=float, default=None, help="手动覆盖阈值")
    parser.add_argument("--threshold_json", type=str, default="", help="包含 best_threshold 的 JSON 文件")
    parser.add_argument("--no_tta", action="store_true", help="关闭 TTA")
    parser.add_argument("--output_csv", type=str, default="", help="输出 CSV 路径")
    parser.add_argument("--model_name", type=str, default=config.MODEL_NAME, help="timm 模型名（可覆盖 ckpt/meta）")
    parser.add_argument("--img_size", type=int, default=config.IMAGE_SIZE, help="输入分辨率（可覆盖 ckpt/meta）")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"), help="运行设备")
    args = parser.parse_args()

    if args.ckpts is not None and len(args.ckpts) > 0:
        ckpts = [p for p in args.ckpts]
    else:
        ckpt_input = args.ckpt or config.CURRENT_RUN_MODELS_DIR
        ckpts = resolve_ckpts(ckpt_input, args.pattern, args.select, args.avg_last_k)

    # 对齐模型名/输入尺寸：CLI -> ckpt meta
    auto_model = args.model_name
    auto_size = args.img_size
    if ckpts:
        meta = get_ckpt_meta(ckpts[0])
        m_model = meta.get("model_name")
        m_size = meta.get("image_size")
        if m_model and m_model != auto_model:
            print(f"[meta] override model_name: {auto_model} -> {m_model} (from ckpt)")
            auto_model = m_model
        if isinstance(m_size, int) and m_size != auto_size:
            print(f"[meta] override image_size: {auto_size} -> {m_size} (from ckpt)")
            auto_size = int(m_size)

    config.MODEL_NAME = auto_model
    config.IMAGE_SIZE = auto_size
    config.DEVICE = args.device
    print(f"[config] CURRENT_STAGE={config.CURRENT_STAGE} IMAGE_SIZE={config.IMAGE_SIZE} MODEL_NAME={config.MODEL_NAME} DEVICE={config.DEVICE}")

    # 依次尝试：CLI 阈值 > meta.json 的 best_thr > 兜底 0.5
    meta = {}
    meta_path = ""
    meta_candidates = []
    if args.threshold_json:
        meta_candidates.append(args.threshold_json)

    ckpt_dirs = sorted({os.path.dirname(p) or "." for p in ckpts})
    for d in ckpt_dirs:
        meta_candidates.extend(sorted(glob.glob(os.path.join(d, f"*stage{config.CURRENT_STAGE}_meta.json"))))
        meta_candidates.extend(sorted(glob.glob(os.path.join(d, "*_meta.json"))))

    for mp in meta_candidates:
        try:
            with open(mp, "r", encoding="utf-8") as f:
                meta = json.load(f)
            meta_path = mp
            break
        except FileNotFoundError:
            continue
        except json.JSONDecodeError:
            print(f"[Meta] Failed to parse: {mp}")
            continue

    meta_thr = meta.get("best_thr") if isinstance(meta, dict) else None
    if args.threshold is not None:
        thr = float(args.threshold)
        print(f"[Thr] Using CLI threshold={thr}")
    elif meta_thr is not None:
        thr = float(meta_thr)
        src = meta_path or "meta"
        print(f"[Thr] Using meta best_thr={thr} (source={src})")
    else:
        thr = 0.5
        print("[Thr] Using fallback threshold=0.5")

    models = [load_model_from_ckpt(p, model_name=auto_model, img_size=auto_size) for p in ckpts]

    _, _, test_tf = get_transforms()
    ds = MedicalDataset(root_dir=config.UNLABELED_TEST_DIR, mode="test", transform=test_tf)
    dl = DataLoader(ds, batch_size=max(1, config.BATCH_SIZE), shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)

    names, probs = predict_probs(models, dl, tta=(not args.no_tta))
    preds = (probs >= thr).astype(np.int64)

    # 竞赛要求首列为 id，这里直接使用文件名（含扩展名）。
    df = pd.DataFrame({"id": names, "label": preds})
    # Default output name includes run/stage/model for traceability & reproducibility.
    if args.output_csv:
        out_csv = args.output_csv
    else:
        stage_tag = os.environ.get("CURRENT_STAGE", "")
        model_tag = os.environ.get("MODEL_NAME", "model")
        if len(ckpts) == 1:
            ckpt_tag = os.path.splitext(os.path.basename(ckpts[0]))[0]
        else:
            ckpt_tag = f"ens{len(ckpts)}"
        out_csv = os.path.join(
            config.OUTPUT_DIR,
            f"submission_{config.RUN_ID}_stage{stage_tag}_{model_tag}_{ckpt_tag}.csv",
        )
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"已保存: {out_csv}")
    print(f"权重数: {len(ckpts)} 阈值={thr:.3f} TTA={not args.no_tta}")


if __name__ == "__main__":
    main()
