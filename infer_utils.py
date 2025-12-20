import os
from typing import Tuple, Dict, Any

import torch
import torch.nn as nn

import config
from model import get_model


def build_model(model_name: str = None, img_size: int = None, drop_path_rate: float = None, device: str = None) -> nn.Module:
    """统一构建模型；允许显式覆盖 model/img_size/drop_path_rate/设备，避免评测阶段依赖环境变量。"""
    dev = device or config.DEVICE
    model = get_model(model_name=model_name, img_size=img_size, drop_path_rate=drop_path_rate).to(dev)
    return model


def _extract_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    """从 ckpt 对象中提取 state_dict，优先返回 EMA 权重。"""
    if not isinstance(obj, dict):
        return obj

    # 优先使用 EMA 权重（若存在）
    for k in ["model_ema", "ema", "ema_state_dict"]:
        v = obj.get(k)
        if isinstance(v, dict):
            return v

    # 其次使用常规模型权重
    for k in ["state_dict", "model", "model_state_dict", "net"]:
        v = obj.get(k)
        if isinstance(v, dict):
            return v

    return obj  # 可能直接是 state_dict


def _strip_prefix(key: str) -> str:
    for p in ("module.", "model."):
        if key.startswith(p):
            return key[len(p):]
    return key


def _maybe_map_backbone_keys(model: nn.Module, k: str) -> str:
    """在 wrapper(backbone+head) 与纯 backbone 权重之间做键名适配。"""
    msd = model.state_dict()
    # 情况1：ckpt 是 backbone-only，但当前模型有前缀 'backbone.'
    if k in msd:
        return k
    bk = f"backbone.{k}"
    if bk in msd:
        return bk
    # 情况2：ckpt 是带有 'backbone.' 前缀，但当前模型是扁平
    if k.startswith("backbone."):
        k2 = k[len("backbone."):]
        if k2 in msd:
            return k2
    return k


def smart_load_state_dict(model: nn.Module, ckpt_path: str, strict: bool = False) -> Tuple[int, int, int]:
    """鲁棒加载：strip 前缀、适配 backbone、过滤尺寸不匹配，并返回 (loaded, missing, unexpected)。"""
    try:
        obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(ckpt_path, map_location="cpu")

    raw = _extract_state_dict(obj)
    msd = model.state_dict()

    new_sd: Dict[str, torch.Tensor] = {}
    for k, v in raw.items():
        k1 = _strip_prefix(k)
        k2 = _maybe_map_backbone_keys(model, k1)
        if k2 in msd and msd[k2].shape == v.shape:
            new_sd[k2] = v

    msg = model.load_state_dict(new_sd, strict=strict)
    missing = len(getattr(msg, "missing_keys", []))
    unexpected = len(getattr(msg, "unexpected_keys", []))

    loaded = len(new_sd)
    total = len(msd)
    ratio = (loaded / max(1, total)) * 100.0
    print(f"[smart_load] loaded={loaded}/{total} ({ratio:.1f}%) missing={missing} unexpected={unexpected} from {os.path.basename(ckpt_path)}")
    return loaded, missing, unexpected


def load_model_from_ckpt(ckpt_path: str, model_name: str = None, img_size: int = None, drop_path_rate: float = None, device: str = None) -> nn.Module:
    model = build_model(model_name=model_name, img_size=img_size, drop_path_rate=drop_path_rate, device=device)
    smart_load_state_dict(model, ckpt_path, strict=False)
    model.eval()
    return model


def get_ckpt_meta(ckpt_path: str) -> Dict[str, Any]:
    """读取 checkpoint 顶层元信息（如 model_name/image_size/stage），若缺失返回空字典。"""
    meta: Dict[str, Any] = {}
    try:
        obj = torch.load(ckpt_path, map_location="cpu")
        if isinstance(obj, dict):
            for k in ["model_name", "image_size", "stage", "epoch"]:
                if k in obj:
                    meta[k] = obj[k]
    except Exception:
        pass
    return meta
