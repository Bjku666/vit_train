import math
import os
from typing import Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from collections import OrderedDict


class GeM(nn.Module):
    """广义均值池化（适用于形状 BxCxHxW 的特征图）。"""
    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.avg_pool2d(x, kernel_size=(x.size(-2), x.size(-1)))
        return x.pow(1.0 / self.p).flatten(1)


class RETFoundViTGeM(nn.Module):
    """RETFound ViT 主干 + token 还原 2D + GeM 池化 + 线性头。

    仅适用于暴露以下模块的 timm ViT：
    - patch_embed, cls_token, pos_embed, pos_drop, blocks, norm
    """
    def __init__(self, backbone: nn.Module, num_classes: int = 1):
        super().__init__()
        self.backbone = backbone
        self.gem = GeM()
        embed_dim = getattr(backbone, "embed_dim", backbone.num_features)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # patch embedding (B,C,H,W)->(B,N,D)
        x = self.backbone.patch_embed(x)
        # 添加 cls token
        cls_token = self.backbone.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        # 位置编码（含 dropout）
        x = x + self.backbone.pos_embed
        x = self.backbone.pos_drop(x)
        # transformer blocks + norm
        for blk in self.backbone.blocks:
            x = blk(x)
        x = self.backbone.norm(x)

        # 去除 CLS，将 patch token 还原为 (B,D,h,w)
        x = x[:, 1:, :]  # (B, N, D)
        B, N, D = x.shape
        h = w = int(math.sqrt(N))
        if h * w != N:
            # 回退：无法整平方则做 1D 平均
            feat = x.mean(dim=1)
        else:
            x = x.transpose(1, 2).reshape(B, D, h, w)
            feat = self.gem(x)
        out = self.head(feat)
        return out.squeeze(-1)


class TimmBinaryClassifier(nn.Module):
    """通用 timm 主干（features）+ 线性头，并挂载 .backbone/.head 以支持 LLRD/冻结。"""
    def __init__(self, backbone: nn.Module, num_classes: int = 1):
        super().__init__()
        self.backbone = backbone
        feat_dim = getattr(backbone, "num_features", None)
        if feat_dim is None:
            raise RuntimeError("backbone.num_features missing")
        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 大多数 timm 模型支持 forward_features；个别模型可能返回 dict/tuple
        if hasattr(self.backbone, "forward_features"):
            feat = self.backbone.forward_features(x)
        else:
            feat = self.backbone(x)

        if isinstance(feat, dict):
            feat = feat.get("x") or next(iter(feat.values()))
        if isinstance(feat, (list, tuple)):
            feat = feat[-1]
        # 若返回特征图，兼容 NCHW 与 NHWC 两种布局
        if isinstance(feat, torch.Tensor) and feat.dim() == 4:
            C = self.head.in_features
            if feat.shape[-1] == C:  # NHWC
                feat = feat.mean(dim=(1, 2))
            elif feat.shape[1] == C:  # NCHW
                feat = feat.mean(dim=(-2, -1))
            else:
                # 回退：尝试将通道维移到最后再做空间平均
                if feat.shape[1] < feat.shape[-1]:
                    feat = feat.permute(0, 2, 3, 1).contiguous().mean(dim=(1, 2))
                else:
                    feat = feat.mean(dim=(-2, -1))

        # 若返回 token/sequence 特征 (B, N, C) 或 (B, C, N)
        # 典型：ViT / Swin 的 forward_features 可能给 (B, L, C)
        if isinstance(feat, torch.Tensor) and feat.dim() == 3:
            C = self.head.in_features
            if feat.shape[-1] == C:      # (B, N, C)
                feat = feat.mean(dim=1)
            elif feat.shape[1] == C:     # (B, C, N)
                feat = feat.mean(dim=2)
            else:
                # 回退：直接对 token 维做平均
                feat = feat.mean(dim=1)
        out = self.head(feat)
        return out.squeeze(-1)


def _load_checkpoint(path: str) -> dict:
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"]
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"]
    return ckpt


def _match_and_filter_state_dict(module: nn.Module, state: dict) -> OrderedDict:
    """根据模块当前参数形状过滤权重：
    - 去除常见前缀（module./model./backbone.）
    - 仅保留名称存在且形状一致的条目，避免 size mismatch 抛错
    返回可安全加载的有序字典。
    """
    msd = module.state_dict()
    out = OrderedDict()
    for k, v in state.items():
        k2 = k
        if k2.startswith("module."):
            k2 = k2[7:]
        if k2.startswith("model."):
            k2 = k2[6:]
        if k2.startswith("backbone."):
            k2 = k2[len("backbone."):]
        if k2 in msd and msd[k2].shape == v.shape:
            out[k2] = v
    return out


def _interpolate_pos_embed(vit: nn.Module, checkpoint_model: dict) -> dict:
    """对 ViT 的位置编码做插值以匹配当前分辨率。"""
    pos_embed_key = "pos_embed"
    if not (hasattr(vit, "pos_embed") and pos_embed_key in checkpoint_model):
        return checkpoint_model

    pos_embed_checkpoint = checkpoint_model[pos_embed_key]
    pos_embed_model = vit.pos_embed
    if pos_embed_checkpoint.shape == pos_embed_model.shape:
        return checkpoint_model

    # checkpoint: (1, old_tokens, D) ; model: (1, new_tokens, D)
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = vit.patch_embed.num_patches
    num_extra_tokens = vit.pos_embed.shape[-2] - num_patches

    # class token 及其他额外 token
    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]

    # 旧网格尺寸
    old_grid = int(math.sqrt(pos_tokens.shape[1]))
    new_grid = int(math.sqrt(num_patches))
    if old_grid * old_grid != pos_tokens.shape[1]:
        return checkpoint_model  # 无法推断则跳过

    pos_tokens = pos_tokens.reshape(1, old_grid, old_grid, embedding_size).permute(0, 3, 1, 2)
    pos_tokens = torch.nn.functional.interpolate(pos_tokens, size=(new_grid, new_grid), mode="bicubic", align_corners=False)
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(1, new_grid * new_grid, embedding_size)

    new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
    checkpoint_model[pos_embed_key] = new_pos_embed
    return checkpoint_model


def load_retfound_weights(vit_backbone: nn.Module, weight_path: str) -> Tuple[int, int]:
    """将 RETFound 权重加载到 ViT 主干（strict=False），返回 missing/unexpected 数量。"""
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"RETFound weights not found: {weight_path}")

    checkpoint_model = _load_checkpoint(weight_path)

    # 如有需要对 pos_embed 插值
    checkpoint_model = _interpolate_pos_embed(vit_backbone, checkpoint_model)

    msg = vit_backbone.load_state_dict(checkpoint_model, strict=False)
    missing = len(getattr(msg, "missing_keys", []))
    unexpected = len(getattr(msg, "unexpected_keys", []))
    return missing, unexpected


def get_model() -> nn.Module:
    name = config.MODEL_NAME.lower()

    # -----------------------------
    # Swin / ConvNeXt 等（默认 ImageNet 预训练）
    # -----------------------------
    if "swin" in name:
        # 不再使用 timm 的 checkpoint_path（其内部为 strict 加载）。
        # 当存在本地权重时：先构建未预训练的模型，再手动 strict=False 加载；
        # 否则：使用 timm 的 ImageNet 预训练权重。
        has_local = os.path.exists(config.SWIN_CHECKPOINT_PATH)
        backbone = timm.create_model(
            config.MODEL_NAME,
            pretrained=(not has_local),
            num_classes=0,
            global_pool="",
            img_size=config.IMAGE_SIZE,  # 确保 patch_embed/grid_size 与当前阶段分辨率一致
        )

        # 允许可变输入尺寸（Stage2 可能使用 448/512）
        try:
            if hasattr(backbone, "patch_embed"):
                backbone.patch_embed.strict_img_size = False
        except Exception:
            pass
        try:
            setattr(backbone, "dynamic_img_size", True)
        except Exception:
            pass

        src = "timm pretrained"
        missing = unexpected = None
        if has_local:
            state = _load_checkpoint(config.SWIN_CHECKPOINT_PATH)
            state = _match_and_filter_state_dict(backbone, state)
            msg = backbone.load_state_dict(state, strict=False)
            missing = len(getattr(msg, "missing_keys", []))
            unexpected = len(getattr(msg, "unexpected_keys", []))
            src = f"manual strict=False from {config.SWIN_CHECKPOINT_PATH} (loaded={len(state)})"

        model = TimmBinaryClassifier(backbone, num_classes=config.NUM_CLASSES)
        print(f"[Model] Swin backbone: {config.MODEL_NAME} | {src} | img={config.IMAGE_SIZE} | num_classes={config.NUM_CLASSES}")
        return model

    # -----------------------------
    # ViT (RETFound)
    # -----------------------------
    if "vit" in name:
        # 创建不带分类头的 timm ViT 主干
        backbone = timm.create_model(config.MODEL_NAME, pretrained=False, num_classes=0, global_pool="")

        # 允许可变输入尺寸（Stage2 可能使用 448/512）
        try:
            if hasattr(backbone, "patch_embed"):
                backbone.patch_embed.strict_img_size = False
        except Exception:
            pass
        try:
            setattr(backbone, "dynamic_img_size", True)
        except Exception:
            pass
        try:
            if hasattr(backbone, "patch_embed"):
                backbone.patch_embed.img_size = None
        except Exception:
            pass
        # 加载 RETFound 权重
        missing, unexpected = load_retfound_weights(backbone, config.RETFOUND_PATH)

        if config.USE_GEM:
            model = RETFoundViTGeM(backbone, num_classes=config.NUM_CLASSES)
            print(f"[Model] RETFound+GeM | {config.MODEL_NAME} | img={config.IMAGE_SIZE} | missing={missing} unexpected={unexpected}")
        else:
            model = TimmBinaryClassifier(backbone, num_classes=config.NUM_CLASSES)
            print(f"[Model] RETFound ViT | {config.MODEL_NAME} | img={config.IMAGE_SIZE} | missing={missing} unexpected={unexpected}")
        return model

    # 回退：任意 timm 模型
    backbone = timm.create_model(config.MODEL_NAME, pretrained=True, num_classes=0, global_pool="")

    # 允许可变输入尺寸（Stage2 可能使用 448/512）
    try:
        if hasattr(backbone, "patch_embed"):
            backbone.patch_embed.strict_img_size = False
    except Exception:
        pass
    try:
        setattr(backbone, "dynamic_img_size", True)
    except Exception:
        pass
    try:
        if hasattr(backbone, "patch_embed"):
            backbone.patch_embed.img_size = None
    except Exception:
        pass
    model = TimmBinaryClassifier(backbone, num_classes=config.NUM_CLASSES)
    print(f"[Model] timm backbone: {config.MODEL_NAME} | img={config.IMAGE_SIZE} | num_classes={config.NUM_CLASSES}")
    return model
