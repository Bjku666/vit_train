import math
import os
from typing import Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

import config


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
        # 大多数 timm 模型支持 forward_features
        if hasattr(self.backbone, "forward_features"):
            feat = self.backbone.forward_features(x)
        else:
            feat = self.backbone(x)
        # 部分模型返回 (B, C, H, W)，需要做池化
        if feat.dim() == 4:
            feat = feat.mean(dim=(-2, -1))
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
        checkpoint_path = config.SWIN_CHECKPOINT_PATH if os.path.exists(config.SWIN_CHECKPOINT_PATH) else ""
        use_pretrained = checkpoint_path == ""
        # 构建去掉分类头的主干；若提供本地权重则不会尝试联网下载。
        backbone = timm.create_model(
            config.MODEL_NAME,
            pretrained=use_pretrained,
            num_classes=0,
            checkpoint_path=checkpoint_path,
        )
        model = TimmBinaryClassifier(backbone, num_classes=config.NUM_CLASSES)
        src = "timm pretrained" if use_pretrained else f"local ckpt: {checkpoint_path}"
        print(f"[Model] Swin backbone: {config.MODEL_NAME} | {src} | img={config.IMAGE_SIZE} | num_classes={config.NUM_CLASSES}")
        return model

    # -----------------------------
    # ViT (RETFound)
    # -----------------------------
    if "vit" in name:
        # 创建不带分类头的 timm ViT 主干
        backbone = timm.create_model(config.MODEL_NAME, pretrained=False, num_classes=0)
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
    backbone = timm.create_model(config.MODEL_NAME, pretrained=True, num_classes=0)
    model = TimmBinaryClassifier(backbone, num_classes=config.NUM_CLASSES)
    print(f"[Model] timm backbone: {config.MODEL_NAME} | img={config.IMAGE_SIZE} | num_classes={config.NUM_CLASSES}")
    return model
