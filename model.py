import math
import os

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

import config


class GeM(nn.Module):
    """GeM (Generalized Mean Pooling)

    中文说明（为什么对眼底/青光眼有用）：
    - 传统 GAP 会把整张图平均掉，视盘/视杯这种小区域的细节很容易被“稀释”。
    - GeM 通过可学习的 p，把“更显著的响应”放大（当 p>1 时更接近 max pooling），
      同时保持可微与稳定，往往能提升细粒度病灶/结构的判别能力。
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = x.clamp(min=self.eps)
        x = x.pow(self.p)
        x = x.mean(dim=(-2, -1))
        x = x.pow(1.0 / self.p)
        return x


class RETFoundViTGeM(nn.Module):
    """RETFound ViT + GeM Pooling 二分类头

    结构“魔改点”（严格按你的要求）：
    1) 取 patch tokens（去掉 cls_token）
    2) reshape 回 2D 特征图 (B, C, H, W)
    3) GeM 聚合得到 (B, C)
    4) Linear 输出 1 个 logit（配合 BCEWithLogitsLoss）

    关键点：H/W 不写死，通过 token 数量动态推导，兼容 384/512 等不同分辨率。
    """

    def __init__(self, backbone: nn.Module, embed_dim: int, num_classes: int = 1):
        super().__init__()
        self.backbone = backbone
        self.gem = GeM()
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- 复刻 timm ViT 的 token 流水线，确保能拿到“完整 token 序列” ---
        # patch embedding: (B, C, H, W) -> (B, N, D)
        x = self.backbone.patch_embed(x)
        # 追加 cls_token: (B, 1+N, D)
        cls_token = self.backbone.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        # 位置编码 + dropout
        x = x + self.backbone.pos_embed
        x = self.backbone.pos_drop(x)
        # transformer blocks
        for blk in self.backbone.blocks:
            x = blk(x)
        x = self.backbone.norm(x)

        # 1) 去掉 cls_token，取 patch tokens
        patch_tokens = x[:, 1:, :]  # (B, N, D)
        b, n, d = patch_tokens.shape

        # 2) N -> (H, W) 动态推导（要求严谨，兼容 384/512）
        # 对于正方形输入且固定 patch_size=16，N 应该是完全平方数：
        # 384: (384/16)^2=24^2=576
        # 512: (512/16)^2=32^2=1024
        h = int(math.sqrt(n))
        w = h
        if h * w != n:
            raise RuntimeError(
                f"patch token 数量不是完全平方数，无法还原2D特征图: N={n}。"
                "请确认输入是正方形且与 patch_size 对齐。"
            )

        # (B, N, D) -> (B, H, W, D) -> (B, D, H, W)
        feat_map = patch_tokens.reshape(b, h, w, d).permute(0, 3, 1, 2).contiguous()

        # 3) GeM pooling: (B, D, H, W) -> (B, D)
        pooled = self.gem(feat_map)

        # 4) Linear head: (B, D) -> (B, 1)
        logits = self.head(pooled)
        # 输出 reshape 成 (B,) 更方便 BCEWithLogitsLoss
        return logits.squeeze(-1)


def _load_retfound_weights_with_pos_embed_interpolation(model: nn.Module, weight_path: str) -> None:
    """加载本地 RETFound 权重，并对 pos_embed 做插值以适配不同分辨率。

    注意：
    - RETFound 通常在 224 预训练（14x14 patch grid），而我们训练/推理用 384/512（24x24/32x32）。
    - 位置编码不插值会 shape mismatch；插值是 ViT 迁移里非常关键的“尺寸对齐”步骤。
    """
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"找不到 RETFound 权重文件: {weight_path}")

    try:
        checkpoint = torch.load(weight_path, map_location='cpu', weights_only=False)
    except TypeError:
        checkpoint = torch.load(weight_path, map_location='cpu')

    checkpoint_model = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint

    # --- 位置编码插值 ---
    pos_embed_key = 'pos_embed'
    state_dict = model.state_dict()
    if pos_embed_key in checkpoint_model and pos_embed_key in state_dict:
        pos_embed_checkpoint = checkpoint_model[pos_embed_key]
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_extra_tokens = 1  # cls token

        orig_size = int((pos_embed_checkpoint.shape[1] - num_extra_tokens) ** 0.5)
        new_size = int((state_dict[pos_embed_key].shape[1] - num_extra_tokens) ** 0.5)

        if orig_size != new_size:
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = F.interpolate(pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            checkpoint_model[pos_embed_key] = torch.cat((extra_tokens, pos_tokens), dim=1)

    # 丢弃不匹配 head（我们的 head 是新建的）
    for k in list(checkpoint_model.keys()):
        if k.startswith('head') or k.startswith('fc') or '.head.' in k:
            del checkpoint_model[k]

    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(f"[RETFound] 权重加载完成 | missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")


def get_model(model_name: str, num_classes: int = 1, pretrained: bool = False) -> nn.Module:
    """构建 RETFound + GeM 的二分类模型。

    兼容 nn.DataParallel：直接返回标准 nn.Module，外部可包 DataParallel。
    """
    _ = pretrained  # 预留参数，保持旧调用兼容

    print(f"[Model] 构建 RETFound+GeM | stage={config.CURRENT_STAGE} | img={config.IMAGE_SIZE} | num_classes={num_classes}")

    # 创建 ViT-Large 主干：num_classes=0 让 timm 去掉原分类头（我们会自建 head）
    backbone = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=0,
        drop_path_rate=0.2,
        img_size=config.IMAGE_SIZE,
    )

    embed_dim = getattr(backbone, 'embed_dim', None)
    if embed_dim is None:
        raise RuntimeError('无法从 timm ViT backbone 读取 embed_dim')

    # 载入 RETFound 预训练权重（本地文件）
    _load_retfound_weights_with_pos_embed_interpolation(backbone, config.RETFOUND_PATH)

    # 包装成“patch token -> GeM -> head”的结构
    model = RETFoundViTGeM(backbone=backbone, embed_dim=embed_dim, num_classes=num_classes)
    return model