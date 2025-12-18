"""train_vit.py

单划分（默认）+ 可选 KFold 的训练脚本。

当前策略目标：
- 二分类（0/1），榜单指标为 Accuracy
- 支持 Swin（推荐）与 RETFound-ViT 主干
- 通过 config.CURRENT_STAGE 控制 Stage1/Stage2 渐进分辨率
- “非最优 epoch 也可能更好”：每个 epoch 都存权重，便于后续挑选
- 可选：Stage2 使用 LLRD + 冻结→解冻 提升稳定性
"""

import argparse
import json
import logging
import os
import random
import shutil
from dataclasses import asdict, dataclass
from glob import glob
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import config
from dataset import MedicalDataset, get_transforms
from model import get_model


# -----------------------------
# 工具函数
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def search_best_threshold(probs: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    """返回 (最佳阈值, 对应准确率)。"""
    best_thr, best_acc = 0.5, -1.0
    # 由粗到细的栅格搜索
    for thr in np.linspace(0.2, 0.8, 121):
        pred = (probs >= thr).astype(np.int64)
        acc = (pred == labels).mean()
        if acc > best_acc:
            best_acc, best_thr = acc, float(thr)
    return best_thr, float(best_acc)


def confusion_from_probs(probs: np.ndarray, labels: np.ndarray, thr: float) -> Tuple[int, int, int, int]:
    pred = (probs >= thr).astype(np.int64)
    tn = int(((pred == 0) & (labels == 0)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    tp = int(((pred == 1) & (labels == 1)).sum())
    return tn, fp, fn, tp


def gather_image_paths_and_labels(root_dirs) -> Tuple[List[str], List[int]]:
    """遍历目录收集图像，标签由父目录名决定：disease=1，否则为0。"""
    if isinstance(root_dirs, (list, tuple)):
        roots = root_dirs
    else:
        roots = [root_dirs]

    paths = []
    labels = []
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    for rd in roots:
        for root, _, files in os.walk(rd):
            for f in files:
                if f.lower().endswith(exts):
                    p = os.path.join(root, f)
                    parent = os.path.basename(os.path.dirname(p)).lower()
                    y = 1 if parent == "disease" else 0
                    paths.append(p)
                    labels.append(y)
    # 保持稳定的排序
    order = np.argsort(paths)
    paths = [paths[i] for i in order]
    labels = [labels[i] for i in order]
    return paths, labels


# -----------------------------
# LLRD（支持 ViT blocks / Swin layers.blocks）
# -----------------------------
def _is_no_decay(name: str, p: nn.Parameter) -> bool:
    if not p.requires_grad:
        return True
    if name.endswith(".bias"):
        return True
    lname = name.lower()
    if "norm" in lname or "bn" in lname or "layernorm" in lname:
        return True
    return False


def _get_backbone_and_head(model: nn.Module) -> Tuple[nn.Module, nn.Module]:
    m = model.module if isinstance(model, nn.DataParallel) else model
    if not hasattr(m, "backbone") or not hasattr(m, "head"):
        raise RuntimeError("模型需要同时具备 .backbone 与 .head 以支持 LLRD/冻结")
    return m.backbone, m.head


def _build_block_id_map_for_swin(backbone: nn.Module) -> Tuple[Dict[Tuple[int, int], int], int]:
    """将 (stage, block) 映射为全局序号，返回映射与总块数。"""
    mapping: Dict[Tuple[int, int], int] = {}
    gid = 0
    layers = getattr(backbone, "layers", None)
    if layers is None:
        return mapping, 0
    for s, layer in enumerate(layers):
        blocks = getattr(layer, "blocks", None)
        if blocks is None:
            continue
        for b in range(len(blocks)):
            mapping[(s, b)] = gid
            gid += 1
    return mapping, gid


def get_optimizer_param_groups(model: nn.Module, base_lr: float, weight_decay: float, layer_decay: float) -> List[Dict]:
    """构建 LLRD 参数组。

    - 对 ViT：backbone.blocks.{i}
    - 对 Swin：backbone.layers.{stage}.blocks.{block}
    """
    backbone, head = _get_backbone_and_head(model)

    # 检测 ViT
    vit_blocks = getattr(backbone, "blocks", None)
    is_vit = vit_blocks is not None

    # 检测 Swin
    swin_layers = getattr(backbone, "layers", None)
    is_swin = (not is_vit) and (swin_layers is not None)

    # 统计层数（展平成块序号）
    if is_vit:
        num_layers = len(vit_blocks)
        def layer_id_from_name(n: str) -> int:
            if "backbone.blocks." in n:
                try:
                    return int(n.split("backbone.blocks.")[1].split(".")[0])
                except Exception:
                    return -1
            return -1
    elif is_swin:
        mapping, num_layers = _build_block_id_map_for_swin(backbone)
        def layer_id_from_name(n: str) -> int:
            # backbone.layers.{s}.blocks.{b}
            if "backbone.layers." in n and ".blocks." in n:
                try:
                    part = n.split("backbone.layers.")[1]
                    s = int(part.split(".")[0])
                    b = int(part.split(".blocks.")[1].split(".")[0])
                    return mapping.get((s, b), -1)
                except Exception:
                    return -1
            return -1
    else:
        # 回退：不做 LLRD，单一参数组
        return [{"params": [p for p in model.parameters() if p.requires_grad], "lr": base_lr, "weight_decay": weight_decay}]

    groups: Dict[Tuple[int, bool], Dict] = {}

    def add_param(layer_id: int, no_decay: bool, p: nn.Parameter, lr: float):
        key = (layer_id, no_decay)
        if key not in groups:
            groups[key] = {
                "params": [],
                "lr": lr,
                "weight_decay": 0.0 if no_decay else weight_decay,
            }
        groups[key]["params"].append(p)

    # 分配参数到各组
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        no_decay = _is_no_decay(n, p)

        # head 始终使用最大 lr
        if n.startswith("head.") or ".head." in n:
            add_param(num_layers - 1, no_decay, p, base_lr)
            continue

        lid = layer_id_from_name(n)
        if lid >= 0 and num_layers > 0:
            scale = layer_decay ** (num_layers - 1 - lid)
            add_param(lid, no_decay, p, base_lr * scale)
        else:
            # embeddings / patch_embed / pos_embed 采用最小 lr
            scale = layer_decay ** num_layers
            add_param(-1, no_decay, p, base_lr * scale)

    # 打印最小/最大学习率用于校验
    lrs = [g["lr"] for g in groups.values()]
    if len(lrs) > 0:
        print(f"[LLRD] param_groups={len(groups)} min_lr={min(lrs):.2e} max_lr={max(lrs):.2e}")

    return list(groups.values())


# -----------------------------
# 冻结相关工具
# -----------------------------
def set_requires_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag


def apply_stage2_freeze(model: nn.Module) -> None:
    """在 Stage2 先冻结前层，便于 warm-start。"""
    backbone, _ = _get_backbone_and_head(model)

    # 冻结 patch embed（timm 的 ViT/Swin 都提供 patch_embed）
    if config.FREEZE_PATCH_EMBED_STAGE2 and hasattr(backbone, "patch_embed"):
        set_requires_grad(backbone.patch_embed, False)

    # ViT blocks
    if hasattr(backbone, "blocks"):
        blocks = backbone.blocks
        n_freeze = min(config.FREEZE_BLOCKS_BEFORE_STAGE2, len(blocks))
        for i in range(n_freeze):
            set_requires_grad(blocks[i], False)
        print(f"[Freeze] ViT: froze patch_embed={config.FREEZE_PATCH_EMBED_STAGE2}, blocks[:{n_freeze}]")
        return

    # Swin layers.blocks（按展平后的块序号冻结）
    if hasattr(backbone, "layers"):
        mapping, total = _build_block_id_map_for_swin(backbone)
        n_freeze = min(config.FREEZE_BLOCKS_BEFORE_STAGE2, total)
        # freeze blocks with gid < n_freeze
        for (s, b), gid in mapping.items():
            if gid < n_freeze:
                set_requires_grad(backbone.layers[s].blocks[b], False)
        print(f"[Freeze] Swin: froze patch_embed={config.FREEZE_PATCH_EMBED_STAGE2}, first {n_freeze}/{total} blocks")
        return


def unfreeze_all(model: nn.Module) -> None:
    backbone, head = _get_backbone_and_head(model)
    set_requires_grad(backbone, True)
    set_requires_grad(head, True)
    print("[Freeze] Unfroze all parameters")


# -----------------------------
# 训练 / 验证循环
# -----------------------------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs = []
    all_y = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.numpy().astype(np.int64)
        logits = model(images).detach().float().cpu().numpy()
        probs = sigmoid_np(logits)
        all_probs.append(probs)
        all_y.append(labels)
    return np.concatenate(all_probs), np.concatenate(all_y)


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer, criterion, device: str, epoch: int) -> float:
    model.train()
    running = 0.0
    optimizer.zero_grad(set_to_none=True)

    use_mixup = config.USE_MIXUP and (config.CURRENT_STAGE == 1 or epoch < (config.EPOCHS - config.MIXUP_DISABLE_LAST_EPOCHS))
    for step, (images, labels) in enumerate(tqdm(loader, desc=f"Train e{epoch}", leave=False)):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()

        # 可选 Mixup（二值软标签）
        if use_mixup and np.random.rand() < config.MIXUP_PROB:
            lam = np.random.beta(config.MIXUP_ALPHA, config.MIXUP_ALPHA)
            index = torch.randperm(images.size(0), device=images.device)
            images = lam * images + (1 - lam) * images[index]
            labels = lam * labels + (1 - lam) * labels[index]

        logits = model(images)
        loss = criterion(logits, labels)

        loss = loss / config.ACCUM_STEPS
        loss.backward()

        if (step + 1) % config.ACCUM_STEPS == 0:
            nn.utils.clip_grad_norm_(model.parameters(), config.CLIP_GRAD_NORM)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        running += loss.item() * config.ACCUM_STEPS

    return running / max(1, len(loader))


def save_checkpoint(path: str, model: nn.Module, epoch: int, extra: dict) -> None:
    m = model.module if isinstance(model, nn.DataParallel) else model
    ckpt = {
        "epoch": epoch,
        "model_name": config.MODEL_NAME,
        "image_size": config.IMAGE_SIZE,
        "stage": config.CURRENT_STAGE,
        "state_dict": m.state_dict(),
        "extra": extra,
    }
    torch.save(ckpt, path)


def maybe_prune_old_ckpts(ckpt_dir: str, prefix: str) -> None:
    if config.KEEP_LAST_N_EPOCHS >= 999:
        return
    paths = sorted(glob(os.path.join(ckpt_dir, f"{prefix}_epoch*.pth")))
    if len(paths) <= config.KEEP_LAST_N_EPOCHS:
        return
    for p in paths[:-config.KEEP_LAST_N_EPOCHS]:
        try:
            os.remove(p)
        except Exception:
            pass


def run_single_split(init_ckpt: str = "") -> None:
    set_seed(config.SEED)
    ensure_dir(config.CURRENT_RUN_MODELS_DIR)
    ensure_dir(config.CURRENT_LOG_DIR)

    # 日志配置
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(config.TEXT_LOG_FILE, mode="a", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger("train")

    writer = SummaryWriter(log_dir=config.CURRENT_LOG_DIR)

    # 构建数据列表并划分
    all_paths, all_labels = gather_image_paths_and_labels(config.TRAIN_DIRS)
    y = np.array(all_labels)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=config.VAL_RATIO, random_state=config.SEED)
    train_idx, val_idx = next(sss.split(np.zeros_like(y), y))

    train_tf, val_tf, _ = get_transforms()
    ds_train = MedicalDataset(root_dir=config.TRAIN_DIRS, mode="train", transform=train_tf, indices=train_idx, image_paths=all_paths)
    ds_val = MedicalDataset(root_dir=config.TRAIN_DIRS, mode="val", transform=val_tf, indices=val_idx, image_paths=all_paths)

    dl_train = DataLoader(ds_train, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True)
    dl_val = DataLoader(ds_val, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)

    # 构建模型
    model = get_model().to(config.DEVICE)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # 可选 warm-start
    if init_ckpt:
        logger.info(f"[Init] Loading init checkpoint: {init_ckpt}")
        ckpt = torch.load(init_ckpt, map_location="cpu")
        sd = ckpt.get("state_dict", ckpt.get("model", ckpt))
        msg = (model.module if isinstance(model, nn.DataParallel) else model).load_state_dict(sd, strict=False)
        logger.info(f"[Init] missing={len(getattr(msg,'missing_keys',[]))} unexpected={len(getattr(msg,'unexpected_keys',[]))}")

    # Stage2 冻结
    if config.CURRENT_STAGE == 2 and config.FREEZE_EPOCHS_STAGE2 > 0:
        apply_stage2_freeze(model)

    criterion = nn.BCEWithLogitsLoss()

    if config.USE_LLRD:
        param_groups = get_optimizer_param_groups(model, base_lr=config.BASE_LR, weight_decay=config.WEIGHT_DECAY, layer_decay=config.LAYER_DECAY)
        optimizer = optim.AdamW(param_groups)
    else:
        optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config.BASE_LR, weight_decay=config.WEIGHT_DECAY)

    # 学习率调度：Cosine，无全局 eta_min（兼容 LLRD）
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS, eta_min=0.0)

    best_val_acc = -1.0
    best_epoch = -1
    best_thr = 0.5

    prefix = f"{config.MODEL_NAME.replace('/','_')}_stage{config.CURRENT_STAGE}"

    for epoch in range(config.EPOCHS):
        # 到期解冻
        if config.CURRENT_STAGE == 2 and epoch == config.FREEZE_EPOCHS_STAGE2:
            unfreeze_all(model)

            # 解冻后重建优化器参数组
            if config.USE_LLRD:
                param_groups = get_optimizer_param_groups(model, base_lr=config.BASE_LR, weight_decay=config.WEIGHT_DECAY, layer_decay=config.LAYER_DECAY)
                optimizer = optim.AdamW(param_groups)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(config.EPOCHS - epoch), eta_min=0.0)

        train_loss = train_one_epoch(model, dl_train, optimizer, criterion, config.DEVICE, epoch)

        probs, labels = evaluate(model, dl_val, config.DEVICE)
        thr, acc = search_best_threshold(probs, labels)
        tn, fp, fn, tp = confusion_from_probs(probs, labels, thr)

        # 日志记录
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("acc/val_bestT", acc, epoch)
        writer.add_scalar("thr/val_bestT", thr, epoch)
        writer.add_scalar("fp/val_bestT", fp, epoch)
        writer.add_scalar("fn/val_bestT", fn, epoch)

        # 额外记录最小/最大学习率
        lrs = [g["lr"] for g in optimizer.param_groups]
        writer.add_scalar("lr/min", float(min(lrs)), epoch)
        writer.add_scalar("lr/max", float(max(lrs)), epoch)

        logger.info(f"[Epoch {epoch:03d}] loss={train_loss:.4f} val_acc@bestT={acc:.4f} bestT={thr:.3f} CM=[[{tn} {fp}] [{fn} {tp}]] lr=[{min(lrs):.2e},{max(lrs):.2e}]")

        # 每个 epoch 都保存，便于后续挑选“非最佳”权重
        if config.SAVE_EVERY_EPOCH:
            ckpt_path = os.path.join(config.CURRENT_RUN_MODELS_DIR, f"{prefix}_epoch{epoch:03d}.pth")
            save_checkpoint(ckpt_path, model, epoch, extra={"val_acc_bestT": acc, "val_thr": thr, "cm": [tn, fp, fn, tp]})
            maybe_prune_old_ckpts(config.CURRENT_RUN_MODELS_DIR, prefix)

        # 仍然跟踪验证集最佳，方便快速定位好模型
        if acc > best_val_acc:
            best_val_acc = acc
            best_epoch = epoch
            best_thr = thr
            best_path = os.path.join(config.CURRENT_RUN_MODELS_DIR, f"{prefix}_best.pth")
            save_checkpoint(best_path, model, epoch, extra={"val_acc_bestT": acc, "val_thr": thr, "cm": [tn, fp, fn, tp]})
            logger.info(f"[Best] epoch={epoch} val_acc@bestT={acc:.4f} saved => {best_path}")

        scheduler.step()

    # 保存最终元数据 json
    meta = {
        "run_id": config.RUN_ID,
        "model_name": config.MODEL_NAME,
        "stage": config.CURRENT_STAGE,
        "image_size": config.IMAGE_SIZE,
        "val_ratio": config.VAL_RATIO,
        "seed": config.SEED,
        "best_epoch": best_epoch,
        "best_val_acc": float(best_val_acc),
        "best_thr": float(best_thr),
        "ckpt_prefix": prefix,
    }
    with open(os.path.join(config.CURRENT_RUN_MODELS_DIR, f"{prefix}_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    logger.info(f"[Done] best_epoch={best_epoch} best_val_acc={best_val_acc:.4f} best_thr={best_thr:.3f}")
    writer.close()


def run_kfold(stage1_models_dir: str = "") -> None:
    """可选：保留旧的 KFold 路径。"""
    set_seed(config.SEED)
    ensure_dir(config.CURRENT_RUN_MODELS_DIR)
    ensure_dir(config.CURRENT_LOG_DIR)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(config.TEXT_LOG_FILE, mode="a", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger("train")

    writer = SummaryWriter(log_dir=config.CURRENT_LOG_DIR)

    all_paths, all_labels = gather_image_paths_and_labels(config.TRAIN_DIRS)
    y = np.array(all_labels)

    skf = StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.SEED)
    train_tf, val_tf, _ = get_transforms()

    prefix = f"{config.MODEL_NAME.replace('/','_')}_stage{config.CURRENT_STAGE}"

    for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros_like(y), y), start=1):
        logger.info(f"========== Fold {fold}/{config.N_SPLITS} ==========")

        ds_train = MedicalDataset(root_dir=config.TRAIN_DIRS, mode="train", transform=train_tf, indices=tr_idx, image_paths=all_paths)
        ds_val = MedicalDataset(root_dir=config.TRAIN_DIRS, mode="val", transform=val_tf, indices=va_idx, image_paths=all_paths)

        dl_train = DataLoader(ds_train, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True)
        dl_val = DataLoader(ds_val, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)

        model = get_model().to(config.DEVICE)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)

        # stage2 warm-start：加载对应 fold 的 stage1 最优
        if config.CURRENT_STAGE == 2 and stage1_models_dir:
            cand = os.path.join(stage1_models_dir, f"{prefix}_fold{fold}_best.pth")
            if os.path.exists(cand):
                ckpt = torch.load(cand, map_location="cpu")
                sd = ckpt.get("state_dict", ckpt.get("model", ckpt))
                (model.module if isinstance(model, nn.DataParallel) else model).load_state_dict(sd, strict=False)
                logger.info(f"[Init] loaded stage1 fold ckpt: {cand}")

        criterion = nn.BCEWithLogitsLoss()
        if config.USE_LLRD:
            param_groups = get_optimizer_param_groups(model, base_lr=config.BASE_LR, weight_decay=config.WEIGHT_DECAY, layer_decay=config.LAYER_DECAY)
            optimizer = optim.AdamW(param_groups)
        else:
            optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config.BASE_LR, weight_decay=config.WEIGHT_DECAY)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS, eta_min=0.0)

        best_val_acc = -1.0
        best_epoch = -1
        best_thr = 0.5

        for epoch in range(config.EPOCHS):
            if config.CURRENT_STAGE == 2 and config.FREEZE_EPOCHS_STAGE2 > 0 and epoch == 0:
                apply_stage2_freeze(model)
            if config.CURRENT_STAGE == 2 and epoch == config.FREEZE_EPOCHS_STAGE2:
                unfreeze_all(model)

            train_loss = train_one_epoch(model, dl_train, optimizer, criterion, config.DEVICE, epoch)
            probs, labels = evaluate(model, dl_val, config.DEVICE)
            thr, acc = search_best_threshold(probs, labels)
            tn, fp, fn, tp = confusion_from_probs(probs, labels, thr)

            writer.add_scalar(f"fold{fold}/loss", train_loss, epoch)
            writer.add_scalar(f"fold{fold}/acc_bestT", acc, epoch)
            writer.add_scalar(f"fold{fold}/thr_bestT", thr, epoch)

            logger.info(f"[Fold {fold} e{epoch:03d}] loss={train_loss:.4f} val_acc@bestT={acc:.4f} bestT={thr:.3f} CM=[[{tn} {fp}] [{fn} {tp}]]")

            if config.SAVE_EVERY_EPOCH:
                ckpt_path = os.path.join(config.CURRENT_RUN_MODELS_DIR, f"{prefix}_fold{fold}_epoch{epoch:03d}.pth")
                save_checkpoint(ckpt_path, model, epoch, extra={"fold": fold, "val_acc_bestT": acc, "val_thr": thr, "cm": [tn, fp, fn, tp]})

            if acc > best_val_acc:
                best_val_acc = acc
                best_epoch = epoch
                best_thr = thr
                best_path = os.path.join(config.CURRENT_RUN_MODELS_DIR, f"{prefix}_fold{fold}_best.pth")
                save_checkpoint(best_path, model, epoch, extra={"fold": fold, "val_acc_bestT": acc, "val_thr": thr, "cm": [tn, fp, fn, tp]})

            scheduler.step()

        logger.info(f"[Fold {fold}] best_epoch={best_epoch} best_acc={best_val_acc:.4f} best_thr={best_thr:.3f}")

    writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_ckpt", type=str, default="", help="Warm-start checkpoint path (single-split).")
    parser.add_argument("--stage1_models_dir", type=str, default="", help="Stage1 models dir (kfold stage2 warm-start).")
    args = parser.parse_args()

    if config.USE_KFOLD:
        run_kfold(stage1_models_dir=args.stage1_models_dir)
    else:
        run_single_split(init_ckpt=args.init_ckpt)


if __name__ == "__main__":
    main()
