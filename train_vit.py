"""train_vit.py

青光眼二分类冠军级训练脚本（按你的策略重构）：
1) Progressive Resizing 两阶段训练：由 config.CURRENT_STAGE 控制（384 -> 512）
2) Loss: BCEWithLogitsLoss（输出单 logit，后续用 OOF 搜索最佳阈值最大化 Accuracy）
3) 保守数据增强：只做轻微颜色扰动+轻微几何+水平翻转，避免引入结构性几何失真
4) Stage2 自动 warm-start：加载 Stage1 对应 fold 的最优权重
5) OOF 保存：每折保存验证集 Preds/Targets 到 oof_fold_X.csv，用于阈值搜索
"""

import argparse
import json
import os
import sys
import logging
import glob

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import config
from model import get_model
from dataset import MedicalDataset


def _apply_mixup(images: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """对一个 batch 做标准 Mixup。

    说明：
    - 二分类 BCEWithLogitsLoss 下，label 是 float (0/1)，mixup 后仍是 float。
    - 这里只实现 Mixup（不做 CutMix/旋转），严格按你的正则化要求“止血”。
    """
    if not getattr(config, 'USE_MIXUP', False):
        return images, labels
    prob = float(getattr(config, 'MIXUP_PROB', 1.0))
    if prob <= 0.0 or np.random.rand() > prob:
        return images, labels

    alpha = float(getattr(config, 'MIXUP_ALPHA', 0.8))
    if alpha <= 0.0:
        return images, labels

    lam = float(np.random.beta(alpha, alpha))
    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)

    mixed_images = images.mul(lam).add(images[index].mul(1.0 - lam))
    labels = labels.view(-1)
    mixed_labels = labels.mul(lam).add(labels[index].mul(1.0 - lam))
    return mixed_images, mixed_labels


def _search_best_threshold_from_oof(oof_paths: list[str]) -> tuple[float, float]:
    """从 OOF 文件搜索全局最佳 Accuracy 阈值（与 inference 对齐：0.2~0.8，步长 0.001）。"""
    if not oof_paths:
        return 0.5, -1.0

    probs_all: list[np.ndarray] = []
    targets_all: list[np.ndarray] = []
    for p in oof_paths:
        df = pd.read_csv(p)
        if 'Preds' not in df.columns or 'Targets' not in df.columns:
            continue
        probs_all.append(df['Preds'].values.astype(np.float32))
        targets_all.append(df['Targets'].values.astype(np.int64))

    if not probs_all:
        return 0.5, -1.0

    probs = np.concatenate(probs_all, axis=0)
    targets = np.concatenate(targets_all, axis=0)

    thresholds = np.linspace(0.2, 0.8, 601, dtype=np.float32)
    best_t = 0.5
    best_acc = -1.0
    for t in thresholds:
        preds = (probs >= t).astype(np.int64)
        acc = float((preds == targets).mean())
        if acc > best_acc or (acc == best_acc and float(t) < best_t):
            best_acc = acc
            best_t = float(t)
    return best_t, best_acc


def _interpolate_pos_embed_in_state_dict(state_dict: dict, key: str, target_shape: torch.Size) -> None:
    """将 state_dict[key] 的 ViT pos_embed 插值到目标形状。

    用途：Stage2 需要加载 Stage1 权重，但 Stage1(384) 的 pos_embed 网格是 24x24，
    Stage2(512) 是 32x32；必须插值，否则无法 strict load 或者特征错位。
    """
    if key not in state_dict:
        return

    pos_embed = state_dict[key]
    if not isinstance(pos_embed, torch.Tensor):
        return
    if tuple(pos_embed.shape) == tuple(target_shape):
        return

    # 形状约定：(1, 1+N, D)
    if pos_embed.ndim != 3:
        return

    num_extra_tokens = 1
    embedding_size = pos_embed.shape[-1]

    n_old = pos_embed.shape[1] - num_extra_tokens
    n_new = target_shape[1] - num_extra_tokens
    old_size = int((n_old) ** 0.5)
    new_size = int((n_new) ** 0.5)
    if old_size * old_size != n_old or new_size * new_size != n_new:
        return

    extra_tokens = pos_embed[:, :num_extra_tokens]
    pos_tokens = pos_embed[:, num_extra_tokens:]
    pos_tokens = pos_tokens.reshape(-1, old_size, old_size, embedding_size).permute(0, 3, 1, 2)
    pos_tokens = torch.nn.functional.interpolate(pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
    state_dict[key] = torch.cat((extra_tokens, pos_tokens), dim=1)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.TEXT_LOG_FILE, mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='RETFound+GeM Training (Binary)')
    # Stage2 时从 Stage1 warm-start 的模型目录（可覆盖 config.STAGE1_MODELS_DIR）
    parser.add_argument('--stage1_models_dir', type=str, default='', help='Stage1 模型目录（包含 vit_stage1_fold{X}.pth 或 vit_fold{X}.pth）')
    return parser.parse_args()


def _read_stage1_step_offset(tb_fold_dir: str) -> int:
    """读取 Stage1 的全局步数末尾，用于让 Stage2 的 TensorBoard 曲线接着画。"""
    meta_path = os.path.join(tb_fold_dir, 'meta_stage1.json')
    if not os.path.exists(meta_path):
        return 0
    try:
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        end_step = int(meta.get('global_step_end', -1))
        return max(end_step + 1, 0)
    except Exception:
        return 0


def _write_stage1_meta(tb_fold_dir: str, global_step_end: int) -> None:
    os.makedirs(tb_fold_dir, exist_ok=True)
    meta_path = os.path.join(tb_fold_dir, 'meta_stage1.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump({'global_step_end': int(global_step_end)}, f, ensure_ascii=False, indent=2)


def build_transforms():
    """按要求使用“保守”增强。

    关键原则：
    - 青光眼判别高度依赖视盘/视杯形态与边界细节，强几何增强（尤其弹性形变）会破坏结构。
    - 只做轻微 Shift/Scale/Rotate + 颜色扰动 + 水平翻转，提升鲁棒性但不扭曲关键解剖结构。
    """
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.02,
            scale_limit=0.05,
            rotate_limit=10,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            p=0.5,
        ),
        A.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.10, p=0.5),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=8, val_shift_limit=5, p=0.3),
        A.Normalize(mean=config.IMG_MEAN, std=config.IMG_STD),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Normalize(mean=config.IMG_MEAN, std=config.IMG_STD),
        ToTensorV2(),
    ])

    return train_transform, val_transform


def main():
    args = parse_args()

    gpu_count = torch.cuda.device_count()
    logger.info(f"启动训练 | Stage={config.CURRENT_STAGE} | Image={config.IMAGE_SIZE} | GPU={gpu_count}")
    logger.info(f"Batch={config.BATCH_SIZE} | Accum={config.ACCUM_STEPS} | EquivBatch={config.BATCH_SIZE * config.ACCUM_STEPS}")
    logger.info(f"LR={config.BASE_LR} | Epochs={config.EPOCHS}")
    logger.info(f"RUN_ID={config.RUN_ID} | TB LogDir={config.CURRENT_LOG_DIR}")

    train_transform, val_transform = build_transforms()

    full_dataset = MedicalDataset(config.TRAIN_DIRS, mode='train', transform=None)
    all_paths = full_dataset.image_paths
    all_labels = [1 if 'disease' in p.lower() else 0 for p in all_paths]
    skf = StratifiedKFold(n_splits=config.K_FOLDS, shuffle=True, random_state=config.SEED)

    # 二分类：logits + BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss()

    for fold, (train_idx, val_idx) in enumerate(skf.split(all_paths, all_labels)):
        fold_idx = fold + 1
        logger.info(f"\n===== Fold {fold_idx}/{config.K_FOLDS} =====")

        # TensorBoard：同一个 RUN_ID 下，fold 写到固定目录。
        # 如果 Stage2 复用同一个 RUN_ID，则会在同一条曲线上继续追加 step。
        tb_fold_dir = os.path.join(config.CURRENT_LOG_DIR, f'fold_{fold_idx}')
        writer = SummaryWriter(log_dir=tb_fold_dir)
        step_offset = _read_stage1_step_offset(tb_fold_dir) if config.CURRENT_STAGE == 2 else 0

        train_ds = MedicalDataset(config.TRAIN_DIRS, mode='train', transform=train_transform, indices=train_idx, image_paths=all_paths)
        val_ds = MedicalDataset(config.TRAIN_DIRS, mode='val', transform=val_transform, indices=val_idx, image_paths=all_paths)

        train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, 
                                  num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, 
                                num_workers=config.NUM_WORKERS, pin_memory=True)

        model = get_model(config.MODEL_NAME, num_classes=config.NUM_CLASSES).to(config.DEVICE)

        # Stage2：自动加载 Stage1 对应 fold 权重作为初始化
        if config.CURRENT_STAGE == 2:
            stage1_dir = (args.stage1_models_dir or config.STAGE1_MODELS_DIR).strip()
            if stage1_dir:
                stage1_path_a = os.path.join(stage1_dir, f'vit_stage1_fold{fold_idx}.pth')
                stage1_path_b = os.path.join(stage1_dir, f'vit_fold{fold_idx}.pth')
                stage1_path = stage1_path_a if os.path.exists(stage1_path_a) else stage1_path_b

                if os.path.exists(stage1_path):
                    logger.info(f"[Stage2] 加载 Stage1 权重初始化: {stage1_path}")
                    try:
                        state_dict = torch.load(stage1_path, map_location='cpu', weights_only=False)
                    except TypeError:
                        state_dict = torch.load(stage1_path, map_location='cpu')
                    # 兼容历史上保存了 DataParallel 的 module. 前缀
                    if isinstance(state_dict, dict) and state_dict and next(iter(state_dict.keys())).startswith('module.'):
                        state_dict = {k[7:]: v for k, v in state_dict.items()}

                    # 关键：pos_embed 插值 (Stage1 384 -> Stage2 512)
                    model_for_keys = model  # 这里尚未 DataParallel
                    target_pos_shape = model_for_keys.backbone.pos_embed.shape
                    _interpolate_pos_embed_in_state_dict(state_dict, 'backbone.pos_embed', target_pos_shape)

                    # 权重健康检查（可选但很关键）：打印 missing/unexpected 的前几项，确保 backbone 对齐
                    incompatible = model.load_state_dict(state_dict, strict=False)
                    missing = list(getattr(incompatible, 'missing_keys', []))
                    unexpected = list(getattr(incompatible, 'unexpected_keys', []))
                    if missing or unexpected:
                        logger.warning(f"[Stage2] load_state_dict 非 strict 检查：missing={len(missing)}, unexpected={len(unexpected)}")
                        if missing:
                            logger.warning(f"[Stage2] missing_keys(前10): {missing[:10]}")
                        if unexpected:
                            logger.warning(f"[Stage2] unexpected_keys(前10): {unexpected[:10]}")

                        # 只允许 head 类小范围不匹配；任何 backbone 缺失/多余都直接报错
                        bad_missing = [k for k in missing if k.startswith('backbone.')]
                        bad_unexpected = [k for k in unexpected if k.startswith('backbone.')]
                        if bad_missing or bad_unexpected:
                            raise RuntimeError(
                                f"[Stage2] Backbone 权重不一致：missing(backbone)={bad_missing[:5]}, unexpected(backbone)={bad_unexpected[:5]}"
                            )
                else:
                    logger.warning(f"[Stage2] 未找到 Stage1 权重: {stage1_path}，将从 RETFound 初始化开始训练")
            else:
                logger.warning("[Stage2] 未指定 Stage1 模型目录（stage1_models_dir / STAGE1_MODELS_DIR），将从 RETFound 初始化开始训练")

        if gpu_count > 1:
            model = nn.DataParallel(model)

        optimizer = optim.AdamW(model.parameters(), lr=config.BASE_LR, weight_decay=config.WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS, eta_min=config.BASE_LR * 0.1)
        
        best_acc = 0.0
        # Stage1/Stage2 命名策略：
        # - Stage1：保留 vit_stage1_foldX.pth，同时写一份 vit_foldX.pth（便于只跑 Stage1 时也能推理）
        # - Stage2：写 vit_foldX.pth 作为最终推理权重（会覆盖 Stage1 的同名文件）
        if config.CURRENT_STAGE == 1:
            model_save_path = os.path.join(config.CURRENT_RUN_MODELS_DIR, f'vit_stage1_fold{fold_idx}.pth')
            model_alias_path = os.path.join(config.CURRENT_RUN_MODELS_DIR, f'vit_fold{fold_idx}.pth')
        else:
            model_save_path = os.path.join(config.CURRENT_RUN_MODELS_DIR, f'vit_fold{fold_idx}.pth')
            model_alias_path = None

        # OOF：Stage1 额外保留一份 stage1 文件；Stage2 覆盖 oof_fold_X.csv 作为最终阈值搜索输入
        oof_save_path = os.path.join(config.OOF_DIR, f'oof_fold_{fold_idx}.csv')
        oof_stage_save_path = os.path.join(config.OOF_DIR, f'oof_stage{config.CURRENT_STAGE}_fold_{fold_idx}.csv')

        for epoch in range(config.EPOCHS):
            model.train()
            train_loss_accum = 0.0
            
            optimizer.zero_grad()
            
            pbar = tqdm(train_loader, desc=f"F{fold_idx} E{epoch+1}", mininterval=10)
            
            for step, (images, labels) in enumerate(pbar):
                images = images.to(config.DEVICE)
                # BCEWithLogitsLoss 需要 float 标签
                labels = labels.to(config.DEVICE).float().view(-1)

                # ===== Mixup 强正则化（按你的 Checklist 强制开启） =====
                images, labels = _apply_mixup(images, labels)

                # 仅在每个 epoch 的第一个 step 记录一次图片，避免日志过大
                # 反归一化 (De-normalization) 以便人眼观看（假设 mean/std = 0.5）
                outputs = model(images)
                if outputs.ndim == 2 and outputs.size(1) == 1:
                    outputs = outputs[:, 0]
                outputs = outputs.view(-1)
                loss = criterion(outputs, labels)
                loss_val = loss.item()
                loss = loss / config.ACCUM_STEPS
                loss.backward()
                
                if (step + 1) % config.ACCUM_STEPS == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), config.CLIP_GRAD_NORM)
                    optimizer.step()
                    optimizer.zero_grad()

                train_loss_accum += loss_val

                # TensorBoard（step 级）
                global_step = step_offset + epoch * len(train_loader) + step
                writer.add_scalar('Loss/train_step', loss_val, global_step)
                writer.add_scalar('LR', optimizer.param_groups[0]['lr'], global_step)

                pbar.set_postfix(loss=f"{loss_val:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
            
            model.eval()

            val_correct = 0
            val_total = 0
            val_loss_accum = 0.0
            oof_probs = []
            oof_targets = []

            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(config.DEVICE)
                    labels_f = labels.to(config.DEVICE).float()

                    logits = model(images)  # (B,)
                    v_loss = criterion(logits, labels_f)
                    val_loss_accum += v_loss.item()

                    probs = torch.sigmoid(logits)
                    preds = (probs >= 0.5).long()

                    val_correct += (preds.cpu() == labels.long()).sum().item()
                    val_total += labels.size(0)

                    oof_probs.extend(probs.detach().cpu().numpy().tolist())
                    oof_targets.extend(labels.detach().cpu().numpy().tolist())

            avg_val_acc = val_correct / max(val_total, 1)
            avg_train_loss = train_loss_accum / max(len(train_loader), 1)
            avg_val_loss = val_loss_accum / max(len(val_loader), 1)

            # TensorBoard（epoch 级，用 step 对齐，曲线连续）
            epoch_end_step = step_offset + (epoch + 1) * len(train_loader) - 1
            writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch_end_step)
            writer.add_scalar('Loss/val', avg_val_loss, epoch_end_step)
            writer.add_scalar('Acc/val@0.5', avg_val_acc, epoch_end_step)

            logger.info(f"Epoch {epoch+1:02d} | TrainLoss={avg_train_loss:.4f} | ValLoss={avg_val_loss:.4f} | ValAcc@0.5={avg_val_acc:.4f}")

            # 每个 epoch 后走一步 scheduler（按 epoch 维度）
            scheduler.step()

            # 保存最优模型 + 对应 OOF（用于后续阈值搜索）
            if avg_val_acc > best_acc:
                best_acc = avg_val_acc
                model_to_save = model.module if isinstance(model, nn.DataParallel) else model
                torch.save(model_to_save.state_dict(), model_save_path)

                if model_alias_path is not None:
                    torch.save(model_to_save.state_dict(), model_alias_path)

                oof_df = pd.DataFrame({'Preds': oof_probs, 'Targets': oof_targets})
                oof_df.to_csv(oof_save_path, index=False)
                oof_df.to_csv(oof_stage_save_path, index=False)

                logger.info(f"   >>> Best Model Saved | Acc@0.5={best_acc:.4f} | {model_save_path}")
                if model_alias_path is not None:
                    logger.info(f"   >>> Model Alias Updated: {model_alias_path}")
                logger.info(f"   >>> OOF Saved: {oof_save_path} (n={len(oof_df)})")

            writer.flush()

        # Stage1 写 meta，供 Stage2 接续 step
        if config.CURRENT_STAGE == 1:
            global_step_end = step_offset + config.EPOCHS * len(train_loader) - 1
            _write_stage1_meta(tb_fold_dir, global_step_end)

        writer.close()

        del model, optimizer
        torch.cuda.empty_cache()

    logger.info("训练完成。")

    # === 训练结束：用 OOF 搜索全局最佳阈值，并固化到 output（供 inference 直接复用） ===
    try:
        oof_paths = sorted(glob.glob(os.path.join(config.OOF_DIR, 'oof_fold_*.csv')))
        if oof_paths:
            best_t, best_acc = _search_best_threshold_from_oof(oof_paths)
            payload = {
                'run_id': config.RUN_ID,
                'stage': int(config.CURRENT_STAGE),
                'threshold': float(best_t),
                'acc': float(best_acc),
                'oof_files': [os.path.basename(p) for p in oof_paths],
            }
            os.makedirs(config.OUTPUT_DIR, exist_ok=True)
            path_run = os.path.join(config.OUTPUT_DIR, f'best_threshold_{config.RUN_ID}.json')
            path_latest = os.path.join(config.OUTPUT_DIR, 'best_threshold.json')
            with open(path_run, 'w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            with open(path_latest, 'w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            logger.info(f"[OOF] 全局最佳阈值已固化：t={best_t:.4f}, acc={best_acc:.6f} | {path_latest}")
    except Exception as e:
        logger.warning(f"[OOF] 阈值固化失败（不影响训练权重保存）：{e}")

if __name__ == '__main__':
    main()