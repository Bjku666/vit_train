"""train_vit.py

青光眼二分类冠军级训练脚本（按你的策略重构）：
1) Progressive Resizing 两阶段训练：由 config.CURRENT_STAGE 控制（384 -> 512）
2) Loss: BCEWithLogitsLoss（输出单 logit，后续用 OOF 搜索最佳阈值最大化 Accuracy）
3) 保守数据增强：只做轻微颜色扰动+轻微几何+水平翻转，避免引入结构性几何失真
4) Stage2 自动 warm-start：加载 Stage1 对应 fold 的最优权重
5) OOF 保存：每折保存验证集 Preds/Targets 到 oof_fold_X.csv，用于阈值搜索
"""

import argparse
import os
import sys
import logging

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import config
from model import get_model
from dataset import MedicalDataset

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
    parser.add_argument('--stage1_models_dir', type=str, default='', help='Stage1 模型目录（包含 vit_fold{X}.pth）')
    return parser.parse_args()


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
                stage1_path = os.path.join(stage1_dir, f'vit_fold{fold_idx}.pth')
                if os.path.exists(stage1_path):
                    logger.info(f"[Stage2] 加载 Stage1 权重初始化: {stage1_path}")
                    try:
                        state_dict = torch.load(stage1_path, map_location='cpu', weights_only=False)
                    except TypeError:
                        state_dict = torch.load(stage1_path, map_location='cpu')
                    # 兼容历史上保存了 DataParallel 的 module. 前缀
                    if isinstance(state_dict, dict) and state_dict and next(iter(state_dict.keys())).startswith('module.'):
                        state_dict = {k[7:]: v for k, v in state_dict.items()}
                    model.load_state_dict(state_dict, strict=True)
                else:
                    logger.warning(f"[Stage2] 未找到 Stage1 权重: {stage1_path}，将从 RETFound 初始化开始训练")
            else:
                logger.warning("[Stage2] 未指定 Stage1 模型目录（stage1_models_dir / STAGE1_MODELS_DIR），将从 RETFound 初始化开始训练")

        if gpu_count > 1:
            model = nn.DataParallel(model)

        optimizer = optim.AdamW(model.parameters(), lr=config.BASE_LR, weight_decay=config.WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS, eta_min=config.BASE_LR * 0.1)
        
        best_acc = 0.0
        model_save_path = os.path.join(config.CURRENT_RUN_MODELS_DIR, f'vit_fold{fold_idx}.pth')
        oof_save_path = os.path.join(config.OOF_DIR, f'oof_fold_{fold_idx}.csv')

        for epoch in range(config.EPOCHS):
            model.train()
            train_loss_accum = 0.0
            
            optimizer.zero_grad()
            
            pbar = tqdm(train_loader, desc=f"F{fold_idx} E{epoch+1}", mininterval=10)
            
            for step, (images, labels) in enumerate(pbar):
                images = images.to(config.DEVICE)
                # BCEWithLogitsLoss 需要 float 标签
                labels = labels.to(config.DEVICE).float()

                # 仅在每个 epoch 的第一个 step 记录一次图片，避免日志过大
                # 反归一化 (De-normalization) 以便人眼观看（假设 mean/std = 0.5）
                outputs = model(images)  # (B,) logits
                loss = criterion(outputs, labels)
                loss_val = loss.item()
                loss = loss / config.ACCUM_STEPS
                loss.backward()
                
                if (step + 1) % config.ACCUM_STEPS == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), config.CLIP_GRAD_NORM)
                    optimizer.step()
                    optimizer.zero_grad()

                train_loss_accum += loss_val

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

            logger.info(f"Epoch {epoch+1:02d} | TrainLoss={avg_train_loss:.4f} | ValLoss={avg_val_loss:.4f} | ValAcc@0.5={avg_val_acc:.4f}")

            # 每个 epoch 后走一步 scheduler（按 epoch 维度）
            scheduler.step()

            # 保存最优模型 + 对应 OOF（用于后续阈值搜索）
            if avg_val_acc > best_acc:
                best_acc = avg_val_acc
                model_to_save = model.module if isinstance(model, nn.DataParallel) else model
                torch.save(model_to_save.state_dict(), model_save_path)

                oof_df = pd.DataFrame({'Preds': oof_probs, 'Targets': oof_targets})
                oof_df.to_csv(oof_save_path, index=False)

                logger.info(f"   >>> Best Model Saved | Acc@0.5={best_acc:.4f} | {model_save_path}")
                logger.info(f"   >>> OOF Saved: {oof_save_path} (n={len(oof_df)})")

        del model, optimizer
        torch.cuda.empty_cache()

    logger.info("训练完成。")

if __name__ == '__main__':
    main()