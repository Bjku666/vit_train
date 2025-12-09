# train_vit.py (最终修正版)
"""
功能：5 折 ViT 训练脚本，内置 EMA、Mixup、Albumentations、Ben Graham 预处理、梯度累积、线性预热+Cosine 学习率、TensorBoard 记录。
用法示例（后台训练）：
    nohup python -u train_vit.py > logs/train.log 2>&1 &
可视化：
    tensorboard --logdir logs --port 6006
依赖配置：超参与路径在 config.py 中设定（TRAIN_DIRS 支持原始+伪标签数据）。
"""

import os
import sys
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from timm.utils import ModelEmaV2
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import config
from model import get_model
from dataset import MedicalDataset, train_transform_alb, val_transform_alb

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.TEXT_LOG_FILE, mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    gpu_count = torch.cuda.device_count()
    logger.info(f"🚀 启动训练 (384px + EMA + BenGraham) | GPU: {gpu_count}")
    logger.info(f"Batch Size: {config.BATCH_SIZE} | Accum Steps: {config.ACCUM_STEPS} (Equiv Batch: {config.BATCH_SIZE * config.ACCUM_STEPS})")
    
    # === 核心修复 1: 使用 Albumentations 专业级增强（替换 torchvision）===
    # 结合 Ben Graham 预处理 + 几何/亮度/噪声扰动，强迫模型关注形状与纹理
    train_transform = train_transform_alb
    val_transform = val_transform_alb

    full_dataset = MedicalDataset(config.TRAIN_DIRS, mode='train', transform=None)
    all_paths = full_dataset.image_paths
    all_labels = [1 if 'disease' in p.lower() else 0 for p in all_paths]
    skf = StratifiedKFold(n_splits=config.K_FOLDS, shuffle=True, random_state=config.SEED)

    if config.USE_MIXUP:
        mixup_fn = Mixup(
            mixup_alpha=config.MIXUP_ALPHA, cutmix_alpha=config.CUTMIX_ALPHA, 
            prob=config.MIXUP_PROB, switch_prob=config.MIXUP_SWITCH_PROB, 
            mode='batch', label_smoothing=config.LABEL_SMOOTHING, num_classes=config.NUM_CLASSES
        )
        train_criterion = SoftTargetCrossEntropy()
    else:
        mixup_fn = None
        train_criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    
    val_criterion = nn.CrossEntropyLoss()

    for fold, (train_idx, val_idx) in enumerate(skf.split(all_paths, all_labels)):
        fold_idx = fold + 1
        logger.info(f"\n===== Fold {fold_idx}/{config.K_FOLDS} =====")
        
        writer = SummaryWriter(log_dir=os.path.join(config.CURRENT_LOG_DIR, f'fold_{fold_idx}'))

        train_ds = MedicalDataset(config.TRAIN_DIRS, mode='train', transform=train_transform, indices=train_idx, image_paths=all_paths)
        val_ds = MedicalDataset(config.TRAIN_DIRS, mode='val', transform=val_transform, indices=val_idx, image_paths=all_paths)

        train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, 
                                  num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, 
                                num_workers=config.NUM_WORKERS, pin_memory=True)

        model = get_model(config.MODEL_NAME, num_classes=config.NUM_CLASSES).to(config.DEVICE)
        
        if gpu_count > 1:
            model = nn.DataParallel(model)
        
        optimizer = optim.AdamW(model.parameters(), lr=config.BASE_LR, weight_decay=config.WEIGHT_DECAY)
        # === 先创建 Cosine 调度器以记录 base_lrs，随后把当前 lr 置 0 做线性预热 ===
        total_steps = config.EPOCHS * len(train_loader)
        warmup_steps = config.WARMUP_EPOCHS * len(train_loader)
        cosine_steps = max(total_steps - warmup_steps, 1)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=1e-6)
        for g in optimizer.param_groups:
            g['lr'] = 0.0  # 预热从 0 开始线性爬升

        model_ema = ModelEmaV2(model, decay=0.999)
        
        best_acc = 0.0
        model_save_path = os.path.join(config.CURRENT_RUN_MODELS_DIR, f'vit_fold{fold_idx}.pth')

        for epoch in range(config.EPOCHS):
            model.train()
            train_loss_accum = 0.0
            
            optimizer.zero_grad()
            
            pbar = tqdm(train_loader, desc=f"F{fold_idx} E{epoch+1}", mininterval=10)
            
            for step, (images, labels) in enumerate(pbar):
                images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)

                if mixup_fn is not None:
                    images, labels = mixup_fn(images, labels)

                outputs = model(images)
                
                loss = train_criterion(outputs, labels)
                loss_val = loss.item()
                loss = loss / config.ACCUM_STEPS
                loss.backward()
                
                if (step + 1) % config.ACCUM_STEPS == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), config.CLIP_GRAD_NORM)
                    optimizer.step()
                    optimizer.zero_grad()
                    model_ema.update(model)

                train_loss_accum += loss_val
                # === 核心修复 2: 补全 Step 级别的日志 + 线性预热 ===
                global_step = epoch * len(train_loader) + step

                # 线性预热：step 级别从 0 -> BASE_LR
                if global_step < warmup_steps:
                    warmup_lr = config.BASE_LR * (global_step + 1) / warmup_steps
                    for g in optimizer.param_groups:
                        g['lr'] = warmup_lr
                else:
                    # 预热完成后交给 CosineAnnealingLR（使用 step 对齐）
                    scheduler.step(global_step - warmup_steps)

                writer.add_scalar('Train/Loss_Step', loss_val, global_step)
                writer.add_scalar('LR_Step', optimizer.param_groups[0]['lr'], global_step)
            
            model_ema.module.eval()
            
            val_correct = 0
            val_total = 0
            val_loss_accum = 0.0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
                    outputs = model_ema.module(images)
                    
                    # === 核心修复 2: 补全 Val Loss 计算 ===
                    v_loss = val_criterion(outputs, labels)
                    val_loss_accum += v_loss.item()

                    _, preds = torch.max(outputs, 1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)
            
            avg_val_acc = val_correct / val_total
            avg_train_loss = train_loss_accum / len(train_loader)
            avg_val_loss = val_loss_accum / len(val_loader)

            # === 核心修复 2: 补全所有 Epoch 级别的日志 ===
            writer.add_scalar('Train/Loss_Epoch', avg_train_loss, epoch)
            writer.add_scalar('Val/Loss', avg_val_loss, epoch)
            writer.add_scalar('Val/Accuracy_EMA', avg_val_acc, epoch) # 名字统一
            writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
            
            logger.info(f"Epoch {epoch+1:02d} | Loss: {avg_train_loss:.4f} | Val Acc (EMA): {avg_val_acc:.4f}")

            if avg_val_acc > best_acc:
                best_acc = avg_val_acc
                torch.save(model_ema.module.state_dict(), model_save_path)
                logger.info(f"   >>> Best EMA Model Saved (Acc: {best_acc:.4f})")

        writer.close()
        del model, model_ema, optimizer
        torch.cuda.empty_cache()

    logger.info("训练完成。")

if __name__ == '__main__':
    main()