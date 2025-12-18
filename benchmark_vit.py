# benchmark_vit.py (最终修正版 - 兼容 DataParallel)
"""
功能：对验证集做多模型 2x TTA 融合评测（仅原图+水平翻转，严禁旋转），自动搜索最佳阈值并输出 benchmark JSON。
用法示例：
    CONFIG_INIT_DIRS=0 python benchmark_vit.py --model_paths "models/run_xxx/vit_fold*.pth"
输出：
    output/benchmark_result_YYYYMMDD_HHMMSS.json，包含阈值/ACC/F1；不创建训练日志目录。
注意：依赖 config.LABELED_TEST_DIR 作为有标签验证集路径。
"""

import os
import argparse
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import glob

# === 评测脚本只读：阻断训练目录自动创建 ===
# 在导入 config 之前关闭目录初始化开关，避免生成 run_* 或 logs/*
os.environ['CONFIG_INIT_DIRS'] = '0'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import config
from model import get_model
from dataset import MedicalDataset, val_transform_alb

def parse_args():
    parser = argparse.ArgumentParser(description="Final ViT Benchmark & Ensemble Script")
    parser.add_argument('--model_paths', nargs='+', required=True, 
                        help="Path to model files. Use quotes for wildcards, e.g., 'models/run_xxx/vit_fold*.pth'")
    return parser.parse_args()

def tta_shift(images: torch.Tensor, shift_px: int):
    # 使用 padding + slice，避免环绕效应
    pad = torch.nn.functional.pad(images, (0, 0, 0, 0, shift_px, shift_px, shift_px, shift_px))
    views = []
    # up
    views.append(pad[:, :, shift_px*2:, shift_px:-shift_px])
    # down
    views.append(pad[:, :, :-shift_px*2, shift_px:-shift_px])
    # left
    views.append(pad[:, :, shift_px:-shift_px, shift_px*2:])
    # right
    views.append(pad[:, :, shift_px:-shift_px, :-shift_px*2])
    return views


def main():
    args = parse_args()
    
    model_files = []
    for path_pattern in args.model_paths:
        model_files.extend(glob.glob(path_pattern))
    
    if not model_files:
        print(f"❌ Error: No model files found matching the pattern: {args.model_paths}")
        return

    print(f"Found {len(model_files)} models for evaluation.")

    # 1. 准备数据（使用验证期同款归一化）
    test_transform = val_transform_alb
    dataset = MedicalDataset(config.LABELED_TEST_DIR, mode='train', transform=test_transform)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)

    # 2. 加载模型
    models = []
    for path in model_files:
        print(f"  -> Loading {os.path.basename(path)}")
        m = get_model(config.MODEL_NAME, num_classes=config.NUM_CLASSES, pretrained=False)
        try:
            # 加载权重字典（兼容不同 torch 版本）
            try:
                state_dict = torch.load(path, map_location=config.DEVICE, weights_only=True)
            except TypeError:
                state_dict = torch.load(path, map_location=config.DEVICE)
            
            # === 核心修复: 智能处理 DataParallel 的 'module.' 前缀 ===
            # 如果 state_dict 的 key 是以 'module.' 开头，就创建一个新的字典去掉这个前缀
            if list(state_dict.keys())[0].startswith('module.'):
                print("     L─ Detected 'module.' prefix, stripping it...")
                new_state_dict = {k[7:]: v for k, v in state_dict.items()}
                state_dict = new_state_dict
            # =========================================================

            m.load_state_dict(state_dict)
            m.to(config.DEVICE)
            m.eval()
            models.append(m)
        except Exception as e:
            print(f"    L─Error loading {os.path.basename(path)}: {e}. Skipping this model.")
            
    if not models:
        print("No models were loaded successfully. Aborting.")
        return

    # 3. 推理 (严格 2x TTA：原图 + 水平翻转)
    all_probs, all_labels = [], []
    
    print("\nRunning Inference with 2x TTA (Identity + HFlip, No Rotation)...")
    shift_tta = int(os.environ.get('SHIFT_TTA', '0')) == 1
    shift_px = int(os.environ.get('SHIFT_PX', '8'))

    with torch.no_grad():
        for images, labels in tqdm(loader):
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            prob_sum = torch.zeros(images.size(0), device=config.DEVICE)
            tta_views = [images, torch.flip(images, dims=[3])]
            if shift_tta:
                tta_views.extend(tta_shift(images, shift_px))
                tta_views.extend([torch.flip(v, dims=[3]) for v in tta_views if v is not images])
            denom = float(len(models) * len(tta_views))
            
            for model in models:
                for view in tta_views:
                    logits = model(view)
                    if logits.ndim == 2 and logits.size(1) == 1:
                        logits = logits[:, 0]
                    prob_sum += torch.sigmoid(logits)

            avg_prob = (prob_sum / denom).detach().cpu().numpy()
            all_probs.extend(avg_prob)
            all_labels.extend(labels.detach().cpu().numpy())

    # 4. 寻找最佳阈值（与 inference 对齐：0.2~0.8，步长 0.001）
    best_threshold, best_acc, best_f1 = 0.5, -1.0, 0.0
    thresholds = np.linspace(0.2, 0.8, 601, dtype=np.float32)
    probs_np = np.asarray(all_probs, dtype=np.float32)
    labels_np = np.asarray(all_labels, dtype=np.int64)
    for t in thresholds:
        preds = (probs_np >= t).astype(np.int64)
        acc = accuracy_score(labels_np, preds)
        f1 = f1_score(labels_np, preds)
        if acc > best_acc or (acc == best_acc and float(t) < best_threshold):
            best_acc, best_f1, best_threshold = acc, f1, float(t)

    # 5. 输出结果
    print(f"\n====== Final Benchmark Results ({len(models)} Models) ======")
    print(f"Best Threshold: {best_threshold:.3f}")
    print(f"Accuracy:       {best_acc:.4f}")
    print(f"F1 Score:       {best_f1:.4f}")
    
    final_preds = (probs_np >= best_threshold).astype(np.int64)
    print("Confusion Matrix:")
    print(confusion_matrix(labels_np, final_preds))

    # 保存结果
    # 仅创建评测输出目录，不触碰训练日志/模型目录
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    res_path = os.path.join(config.OUTPUT_DIR, f"benchmark_result_{config.RUN_ID}.json")
    with open(res_path, 'w') as f:
        json.dump({
            "models_used": [os.path.basename(p) for p in model_files],
            "threshold": best_threshold,
            "accuracy": best_acc,
            "f1_score": best_f1
        }, f, indent=4)
    print(f"\n Results saved to {res_path}")

if __name__ == '__main__':
    main()