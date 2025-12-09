# benchmark_vit.py (最终修正版 - 兼容 DataParallel)
"""
功能：对验证集做多模型 8x TTA 融合评测，自动搜索最佳阈值并输出 benchmark JSON。
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
            # 加载权重字典
            state_dict = torch.load(path, map_location=config.DEVICE, weights_only=True)
            
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
            print(f"    L─❌ Error loading {os.path.basename(path)}: {e}. Skipping this model.")
            
    if not models:
        print("No models were loaded successfully. Aborting.")
        return

    # 3. 推理 (8x TTA)
    all_probs, all_labels = [], []
    
    print("\nRunning Inference with 8x TTA (Rotation + Flip)...")
    with torch.no_grad():
        for images, labels in tqdm(loader):
            images = images.to(config.DEVICE)
            batch_probs = torch.zeros(images.size(0), 2).to(config.DEVICE)
            rotations = [0, 1, 2, 3]
            
            for model in models:
                for k in rotations:
                    img_rot = torch.rot90(images, k=k, dims=[2, 3])
                    logits = model(img_rot)
                    batch_probs += torch.softmax(logits, dim=1)
                    
                    img_rot_flip = torch.flip(img_rot, dims=[3])
                    logits_flip = model(img_rot_flip)
                    batch_probs += torch.softmax(logits_flip, dim=1)
            
            batch_probs /= (len(models) * 8)
            all_probs.extend(batch_probs[:, 1].cpu().numpy())
            all_labels.extend(labels.numpy())

    # 4. 寻找最佳阈值
    best_threshold, best_acc, best_f1 = 0.5, 0.0, 0.0
    for t in np.arange(0.1, 0.9, 0.01):
        preds = [1 if p > t else 0 for p in all_probs]
        acc = accuracy_score(all_labels, preds)
        if acc > best_acc:
            best_acc, best_f1, best_threshold = acc, f1_score(all_labels, preds), t

    # 5. 输出结果
    print(f"\n====== Final Benchmark Results ({len(models)} Models) ======")
    print(f"Best Threshold: {best_threshold:.3f}")
    print(f"Accuracy:       {best_acc:.4f}")
    print(f"F1 Score:       {best_f1:.4f}")
    
    final_preds = [1 if p > best_threshold else 0 for p in all_probs]
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, final_preds))

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