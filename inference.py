# inference.py (最终提交脚本)
"""
功能：使用多模型 8x TTA 对无标签测试集推理，按 benchmark JSON 阈值生成提交文件。
用法示例：
    CONFIG_INIT_DIRS=0 python inference.py \
        --model_paths "models/run_xxx/vit_fold*.pth" \
        --benchmark_json output/benchmark_result_YYYYMMDD_HHMMSS.json
输出：
    output/submission_YYYYMMDD_HHMMSS.csv
依赖：benchmark JSON 提供最佳阈值；测试集路径 config.UNLABELED_TEST_DIR。
"""

import os
import argparse
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import glob
from datetime import datetime

# 在导入 config 之前关闭目录初始化开关
os.environ['CONFIG_INIT_DIRS'] = '0'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import config
from model import get_model
from dataset import MedicalDataset, val_transform_alb

def parse_args():
    parser = argparse.ArgumentParser(description="Final Submission Inference Script")
    
    # 接收模型路径，支持通配符
    parser.add_argument('--model_paths', nargs='+', required=True, 
                        help="Path to model files. Use quotes for wildcards, e.g., 'models/run_xxx/vit_fold*.pth'")
    
    # 接收 benchmark.json 文件路径，用于自动读取最佳阈值
    parser.add_argument('--benchmark_json', type=str, required=True,
                        help="Path to the benchmark_result_xxx.json file to get the best threshold.")
                        
    return parser.parse_args()

def main():
    args = parse_args()
    
    # --- 1. 智能处理模型路径 ---
    model_files = []
    for path_pattern in args.model_paths:
        model_files.extend(glob.glob(path_pattern))
    
    if not model_files:
        print(f"Error: No model files found matching pattern: {args.model_paths}")
        return
    print(f"Found {len(model_files)} models for inference.")

    # --- 2. 从 JSON 文件中自动读取最佳阈值 ---
    try:
        with open(args.benchmark_json, 'r') as f:
            benchmark_data = json.load(f)
        best_threshold = benchmark_data['threshold']
        print(f"Successfully loaded best threshold: {best_threshold:.4f} from {args.benchmark_json}")
    except Exception as e:
        print(f"Error loading benchmark JSON file: {e}. Please check the path.")
        return

    # --- 3. 准备无标签测试集 ---
    # 使用与验证集完全相同的预处理
    test_transform = val_transform_alb 
    # 【关键】读取无标签测试集，并将 mode 设为 'test'
    dataset = MedicalDataset(config.UNLABELED_TEST_DIR, mode='test', transform=test_transform)
    # Batch size 可以设大一点加速推理
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE * 2, shuffle=False, num_workers=4)

    # --- 4. 加载模型 ---
    models = []
    for path in model_files:
        print(f"  -> Loading {os.path.basename(path)}")
        m = get_model(config.MODEL_NAME, num_classes=config.NUM_CLASSES, pretrained=False)
        state_dict = torch.load(path, map_location=config.DEVICE, weights_only=True)
        if list(state_dict.keys())[0].startswith('module.'):
            new_state_dict = {k[7:]: v for k, v in state_dict.items()}
            state_dict = new_state_dict
        m.load_state_dict(state_dict)
        m.to(config.DEVICE)
        m.eval()
        models.append(m)

    # --- 5. 执行推理 (8x TTA) ---
    predictions = []
    
    print("\nRunning Inference on unlabeled test set with 8x TTA...")
    with torch.no_grad():
        # 【关键】loader 现在返回 (images, filenames)
        for images, filenames in tqdm(loader):
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
            
            # 根据最佳阈值生成 0/1 标签
            final_preds = (batch_probs[:, 1] > best_threshold).int().cpu().numpy()
            
            # 记录文件名和对应的预测标签
            for fname, label in zip(filenames, final_preds):
                predictions.append({"id": fname, "label": label})

    # --- 6. 生成 submission.csv 文件 ---
    if not predictions:
        print("No predictions were generated. Check your test set directory.")
        return

    # 创建 DataFrame
    submission_df = pd.DataFrame(predictions)
    
    # 生成带时间戳的文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    submission_filename = f"submission_{timestamp}.csv"
    submission_path = os.path.join(config.OUTPUT_DIR, submission_filename)
    
    # 保存为 CSV
    submission_df.to_csv(submission_path, index=False)
    
    print(f"\n🎉 Submission file created successfully!")
    print(f"   Total predictions: {len(submission_df)}")
    print(f"   Saved to: {submission_path}")

if __name__ == '__main__':
    main()