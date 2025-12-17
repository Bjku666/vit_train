"""inference.py

按你的冠军级策略实现两大核心功能：

功能 1：搜索最佳阈值（Accuracy 最大化）
- 读取 5 个折的 oof_fold_X.csv（来自 train_vit.py 保存的 Preds/Targets）
- 在 [0.2, 0.8] 搜索 best_threshold

功能 2：生成提交（5折模型集成 + 8x TTA）
- 使用 best_threshold
- 5fold 模型对无标签测试集推理
- 每个模型做 8x TTA（4个旋转 * 是否水平翻转）
- 取概率均值 Avg_Prob，最终标签：Avg_Prob >= best_threshold -> 1 else 0

注意：本项目是二分类 BCEWithLogitsLoss，所以模型输出是单 logit，推理用 sigmoid 得到概率。
"""

import os
import argparse
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
from dataset import MedicalDataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

def parse_args():
    parser = argparse.ArgumentParser(description="Final Submission Inference Script")
    
    # 接收模型路径，支持通配符
    parser.add_argument('--model_paths', nargs='+', required=True, 
                        help="Path to model files. Use quotes for wildcards, e.g., 'models/run_xxx/vit_fold*.pth'")

    # OOF 文件目录或通配符（默认读 config.OOF_DIR 下的 oof_fold_*.csv）
    parser.add_argument('--oof_paths', nargs='+', default=[],
                        help="OOF csv paths (supports wildcards). Default: output/oof/oof_fold_*.csv")
                        
    return parser.parse_args()


def build_test_transform():
    # 推理保持与验证一致：只归一化，不做额外扰动
    return A.Compose([
        A.Normalize(mean=config.IMG_MEAN, std=config.IMG_STD),
        ToTensorV2(),
    ])


def load_oof(oof_paths):
    """读取并合并 5 折 OOF，用于阈值搜索。"""
    probs = []
    targets = []
    for p in oof_paths:
        df = pd.read_csv(p)
        if 'Preds' not in df.columns or 'Targets' not in df.columns:
            raise ValueError(f"OOF 文件列名必须包含 Preds/Targets，但在 {p} 中未找到")
        probs.append(df['Preds'].values.astype(np.float32))
        targets.append(df['Targets'].values.astype(np.int64))
    probs = np.concatenate(probs, axis=0)
    targets = np.concatenate(targets, axis=0)
    return probs, targets


def search_best_threshold(probs: np.ndarray, targets: np.ndarray) -> float:
    """在 [0.2,0.8] 搜索能最大化 Accuracy 的阈值。"""
    thresholds = np.linspace(0.2, 0.8, 601, dtype=np.float32)  # step=0.001
    best_t = 0.5
    best_acc = -1.0
    for t in thresholds:
        preds = (probs >= t).astype(np.int64)
        acc = (preds == targets).mean()
        # 若相同 acc，优先选择更“保守”的阈值（更小/更大都可，这里取更小，便于 recall）
        if acc > best_acc or (acc == best_acc and t < best_t):
            best_acc = acc
            best_t = float(t)
    print(f"[Threshold] best_threshold={best_t:.4f} | best_acc={best_acc:.6f} | n={len(targets)}")
    return best_t


def tta_8x(images: torch.Tensor):
    """8x TTA: 4 rotations * (no flip / hflip)
	
    images: (B,3,H,W)
    """
    outs = []
    for k in (0, 1, 2, 3):
        rot = torch.rot90(images, k=k, dims=[2, 3])
        outs.append(rot)
        outs.append(torch.flip(rot, dims=[3]))  # 水平翻转
    return outs

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

    # --- 2. 阈值搜索：读取 5 折 OOF ---
    oof_files = []
    if args.oof_paths:
        for p in args.oof_paths:
            oof_files.extend(glob.glob(p))
    else:
        oof_files.extend(glob.glob(os.path.join(config.OOF_DIR, 'oof_fold_*.csv')))

    oof_files = sorted(oof_files)
    if len(oof_files) < config.K_FOLDS:
        raise RuntimeError(
            f"未找到足够的 OOF 文件用于阈值搜索：找到 {len(oof_files)} 个，期望至少 {config.K_FOLDS} 个。"
            f"请检查 --oof_paths 或目录 {config.OOF_DIR}"
        )
    print(f"Found {len(oof_files)} OOF files for threshold search.")

    probs, targets = load_oof(oof_files[:config.K_FOLDS])
    best_threshold = search_best_threshold(probs, targets)

    # --- 3. 准备无标签测试集 ---
    # 使用与验证集一致的预处理：仅 Normalize
    test_transform = build_test_transform()
    # 【关键】读取无标签测试集，并将 mode 设为 'test'
    dataset = MedicalDataset(config.UNLABELED_TEST_DIR, mode='test', transform=test_transform)
    # Batch size 可以设大一点加速推理
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE * 2, shuffle=False, num_workers=4)

    # --- 4. 加载模型 ---
    models = []
    for path in model_files:
        print(f"  -> Loading {os.path.basename(path)}")
        m = get_model(config.MODEL_NAME, num_classes=config.NUM_CLASSES, pretrained=False)
        try:
            state_dict = torch.load(path, map_location=config.DEVICE, weights_only=True)
        except TypeError:
            state_dict = torch.load(path, map_location=config.DEVICE)
        if isinstance(state_dict, dict) and state_dict and next(iter(state_dict.keys())).startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        m.load_state_dict(state_dict)
        m.to(config.DEVICE)
        m.eval()
        models.append(m)

    # --- 5. 执行推理 (5fold + 8x TTA) ---
    predictions = []

    print("\nRunning inference on unlabeled test set (5fold ensemble + 8x TTA)...")
    with torch.no_grad():
        for images, filenames in tqdm(loader):
            images = images.to(config.DEVICE)

            # 累积所有模型与 TTA 的概率
            prob_sum = torch.zeros(images.size(0), device=config.DEVICE)
            tta_views = tta_8x(images)
            denom = float(len(models) * len(tta_views))

            for model in models:
                for view in tta_views:
                    logits = model(view)  # (B,) logit
                    prob_sum += torch.sigmoid(logits)

            avg_prob = (prob_sum / denom).detach().cpu().numpy()
            final_preds = (avg_prob >= best_threshold).astype(np.int64)

            for fname, label in zip(filenames, final_preds):
                predictions.append({"id": fname, "label": int(label)})

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

    # CONFIG_INIT_DIRS=0 会阻止 config.py 自动创建目录；推理需要显式确保输出目录存在
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # 保存为 CSV
    submission_df.to_csv(submission_path, index=False)
    
    print("\nSubmission file created successfully!")
    print(f"Total predictions: {len(submission_df)}")
    print(f"Saved to: {submission_path}")

if __name__ == '__main__':
    main()