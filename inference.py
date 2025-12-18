"""inference.py
功能升级版：
1. 支持 --only_eval 参数：仅读取 OOF 计算最佳阈值并打印，不进行后续推理（秒级反馈）。
2. 推理使用严格 2x TTA：仅原图 + 水平翻转（严禁 90° 旋转）。
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
    
    # 接收模型路径
    parser.add_argument('--model_paths', nargs='+', required=True, 
                        help="Path to model files. Use quotes for wildcards.")

    # OOF 文件路径 (默认读取 output/oof/oof_fold_*.csv)
    parser.add_argument('--oof_paths', nargs='+', default=[],
                        help="OOF csv paths (supports wildcards).")
    
    # [关键] 仅评测模式开关
    parser.add_argument('--only_eval', action='store_true', 
                        help="If set, only search best threshold based on OOF and exit.")
                        
    return parser.parse_args()


def build_test_transform():
    return A.Compose([
        A.Normalize(mean=config.IMG_MEAN, std=config.IMG_STD),
        ToTensorV2(),
    ])


def load_oof(oof_paths):
    """读取并合并 OOF CSV"""
    probs = []
    targets = []
    if not oof_paths:
        return np.array([]), np.array([])
        
    for p in oof_paths:
        df = pd.read_csv(p)
        if 'Preds' not in df.columns or 'Targets' not in df.columns:
            raise ValueError(f"OOF 文件列名必须包含 Preds/Targets，但在 {p} 中未找到")
        probs.append(df['Preds'].values.astype(np.float32))
        targets.append(df['Targets'].values.astype(np.int64))
    
    if not probs:
        return np.array([]), np.array([])
        
    probs = np.concatenate(probs, axis=0)
    targets = np.concatenate(targets, axis=0)
    return probs, targets


def search_best_threshold(probs: np.ndarray, targets: np.ndarray) -> float:
    """在 [0.2, 0.8] 搜索最佳阈值"""
    thresholds = np.linspace(0.2, 0.8, 601, dtype=np.float32)
    best_t = 0.5
    best_acc = -1.0
    
    # 简单的向量化计算加速
    for t in thresholds:
        preds = (probs >= t).astype(np.int64)
        acc = (preds == targets).mean()
        if acc > best_acc or (acc == best_acc and t < best_t):
            best_acc = acc
            best_t = float(t)
    
    print("\n" + "="*45)
    print(f" 📊 [OOF Evaluation] 内部验证集评测报告")
    print(f" ---------------------------------------------")
    print(f" 样本总数: {len(targets)}")
    print(f" 最佳阈值: {best_t:.4f}")
    print(f" 最佳 Acc: {best_acc:.6f}  (这是预期的上线)")
    print("="*45 + "\n")
    return best_t


def _load_solid_threshold():
    """从 output/best_threshold.json 读取固化阈值（由训练/评测脚本写入）。"""
    path = os.path.join(config.OUTPUT_DIR, 'best_threshold.json')
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_json(path)
        # 兼容 dict/json
        if isinstance(df, pd.Series):
            t = float(df.get('threshold', np.nan))
        else:
            # 不太可能走到这里
            t = float(df['threshold'].iloc[0])
        if np.isfinite(t):
            return t
    except Exception:
        pass
    try:
        import json
        with open(path, 'r', encoding='utf-8') as f:
            obj = json.load(f)
        t = float(obj.get('threshold', float('nan')))
        return t if np.isfinite(t) else None
    except Exception:
        return None


def _save_solid_threshold(best_threshold: float, oof_files: list[str]) -> None:
    """将 OOF 搜出来的阈值固化到 output/best_threshold.json，供后续推理直接复用。"""
    try:
        import json
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        payload = {
            'run_id': config.RUN_ID,
            'stage': int(getattr(config, 'CURRENT_STAGE', 0)),
            'threshold': float(best_threshold),
            'oof_files': [os.path.basename(p) for p in oof_files],
        }
        path_latest = os.path.join(config.OUTPUT_DIR, 'best_threshold.json')
        with open(path_latest, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        return


def tta_2x(images: torch.Tensor):
    """严格 2x TTA：仅原图 + 水平翻转（不做任何旋转）"""
    return [images, torch.flip(images, dims=[3])]


def tta_shift(images: torch.Tensor, shift_px: int):
    pad = torch.nn.functional.pad(images, (0, 0, 0, 0, shift_px, shift_px, shift_px, shift_px))
    views = []
    views.append(pad[:, :, shift_px*2:, shift_px:-shift_px])  # up
    views.append(pad[:, :, :-shift_px*2, shift_px:-shift_px])  # down
    views.append(pad[:, :, shift_px:-shift_px, shift_px*2:])  # left
    views.append(pad[:, :, shift_px:-shift_px, :-shift_px*2])  # right
    return views

def main():
    args = parse_args()
    
    # --- 1. 确定 OOF 文件 ---
    oof_files = []
    if args.oof_paths:
        for p in args.oof_paths:
            oof_files.extend(glob.glob(p))
    else:
        # 默认去 config 目录找
        oof_files.extend(glob.glob(os.path.join(config.OOF_DIR, 'oof_fold_*.csv')))

    oof_files = sorted(oof_files)
    best_threshold = None
    if not args.only_eval:
        best_threshold = _load_solid_threshold()
        if best_threshold is not None:
            print(f"[Info] 使用固化阈值 output/best_threshold.json: {best_threshold:.4f}")

    if best_threshold is None:
        if not oof_files:
            print(f"⚠️  警告: 未找到 OOF 文件，将使用默认阈值 0.5。请检查 {config.OOF_DIR}")
            best_threshold = 0.5
        else:
            print(f"[Info] 加载 {len(oof_files)} 个 OOF 文件进行阈值搜索...")
            probs, targets = load_oof(oof_files)
            if len(targets) == 0:
                print("⚠️  OOF文件为空，使用默认阈值0.5")
                best_threshold = 0.5
            else:
                best_threshold = search_best_threshold(probs, targets)
                if not args.only_eval:
                    _save_solid_threshold(best_threshold, oof_files)

    # === [核心逻辑] 如果只是评测，到这里就结束 ===
    if args.only_eval:
        print("✅ 评测完成 (--only_eval)。不执行推理提交。")
        return

    # --- 2. 正式推理流程 ---
    model_files = []
    for path_pattern in args.model_paths:
        model_files.extend(glob.glob(path_pattern))

    if not model_files:
        print(f"❌ Error: No model files found matching: {args.model_paths}")
        return
    
    print(f"🚀 开始执行推理，使用 {len(model_files)} 个模型...")
    print(f"📌 使用阈值: {best_threshold:.4f}")

    test_transform = build_test_transform()
    dataset = MedicalDataset(config.UNLABELED_TEST_DIR, mode='test', transform=test_transform)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE * 2, shuffle=False, num_workers=4)

    # 加载模型
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

    shift_tta = int(os.environ.get('SHIFT_TTA', '0')) == 1
    shift_px = int(os.environ.get('SHIFT_PX', '8'))

    # 推理
    predictions = []
    print("\nRunning inference on unlabeled test set (TTA: identity + hflip{} )...".format(" + shift" if shift_tta else ""))
    with torch.no_grad():
        for images, filenames in tqdm(loader):
            images = images.to(config.DEVICE)

            prob_sum = torch.zeros(images.size(0), device=config.DEVICE)
            tta_views = tta_2x(images)
            if shift_tta:
                tta_views.extend(tta_shift(images, shift_px))
                tta_views.extend([torch.flip(v, dims=[3]) for v in tta_views if v is not images])
            denom = float(len(models) * len(tta_views))

            for model in models:
                for view in tta_views:
                    logits = model(view)
                    prob_sum += torch.sigmoid(logits)

            avg_prob = (prob_sum / denom).detach().cpu().numpy()
            final_preds = (avg_prob >= best_threshold).astype(np.int64)

            for fname, label in zip(filenames, final_preds):
                predictions.append({"id": fname, "label": int(label)})

    # 保存
    submission_df = pd.DataFrame(predictions)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    submission_filename = f"submission_{timestamp}.csv"
    submission_path = os.path.join(config.OUTPUT_DIR, submission_filename)

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    submission_df.to_csv(submission_path, index=False)
    
    print(f"\n🎉 Submission file created: {submission_path}")

if __name__ == '__main__':
    main()