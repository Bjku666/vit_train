"""
功能：为无标签测试集生成高置信度伪标签，复制到 data/pseudo_labeled_set/disease|normal。
核心策略：多模型 8x TTA 融合，与 benchmark_vit.py 一致。
用法示例（可调阈值与多卡推理）：
    CONFIG_INIT_DIRS=0 python generate_pseudo_labels.py \
        --model_paths "models/run_xxx/vit_fold*.pth" \
        --threshold 0.9 \
        --batch_size 64
输出：统计信息 + 复制文件到 pseudo_labeled_set；不创建训练日志目录。
"""
import os
import argparse
import shutil
from typing import List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# 关闭训练目录自动创建，确保脚本只读（仅会手动创建伪标签输出目录）
os.environ["CONFIG_INIT_DIRS"] = "0"
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import config
from model import get_model
from dataset import MedicalDataset, val_transform_alb

def parse_args():
    parser = argparse.ArgumentParser(description="Generate pseudo labels with high confidence")
    parser.add_argument(
        '--model_paths', nargs='+', required=True,
        help="Path(s) to trained fold models, e.g. models/run_xxx/vit_fold*.pth"
    )
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE, help='Inference batch size')
    parser.add_argument('--num_workers', type=int, default=config.NUM_WORKERS, help='DataLoader workers')
    parser.add_argument('--threshold', type=float, default=0.99, help='High confidence threshold for pseudo labels')
    return parser.parse_args()


def load_models(model_paths: List[str]):
    """
    加载多折模型；若有多卡则自动 DataParallel 以提升吞吐。
    先加载到单卡模型，再根据 GPU 数量包裹 DataParallel，避免 state_dict 前缀不匹配。
    """
    models = []
    gpu_count = torch.cuda.device_count()
    for path in model_paths:
        print(f"加载模型: {os.path.basename(path)}")
        model = get_model(config.MODEL_NAME, num_classes=config.NUM_CLASSES, pretrained=False)
        state = torch.load(path, map_location=config.DEVICE)
        # 兼容 DataParallel 的 'module.' 前缀
        if list(state.keys())[0].startswith('module.'):
            state = {k[7:]: v for k, v in state.items()}
        model.load_state_dict(state, strict=True)

        # 多卡自动 DataParallel
        if gpu_count > 1:
            model = torch.nn.DataParallel(model)

        model.to(config.DEVICE)
        model.eval()
        models.append(model)
    return models


def ensure_output_dirs(base_dir: str):
    disease_dir = os.path.join(base_dir, 'disease')
    normal_dir = os.path.join(base_dir, 'normal')
    os.makedirs(disease_dir, exist_ok=True)
    os.makedirs(normal_dir, exist_ok=True)
    return disease_dir, normal_dir


def main():
    args = parse_args()
    models = load_models(args.model_paths)

    # 无标签集：MedImage-TestSet
    dataset = MedicalDataset(config.UNLABELED_TEST_DIR, mode='test', transform=val_transform_alb)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    pseudo_root = os.path.join(config.DATA_DIR, 'pseudo_labeled_set')
    disease_dir, normal_dir = ensure_output_dirs(pseudo_root)

    total = 0
    high_conf = 0
    disease_count = 0
    normal_count = 0
    # 统计分布，便于调阈值
    bucket_095 = 0
    bucket_099 = 0

    rotations = [0, 1, 2, 3]
    print(f"开始伪标签生成：数据量 {len(dataset)}，模型数 {len(models)}，8x TTA，高置信度阈值 {args.threshold}")

    with torch.no_grad():
        for batch_idx, (images, filenames) in enumerate(tqdm(loader)):
            images = images.to(config.DEVICE)
            batch_size = images.size(0)
            total += batch_size

            # 累积所有模型 + TTA 的概率
            probs = torch.zeros(batch_size, config.NUM_CLASSES, device=config.DEVICE)
            for model in models:
                for k in rotations:
                    img_rot = torch.rot90(images, k=k, dims=[2, 3])
                    logits = model(img_rot)
                    probs += torch.softmax(logits, dim=1)

                    img_rot_flip = torch.flip(img_rot, dims=[3])
                    logits_flip = model(img_rot_flip)
                    probs += torch.softmax(logits_flip, dim=1)

            probs /= (len(models) * 8)
            probs_cpu = probs.cpu()

            # 将批次内每个样本映射回原始路径
            for i in range(batch_size):
                global_idx = batch_idx * args.batch_size + i
                img_path = dataset.image_paths[global_idx]
                filename = filenames[i]
                p1 = float(probs_cpu[i, 1])
                p0 = 1.0 - p1

                if p1 >= 0.95:
                    bucket_095 += 1
                if p1 >= 0.99:
                    bucket_099 += 1

                if p1 >= args.threshold:
                    # 高置信疾病
                    shutil.copy2(img_path, os.path.join(disease_dir, os.path.basename(filename)))
                    high_conf += 1
                    disease_count += 1
                elif p0 >= args.threshold:
                    # 高置信正常
                    shutil.copy2(img_path, os.path.join(normal_dir, os.path.basename(filename)))
                    high_conf += 1
                    normal_count += 1
                # 其他样本忽略

    print("\n===== 伪标签统计 =====")
    print(f"共处理: {total} 张")
    print(f"高质量伪标签: {high_conf} 张 (Disease: {disease_count}, Normal: {normal_count})")
    print(f"分布参考: p>=0.95 的阳性预测 {bucket_095} 张；p>=0.99 的阳性预测 {bucket_099} 张")
    print(f"伪标签已保存至: {pseudo_root}")


if __name__ == "__main__":
    main()
