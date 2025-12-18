import os
import torch
from datetime import datetime

# ===== 基础路径配置 =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 假设 dataset 在 personal 的上级目录 (根据你之前的调试修改)
PROJECT_ROOT = os.path.dirname(BASE_DIR) 
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
LOG_DIR = os.path.join(BASE_DIR, 'logs')

# ===== RETFound 权重路径 =====
# 指向本地 RETFound 权重文件（请确保文件存在且完整）
PRETRAINED_DIR = os.path.join(BASE_DIR, 'pretrained')
RETFOUND_PATH = os.path.join(PRETRAINED_DIR, 'RETFound_cfp_weights.pth')

# 其他预训练权重（如仍需使用 vit_base_384.bin）
VIT_BASE_384_PATH = os.path.join(PRETRAINED_DIR, 'vit_base_384.bin')

# 兼容旧路径：如果你仍然把权重放在 models/ 里，也能自动找到
if not os.path.exists(RETFOUND_PATH):
	_old_path = os.path.join(MODELS_DIR, 'RETFound_cfp_weights.pth')
	if os.path.exists(_old_path):
		RETFOUND_PATH = _old_path

if not os.path.exists(VIT_BASE_384_PATH):
	_old_path = os.path.join(MODELS_DIR, 'vit_base_384.bin')
	if os.path.exists(_old_path):
		VIT_BASE_384_PATH = _old_path

# 1. 生成本次运行的唯一 ID
# 说明：为了让 Stage1/Stage2 在 TensorBoard 上“无缝衔接”，允许你手动指定同一个 RUN_ID。
# 用法：
# - Stage1：RUN_ID=20251217_120000 CURRENT_STAGE=1 python train_vit.py
# - Stage2：RUN_ID=20251217_120000 CURRENT_STAGE=2 python train_vit.py --stage1_models_dir models/run_20251217_120000
RUN_ID = os.environ.get('RUN_ID', '').strip() or datetime.now().strftime('%Y%m%d_%H%M%S')

# 2. 目录配置
CURRENT_RUN_MODELS_DIR = os.path.join(MODELS_DIR, f'run_{RUN_ID}')
CURRENT_LOG_DIR = os.path.join(LOG_DIR, RUN_ID)
# 训练文本日志放在 logs/ 目录
TEXT_LOG_FILE = os.path.join(LOG_DIR, f'train_vit_{RUN_ID}.log')

# === 修复：确保 OUTPUT_DIR 存在 ===
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# ===== 自动创建目录 =====
# 评测脚本必须“只读”，因此提供可控开关，防止基准测试时误生成空目录
INIT_OUTPUT_DIRS = os.environ.get("CONFIG_INIT_DIRS", "1") == "1"
if INIT_OUTPUT_DIRS:
	os.makedirs(CURRENT_RUN_MODELS_DIR, exist_ok=True)
	os.makedirs(CURRENT_LOG_DIR, exist_ok=True)
	os.makedirs(LOG_DIR, exist_ok=True)
	os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===== 数据路径 =====
# 说明：你表示“伪标签过程可以保留，但现在不需要跑这一步”。
# 因此这里把伪标签数据做成可选开关：默认不参与训练；需要时再通过环境变量打开。

# 真实标注训练集
LABELED_TRAIN_DIR = os.path.join(DATA_DIR, '2-MedImage-TrainSet')

# 伪标签数据根目录（按 disease/normal 组织）
# 兼容两种常见位置：data/pseudo_labeled_set 或 rub/pseudo_labeled_set
PSEUDO_LABELED_DIR = os.path.join(DATA_DIR, 'pseudo_labeled_set')
_rub_pseudo = os.path.join(BASE_DIR, 'rub', 'pseudo_labeled_set')
if not os.path.exists(PSEUDO_LABELED_DIR) and os.path.exists(_rub_pseudo):
	PSEUDO_LABELED_DIR = _rub_pseudo

# 是否启用伪标签（默认关闭）
USE_PSEUDO_LABELS = os.environ.get('USE_PSEUDO_LABELS', '0') == '1'

TRAIN_DIRS = [LABELED_TRAIN_DIR]
if USE_PSEUDO_LABELS and os.path.exists(PSEUDO_LABELED_DIR):
	TRAIN_DIRS.append(PSEUDO_LABELED_DIR)
LABELED_TEST_DIR = os.path.join(DATA_DIR, '2-MedImage-TestSet')
# 无标签测试集，用于生成伪标签
UNLABELED_TEST_DIR = os.path.join(DATA_DIR, 'MedImage-TestSet')

# ===== 模型与训练参数 (冠军级策略：渐进式分辨率 Progressive Resizing) =====
# 说明：
# - 青光眼二分类最终以 Accuracy 排名为主，因此后续会做 OOF 阈值搜索。
# - 训练采用两阶段：先用较低分辨率稳定收敛，再用更高分辨率低学习率微调，保护 RETFound 迁移来的表征。

# 手动切换训练阶段：
# - Stage 1: 384, 30 epochs, LR=1e-5
# - Stage 2: 512, 10 epochs, LR=2e-6
CURRENT_STAGE = int(os.environ.get('CURRENT_STAGE', '1'))

# ViT-Large (RETFound 底座)
# 注意：这里定义 timm 模型名，实际输入分辨率由 IMAGE_SIZE 决定；model.py 内会进行位置编码插值。
MODEL_NAME = 'vit_large_patch16_384'

# 二分类：使用 BCEWithLogitsLoss，因此模型输出维度为 1（logit），标签为 0/1 float。
NUM_CLASSES = 1

# === Stage2 Solver 可配置开关与超参 ===
# LLRD 开关 + 衰减因子
USE_LLRD = os.environ.get('USE_LLRD', '1').lower() not in ['0', 'false']
LAYER_DECAY = float(os.environ.get('LAYER_DECAY', 0.75))

# Stage2 专用 LR/WD（不再用 1e-6“学不动”）
STAGE2_BASE_LR = float(os.environ.get('STAGE2_BASE_LR', 2e-5))
STAGE2_WEIGHT_DECAY = float(os.environ.get('STAGE2_WEIGHT_DECAY', 0.1))

# Stage2 冻结→解冻策略
FREEZE_PATCH_EMBED_STAGE2 = os.environ.get('FREEZE_PATCH_EMBED_STAGE2', '1').lower() not in ['0', 'false']
FREEZE_BLOCKS_BEFORE_STAGE2 = int(os.environ.get('FREEZE_BLOCKS_BEFORE_STAGE2', 12))
FREEZE_EPOCHS_STAGE2 = int(os.environ.get('FREEZE_EPOCHS_STAGE2', 2))

# EMA
USE_EMA = os.environ.get('USE_EMA', '1').lower() not in ['0', 'false']
EMA_DECAY = float(os.environ.get('EMA_DECAY', 0.9999))

# === 动态参数配置 (根据阶段自动调整) ===
if CURRENT_STAGE == 1:
	# Stage 1: 384 分辨率
	IMAGE_SIZE = 384
	EPOCHS = 30
	BASE_LR = 1e-5

	# 显存配置 (384px)
	# 单卡 4090 24G 通常可以稳定跑到 BS=8
	BATCH_SIZE = 8
	ACCUM_STEPS = 8        # 等效 BS = 8 * 8 = 64
	WEIGHT_DECAY = 0.05
elif CURRENT_STAGE == 2:
	# Stage 2: 512 分辨率 (显存压力显著增加)
	IMAGE_SIZE = 512
	EPOCHS = 10
	# 使用专门为 Stage2 设计的 solver（LLRD + 更高 base lr）
	BASE_LR = STAGE2_BASE_LR

	# 显存配置 (512px) - 建议调小 batch 防止 OOM
	BATCH_SIZE = 4
	ACCUM_STEPS = 16    # 等效 BS = 4 * 16 = 64
	WEIGHT_DECAY = STAGE2_WEIGHT_DECAY
else:
	raise ValueError(f"CURRENT_STAGE 只能是 1 或 2，但得到 {CURRENT_STAGE}")
IMG_MEAN = [0.5, 0.5, 0.5]
IMG_STD = [0.5, 0.5, 0.5]

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 16

TRAIN_SPLIT = 0.85
K_FOLDS = 5

# === 显存与梯度累积 ===
# BATCH_SIZE / ACCUM_STEPS 已在上方按 CURRENT_STAGE 动态指定。

# === 学习率设置 ===
# Stage1/Stage2 的 BASE_LR 已在上方按 CURRENT_STAGE 指定。
# 如果你后续仍想保留 warmup/cosine 等策略，可以在 train_vit.py 使用 WARMUP_EPOCHS。
WARMUP_EPOCHS = 0
CLIP_GRAD_NORM = 1.0

# ===== Stage 2 初始化：从 Stage 1 权重 warm-start =====
# 说明：
# - 当 CURRENT_STAGE==2 时，train_vit.py 会尝试加载 Stage 1 对应 fold 的最优权重作为初始化。
# - 你可以手动把 Stage 1 的模型目录填到这里，或通过命令行参数覆盖。
STAGE1_MODELS_DIR = os.environ.get('STAGE1_MODELS_DIR', '')

# OOF 输出目录（用于阈值搜索）
OOF_DIR = os.path.join(OUTPUT_DIR, 'oof')
if INIT_OUTPUT_DIRS:
	os.makedirs(OOF_DIR, exist_ok=True)

# ===== Mixup 强正则化 =====
USE_MIXUP = True
MIXUP_ALPHA = 0.8
CUTMIX_ALPHA = 1.0
MIXUP_PROB = 1.0
MIXUP_SWITCH_PROB = 0.5
LABEL_SMOOTHING = 0.1