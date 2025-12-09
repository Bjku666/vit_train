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

# 1. 生成本次运行的唯一 ID
RUN_ID = datetime.now().strftime('%Y%m%d_%H%M%S')

# 2. 目录配置
CURRENT_RUN_MODELS_DIR = os.path.join(MODELS_DIR, f'run_{RUN_ID}')
CURRENT_LOG_DIR = os.path.join(LOG_DIR, RUN_ID)
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

# ===== 数据路径 (请确保这里是你真实的绝对路径) =====
# 建议填入绝对路径以防万一
TRAIN_DIRS = [
	os.path.join(DATA_DIR, '2-MedImage-TrainSet'),
	os.path.join(DATA_DIR, 'pseudo_labeled_set'),  # 伪标签数据根目录（按 disease/normal 组织）
]
LABELED_TEST_DIR = os.path.join(DATA_DIR, '2-MedImage-TestSet')
# 无标签测试集，用于生成伪标签
UNLABELED_TEST_DIR = os.path.join(DATA_DIR, 'MedImage-TestSet')

# ===== 模型与训练参数 (升级版) =====
# 使用支持 384 分辨率的权重
MODEL_NAME = 'vit_base_patch16_384.augreg_in21k_ft_in1k'
NUM_CLASSES = 2

# === 核心升级 1: 提升分辨率 ===
IMAGE_SIZE = 384  
IMG_MEAN = [0.5, 0.5, 0.5]
IMG_STD = [0.5, 0.5, 0.5]

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 16

TRAIN_SPLIT = 0.85
K_FOLDS = 5
# === 稳定收敛：延长训练并降低初始学习率 ===
EPOCHS = 60  # 延长到 60 轮，配合预热与 EMA 稳定收敛

# === 核心升级 1.1: 显存控制与梯度累积 ===
# 384分辨率下显存占用大，BatchSize 调小，通过 Accumulation 补回
BATCH_SIZE = 64       # 单次前向传播的图片数
ACCUM_STEPS = 1       # 梯度累积步数 (等效 Batch Size = 16 * 2 = 32)

BASE_LR = 1.5e-5        # 学习率下调，降低各 Fold 方差
WARMUP_EPOCHS = 10    # 新增线性预热轮数（按 step 级计算）
WEIGHT_DECAY = 0.05
CLIP_GRAD_NORM = 1.0

# ===== Mixup 强正则化 =====
USE_MIXUP = True
MIXUP_ALPHA = 0.8
CUTMIX_ALPHA = 1.0
MIXUP_PROB = 1.0
MIXUP_SWITCH_PROB = 0.5
LABEL_SMOOTHING = 0.1