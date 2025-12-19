import os
from datetime import datetime

import torch


# =============================
# 路径配置
# =============================
# 以当前文件为基准定位项目目录结构
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")  # 数据目录（期望包含 disease/normal 子文件夹）
MODELS_DIR = os.path.join(BASE_DIR, "models")  # 模型权重输出目录
LOG_DIR = os.path.join(BASE_DIR, "logs")  # 日志目录（含 TensorBoard）
OUTPUT_DIR = os.path.join(BASE_DIR, "output")  # 推理/评测输出目录
PRETRAINED_DIR = os.path.join(BASE_DIR, "pretrained")  # 预训练权重目录

# RETFound 预训练权重（仅 ViT 主干会用到）
RETFOUND_PATH = os.path.join(PRETRAINED_DIR, "RETFound_cfp_weights.pth")


# =============================
# 运行 ID 与阶段（Stage1/Stage2）
# =============================
RUN_ID = os.environ.get("RUN_ID", "").strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
CURRENT_STAGE = int(os.environ.get("CURRENT_STAGE", "1"))

CURRENT_RUN_MODELS_DIR = os.path.join(MODELS_DIR, f"run_{RUN_ID}")
CURRENT_LOG_DIR = os.path.join(LOG_DIR, RUN_ID)
TEXT_LOG_FILE = os.path.join(LOG_DIR, f"train_{RUN_ID}.log")

# 注意：评测/推理时默认也创建输出目录，便于落盘
INIT_OUTPUT_DIRS = os.environ.get("CONFIG_INIT_DIRS", "1") == "1"
if INIT_OUTPUT_DIRS:
    os.makedirs(CURRENT_RUN_MODELS_DIR, exist_ok=True)
    os.makedirs(CURRENT_LOG_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

# OOF 目录仅在需要时创建（默认关闭）
OOF_DIR = os.path.join(OUTPUT_DIR, "oof")
CREATE_OOF_DIR = os.environ.get("CREATE_OOF", "0").lower() in ["1", "true", "yes", "y"]
if INIT_OUTPUT_DIRS and CREATE_OOF_DIR:
    os.makedirs(OOF_DIR, exist_ok=True)


# =============================
# 数据集目录（自动兼容你的 data 结构）
# =============================
def _first_existing_dir(base: str, candidates):
    """在 base 下按顺序返回第一个存在的目录。"""
    for name in candidates:
        p = os.path.join(base, name)
        if os.path.isdir(p):
            return p
    return os.path.join(base, candidates[0])  # 若都不存在，回退到首选项（便于后续报错信息统一）

# 训练集可能被命名为 TrainSet 或 TestSet，这里做健壮处理
_TRAIN_CANDIDATES = [
    os.environ.get("TRAIN_DIR_NAME", "2-MedImage-TrainSet"),
    "MedImage-TrainSet",
    "2-MedImage-TestSet",   # 你的当前目录名
    "MedImage-TestSet",
]

LABELED_TRAIN_DIR = _first_existing_dir(DATA_DIR, _TRAIN_CANDIDATES)

# 推理/提交时所用的无标签测试集目录（同样做名称兼容）
_TEST_CANDIDATES = [
    os.environ.get("TEST_DIR_NAME", "MedImage-TestSet"),
    "2-MedImage-TestSet",
    "MedImage-TrainSet",
    "2-MedImage-TrainSet",
]
UNLABELED_TEST_DIR = _first_existing_dir(DATA_DIR, _TEST_CANDIDATES)

# 可同时接入多个训练目录（例如加入伪标签集），默认只包含标注训练集
TRAIN_DIRS = [LABELED_TRAIN_DIR]


# =============================
# 模型设置
# =============================
# 主干模型：
# - Swin（推荐基线）：swin_base_patch4_window12_384
# - ViT（RETFound）：vit_base_patch16_384 / vit_large_patch16_384 等
"""Project config.

NOTE (important for Swin):
`swin_*_window12_384` models in timm are often hard-bound to img_size=384.
Even if you pass `img_size=...`, some internal asserts / mask logic can still
break when you feed 448/480/512.

For a stable 2-stage pipeline (stage1=224 -> stage2=448), use a window7-224
Swin as default:
  - stage1: 224
  - stage2: 448
These sizes are multiples of 224, and they keep all Swin stages aligned.
"""

MODEL_NAME = os.environ.get("MODEL_NAME", "swin_base_patch4_window7_224")

# 二分类（BCEWithLogitsLoss）：输出 1 维 logit
NUM_CLASSES = 1

# 可选：是否使用 GeM 池化（主要针对 ViT 类 token 模型）
USE_GEM = os.environ.get("USE_GEM", "0").lower() in ["1", "true", "yes", "y"]

# 可选：本地 Swin 权重路径（避免联网下载）
SWIN_CHECKPOINT_PATH = os.environ.get(
    "SWIN_CHECKPOINT_PATH", os.path.join(PRETRAINED_DIR, f"{MODEL_NAME}.pth")
)


# =============================
# 划分策略
# =============================
# 默认单划分；通过环境变量开启 K 折
USE_KFOLD = os.environ.get("USE_KFOLD", "0").lower() in ["1", "true", "yes", "y"]
N_SPLITS = int(os.environ.get("N_SPLITS", "5"))
VAL_RATIO = float(os.environ.get("VAL_RATIO", "0.2"))


# =============================
# 训练基础参数
# =============================
SEED = int(os.environ.get("SEED", "42"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "8"))

# 固定单划分的落盘位置（按 seed / val_ratio 区分）
SPLIT_DIR = os.path.join(BASE_DIR, "splits")
SPLIT_FILE = os.path.join(SPLIT_DIR, f"single_split_seed{SEED}_val{int(VAL_RATIO*100)}.json")

# 渐进分辨率
if CURRENT_STAGE == 1:
    # Stage 1 (warmup stage): train at 224.
    IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", "224"))
    EPOCHS = int(os.environ.get("EPOCHS", "25"))
    BASE_LR = float(os.environ.get("BASE_LR", "9e-5"))
    WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "0.05"))
    BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))
else:
    # Stage 2 (hi-res finetune):
    # For Swin window7/patch4, 448 (=224*2) is the safest choice.
    IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", "448"))
    EPOCHS = int(os.environ.get("EPOCHS", "15"))
    # 第二阶段短训微调
    BASE_LR = float(os.environ.get("BASE_LR", "1e-5"))
    WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "0.05"))
    BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))

ACCUM_STEPS = int(os.environ.get("ACCUM_STEPS", "1"))
CLIP_GRAD_NORM = float(os.environ.get("CLIP_GRAD_NORM", "1.0"))


# =============================
# LLRD / 第二阶段冻结策略
# =============================
USE_LLRD = os.environ.get("USE_LLRD", "1").lower() not in ["0", "false"]
LAYER_DECAY = float(os.environ.get("LAYER_DECAY", "0.9"))

# 冻结 → 解冻（仅 Stage2 生效）
FREEZE_EPOCHS_STAGE2 = int(os.environ.get("FREEZE_EPOCHS_STAGE2", "0"))
# ViT：冻结前 N 个 blocks；Swin：按 stage.blocks 展平后的前 N 个块
FREEZE_BLOCKS_BEFORE_STAGE2 = int(os.environ.get("FREEZE_BLOCKS_BEFORE_STAGE2", "0"))
FREEZE_PATCH_EMBED_STAGE2 = os.environ.get("FREEZE_PATCH_EMBED_STAGE2", "0").lower() not in ["0", "false"]


# =============================
# EMA 设置
# =============================
USE_EMA = os.environ.get("USE_EMA", "1").lower() not in ["0", "false"]
EMA_DECAY = float(os.environ.get("EMA_DECAY", "0.9999"))


# =============================
# Mixup（可选；青光眼数据建议保守使用）
# =============================
USE_MIXUP = os.environ.get("USE_MIXUP", "0").lower() in ["1", "true", "yes", "y"]
MIXUP_ALPHA = float(os.environ.get("MIXUP_ALPHA", "0.8"))
MIXUP_PROB = float(os.environ.get("MIXUP_PROB", "0.3"))
MIXUP_DISABLE_LAST_EPOCHS = int(os.environ.get("MIXUP_DISABLE_LAST_EPOCHS", "2"))


# =============================
# 图像归一化
# =============================
IMG_MEAN = (0.485, 0.456, 0.406)
IMG_STD = (0.229, 0.224, 0.225)


# =============================
# 预处理开关（Ben Graham / CLAHE）
# =============================
USE_CLAHE = os.environ.get("USE_CLAHE", "1").lower() not in ["0", "false"]
CLAHE_CLIPLIMIT = float(os.environ.get("CLAHE_CLIPLIMIT", "2.0"))
CLAHE_TILEGRIDSIZE = int(os.environ.get("CLAHE_TILEGRIDSIZE", "8"))


# =============================
# 模型保存
# =============================
# 每个 epoch 都保存，便于后续挑选“非最佳”权重
SAVE_EVERY_EPOCH = os.environ.get("SAVE_EVERY_EPOCH", "1").lower() not in ["0", "false"]
KEEP_LAST_N_EPOCHS = int(os.environ.get("KEEP_LAST_N_EPOCHS", "999"))
