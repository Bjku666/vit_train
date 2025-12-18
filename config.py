import os
from datetime import datetime
import torch
# =============================
# 路径配置
# =============================
# =============================
# RETFound（仅用于 ViT 主干）
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
# 运行 / 阶段
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
# 注意：benchmark/inference 默认不应创建目录

# RETFound (only used for ViT backbones)
RETFOUND_PATH = os.path.join(PRETRAINED_DIR, "RETFound_cfp_weights.pth")
# =============================
# 数据目录
# =============================
# =============================
# 有标签训练集：期望结构 data/2-MedImage-TrainSet/{disease,normal}/*.jpg
CURRENT_STAGE = int(os.environ.get("CURRENT_STAGE", "1"))
# 可选伪标签集（默认关闭）
CURRENT_RUN_MODELS_DIR = os.path.join(MODELS_DIR, f"run_{RUN_ID}")
CURRENT_LOG_DIR = os.path.join(LOG_DIR, RUN_ID)
TEXT_LOG_FILE = os.path.join(LOG_DIR, f"train_{RUN_ID}.log")
# =============================
# 模型
# =============================
if INIT_OUTPUT_DIRS:
# 在此切换主干：
    os.makedirs(CURRENT_LOG_DIR, exist_ok=True)
# - Swin（推荐基线）：swin_base_patch4_window12_384
    os.makedirs(OUTPUT_DIR, exist_ok=True)
# - ViT（RETFound）：vit_base_patch16_384 / vit_large_patch16_384 ...
OOF_DIR = os.path.join(OUTPUT_DIR, "oof")
# 二分类，BCEWithLogitsLoss，对应输出 1 个 logit
    os.makedirs(OOF_DIR, exist_ok=True)
# 可选：使用 GeM 池化（适用于 ViT 类 token 模型）
# =============================
# Data dirs
# =============================
# =============================
# 划分策略
# =============================
UNLABELED_TEST_DIR = os.path.join(DATA_DIR, "MedImage-TestSet")  # unlabeled testset (for submission / pseudo label)
# 目前默认单划分，可切换 KFold。
# Optional pseudo-labeled set (disabled by default)
USE_PSEUDO_LABELS = os.environ.get("USE_PSEUDO_LABELS", "0").lower() in ["1", "true", "yes", "y"]
PSEUDO_LABELED_DIR = os.path.join(DATA_DIR, "pseudo_labeled_set")
# =============================
# 训练基础参数
# =============================
    TRAIN_DIRS.append(PSEUDO_LABELED_DIR)
# 渐进式分辨率
# =============================
    # Stage2 微调用的 solver
# =============================
# Switch backbone here:
# - Swin (recommended baseline): swin_base_patch4_window12_384
# =============================
# LLRD / Stage2 冻结配置
# =============================
# Binary classification with BCEWithLogitsLoss -> output 1 logit
# 冻结→解冻（Stage2）

# 对 ViT：冻结前 N 个 block；对 Swin：可按 stage/block 冻结前段。
USE_GEM = os.environ.get("USE_GEM", "0").lower() in ["1", "true", "yes", "y"]

# =============================
# =============================
# EMA
# =============================
USE_KFOLD = os.environ.get("USE_KFOLD", "0").lower() in ["1", "true", "yes", "y"]
# Mixup（可选，建议适度以免影响边界）
VAL_RATIO = float(os.environ.get("VAL_RATIO", "0.2"))

# =============================
# =============================
# 图像归一化
# =============================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# 预处理开关（Ben Graham / CLAHE）

# Progressive resizing
if CURRENT_STAGE == 1:
# =============================
# 检查点
# =============================
    WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "0.05"))
# 每个 epoch 保存，便于后续尝试“非最佳”权重。
else:
    IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", "512"))
    EPOCHS = int(os.environ.get("EPOCHS", "8"))
    # Stage2 solver (fine-tune)
    BASE_LR = float(os.environ.get("BASE_LR", "1e-5"))
    WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "0.1"))
    BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "4"))

ACCUM_STEPS = int(os.environ.get("ACCUM_STEPS", "1"))
CLIP_GRAD_NORM = float(os.environ.get("CLIP_GRAD_NORM", "1.0"))

# =============================
# LLRD / Freeze for stage2
# =============================
USE_LLRD = os.environ.get("USE_LLRD", "1").lower() not in ["0", "false"]
LAYER_DECAY = float(os.environ.get("LAYER_DECAY", "0.75"))

# Freeze → unfreeze (stage2)
FREEZE_EPOCHS_STAGE2 = int(os.environ.get("FREEZE_EPOCHS_STAGE2", "2"))
# For ViT: freeze first N blocks; for Swin: freeze first N stages (0-3) + early blocks in stage if desired.
FREEZE_BLOCKS_BEFORE_STAGE2 = int(os.environ.get("FREEZE_BLOCKS_BEFORE_STAGE2", "12"))
FREEZE_PATCH_EMBED_STAGE2 = os.environ.get("FREEZE_PATCH_EMBED_STAGE2", "1").lower() not in ["0", "false"]

# =============================
# EMA
# =============================
USE_EMA = os.environ.get("USE_EMA", "1").lower() not in ["0", "false"]
EMA_DECAY = float(os.environ.get("EMA_DECAY", "0.9999"))

# =============================
# Mixup (optional; you can keep it light for glaucoma)
# =============================
USE_MIXUP = os.environ.get("USE_MIXUP", "0").lower() in ["1", "true", "yes", "y"]
MIXUP_ALPHA = float(os.environ.get("MIXUP_ALPHA", "0.8"))
MIXUP_PROB = float(os.environ.get("MIXUP_PROB", "0.3"))
MIXUP_DISABLE_LAST_EPOCHS = int(os.environ.get("MIXUP_DISABLE_LAST_EPOCHS", "2"))

# =============================
# Image normalization
# =============================
IMG_MEAN = (0.485, 0.456, 0.406)
IMG_STD = (0.229, 0.224, 0.225)

# =============================
# Preprocessing toggles (Ben Graham / CLAHE)
# =============================
USE_CLAHE = os.environ.get("USE_CLAHE", "1").lower() not in ["0", "false"]
CLAHE_CLIPLIMIT = float(os.environ.get("CLAHE_CLIPLIMIT", "2.0"))
CLAHE_TILEGRIDSIZE = int(os.environ.get("CLAHE_TILEGRIDSIZE", "8"))

# =============================
# Checkpoints
# =============================
# Save each epoch so you can try "non-best" epochs on the leaderboard.
SAVE_EVERY_EPOCH = os.environ.get("SAVE_EVERY_EPOCH", "1").lower() not in ["0", "false"]
KEEP_LAST_N_EPOCHS = int(os.environ.get("KEEP_LAST_N_EPOCHS", "999"))
