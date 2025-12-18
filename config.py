import os
from datetime import datetime
import torch

# =============================
# Paths
# =============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOG_DIR = os.path.join(BASE_DIR, "logs")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
PRETRAINED_DIR = os.path.join(BASE_DIR, "pretrained")

# RETFound (only used for ViT backbones)
RETFOUND_PATH = os.path.join(PRETRAINED_DIR, "RETFound_cfp_weights.pth")

# =============================
# Run / stage
# =============================
RUN_ID = os.environ.get("RUN_ID", "").strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
CURRENT_STAGE = int(os.environ.get("CURRENT_STAGE", "1"))

CURRENT_RUN_MODELS_DIR = os.path.join(MODELS_DIR, f"run_{RUN_ID}")
CURRENT_LOG_DIR = os.path.join(LOG_DIR, RUN_ID)
TEXT_LOG_FILE = os.path.join(LOG_DIR, f"train_{RUN_ID}.log")

# IMPORTANT: benchmark/inference should not create folders by default
INIT_OUTPUT_DIRS = os.environ.get("CONFIG_INIT_DIRS", "1") == "1"
if INIT_OUTPUT_DIRS:
    os.makedirs(CURRENT_RUN_MODELS_DIR, exist_ok=True)
    os.makedirs(CURRENT_LOG_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

OOF_DIR = os.path.join(OUTPUT_DIR, "oof")
if INIT_OUTPUT_DIRS:
    os.makedirs(OOF_DIR, exist_ok=True)

# =============================
# Data dirs
# =============================
# Labeled train set: expect subfolders like data/2-MedImage-TrainSet/{disease,normal}/*.jpg
LABELED_TRAIN_DIR = os.path.join(DATA_DIR, "2-MedImage-TrainSet")
LABELED_TEST_DIR = os.path.join(DATA_DIR, "2-MedImage-TestSet")  # labeled testset (for local benchmarking)
UNLABELED_TEST_DIR = os.path.join(DATA_DIR, "MedImage-TestSet")  # unlabeled testset (for submission / pseudo label)

# Optional pseudo-labeled set (disabled by default)
USE_PSEUDO_LABELS = os.environ.get("USE_PSEUDO_LABELS", "0").lower() in ["1", "true", "yes", "y"]
PSEUDO_LABELED_DIR = os.path.join(DATA_DIR, "pseudo_labeled_set")

TRAIN_DIRS = [LABELED_TRAIN_DIR]
if USE_PSEUDO_LABELS and os.path.exists(PSEUDO_LABELED_DIR):
    TRAIN_DIRS.append(PSEUDO_LABELED_DIR)

# =============================
# Model
# =============================
# Switch backbone here:
# - Swin (recommended baseline): swin_base_patch4_window12_384
# - ViT (RETFound): vit_base_patch16_384 / vit_large_patch16_384 ...
MODEL_NAME = os.environ.get("MODEL_NAME", "swin_base_patch4_window12_384")

# Binary classification with BCEWithLogitsLoss -> output 1 logit
NUM_CLASSES = 1

# Optional: use GeM pooling (only for ViT-style token models)
USE_GEM = os.environ.get("USE_GEM", "0").lower() in ["1", "true", "yes", "y"]

# =============================
# Split strategy
# =============================
# You said you want to keep a single split instead of k-fold.
USE_KFOLD = os.environ.get("USE_KFOLD", "0").lower() in ["1", "true", "yes", "y"]
N_SPLITS = int(os.environ.get("N_SPLITS", "5"))
VAL_RATIO = float(os.environ.get("VAL_RATIO", "0.2"))

# =============================
# Training basics
# =============================
SEED = int(os.environ.get("SEED", "42"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "8"))

# Progressive resizing
if CURRENT_STAGE == 1:
    IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", "384"))
    EPOCHS = int(os.environ.get("EPOCHS", "10"))
    BASE_LR = float(os.environ.get("BASE_LR", "2e-5"))  # Swin/Vit-base friendly
    WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "0.05"))
    BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "8"))
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
