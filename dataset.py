import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    _HAS_ALB = True
except Exception:
    A = None
    ToTensorV2 = None
    _HAS_ALB = False
import config  # 导入配置以获取 target size

def get_transforms():
    """集中定义 train/val/test 变换，供 train/inference 复用。"""
    if not _HAS_ALB:
        return None, None, None

    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        # 使用 Affine 并采用新版本参数名：border_mode/value，避免 mode/cval 警告
        A.Affine(
            translate_percent={"x": (-0.03, 0.03), "y": (-0.03, 0.03)},
            scale=(0.95, 1.05),
            rotate=(-10, 10),
            shear=(-5, 5),
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            p=0.7,
        ),
        A.RandomBrightnessContrast(brightness_limit=0.10, contrast_limit=0.10, p=0.5),
        A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=8, val_shift_limit=5, p=0.3),
        A.Normalize(mean=config.IMG_MEAN, std=config.IMG_STD),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        A.Normalize(mean=config.IMG_MEAN, std=config.IMG_STD),
        ToTensorV2(),
    ])

    # 测试/推理沿用验证流程，保持归一化一致
    test_transform = val_transform

    return train_transform, val_transform, test_transform

def ben_graham_preprocessing(image, target_size=config.IMAGE_SIZE):
    """
    Ben Graham 预处理流程 (Kaggle 金牌标准):
    1. 自动切除眼球周围的黑边 (Circle Crop)
    2. Resize 到目标尺寸
    3. CLAHE (限制对比度自适应直方图均衡化) 增强纹理
    """
    # 将 PIL 转为 Numpy (RGB)
    img = np.array(image)
    
    # --- 步骤 1: 自动裁剪黑边 ---
    # 转灰度
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    
    # 二值化找前景 (阈值设为 7 过滤极暗背景)
    mask = gray > 7
    
    # 检查是否有前景，如果没有则返回原图
    if mask.sum() == 0:
        return image
        
    # 获取前景的边界框
    coords = np.argwhere(mask)
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1   
    
    # 裁剪
    img_cropped = img[x0:x1, y0:y1]
    
    # --- 步骤 2: Resize ---
    # 在做 CLAHE 之前 Resize，可以大幅节省计算时间
    img_resized = cv2.resize(img_cropped, (target_size, target_size))

    # --- 步骤 3: Ben Graham 高斯模糊差分，sigma 随分辨率缩放 ---
    sigma = max(float(config.BEN_GRAHAM_MIN_SIGMA), float(target_size) / float(config.BEN_GRAHAM_SIGMA_DIVISOR))
    blurred = cv2.GaussianBlur(img_resized, (0, 0), sigmaX=sigma, sigmaY=sigma)
    img_resized = cv2.addWeighted(img_resized, 4.0, blurred, -4.0, 128)
    img_resized = np.clip(img_resized, 0, 255).astype(np.uint8)

    # --- 步骤 4: Ben Graham 方法 (LAB + CLAHE) ---
    # 转到 LAB 空间
    lab = cv2.cvtColor(img_resized, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # 对 L 通道 (亮度) 应用 CLAHE
    # CLAHE 会放大局部对比度，也可能放大噪声：提供开关与强度可控
    if config.USE_CLAHE:
        clahe = cv2.createCLAHE(
            clipLimit=float(config.CLAHE_CLIPLIMIT),
            tileGridSize=(int(config.CLAHE_TILEGRIDSIZE), int(config.CLAHE_TILEGRIDSIZE)),
        )
        l = clahe.apply(l)
    
    # 合并并转回 RGB
    lab = cv2.merge((l, a, b))
    img_final = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    return Image.fromarray(img_final)

class MedicalDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None, indices=None, image_paths=None):
        """
        root_dir: 可为单个字符串或字符串列表，支持原始数据 + 伪标签数据的多源读取。
        """
        self.root_dir = root_dir
        self.mode = mode
        self.transform = transform

        if image_paths is not None:
            self.image_paths = list(image_paths)
        else:
            # 支持多数据源：将单个路径统一转为列表遍历
            root_list = root_dir if isinstance(root_dir, (list, tuple)) else [root_dir]
            self.image_paths = []
            for rd in root_list:
                for root, dirs, files in os.walk(rd):
                    for file in files:
                        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                            self.image_paths.append(os.path.join(root, file))
        self.image_paths.sort()
        
        # 允许传入 indices 进行划分
        if indices is None:
            self.indices = list(range(len(self.image_paths)))
        else:
            self.indices = list(indices)
        
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img_path = self.image_paths[real_idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            # === 核心改动: 应用 Ben Graham 预处理 ===
            image = ben_graham_preprocessing(image, target_size=config.IMAGE_SIZE)
            img_np = np.array(image)
            if self.transform:
                transformed = self.transform(image=img_np)
                image = transformed["image"]
            else:
                image = torch.from_numpy(np.array(image).transpose(2,0,1)).float() / 255.0

            # 标签处理：严格用父目录名判断
            if self.mode in ['train', 'val']:
                parent = os.path.basename(os.path.dirname(img_path)).lower()
                label = 1 if parent == 'disease' else 0
                return image, torch.tensor(label, dtype=torch.long)
            else:
                filename = os.path.basename(img_path)
                return image, filename
                
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            return torch.zeros(3, config.IMAGE_SIZE, config.IMAGE_SIZE), 0