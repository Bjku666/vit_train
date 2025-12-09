import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import config  # 导入配置以获取 target size

# === 专业级数据增强：Albumentations 版 ===
# 说明：在 Ben Graham 预处理后进一步做几何与噪声扰动，缓解域偏移
train_transform_alb = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=20, p=0.7),
    A.RandomBrightnessContrast(p=0.7),
    A.GaussNoise(p=0.2),
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ToTensorV2(),
])

# 验证 / 测试仅做归一化，保持分布稳定
val_transform_alb = A.Compose([
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ToTensorV2(),
])

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
    
    # --- 步骤 3: Ben Graham 方法 (LAB + CLAHE) ---
    # 转到 LAB 空间
    lab = cv2.cvtColor(img_resized, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # 对 L 通道 (亮度) 应用 CLAHE
    # ClipLimit=2.0 是经验值，既增强血管又不过分放大噪声
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
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
            # 这会把不同医院、不同光照的图统一成“橙色调、高对比度”的标准图
            image = ben_graham_preprocessing(image, target_size=config.IMAGE_SIZE)
            # Albumentations 期望 numpy -> dict 输入
            img_np = np.array(image)
            if self.transform:
                transformed = self.transform(image=img_np)
                image = transformed["image"]
            else:
                # 兜底：无增强时使用最简单的张量转换
                image = transforms.ToTensor()(image)

            # 标签处理
            if self.mode in ['train', 'val']:
                if 'disease' in img_path.lower():
                    label = 1
                else:
                    label = 0
                return image, torch.tensor(label, dtype=torch.long)
            else:
                filename = os.path.basename(img_path)
                return image, filename
                
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # 返回全黑图防止 Crash
            return torch.zeros(3, config.IMAGE_SIZE, config.IMAGE_SIZE), 0