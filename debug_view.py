import cv2
import numpy as np
from PIL import Image
import os
import glob

# === 这里直接复制了 dataset.py 里的逻辑，完全独立运行 ===
def ben_graham_preprocessing(image, target_size=384):
    img = np.array(image)
    
    # 1. 自动裁剪黑边
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img
    
    # 关键点：这里阈值是 7。如果图本身很暗，mask 可能全为 False
    mask = gray > 7
    if mask.sum() == 0:
        print("⚠️ 警告：整张图过暗，被判定为全黑！")
        return image # 返回原图
        
    coords = np.argwhere(mask)
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1   
    img_cropped = img[x0:x1, y0:y1]
    
    # 2. Resize
    img_resized = cv2.resize(img_cropped, (target_size, target_size))
    
    # 3. CLAHE
    lab = cv2.cvtColor(img_resized, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img_final = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    return Image.fromarray(img_final)

# === 主程序 ===
# 请把这里的路径改成你那个“带标签测试集”里的任意一张图的路径
TEST_IMG_PATH = "data/MedImage-TestSet/6.jpg" 
# 或者用通配符自动找一张
if not os.path.exists(TEST_IMG_PATH):
    # 尝试自动找一张图
    search_path = "data/2-MedImage-TestSet/*/*.png"
    files = glob.glob(search_path)
    if files:
        TEST_IMG_PATH = files[0]
    else:
        print("❌ 找不到测试图片，请手动修改代码里的 TEST_IMG_PATH")
        exit()

print(f"正在诊断图片: {TEST_IMG_PATH}")

# 1. 加载原图
orig_img = Image.open(TEST_IMG_PATH).convert('RGB')
orig_img.save("debug_original.jpg")
print(f"✅ 原图已保存为 debug_original.jpg")

# 2. 执行预处理
proc_img = ben_graham_preprocessing(orig_img, target_size=384)
proc_img.save("debug_processed.jpg")
print(f"✅ 处理后的图已保存为 debug_processed.jpg")

# 3. 统计像素（判断是否全黑）
arr = np.array(proc_img)
print(f"📊 统计信息: Min像素={arr.min()}, Max像素={arr.max()}, Mean={arr.mean():.2f}")

if arr.mean() < 10:
    print("\n🚨🚨🚨 诊断结果：图黑了！预处理有问题！🚨🚨🚨")
else:
    print("\n✅ 诊断结果：图看起来挺亮，可能是其他原因（如TTA）。")