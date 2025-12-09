import timm
import torch
import torch.nn as nn
import os
import config

def get_model(model_name, num_classes=2, pretrained=True, drop_path_rate=0.1):
    """
    通用模型构建函数 (修正版):
    - 优先加载本地权重。
    - 自动处理分类头不匹配的问题 (迁移学习核心)。
    """
    print(f"[Model] Creating {model_name} for {num_classes} classes...")
    
    # 1. 先创建我们自己任务的模型 (num_classes=2)
    #    pretrained=False 确保它不会自己去联网下载
    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=num_classes,
        drop_path_rate=drop_path_rate,
    )
    
    # 定义本地权重路径
    local_weight_path = os.path.join(config.MODELS_DIR, 'vit_base_384.bin')
    
    # 2. 如果本地有权重文件，就加载它
    if os.path.exists(local_weight_path):
        print(f"[Info] Found local weights: {local_weight_path}, performing transfer learning...")
        
        # 加载预训练模型的 state_dict
        state_dict = torch.load(local_weight_path, map_location='cpu')

        # 检查是否是 timm 的 checkpoint 格式
        if 'model' in state_dict:
            state_dict = state_dict['model']

        # --- 核心修复：处理分类头不匹配 ---
        # 1. 获取我们自己模型的 state_dict
        model_dict = model.state_dict()
        
        # 2. 筛选出预训练模型中，除了分类头之外的所有权重
        #    我们只加载那些 key 存在于我们模型中，并且尺寸也匹配的权重
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        
        # 3. 更新我们模型的权重
        model_dict.update(pretrained_dict)
        
        # 4. 将更新后的权重加载回模型
        #    strict=True 确保所有我们期望的权重都正确加载了
        model.load_state_dict(model_dict, strict=True)
        
        print(f"[Success] Loaded {len(pretrained_dict)} matching layers from checkpoint.")
        
    elif pretrained:
        # 如果没有本地文件，并且要求 pretrained=True，才尝试联网
        print(f"[Warn] Local weights not found, trying to download from Hugging Face Hub...")
        # 这里会利用 timm 的内置逻辑自动处理分类头
        model = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    
    return model