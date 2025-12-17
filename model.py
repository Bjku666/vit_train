import timm
import torch
import torch.nn as nn
import os
import config

def get_model(model_name, num_classes=2, pretrained=False):
    """
    RETFound 专用构建函数:
    1. 创建 ViT-Large-384 骨架
    2. 加载 RETFound-224 权重
    3. 自动执行 Positional Embedding 插值 (224 -> 384)
    """
    print(f"[Model] 初始化 RETFound (ViT-Large) for {num_classes} classes...")
    
    # 1. 创建骨架 (不加载 timm 的 ImageNet 权重)
    model = timm.create_model(
        config.MODEL_NAME,        # 'vit_large_patch16_384'
        pretrained=False,
        num_classes=num_classes,
        drop_path_rate=0.2,       # 增加一点 drop_path 防止过拟合
    )
    
    # 2. 加载本地 RETFound 权重
    weight_path = config.RETFOUND_PATH
    if not os.path.exists(weight_path):
        print(f"❌ 错误: 找不到 RETFound 权重文件: {weight_path}")
        print("请确保已下载 RETFound_cfp_weights.pth 并放入 models 目录。")
        raise FileNotFoundError("RETFound weights missing")
        
    print(f"[Model] 正在加载 RETFound 权重: {os.path.basename(weight_path)}")
    checkpoint = torch.load(weight_path, map_location='cpu', weights_only=False)
    
    # 处理不同的权重格式 (有些是 {'model': ...}, 有些直接是 state_dict)
    checkpoint_model = checkpoint['model'] if 'model' in checkpoint else checkpoint

    # --- 核心黑魔法: 位置编码插值 (Interpolation) ---
    # 解决 RETFound(224) -> 当前模型(384) 的尺寸不匹配问题
    state_dict = model.state_dict()
    pos_embed_key = 'pos_embed'
    
    if pos_embed_key in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model[pos_embed_key]
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_extra_tokens = 1 # cls token
        
        # 计算原版 (224) 和 目标 (384) 的 grid size
        # 224 / 16 = 14 (orig_size)
        # 384 / 16 = 24 (new_size)
        orig_size = int((pos_embed_checkpoint.shape[1] - num_extra_tokens) ** 0.5)
        new_size = int((state_dict[pos_embed_key].shape[1] - num_extra_tokens) ** 0.5)
        
        if orig_size != new_size:
            print(f"[Info] 执行位置编码插值: {orig_size}x{orig_size} -> {new_size}x{new_size}")
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            
            # Reshape -> Permute -> Interpolate -> Permute -> Flatten
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False
            )
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            
            # 合并并更新权重
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model[pos_embed_key] = new_pos_embed
    
    # --- 移除不匹配的 Head ---
    # RETFound 预训练时的分类头通常不匹配，直接丢弃
    for k in list(checkpoint_model.keys()):
        if 'head' in k or 'fc' in k:
            del checkpoint_model[k]

    # 3. 加载权重 (strict=False 允许忽略 head)
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(f"[Success] 权重加载完成。Missing keys (主要是head): {len(msg.missing_keys)}")
    
    return model