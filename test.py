import torch

def load_sd(path):
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model" in ckpt:
        sd = ckpt["model"]
    else:
        sd = ckpt
    # 去掉可能的 "module." 前缀
    new_sd = {}
    for k,v in sd.items():
        if k.startswith("module."):
            k = k[len("module."):]
        new_sd[k] = v
    return new_sd

def diff(sd1, sd2, key):
    a = sd1[key].float()
    b = sd2[key].float()
    return (a-b).abs().max().item(), torch.norm(a-b).item()

stage1 = load_sd("models/run_20251219_215519/swin_base_patch4_window7_224_stage1_epoch012.pth")
stage2 = load_sd("models/run_20251219_215519/swin_base_patch4_window7_224_stage2_best.pth")
pretr = load_sd("pretrained/swin_base_patch4_window7_224.pth")

# 选几层代表性的 key（你可以按你模型state_dict实际key改一下）
keys = [
    "head.weight",
    "head.bias",
    "patch_embed.proj.weight",
    "layers.0.blocks.0.attn.qkv.weight",
    "layers.3.blocks.1.mlp.fc2.weight",
]

for k in keys:
    if k not in stage1 or k not in stage2 or k not in pretr:
        print("missing key:", k)
        continue
    m12, l12 = diff(stage1, pretr, k)
    m21, l21 = diff(stage2, stage1, k)
    m2p, l2p = diff(stage2, pretr, k)
    print(f"{k:40s}  stage1-pretr max={m12:.6g}  stage2-stage1 max={m21:.6g}  stage2-pretr max={m2p:.6g}")
