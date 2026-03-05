
import torch
import torch.nn as nn

def test_manual_inflation():
    print("Testing Manual Inflation Logic...")
    
    # 模拟 ImageNet 预训练权重 (2D)
    # Shape: [Out=192, In=3, KH=16, KW=16]
    C_out, C_in, KH, KW = 192, 3, 16, 16
    weight_2d = torch.randn(C_out, C_in, KH, KW)
    
    # 模拟 VideoMamba 模型中的 PatchEmbed 权重 (3D)
    # Shape: [Out=192, In=3, KT=2, KH=16, KW=16]
    KT = 2
    weight_3d_target_shape = (C_out, C_in, KT, KH, KW)
    
    # ---------------------------------------------------------
    # 复制自 run_class_finetuning.py 的逻辑
    # ---------------------------------------------------------
    print(f"Inflating patch_embed.proj.weight from {weight_2d.shape} to {weight_3d_target_shape}")
    
    # 获取时间维度 T
    time_dim = weight_3d_target_shape[2] # 2
    
    # 执行中心帧初始化 (Center Frame Initialization)
    weight_3d = torch.zeros(weight_3d_target_shape)
    
    # 将 2D 权重放置在时间维度的中间位置
    # time_dim // 2 = 1. So index 1 is filled. Index 0 is zero.
    center_idx = time_dim // 2
    print(f"Center Index: {center_idx}")
    
    weight_3d[:, :, center_idx, :, :] = weight_2d
    
    # ---------------------------------------------------------
    # 验证
    # ---------------------------------------------------------
    # 检查是否成功赋值
    slice_center = weight_3d[:, :, center_idx, :, :]
    print(f"Center slice equals 2D weight? {torch.equal(slice_center, weight_2d)}")
    
    if center_idx > 0:
        slice_zero = weight_3d[:, :, 0, :, :]
        print(f"Index 0 slice is all zeros? {torch.all(slice_zero == 0)}")
        print(f"Max value in Index 0: {slice_zero.max().item()}")
    
    print(f"Weight 3D Max: {weight_3d.max().item()}")
    print(f"Weight 3D Min: {weight_3d.min().item()}")
    print(f"Weight 3D Mean: {weight_3d.mean().item()}")
    
    # 测试 Mean Inflation (均值膨胀)
    print("\nTesting Mean Inflation Logic (Alternative)...")
    weight_3d_mean = weight_2d.unsqueeze(2).repeat(1, 1, KT, 1, 1) / KT
    print(f"Mean Inflation Max: {weight_3d_mean.max().item()}")
    print(f"Mean Inflation Sum of temporal dim equals 2D? {torch.allclose(weight_3d_mean.sum(dim=2), weight_2d, atol=1e-6)}")

if __name__ == "__main__":
    test_manual_inflation()
