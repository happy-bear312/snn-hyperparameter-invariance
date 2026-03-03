"""
DEER-compatible Spiking Neuron for Spikformer

将Accumulator-LIF集成到Spikformer架构中
"""
import torch
import torch.nn as nn
import sys
import os

# 添加父目录到path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from accumulator_lif import AccumulatorLIF


class DEER_LIF_Layer(nn.Module):
    """
    DEER-compatible LIF layer，兼容Multi Step接口
    
    输入: (T, B, C, H, W) 或 (T, B, F)
    输出: 同shape
    """
    
    def __init__(self, tau=2.0, theta_base=0.5, detach_reset=True, backend='torch'):
        """
        Args:
            tau: 膜时间常数
            theta_base: 基础阈值
            detach_reset: 兼容参数（DEER不需要reset）
            backend: 兼容参数
        """
        super().__init__()
        self.tau = tau
        self.theta_base = theta_base
        self.detach_reset = detach_reset  # 保留以兼容接口
        self.backend = backend
        
        # 创建AccumulatorLIF（作为nn.Module的一部分）
        self.lif_model = AccumulatorLIF(
            tau=self.tau,
            theta_base=self.theta_base
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: (T, B, ...) 任意shape
        
        Returns:
            spike: 同shape
        """
        original_shape = x.shape
        T, B = x.shape[0], x.shape[1]
        
        # Reshape到3D: (T, B, F)
        x_flat = x.reshape(T, B, -1)
        
        # 前向计算（使用并行版本）
        spike_flat, _ = self.lif_model.forward_parallel(x_flat)
        
        # Reshape回原shape
        spike = spike_flat.reshape(original_shape)
        
        return spike


class DEER_SPS(nn.Module):
    """
    DEER版本的Spiking Patch Splitting
    
    与原版SPS相同的结构，但使用DEER_LIF_Layer代替MultiStepLIFNode
    """
    
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2, embed_dims=256):
        super().__init__()
        from timm.models.layers import to_2tuple
        
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        
        # 第一层
        self.proj_conv = nn.Conv2d(in_channels, embed_dims//8, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims//8)
        self.proj_lif = DEER_LIF_Layer(tau=2.0, theta_base=0.5)
        
        # 第二层
        self.proj_conv1 = nn.Conv2d(embed_dims//8, embed_dims//4, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn1 = nn.BatchNorm2d(embed_dims//4)
        self.proj_lif1 = DEER_LIF_Layer(tau=2.0, theta_base=0.5)
        
        # 第三层
        self.proj_conv2 = nn.Conv2d(embed_dims//4, embed_dims//2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn2 = nn.BatchNorm2d(embed_dims//2)
        self.proj_lif2 = DEER_LIF_Layer(tau=2.0, theta_base=0.5)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        
        # 第四层
        self.proj_conv3 = nn.Conv2d(embed_dims//2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn3 = nn.BatchNorm2d(embed_dims)
        self.proj_lif3 = DEER_LIF_Layer(tau=2.0, theta_base=0.5)
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        
        # RPE层
        self.rpe_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.rpe_bn = nn.BatchNorm2d(embed_dims)
        self.rpe_lif = DEER_LIF_Layer(tau=2.0, theta_base=0.5)
    
    def forward(self, x):
        """
        Args:
            x: (T, B, C, H, W)
        Returns:
            output: (T, B, embed_dims, H//4, W//4) + RPE
        """
        T, B, C, H, W = x.shape
        
        # 第一层
        x = self.proj_conv(x.flatten(0, 1))
        x = self.proj_bn(x).reshape(T, B, -1, H, W).contiguous()
        x = self.proj_lif(x).flatten(0, 1).contiguous()
        
        # 第二层
        x = self.proj_conv1(x)
        x = self.proj_bn1(x).reshape(T, B, -1, H, W).contiguous()
        x = self.proj_lif1(x).flatten(0, 1).contiguous()
        
        # 第三层
        x = self.proj_conv2(x)
        x = self.proj_bn2(x).reshape(T, B, -1, H, W).contiguous()
        x = self.proj_lif2(x).flatten(0, 1).contiguous()
        x = self.maxpool2(x)
        
        # 第四层
        x = self.proj_conv3(x)
        x = self.proj_bn3(x).reshape(T, B, -1, H//2, W//2).contiguous()
        x = self.proj_lif3(x).flatten(0, 1).contiguous()
        x = self.maxpool3(x)
        
        # RPE层
        x_feat = x.reshape(T, B, -1, H//4, W//4).contiguous()
        x = self.rpe_conv(x)
        x = self.rpe_bn(x).reshape(T, B, -1, H//4, W//4).contiguous()
        x = self.rpe_lif(x)
        x = x + x_feat
        
        H, W = H // self.patch_size[0], W // self.patch_size[1]
        return x, (H, W)


# 测试代码
if __name__ == "__main__":
    print("="*70)
    print("测试DEER_LIF_Layer")
    print("="*70)
    
    torch.manual_seed(42)
    T, B, C, H, W = 4, 2, 32, 8, 8
    
    # 创建layer
    layer = DEER_LIF_Layer(tau=2.0, theta_base=0.5)
    layer.train()
    
    # 测试5D输入
    x_5d = torch.randn(T, B, C, H, W, requires_grad=True) * 0.5
    out_5d = layer(x_5d)
    
    print(f"\n5D输入测试:")
    print(f"  输入shape: {x_5d.shape}")
    print(f"  输出shape: {out_5d.shape}")
    print(f"  输出范围: [{out_5d.min().item():.3f}, {out_5d.max().item():.3f}]")
    print(f"  脉冲率: {out_5d.mean().item():.4f}")
    
    # 测试梯度
    loss = out_5d.sum()
    loss.backward()
    print(f"  ✅ 梯度反传成功")
    
    # 测试3D输入
    layer2 = DEER_LIF_Layer(tau=2.0, theta_base=0.5)
    layer2.train()
    
    x_3d = torch.randn(T, B, 256, requires_grad=True) * 0.5
    out_3d = layer2(x_3d)
    
    print(f"\n3D输入测试:")
    print(f"  输入shape: {x_3d.shape}")
    print(f"  输出shape: {out_3d.shape}")
    print(f"  脉冲率: {out_3d.mean().item():.4f}")
    
    loss2 = out_3d.sum()
    loss2.backward()
    print(f"  ✅ 梯度反传成功")
    
    print("\n" + "="*70)
    print("测试DEER_SPS完整模块")
    print("="*70)
    
    # 创建SPS
    sps = DEER_SPS(img_size_h=32, img_size_w=32, patch_size=4, in_channels=2, embed_dims=256)
    sps.train()
    sps = sps.cuda() if torch.cuda.is_available() else sps
    
    # 输入
    x_in = torch.randn(4, 2, 2, 32, 32) * 0.5
    x_in = x_in.cuda() if torch.cuda.is_available() else x_in
    x_in.requires_grad = True
    
    # 前向
    out, (H, W) = sps(x_in)
    
    print(f"\nSPS输出:")
    print(f"  输入shape: {x_in.shape}")
    print(f"  输出shape: {out.shape}")
    print(f"  输出(H, W): ({H}, {W})")
    print(f"  脉冲率: {out.mean().item():.4f}")
    
    # 梯度
    loss3 = out.sum()
    loss3.backward()
    print(f"  ✅ 完整SPS梯度反传成功！")
    
    print("\n" + "="*70)
    print("✅ 所有测试通过！DEER_SPS可以集成到Spikformer")
    print("="*70)
