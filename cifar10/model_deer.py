"""
Spikformer with DEER-compatible LIF neurons

复制并修改原始Spikformer，替换为DEER_LIF
"""
import torch
import torch.nn as nn
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import torch.nn.functional as F
from functools import partial
import sys
import os

# 导入DEER模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from deer_sps import DEER_SPS, DEER_LIF_Layer


class DEER_Block(nn.Module):
    """
    Spikformer Block with DEER_LIF
    
    与原始Block相同，但LIF节点替换为DEER_LIF_Layer
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = DEER_Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = DEER_MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DEER_Attention(nn.Module):
    """Attention with DEER_LIF"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125  # Spikformer默认值
        self.q_linear = nn.Linear(dim, dim)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = DEER_LIF_Layer(tau=2.0, theta_base=0.5)

        self.k_linear = nn.Linear(dim, dim)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = DEER_LIF_Layer(tau=2.0, theta_base=0.5)

        self.v_linear = nn.Linear(dim, dim)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = DEER_LIF_Layer(tau=2.0, theta_base=0.5)
        self.attn_lif = DEER_LIF_Layer(tau=2.0, theta_base=0.5)

        self.proj = nn.Linear(dim, dim)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = DEER_LIF_Layer(tau=2.0, theta_base=0.5)
        
        # SR (Spatial Reduction)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.sr_bn = nn.BatchNorm2d(dim)
            self.sr_lif = DEER_LIF_Layer(tau=2.0, theta_base=0.5)

    def forward(self, x):
        T, B, N, C = x.shape

        # Q
        q = self.q_linear(x.flatten(0, 1)).reshape(T, B, N, C).contiguous()
        q = self.q_bn(q.flatten(0, 1).transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
        q = self.q_lif(q)
        q = q.reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        # K, V (可能带SR)
        if self.sr_ratio > 1:
            x_ = x.reshape(T, B, int(N**0.5), int(N**0.5), C).permute(0, 1, 4, 2, 3).contiguous()
            x_ = self.sr(x_.flatten(0, 1))
            x_ = self.sr_bn(x_).reshape(T, B, C, -1).permute(0, 1, 3, 2).contiguous()
            x_ = self.sr_lif(x_)
        else:
            x_ = x

        # K
        k = self.k_linear(x_.flatten(0, 1)).reshape(T, B, -1, C).contiguous()
        k = self.k_bn(k.flatten(0, 1).transpose(-1, -2)).transpose(-1, -2).reshape(T, B, -1, C).contiguous()
        k = self.k_lif(k)
        k = k.reshape(T, B, -1, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        # V
        v = self.v_linear(x_.flatten(0, 1)).reshape(T, B, -1, C).contiguous()
        v = self.v_bn(v.flatten(0, 1).transpose(-1, -2)).transpose(-1, -2).reshape(T, B, -1, C).contiguous()
        v = self.v_lif(v)
        v = v.reshape(T, B, -1, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        x = attn @ v
        x = x.transpose(2, 3).reshape(T, B, N, C).contiguous()
        x = self.attn_lif(x)

        # Projection
        x = self.proj(x.flatten(0, 1)).reshape(T, B, N, C).contiguous()
        x = self.proj_bn(x.flatten(0, 1).transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
        x = self.proj_lif(x)

        return x


class DEER_MLP(nn.Module):
    """MLP with DEER_LIF"""
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_lif = DEER_LIF_Layer(tau=2.0, theta_base=0.5)

        self.fc2 = nn.Linear(hidden_features, out_features)
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_lif = DEER_LIF_Layer(tau=2.0, theta_base=0.5)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        T, B, N, C = x.shape
        x = self.fc1(x.flatten(0, 1))
        x = self.fc1_bn(x.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, -1).contiguous()
        x = self.fc1_lif(x)
        x = self.drop(x)

        x = self.fc2(x.flatten(0, 1))
        x = self.fc2_bn(x.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
        x = self.fc2_lif(x)
        x = self.drop(x)
        return x


class Spikformer_DEER(nn.Module):
    """
    Spikformer with DEER-compatible LIF neurons
    
    与原始Spikformer API完全兼容，只是内部LIF替换为DEER版本
    """
    def __init__(self,
                 img_size_h=128, img_size_w=128, patch_size=16, in_channels=2, num_classes=11,
                 embed_dims=[64, 128, 256], num_heads=[1, 2, 4], mlp_ratios=[4, 4, 4], qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[6, 8, 6], sr_ratios=[8, 4, 2], T=4):
        super().__init__()
        self.T = T
        self.num_classes = num_classes
        self.depths = depths

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]

        # Patch embedding with DEER
        patch_embed = DEER_SPS(img_size_h=img_size_h,
                               img_size_w=img_size_w,
                               patch_size=patch_size,
                               in_channels=in_channels,
                               embed_dims=embed_dims)

        # Transformer blocks with DEER
        block = nn.ModuleList([DEER_Block(
            dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer, sr_ratio=sr_ratios)
            for j in range(depths)])

        setattr(self, f"patch_embed", patch_embed)
        setattr(self, f"block", block)

        # Classification head
        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        block = getattr(self, f"block")
        patch_embed = getattr(self, f"patch_embed")

        x, (H, W) = patch_embed(x)
        
        # Flatten spatial dimensions: (T, B, C, H, W) -> (T, B, N, C)
        T, B, C, _, _ = x.shape
        x = x.flatten(3).transpose(2, 3)  # (T, B, C, H*W) -> (T, B, H*W, C)
        
        for blk in block:
            x = blk(x)
        return x.mean(2)  # 对N维度平均

    def forward(self, x):
        x = x.permute(1, 0, 2, 3, 4)  # (B, T, C, H, W) -> (T, B, C, H, W)
        x = self.forward_features(x)
        x = self.head(x.mean(0))  # batch维度平均
        return x


# 测试代码
if __name__ == "__main__":
    print("="*70)
    print("测试Spikformer_DEER完整模型")
    print("="*70)
    
    # 创建模型
    model = Spikformer_DEER(
        img_size_h=32, img_size_w=32,
        patch_size=4,
        in_channels=2,
        num_classes=10,
        embed_dims=256,
        num_heads=8,
        mlp_ratios=4,
        depths=2,  # 小模型测试
        sr_ratios=4,
        T=4
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.train()
    
    # 输入
    x = torch.randn(2, 4, 2, 32, 32).to(device)  # (B, T, C, H, W)
    
    # 前向
    print(f"\n输入shape: {x.shape}")
    out = model(x)
    print(f"输出shape: {out.shape}")
    print(f"输出范围: [{out.min().item():.3f}, {out.max().item():.3f}]")
    
    # 梯度
    loss = out.sum()
    loss.backward()
    print(f"\n✅ 梯度反传成功！")
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数:")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    
    print("\n" + "="*70)
    print("✅ Spikformer_DEER模型构建成功！")
    print("="*70)
