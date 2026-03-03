"""
DEER-Spikformer: 集成DEER并行化的Spikformer模型

替换MLP层中的LIF为DEER-LIF，保持SSA层使用串行LIF
"""

import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import torch.nn.functional as F
from functools import partial
from torch.utils.checkpoint import checkpoint

# 导入DEER-LIF
from deer_lif_node import DEERLIFNode

__all__ = ['deer_spikformer']


class DEER_MLP(nn.Module):
    """
    使用DEER并行化LIF的MLP模块
    
    **关键修正 (2024-11-03)**:
    - DEER并行化的是TIME维度T，不是batch维度B！
    - 应将(T,B,N,C)合并为(T,B*N,C)一起处理
    - DEER-LIF在T维度上并行化递归求解
    - 移除所有micro-batch循环，避免串行化
    - 添加梯度检查点支持（用于T>8时减少内存）
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., 
                 deer_max_iter=10, deer_tol=1e-6, use_checkpoint=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.use_checkpoint = use_checkpoint
        
        self.fc1_linear = nn.Linear(in_features, hidden_features)
        self.fc1_bn = nn.BatchNorm1d(hidden_features)
        self.fc1_lif = DEERLIFNode(
            tau=2.0, 
            max_iter=deer_max_iter, 
            tol=deer_tol, 
            use_diagonal_approx=True,
            check_convergence=False  # 训练时不检查收敛，避免GPU-CPU同步
        )

        self.fc2_linear = nn.Linear(hidden_features, out_features)
        self.fc2_bn = nn.BatchNorm1d(out_features)
        self.fc2_lif = DEERLIFNode(
            tau=2.0, 
            max_iter=deer_max_iter, 
            tol=deer_tol, 
            use_diagonal_approx=True,
            check_convergence=False
        )

        self.c_hidden = hidden_features
        self.c_output = out_features

    def _forward_fc1(self, x_flat):
        """FC1-LIF的checkpoint包装函数"""
        return self.fc1_lif(x_flat)
    
    def _forward_fc2(self, x_flat):
        """FC2-LIF的checkpoint包装函数"""
        return self.fc2_lif(x_flat)
    
    def forward(self, x):
        """
        关键修正：
        1. 不再循环batch维度B和patch维度N
        2. 将(T,B,N,C)展平为(T,B*N,C)一起处理
        3. DEER-LIF在T维度并行化，B*N作为batch一起处理
        4. 使用梯度检查点减少内存（T>8时启用）
        
        这才是DEER的正确用法：并行TIME STEPS，不是并行BATCH！
        """
        T, B, N, C = x.shape
        x_ = x.flatten(0, 1)  # (T*B, N, C)
        
        # FC1
        x = self.fc1_linear(x_)
        x = self.fc1_bn(x.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, self.c_hidden).contiguous()
        
        # **关键修正**: 将B和N合并，一起处理
        # (T, B, N, C) -> (T, B*N, C)
        x_flat = x.reshape(T, B * N, self.c_hidden)
        
        # DEER-LIF: 在T维度并行化！（使用checkpoint减少内存）
        if not hasattr(self, '_fc1_lif_called'):
            print(f"[DEER-MLP] FC1-LIF层: T={T}, B*N={B*N}, C={self.c_hidden}, checkpoint={self.use_checkpoint}")
            self._fc1_lif_called = True
        
        if self.use_checkpoint:
            x_flat = checkpoint(self._forward_fc1, x_flat, use_reentrant=False)
        else:
            x_flat = self.fc1_lif(x_flat)
        
        if not hasattr(self, '_fc1_lif_done'):
            print(f"[DEER-MLP] FC1-LIF完成")
            self._fc1_lif_done = True
        
        # 恢复形状
        x = x_flat.reshape(T, B, N, self.c_hidden)
        
        # FC2
        x = self.fc2_linear(x.flatten(0, 1))
        x = self.fc2_bn(x.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, self.c_output).contiguous()
        
        # **关键修正**: 再次将B和N合并
        x_flat = x.reshape(T, B * N, self.c_output)
        
        # DEER-LIF: 在T维度并行化！（使用checkpoint减少内存）
        if not hasattr(self, '_fc2_lif_called'):
            print(f"[DEER-MLP] FC2-LIF层: T={T}, B*N={B*N}, C={self.c_output}")
            self._fc2_lif_called = True
        
        if self.use_checkpoint:
            x_flat = checkpoint(self._forward_fc2, x_flat, use_reentrant=False)
        else:
            x_flat = self.fc2_lif(x_flat)
        
        if not hasattr(self, '_fc2_lif_done'):
            print(f"[DEER-MLP] FC2-LIF完成")
            self._fc2_lif_done = True
        
        # 恢复形状
        x = x_flat.reshape(T, B, N, self.c_output)
        
        return x


class SSA(nn.Module):
    """
    保持原始的串行SSA (脉冲率高，暂不并行化)
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.scale = 0.125
        self.q_linear = nn.Linear(dim, dim)
        self.q_bn = nn.BatchNorm1d(dim)
        self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

        self.k_linear = nn.Linear(dim, dim)
        self.k_bn = nn.BatchNorm1d(dim)
        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

        self.v_linear = nn.Linear(dim, dim)
        self.v_bn = nn.BatchNorm1d(dim)
        self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='torch')

        self.proj_linear = nn.Linear(dim, dim)
        self.proj_bn = nn.BatchNorm1d(dim)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

    def forward(self, x):
        T, B, N, C = x.shape

        x_for_qkv = x.flatten(0, 1)  # TB, N, C
        q_linear_out = self.q_linear(x_for_qkv)  # [TB, N, C]
        q_linear_out = self.q_bn(q_linear_out.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
        q_linear_out = self.q_lif(q_linear_out)
        q = q_linear_out.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        k_linear_out = self.k_linear(x_for_qkv)
        k_linear_out = self.k_bn(k_linear_out.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
        k_linear_out = self.k_lif(k_linear_out)
        k = k_linear_out.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        v_linear_out = self.v_linear(x_for_qkv)
        v_linear_out = self.v_bn(v_linear_out.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
        v_linear_out = self.v_lif(v_linear_out)
        v = v_linear_out.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

        attn = (q @ k.transpose(-2, -1)) * self.scale
        x = attn @ v
        x = x.transpose(2, 3).reshape(T, B, N, C).contiguous()
        x = self.attn_lif(x)
        x = x.flatten(0, 1)
        x = self.proj_lif(self.proj_bn(self.proj_linear(x).transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C))
        return x


class DEER_Block(nn.Module):
    """使用DEER_MLP的Block"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1, deer_max_iter=10, deer_tol=1e-6, use_checkpoint=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = SSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = DEER_MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop,
                           deer_max_iter=deer_max_iter, deer_tol=deer_tol, use_checkpoint=use_checkpoint)

    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class SPS(nn.Module):
    """保持原始的SPS (Spiking Patch Splitting)"""
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2, embed_dims=256):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj_conv = nn.Conv2d(in_channels, embed_dims//8, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims//8)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

        self.proj_conv1 = nn.Conv2d(embed_dims//8, embed_dims//4, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn1 = nn.BatchNorm2d(embed_dims//4)
        self.proj_lif1 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

        self.proj_conv2 = nn.Conv2d(embed_dims//4, embed_dims//2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn2 = nn.BatchNorm2d(embed_dims//2)
        self.proj_lif2 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.proj_conv3 = nn.Conv2d(embed_dims//2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn3 = nn.BatchNorm2d(embed_dims)
        self.proj_lif3 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.rpe_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.rpe_bn = nn.BatchNorm2d(embed_dims)
        self.rpe_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='torch')

    def forward(self, x):
        T, B, C, H, W = x.shape
        x = self.proj_conv(x.flatten(0, 1))
        x = self.proj_bn(x).reshape(T, B, -1, H, W).contiguous()
        x = self.proj_lif(x).flatten(0, 1).contiguous()

        x = self.proj_conv1(x)
        x = self.proj_bn1(x).reshape(T, B, -1, H, W).contiguous()
        x = self.proj_lif1(x).flatten(0, 1).contiguous()

        x = self.proj_conv2(x)
        x = self.proj_bn2(x).reshape(T, B, -1, H, W).contiguous()
        x = self.proj_lif2(x).flatten(0, 1).contiguous()
        x = self.maxpool2(x)

        x = self.proj_conv3(x)
        x = self.proj_bn3(x).reshape(T, B, -1, H//2, W//2).contiguous()
        x = self.proj_lif3(x).flatten(0, 1).contiguous()
        x = self.maxpool3(x)

        x_feat = x.reshape(T, B, -1, H//4, W//4).contiguous()
        x = self.rpe_conv(x)
        x = self.rpe_bn(x).reshape(T, B, -1, H//4, W//4).contiguous()
        x = self.rpe_lif(x)
        x = x + x_feat

        x = x.flatten(-2).transpose(-1, -2)  # T,B,N,C
        return x


class DEERSpikformer(nn.Module):
    """
    DEER-Spikformer: 集成DEER并行化的Spikformer
    
    创新点：
    1. MLP层使用DEER-LIF并行化 (脉冲率低，适合DEER)
    2. SSA层保持串行 (脉冲率高，暂不并行)
    3. 兼容原始Spikformer的接口
    """
    def __init__(self,
                 img_size_h=128, img_size_w=128, patch_size=16, in_channels=2, num_classes=11,
                 embed_dims=[64, 128, 256], num_heads=[1, 2, 4], mlp_ratios=[4, 4, 4], qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[6, 8, 6], sr_ratios=[8, 4, 2], T=4,
                 deer_max_iter=10, deer_tol=1e-6, use_checkpoint=False):
        super().__init__()
        self.T = T  # time step
        self.num_classes = num_classes
        self.depths = depths
        self.deer_max_iter = deer_max_iter
        self.deer_tol = deer_tol
        self.use_checkpoint = use_checkpoint

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule

        patch_embed = SPS(img_size_h=img_size_h,
                         img_size_w=img_size_w,
                         patch_size=patch_size,
                         in_channels=in_channels,
                         embed_dims=embed_dims)

        block = nn.ModuleList([DEER_Block(
            dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer, sr_ratio=sr_ratios,
            deer_max_iter=deer_max_iter, deer_tol=deer_tol, use_checkpoint=use_checkpoint)
            for j in range(depths)])

        setattr(self, f"patch_embed", patch_embed)
        setattr(self, f"block", block)

        # classification head
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

        x = patch_embed(x)
        for blk in block:
            x = blk(x)
        return x.mean(2)

    def forward(self, x):
        x = (x.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)
        x = self.forward_features(x)
        x = self.head(x.mean(0))
        return x
    
    def get_deer_stats(self):
        """获取所有DEER-LIF节点的统计信息"""
        stats = []
        for block in self.block:
            if hasattr(block.mlp, 'fc1_lif'):
                stats.append({
                    'layer': 'fc1_lif',
                    'stats': block.mlp.fc1_lif.get_stats_summary()
                })
            if hasattr(block.mlp, 'fc2_lif'):
                stats.append({
                    'layer': 'fc2_lif',
                    'stats': block.mlp.fc2_lif.get_stats_summary()
                })
        return stats


@register_model
def deer_spikformer(pretrained=False, **kwargs):
    """DEER-Spikformer模型构造函数"""
    # 移除 timm 传递的额外参数
    kwargs.pop('pretrained_cfg', None)
    kwargs.pop('pretrained_cfg_overlay', None)
    
    model = DEERSpikformer(**kwargs)
    model.default_cfg = _cfg()
    return model
