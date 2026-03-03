"""
τ/θ 超参数不变性测试 - 核心创新验证

测试 DEER-Spikformer 在不同 τ 和 θ 下的准确率稳定性：
1. τ 维度：[1.0, 1.5, 2.0, 2.5, 3.0]（固定 θ=0.5）
2. θ 维度：[0.3, 0.4, 0.5, 0.6, 0.7]（固定 τ=2.0）

目标：验证 std < 0.5%，证明超参数不变性
"""

import os
import sys
import time
import json
import argparse
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cifar10.model_deer import Spikformer_DEER


def get_args():
    parser = argparse.ArgumentParser('DEER Hyperparameter Invariance Test')
    parser.add_argument('--checkpoint', default='./output_deer_cifar100_256_fixed/20251203-121948/best_model.pth', type=str)
    parser.add_argument('--data_dir', default='./data', type=str)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--T', default=4, type=int)
    parser.add_argument('--embed_dims', default=256, type=int)
    parser.add_argument('--num_heads', default=8, type=int)
    parser.add_argument('--depths', default=4, type=int)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--output_dir', default='./hyperparameter_invariance_results', type=str)
    return parser.parse_args()


def get_test_loader(args):
    """加载 CIFAR-100 测试集"""
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    testset = torchvision.datasets.CIFAR100(
        root=args.data_dir, train=False, download=True, transform=transform_test
    )
    testloader = DataLoader(
        testset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    return testloader


def build_model(args):
    """构建模型"""
    model = Spikformer_DEER(
        img_size_h=32,
        img_size_w=32,
        patch_size=4,
        embed_dims=args.embed_dims,
        num_heads=args.num_heads,
        mlp_ratios=4,
        in_channels=2,
        num_classes=100,
        qkv_bias=False,
        depths=args.depths,
        sr_ratios=1,
    )
    return model.to(args.device)


def set_model_tau_theta(model, tau, theta):
    """
    设置模型中所有 DEER_LIF_Layer 的 τ 和 θ 参数
    
    注意：需要找到模型中所有使用 AccumulatorLIF 的地方并更新参数
    """
    # 遍历模型所有模块
    for name, module in model.named_modules():
        # 检查是否是 DEER_LIF_Layer 或 AccumulatorLIF
        if hasattr(module, 'lif_model'):  # DEER_LIF_Layer 包含 lif_model (AccumulatorLIF)
            lif = module.lif_model
            if hasattr(lif, 'tau'):
                lif.tau = tau
            if hasattr(lif, 'theta'):
                lif.theta = theta
            # print(f"Updated {name}: tau={tau}, theta={theta}")
        elif hasattr(module, 'tau') and hasattr(module, 'theta'):  # 直接是 AccumulatorLIF
            module.tau = tau
            module.theta = theta
            # print(f"Updated {name}: tau={tau}, theta={theta}")


@torch.no_grad()
def evaluate_with_params(model, testloader, tau, theta, args):
    """使用特定 τ 和 θ 评估模型"""
    # 设置超参数
    set_model_tau_theta(model, tau, theta)
    
    model.train()  # 使用 train 模式（小 batch BN）
    
    correct = 0
    total = 0
    
    for images, labels in testloader:
        # 转换为 2 通道 + 时间维度
        B, C, H, W = images.shape
        images_2ch = images[:, :2, :, :]
        images_t = images_2ch.unsqueeze(1).repeat(1, args.T, 1, 1, 1)
        
        images_t = images_t.to(args.device, non_blocking=True)
        labels = labels.to(args.device, non_blocking=True)
        
        # 前向传播
        outputs = model(images_t)
        _, predicted = outputs.max(1)
        
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    return accuracy


def test_tau_invariance(model, testloader, args):
    """测试 τ 不变性（固定 θ=0.5）"""
    tau_values = [1.0, 1.5, 2.0, 2.5, 3.0]
    theta_fixed = 0.5
    
    print("\n" + "="*70)
    print("测试 τ 不变性（固定 θ=0.5）")
    print("="*70)
    
    results = []
    for tau in tau_values:
        print(f"\n测试 τ={tau}, θ={theta_fixed}...")
        start_time = time.time()
        acc = evaluate_with_params(model, testloader, tau, theta_fixed, args)
        elapsed = time.time() - start_time
        
        results.append({
            'tau': tau,
            'theta': theta_fixed,
            'accuracy': acc
        })
        
        print(f"  准确率: {acc:.2f}%  (耗时: {elapsed:.1f}s)")
    
    # 统计分析
    accs = [r['accuracy'] for r in results]
    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    min_acc = np.min(accs)
    max_acc = np.max(accs)
    
    print("\n" + "-"*70)
    print("τ 不变性统计:")
    print(f"  平均准确率: {mean_acc:.2f}%")
    print(f"  标准差: {std_acc:.3f}%")
    print(f"  范围: [{min_acc:.2f}%, {max_acc:.2f}%]")
    print(f"  变化幅度: {max_acc - min_acc:.2f}%")
    
    if std_acc < 0.5:
        print(f"  ✅ 标准差 < 0.5%，τ 不变性验证成功！")
    else:
        print(f"  ⚠️  标准差 ≥ 0.5%，τ 敏感性较高")
    
    return results


def test_theta_invariance(model, testloader, args):
    """测试 θ 不变性（固定 τ=2.0）"""
    tau_fixed = 2.0
    theta_values = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    print("\n" + "="*70)
    print("测试 θ 不变性（固定 τ=2.0）")
    print("="*70)
    
    results = []
    for theta in theta_values:
        print(f"\n测试 τ={tau_fixed}, θ={theta}...")
        start_time = time.time()
        acc = evaluate_with_params(model, testloader, tau_fixed, theta, args)
        elapsed = time.time() - start_time
        
        results.append({
            'tau': tau_fixed,
            'theta': theta,
            'accuracy': acc
        })
        
        print(f"  准确率: {acc:.2f}%  (耗时: {elapsed:.1f}s)")
    
    # 统计分析
    accs = [r['accuracy'] for r in results]
    mean_acc = np.mean(accs)
    std_acc = np.std(accs)
    min_acc = np.min(accs)
    max_acc = np.max(accs)
    
    print("\n" + "-"*70)
    print("θ 不变性统计:")
    print(f"  平均准确率: {mean_acc:.2f}%")
    print(f"  标准差: {std_acc:.3f}%")
    print(f"  范围: [{min_acc:.2f}%, {max_acc:.2f}%]")
    print(f"  变化幅度: {max_acc - min_acc:.2f}%")
    
    if std_acc < 0.5:
        print(f"  ✅ 标准差 < 0.5%，θ 不变性验证成功！")
    else:
        print(f"  ⚠️  标准差 ≥ 0.5%，θ 敏感性较高")
    
    return results


def plot_invariance_results(tau_results, theta_results, output_dir):
    """绘制不变性测试结果"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. τ 不变性
    ax = axes[0]
    tau_vals = [r['tau'] for r in tau_results]
    tau_accs = [r['accuracy'] for r in tau_results]
    tau_mean = np.mean(tau_accs)
    tau_std = np.std(tau_accs)
    
    ax.plot(tau_vals, tau_accs, marker='o', linewidth=2, markersize=10, label='Accuracy')
    ax.axhline(y=tau_mean, color='r', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Mean ({tau_mean:.2f}%)')
    ax.fill_between(tau_vals, tau_mean - tau_std, tau_mean + tau_std, alpha=0.2, color='r', label=f'±1 std ({tau_std:.3f}%)')
    ax.set_xlabel('τ (Time Constant)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title(f'τ Invariance Test (θ=0.5)\nStd={tau_std:.3f}%', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([tau_mean - 2, tau_mean + 2])
    
    # 添加数值标注
    for tau, acc in zip(tau_vals, tau_accs):
        ax.text(tau, acc + 0.15, f'{acc:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 2. θ 不变性
    ax = axes[1]
    theta_vals = [r['theta'] for r in theta_results]
    theta_accs = [r['accuracy'] for r in theta_results]
    theta_mean = np.mean(theta_accs)
    theta_std = np.std(theta_accs)
    
    ax.plot(theta_vals, theta_accs, marker='s', linewidth=2, markersize=10, color='green', label='Accuracy')
    ax.axhline(y=theta_mean, color='r', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Mean ({theta_mean:.2f}%)')
    ax.fill_between(theta_vals, theta_mean - theta_std, theta_mean + theta_std, alpha=0.2, color='r', label=f'±1 std ({theta_std:.3f}%)')
    ax.set_xlabel('θ (Threshold)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Test Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title(f'θ Invariance Test (τ=2.0)\nStd={theta_std:.3f}%', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([theta_mean - 2, theta_mean + 2])
    
    # 添加数值标注
    for theta, acc in zip(theta_vals, theta_accs):
        ax.text(theta, acc + 0.15, f'{acc:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/hyperparameter_invariance.png', dpi=150, bbox_inches='tight')
    plt.savefig(f'{output_dir}/hyperparameter_invariance.pdf', bbox_inches='tight')
    print(f"\n✅ 不变性曲线已保存: {output_dir}/hyperparameter_invariance.png (and .pdf)")


def save_results(tau_results, theta_results, output_dir):
    """保存结果到 JSON"""
    results = {
        'tau_invariance': {
            'fixed_theta': 0.5,
            'tau_values': [r['tau'] for r in tau_results],
            'accuracies': [r['accuracy'] for r in tau_results],
            'mean': float(np.mean([r['accuracy'] for r in tau_results])),
            'std': float(np.std([r['accuracy'] for r in tau_results])),
            'min': float(np.min([r['accuracy'] for r in tau_results])),
            'max': float(np.max([r['accuracy'] for r in tau_results])),
        },
        'theta_invariance': {
            'fixed_tau': 2.0,
            'theta_values': [r['theta'] for r in theta_results],
            'accuracies': [r['accuracy'] for r in theta_results],
            'mean': float(np.mean([r['accuracy'] for r in theta_results])),
            'std': float(np.std([r['accuracy'] for r in theta_results])),
            'min': float(np.min([r['accuracy'] for r in theta_results])),
            'max': float(np.max([r['accuracy'] for r in theta_results])),
        },
        'test_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'checkpoint': args.checkpoint
    }
    
    with open(f'{output_dir}/invariance_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ 结果已保存: {output_dir}/invariance_results.json")


def main():
    global args
    args = get_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print("🔬 DEER 超参数不变性测试")
    print("="*80)
    print(f"\nCheckpoint: {args.checkpoint}")
    print(f"输出目录: {args.output_dir}")
    print(f"设备: {args.device}")
    
    # 加载测试数据
    print("\n加载 CIFAR-100 测试集...")
    testloader = get_test_loader(args)
    print(f"测试集样本数: {len(testloader.dataset)}")
    
    # 构建模型
    print("\n构建模型...")
    model = build_model(args)
    
    # 加载 checkpoint
    print(f"\n加载 checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Checkpoint 加载成功 (trained epoch: {checkpoint['epoch']})")
    
    # 测试 τ 不变性
    tau_results = test_tau_invariance(model, testloader, args)
    
    # 测试 θ 不变性
    theta_results = test_theta_invariance(model, testloader, args)
    
    # 绘制结果
    print("\n绘制不变性曲线...")
    plot_invariance_results(tau_results, theta_results, args.output_dir)
    
    # 保存结果
    print("\n保存结果...")
    save_results(tau_results, theta_results, args.output_dir)
    
    # 最终总结
    print("\n" + "="*80)
    print("📊 超参数不变性测试总结")
    print("="*80)
    
    tau_accs = [r['accuracy'] for r in tau_results]
    theta_accs = [r['accuracy'] for r in theta_results]
    
    tau_std = np.std(tau_accs)
    theta_std = np.std(theta_accs)
    
    print(f"\nτ 不变性 (θ=0.5):")
    print(f"  准确率范围: [{np.min(tau_accs):.2f}%, {np.max(tau_accs):.2f}%]")
    print(f"  标准差: {tau_std:.3f}%")
    print(f"  结论: {'✅ 通过（std < 0.5%）' if tau_std < 0.5 else '⚠️ 敏感性偏高'}")
    
    print(f"\nθ 不变性 (τ=2.0):")
    print(f"  准确率范围: [{np.min(theta_accs):.2f}%, {np.max(theta_accs):.2f}%]")
    print(f"  标准差: {theta_std:.3f}%")
    print(f"  结论: {'✅ 通过（std < 0.5%）' if theta_std < 0.5 else '⚠️ 敏感性偏高'}")
    
    if tau_std < 0.5 and theta_std < 0.5:
        print(f"\n🎉 恭喜！超参数不变性验证成功！")
        print(f"   τ 和 θ 的标准差均 < 0.5%，证明了方法的稳健性")
        print(f"   这是论文的核心创新点，可以写入实验结果部分")
    else:
        print(f"\n⚠️  部分超参数敏感性偏高")
        print(f"   建议：分析敏感性来源，或在论文中讨论这一现象")
    
    print("="*80)


if __name__ == '__main__':
    main()
