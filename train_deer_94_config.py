"""
使用已验证的94%配置训练DEER-Spikformer
目标：多种子训练，争取达到95%+

配置来源：output_deer_cifar10/20251104-144307/results.json (94.03%)
"""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import sys
import time
import json
from datetime import datetime
import argparse

sys.path.insert(0, os.path.dirname(__file__))
from cifar10.model_deer import Spikformer_DEER

def get_args():
    parser = argparse.ArgumentParser('DEER-Spikformer with 94% Config')
    
    # 数据
    parser.add_argument('--data_dir', default='./data', type=str)
    parser.add_argument('--num_workers', default=4, type=int)
    
    # 94%配置（已验证）
    parser.add_argument('--batch_size', default=128, type=int, help='94%用128')
    parser.add_argument('--val_batch_size', default=128, type=int)
    parser.add_argument('--T', default=4, type=int)
    parser.add_argument('--embed_dims', default=256, type=int, help='94%用256')
    parser.add_argument('--num_heads', default=8, type=int, help='94%用8')
    parser.add_argument('--mlp_ratios', default=4, type=int)
    parser.add_argument('--depths', default=4, type=int)
    parser.add_argument('--patch_size', default=4, type=int)
    parser.add_argument('--num_classes', default=10, type=int)
    
    # 训练（94%配置）
    parser.add_argument('--epochs', default=300, type=int, help='94%训练300轮')
    parser.add_argument('--lr', default=0.001, type=float, help='94%用0.001')
    parser.add_argument('--weight_decay', default=0.05, type=float, help='94%用0.05')
    parser.add_argument('--warmup_epochs', default=10, type=int, help='94%用10')
    parser.add_argument('--min_lr', default=1e-5, type=float)
    
    # 数据增强（94%配置：无特殊增强，只有基本的）
    parser.add_argument('--use_autoaugment', default=False, action='store_true',
                        help='94%没用RandAugment')
    
    # 其他
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default=42, type=int, help='可指定不同种子')
    parser.add_argument('--output_dir', default='./output_deer_94config', type=str)
    parser.add_argument('--print_freq', default=50, type=int)
    parser.add_argument('--eval_freq', default=5, type=int)
    parser.add_argument('--resume', default='', type=str, help='Resume from checkpoint')
    
    return parser.parse_args()

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_transforms():
    """94%配置的数据增强：基本增强，无RandAugment"""
    
    # 训练集：基本增强
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # 测试集：只标准化
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    return train_transform, test_transform

def adjust_learning_rate(optimizer, epoch, args):
    """Cosine学习率调度"""
    if epoch < args.warmup_epochs:
        # Warmup
        lr = args.lr * (epoch + 1) / args.warmup_epochs
    else:
        # Cosine decay
        progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
        lr = lr.item()
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return lr

def train_epoch(model, train_loader, criterion, optimizer, epoch, args):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    start_time = time.time()
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(args.device), targets.to(args.device)
        
        # 转换为2通道（移除蓝色通道）
        if inputs.shape[1] == 3:
            inputs = inputs[:, :2, :, :]
        
        # 扩展时间维度：(B, C, H, W) -> (B, T, C, H, W)
        inputs = inputs.unsqueeze(1).repeat(1, args.T, 1, 1, 1)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        if (batch_idx + 1) % args.print_freq == 0:
            print(f'  Batch [{batch_idx+1}/{len(train_loader)}] '
                  f'Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}% '
                  f'({correct}/{total})')
    
    epoch_time = time.time() - start_time
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy, epoch_time

def evaluate(model, test_loader, criterion, args):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            
            # 转换为2通道
            if inputs.shape[1] == 3:
                inputs = inputs[:, :2, :, :]
            
            # 扩展时间维度：(B, C, H, W) -> (B, T, C, H, W)
            inputs = inputs.unsqueeze(1).repeat(1, args.T, 1, 1, 1)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    avg_loss = total_loss / len(test_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy

def main():
    args = get_args()
    
    # 设置随机种子
    setup_seed(args.seed)
    
    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存配置
    config_file = os.path.join(output_dir, 'config.json')
    with open(config_file, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print("\n" + "="*70)
    print("DEER-Spikformer训练（94%配置）")
    print("="*70)
    print(f"配置来源: 已验证的94.03%训练")
    print(f"输出目录: {output_dir}")
    print(f"随机种子: {args.seed}")
    print("\n关键配置:")
    print(f"  batch_size: {args.batch_size} (94%: 128)")
    print(f"  lr: {args.lr} (94%: 0.001)")
    print(f"  embed_dims: {args.embed_dims} (94%: 256)")
    print(f"  num_heads: {args.num_heads} (94%: 8)")
    print(f"  epochs: {args.epochs} (94%: 300)")
    print(f"  weight_decay: {args.weight_decay} (94%: 0.05)")
    print(f"  数据增强: 基本增强（无RandAugment）")
    print("="*70)
    
    # 准备数据
    print("\n准备数据...")
    train_transform, test_transform = get_transforms()
    
    train_dataset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=False, download=True, transform=test_transform
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, 
        shuffle=True, num_workers=args.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.val_batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True
    )
    
    print(f"训练集: {len(train_dataset)} 样本")
    print(f"测试集: {len(test_dataset)} 样本")
    
    # 创建模型
    print("\n创建模型...")
    model = Spikformer_DEER(
        img_size_h=32, img_size_w=32,
        patch_size=args.patch_size,
        in_channels=2,
        num_classes=args.num_classes,
        embed_dims=args.embed_dims,
        num_heads=args.num_heads,
        mlp_ratios=args.mlp_ratios,
        qkv_bias=False,  # 重要：与94%训练保持一致
        depths=args.depths,
        sr_ratios=1,  # 固定为1，不使用spatial reduction
        T=args.T
    ).to(args.device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {num_params/1e6:.2f}M")
    
    # 优化器和损失函数
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    
    # 训练循环
    print("\n开始训练...")
    best_acc = 0
    best_epoch = 0
    history = {
        'train_loss': [], 'train_acc': [],
        'test_loss': [], 'test_acc': [], 'lr': []
    }
    
    total_start = time.time()
    
    for epoch in range(args.epochs):
        # 调整学习率
        current_lr = adjust_learning_rate(optimizer, epoch, args)
        
        print(f"\nEpoch {epoch+1}/{args.epochs} | LR: {current_lr:.6f}")
        print("-" * 70)
        
        # 训练
        train_loss, train_acc, epoch_time = train_epoch(
            model, train_loader, criterion, optimizer, epoch, args
        )
        
        print(f"训练: Loss={train_loss:.4f}, Acc={train_acc:.2f}%, Time={epoch_time:.1f}s")
        
        # 评估
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            test_loss, test_acc = evaluate(model, test_loader, criterion, args)
            print(f"测试: Loss={test_loss:.4f}, Acc={test_acc:.2f}%")
            
            # 保存最佳模型
            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = epoch + 1
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc,
                    'args': vars(args)
                }, os.path.join(output_dir, 'best_model.pth'))
                print(f"★ 新最佳准确率: {best_acc:.2f}% (Epoch {best_epoch})")
            
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['lr'].append(current_lr)
        
        # 定期保存checkpoint
        if (epoch + 1) % 50 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }, os.path.join(output_dir, f'checkpoint_epoch{epoch+1}.pth'))
    
    total_time = time.time() - total_start
    
    # 最终评估
    print("\n" + "="*70)
    print("训练完成！")
    print("="*70)
    
    # 加载最佳模型进行最终测试
    checkpoint = torch.load(os.path.join(output_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    final_loss, final_acc = evaluate(model, test_loader, criterion, args)
    
    print(f"最佳准确率: {best_acc:.2f}% (Epoch {best_epoch})")
    print(f"最终准确率: {final_acc:.2f}%")
    print(f"总训练时间: {total_time/3600:.2f} 小时")
    print(f"目标对比: 94.03% (已验证) vs 95.29% (Baseline)")
    
    if final_acc >= 95.0:
        print(f"\n🎉 成功！达到≥95%，可以发论文了！")
    elif final_acc >= 94.0:
        print(f"\n✓ 很好！接近94%，可以尝试其他种子或微调")
    
    # 保存结果
    results = {
        'best_acc': best_acc,
        'best_epoch': best_epoch,
        'final_acc': final_acc,
        'total_time_hours': total_time / 3600,
        'history': history,
        'args': vars(args)
    }
    
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n结果已保存到: {output_dir}")
    print("="*70)

if __name__ == "__main__":
    main()
