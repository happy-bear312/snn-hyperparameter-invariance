"""
分析多种子训练结果
计算均值、标准差、最高/最低精度等统计信息
"""
import json
import os
import glob
import numpy as np
from pathlib import Path

def load_results(output_dir):
    """加载某个seed的results.json"""
    # 先检查目录本身是否有results.json
    direct_file = os.path.join(output_dir, "results.json")
    if os.path.exists(direct_file):
        with open(direct_file, 'r') as f:
            return json.load(f)
    
    # 否则在子目录中查找
    pattern = os.path.join(output_dir, "*/results.json")
    files = glob.glob(pattern)
    
    if not files:
        return None
    
    # 如果有多个结果，选最新的
    latest_file = max(files, key=os.path.getmtime)
    
    with open(latest_file, 'r') as f:
        return json.load(f)

def main():
    # 所有seeds目录（需要找到seed=42的正确路径）
    seed_dirs = [
        "./output_deer_94config_seeds/seed_2024",
        "./output_deer_94config_seeds/seed_3407",
        "./output_deer_94config_seeds/seed_12345",
        "./output_deer_94config_seeds/seed_99999",
    ]
    
    # 尝试找seed=42的结果
    seed_42_candidates = [
        "./output_deer_cifar10/20251104-144307",
        "./output_deer_94config_seeds/seed_42",
    ]
    
    for candidate in seed_42_candidates:
        if os.path.exists(candidate):
            result = load_results(candidate)
            if result:
                seed_dirs.insert(0, candidate)
                break
    
    # 收集结果
    all_results = []
    for seed_dir in seed_dirs:
        if os.path.exists(seed_dir):
            result = load_results(seed_dir)
            if result:
                all_results.append(result)
                print(f"✅ 加载成功: {seed_dir}")
                print(f"   Best Acc: {result['best_acc']:.2f}%, Final Acc: {result['final_acc']:.2f}%")
            else:
                print(f"⚠️  无结果: {seed_dir}")
        else:
            print(f"❌ 不存在: {seed_dir}")
    
    if len(all_results) < 3:
        print(f"\n错误：至少需要3个有效结果，当前只有{len(all_results)}个")
        return
    
    # 统计分析
    best_accs = [r['best_acc'] for r in all_results]
    final_accs = [r['final_acc'] for r in all_results]
    times = [r['total_time_hours'] for r in all_results]
    
    print("\n" + "="*60)
    print("多种子训练统计结果")
    print("="*60)
    print(f"有效实验数: {len(all_results)}")
    print()
    print("Best Accuracy:")
    print(f"  均值: {np.mean(best_accs):.2f}%")
    print(f"  标准差: {np.std(best_accs):.2f}%")
    print(f"  最高: {np.max(best_accs):.2f}%")
    print(f"  最低: {np.min(best_accs):.2f}%")
    print()
    print("Final Accuracy:")
    print(f"  均值: {np.mean(final_accs):.2f}%")
    print(f"  标准差: {np.std(final_accs):.2f}%")
    print()
    print("训练时间:")
    print(f"  均值: {np.mean(times):.2f}小时")
    print(f"  标准差: {np.std(times):.2f}小时")
    print("="*60)
    
    # 详细列表
    print("\n详细结果:")
    print(f"{'Seed':<10} {'Best Acc':<12} {'Final Acc':<12} {'Time (h)':<10}")
    print("-" * 50)
    for i, result in enumerate(all_results):
        seed = result['args']['seed']
        print(f"{seed:<10} {result['best_acc']:<12.2f} {result['final_acc']:<12.2f} {result['total_time_hours']:<10.2f}")
    
    # 与baseline对比
    baseline_acc = 95.29  # Spikformer-4-256原文结果
    gap = np.mean(best_accs) - baseline_acc
    print(f"\n与Spikformer基线对比:")
    print(f"  Baseline: {baseline_acc}%")
    print(f"  Ours: {np.mean(best_accs):.2f}% ± {np.std(best_accs):.2f}%")
    print(f"  Gap: {gap:+.2f}%")
    print("="*60)
    
    # 保存统计结果
    summary = {
        'num_experiments': len(all_results),
        'best_acc': {
            'mean': float(np.mean(best_accs)),
            'std': float(np.std(best_accs)),
            'max': float(np.max(best_accs)),
            'min': float(np.min(best_accs)),
        },
        'final_acc': {
            'mean': float(np.mean(final_accs)),
            'std': float(np.std(final_accs)),
        },
        'training_time': {
            'mean': float(np.mean(times)),
            'std': float(np.std(times)),
        },
        'baseline_comparison': {
            'baseline': baseline_acc,
            'ours': float(np.mean(best_accs)),
            'gap': float(gap),
        },
        'all_results': all_results,
    }
    
    output_file = "./output_deer_94config_seeds/summary_statistics.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n统计结果已保存至: {output_file}")

if __name__ == "__main__":
    main()
