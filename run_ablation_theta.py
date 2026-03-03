"""
Theta ablation experiment script (Stage 2)
Tests 5 different v_threshold values: 0.3, 0.4, 0.5, 0.6, 0.7
Each runs 300 epochs (~13h), total ~65h
"""

import os
import subprocess
import time
import re

# 切换到脚本所在目录
os.chdir(os.path.dirname(os.path.abspath(__file__)))

THETA_VALUES = [0.3, 0.4, 0.5, 0.6, 0.7]

BASE_CONFIG = {
    'batch_size': 128,
    'epochs': 300,
    'lr': 0.001,  # 修复：与τ消融保持一致
    'weight_decay': 0.05,  # 修复：与τ消融保持一致
    'seed': 42,
}

def modify_theta_in_code(new_theta):
    """修改deer_lif_node.py中的默认v_threshold值"""
    filepath = "deer_lif_node.py"
    
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    pattern = r'v_threshold: float = [\d\.]+'
    replacement = f'v_threshold: float = {new_theta}'
    content = re.sub(pattern, replacement, content, count=1)
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    
    print(f"[OK] 已修改 deer_lif_node.py: v_threshold={new_theta}")

def run_experiment(theta):
    """运行单个θ值的实验"""
    output_dir = f"./output_ablation_theta/theta_{theta}"
    
    # 修改代码中的theta值
    modify_theta_in_code(theta)
    
    cmd = f"conda activate SPIKE && python train_deer_cifar10.py --data_dir ./data --batch_size {BASE_CONFIG['batch_size']} --epochs {BASE_CONFIG['epochs']} --lr {BASE_CONFIG['lr']} --weight_decay {BASE_CONFIG['weight_decay']} --seed {BASE_CONFIG['seed']} --output_dir {output_dir}"
    
    print(f"\n{'='*60}")
    print(f"开始实验: θ = {theta}")
    print(f"输出目录: {output_dir}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    result = subprocess.run(cmd, shell=True)
    elapsed = time.time() - start_time
    
    print(f"\n实验完成: θ = {theta}, 耗时: {elapsed/3600:.2f}小时")
    return result.returncode == 0

if __name__ == "__main__":
    print("="*60)
    print("消融实验：θ阈值影响（继续）")
    print(f"测试θ值: {THETA_VALUES}")
    print(f"基础配置: batch={BASE_CONFIG['batch_size']}, epochs={BASE_CONFIG['epochs']}")
    print("="*60)
    
    results = {}
    for theta in THETA_VALUES:
        # 检查是否已完成
        result_file = f"./output_ablation_theta/theta_{theta}/*/results.json"
        import glob
        if glob.glob(result_file):
            print(f"\n[跳过] θ = {theta} 已完成")
            results[theta] = "[OK] 已完成"
            continue
        
        success = run_experiment(theta)
        results[theta] = "[OK] 成功" if success else "[FAIL] 失败"
        
        if theta != THETA_VALUES[-1]:
            time.sleep(2)
    
    # 打印总结
    print("\n" + "="*60)
    print("实验总结")
    print("="*60)
    for theta, status in results.items():
        print(f"θ = {theta}: {status}")
    print("="*60)
