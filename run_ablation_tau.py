"""
消融实验：τ衰减系数影响
测试不同τ值对Accumulator-LIF性能的影响

方法：修改deer_lif_node.py中DEERLIFNode的默认tau值，然后训练
"""
import os
import subprocess
import time
import shutil

# 配置
TAU_VALUES = [1.0, 1.5, 2.0, 2.5, 3.0]
ORIGINAL_TAU = 2.0  # 原始默认值

BASE_CONFIG = {
    'batch_size': 128,
    'epochs': 300,
    'lr': 0.001,
    'weight_decay': 0.05,
    'seed': 42,
}

def modify_tau_in_code(new_tau):
    """修改deer_lif_node.py中的默认tau值"""
    filepath = "deer_lif_node.py"
    
    # 读取文件
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 替换tau默认值（在__init__函数中）
    # 查找: tau: float = X.X,
    import re
    pattern = r'tau: float = \d+\.?\d*,'
    replacement = f'tau: float = {new_tau},'
    
    new_content = re.sub(pattern, replacement, content, count=1)
    
    # 写回文件
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"[OK] 已修改 deer_lif_node.py: tau={new_tau}")

def run_experiment(tau):
    """运行单个τ值的实验"""
    output_dir = f"./output_ablation_tau/tau_{tau:.1f}"
    
    # 修改代码中的tau值
    modify_tau_in_code(tau)
    
    cmd = f"python train_deer_94_config.py --data_dir ./data --batch_size {BASE_CONFIG['batch_size']} --epochs {BASE_CONFIG['epochs']} --lr {BASE_CONFIG['lr']} --weight_decay {BASE_CONFIG['weight_decay']} --seed {BASE_CONFIG['seed']} --output_dir {output_dir}"
    
    print(f"\n{'='*60}")
    print(f"开始实验: τ = {tau}")
    print(f"输出目录: {output_dir}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    result = subprocess.run(cmd, shell=True)
    elapsed = time.time() - start_time
    
    print(f"\n实验完成: τ = {tau}, 耗时: {elapsed/3600:.2f}小时")
    return result.returncode == 0

def main():
    print("="*60)
    print("消融实验：τ衰减系数影响")
    print(f"测试τ值: {TAU_VALUES}")
    print(f"基础配置: batch={BASE_CONFIG['batch_size']}, epochs={BASE_CONFIG['epochs']}")
    print("="*60)
    
    # 备份原始deer_lif_node.py
    backup_file = "deer_lif_node.py.backup_tau"
    if not os.path.exists(backup_file):
        shutil.copy("deer_lif_node.py", backup_file)
        print(f"[OK] 已备份 deer_lif_node.py -> {backup_file}\n")
    
    results = {}
    for tau in TAU_VALUES:
        success = run_experiment(tau)
        results[tau] = "[OK] 成功" if success else "[FAIL] 失败"
        
        # 实验间恢复原始tau值（避免影响）
        if tau != TAU_VALUES[-1]:
            time.sleep(2)  # 等待文件写入完成
    
    # 恢复原始tau值
    modify_tau_in_code(ORIGINAL_TAU)
    print(f"\n[OK] 已恢复原始 tau={ORIGINAL_TAU}\n")
    
    # 打印总结
    print("\n" + "="*60)
    print("实验总结")
    print("="*60)
    for tau, status in results.items():
        print(f"τ = {tau:.1f}: {status}")
    print("="*60)
    print(f"\n备份文件位于: {backup_file}")
    print("如需恢复: mv {backup_file} deer_lif_node.py")

if __name__ == "__main__":
    main()
