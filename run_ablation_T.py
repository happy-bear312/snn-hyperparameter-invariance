import os
import sys
import subprocess
import glob
import json
import time
from datetime import datetime

os.chdir(r'c:\Users\dell\SPIKE\spikformer')

T_VALUES = [2, 4, 8]

BASE_CONFIG = {
    'data_dir': './data',
    'batch_size': 128,
    'num_workers': 4,
    'embed_dims': 256,
    'num_heads': 8,
    'epochs': 300,
    'lr': 0.001,
    'weight_decay': 0.05,
    'warmup_epochs': 20,
    'device': 'cuda',
    'seed': 42,
    'print_freq': 50,
    'eval_freq': 5
}

def build_command(T_value, output_dir):
    cmd_parts = [
        'conda activate SPIKE',
        '&&',
        'python train_deer_cifar10.py',
        f'--T {T_value}',
        f'--output_dir {output_dir}',
    ]
    
    for key, value in BASE_CONFIG.items():
        cmd_parts.append(f'--{key} {value}')
    
    return ' '.join(cmd_parts)

def run_experiment(T_value):
    output_dir = f'./output_ablation_T/T_{T_value}'
    
    result_file = os.path.join(output_dir, '**', 'results.json')
    if glob.glob(result_file, recursive=True):
        print(f'Skip T = {T_value} completed')
        return True
    
    print(f'\n{"="*60}')
    print(f'Start: T = {T_value}')
    print(f'Output: {output_dir}')
    print(f'{"="*60}\n')
    
    cmd = build_command(T_value, output_dir)
    print(f'Command: {cmd}\n')
    
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=False,
            text=True
        )
        elapsed = time.time() - start_time
        print(f'\nSuccess T={T_value}, time: {elapsed/3600:.2f} hours')
        return True
        
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        print(f'\nFailed T={T_value}, time: {elapsed/3600:.2f} hours')
        print(f'Error: {e}')
        return False

def main():
    print('='*60)
    print('T Ablation - Accumulator-LIF Time Steps Sensitivity')
    print('='*60)
    print(f'T values: {T_VALUES}')
    print(f'Total experiments: {len(T_VALUES)}')
    print(f'Estimated time: ~{len(T_VALUES) * 13:.1f} hours')
    print('='*60)
    
    total_start = time.time()
    success_count = 0
    
    for i, T_value in enumerate(T_VALUES, 1):
        print(f'\nProgress: [{i}/{len(T_VALUES)}]')
        
        if run_experiment(T_value):
            success_count += 1
        else:
            print(f'Warning: T={T_value} failed, continue...')
            continue
    
    total_elapsed = time.time() - total_start
    print('\n' + '='*60)
    print('T Ablation Summary')
    print('='*60)
    print(f'Success: {success_count}/{len(T_VALUES)}')
    print(f'Total time: {total_elapsed/3600:.2f} hours')
    print('='*60)
    
    print('\nCollecting results...')
    results = []
    for T_value in T_VALUES:
        result_files = glob.glob(f'./output_ablation_T/T_{T_value}/**/results.json', recursive=True)
        if result_files:
            with open(result_files[0], 'r') as f:
                data = json.load(f)
                results.append({
                    'T': T_value,
                    'best_acc': data['best_acc'],
                    'final_acc': data['final_acc'],
                    'time_hours': data.get('total_time_hours', 0)
                })
    
    if results:
        print('\nT Ablation Results:')
        print('-'*60)
        print(f'{"T":<10} {"Best Acc":<15} {"Final Acc":<15} {"Time(h)":<15}')
        print('-'*60)
        for r in results:
            print(f'{r["T"]:<10} {r["best_acc"]:<15.2f} {r["final_acc"]:<15.2f} {r["time_hours"]:<15.2f}')
        print('-'*60)
        
        summary_file = './output_ablation_T/T_ablation_summary.json'
        os.makedirs(os.path.dirname(summary_file), exist_ok=True)
        with open(summary_file, 'w') as f:
            json.dump({
                'experiment': 'T_ablation',
                'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'T_values': T_VALUES,
                'config': BASE_CONFIG,
                'results': results,
                'total_time_hours': total_elapsed/3600
            }, f, indent=2)
        print(f'\nSummary saved to: {summary_file}')

if __name__ == '__main__':
    main()
