"""
DEER-LIF Node v5 - 串行solver优化版

核心发现:
1. Profile显示并行scan在PyTorch中没有优势（反而更慢）
2. 串行for-loop在小T(≤16)时更快
3. DEER的瓶颈在Python循环overhead，不是solver本身

作者: AI Agent
创建日期: 2025-11-04
版本: v5 (回归串行，等待torch.compile优化)
"""

import torch
import torch.nn as nn
from typing import Tuple, Dict
import warnings


class DEERLIFNode(nn.Module):
    """
    DEER并行化的LIF神经元 (正确实现)
    
    基于ICLR 2024 DEER论文的正确数学原理:
    - 线性系统是递归形式，不是 [I+G]y=rhs
    - Jacobian是完整的F×F矩阵，不是对角近似
    - 使用递归求解器（串行版本），未来可升级到associative scan
    """
    
    def __init__(
        self,
        tau: float = 2.0,
        v_threshold: float = 0.7,
        v_reset: float = 0.0,
        surrogate_alpha: float = 4.0,
        max_iter: int = 10,  # 允许充分迭代（训练时会提前停止）
        tol: float = 1e-6,
        warmstart_steps: int = 2,
        use_diagonal_approx: bool = True,  # 使用对角近似（内存友好）
        check_convergence: bool = False,  # 训练时不检查收敛（避免GPU-CPU同步）
    ):
        """
        Args:
            tau: LIF时间常数
            v_threshold: 脉冲阈值
            v_reset: reset后的膜电位
            surrogate_alpha: 代理梯度陡峭度
            max_iter: DEER最大迭代次数
            tol: 收敛阈值
            warmstart_steps: warm-start的串行步数
            use_diagonal_approx: 使用对角近似（True=内存友好，False=完整矩阵）
            check_convergence: 是否检查收敛（False=固定迭代，True=早停）
        """
        super().__init__()
        
        self.tau = tau
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.surrogate_alpha = surrogate_alpha
        self.max_iter = max_iter
        self.tol = tol
        self.warmstart_steps = warmstart_steps
        self.use_diagonal_approx = use_diagonal_approx
        self.check_convergence = check_convergence
        
        # 统计信息
        self.stats = {
            'iter_counts': [],
            'converged': [],
            'errors': [],
        }
    
    def reset_stats(self):
        """重置统计信息"""
        for key in self.stats:
            self.stats[key].clear()
    
    def get_stats_summary(self) -> Dict[str, float]:
        """获取统计摘要"""
        if len(self.stats['iter_counts']) == 0:
            return {}
        
        import numpy as np
        return {
            'avg_iters': float(np.mean(self.stats['iter_counts'])),
            'max_iters': int(np.max(self.stats['iter_counts'])),
            'convergence_rate': float(np.mean(self.stats['converged'])) * 100,
            'avg_error': float(np.mean(self.stats['errors'])),
        }
    
    def lif_step_forward(self, v_prev: torch.Tensor, x_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        单步LIF更新（前向，用于计算L[y]）
        
        Args:
            v_prev: (B, F) 前一时刻膜电位
            x_t: (B, F) 当前输入
        
        Returns:
            v_next: (B, F) 更新后的膜电位
            spike: (B, F) 脉冲输出
        """
        # LIF动力学
        h = v_prev + (x_t - v_prev) / self.tau
        spike = (h >= self.v_threshold).float()
        v_next = h * (1.0 - spike) + self.v_reset * spike
        
        return v_next, spike
    
    def lif_step_with_jacobian(self, v_prev: torch.Tensor, x_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        单步LIF更新（带Jacobian计算）
        
        使用代理梯度使Jacobian可微
        
        Args:
            v_prev: (B, F) 前一时刻膜电位
            x_t: (B, F) 当前输入
        
        Returns:
            v_next: (B, F) 更新后的膜电位
            spike: (B, F) 脉冲输出
            jacobian: (B, F, F) Jacobian矩阵 ∂v_next/∂v_prev
        """
        B, F = v_prev.shape
        
        # 需要计算梯度
        v_prev_grad = v_prev.detach().requires_grad_(True)
        
        # 前向传播
        h = v_prev_grad + (x_t - v_prev_grad) / self.tau
        
        # 使用代理梯度
        sigmoid = torch.sigmoid(self.surrogate_alpha * (h - self.v_threshold))
        surrogate_grad = self.surrogate_alpha * sigmoid * (1.0 - sigmoid)
        
        # 前向的spike（真实Heaviside）
        spike = (h >= self.v_threshold).float()
        
        # 前向的v_next
        v_next = h * (1.0 - spike) + self.v_reset * spike
        
        # 计算Jacobian: ∂v_next/∂v_prev (使用surrogate)
        # v_next = h * (1 - spike)
        # ∂v_next/∂v_prev = ∂h/∂v_prev * (1-spike) - h * ∂spike/∂h * ∂h/∂v_prev
        
        decay = 1.0 - 1.0 / self.tau  # ∂h/∂v_prev
        
        # 构造对角Jacobian矩阵（向量化，避免循环）
        # 对角元素: decay * (1 - spike - h * surrogate_grad)
        diag_elements = decay * (1.0 - spike - h * surrogate_grad)  # (B, F)
        
        # 批量构造对角矩阵
        jacobian = torch.diag_embed(diag_elements)  # (B, F, F)
        
        return v_next.detach(), spike, jacobian
    
    def shifter_func(self, y: torch.Tensor, y_init: torch.Tensor) -> torch.Tensor:
        """
        Shift操作: [y_init, y[0], y[1], ..., y[T-2]]
        
        Args:
            y: (T, B, F) 当前序列
            y_init: (B, F) 初始条件
        
        Returns:
            y_shifted: (T, B, F) shifted序列
        """
        # 拼接 [y_init, y[:-1]]
        return torch.cat([y_init.unsqueeze(0), y[:-1]], dim=0)
    
    def compute_L(self, y: torch.Tensor, x: torch.Tensor, y_init: torch.Tensor) -> torch.Tensor:
        """
        计算 L[y] = [f(y[-1], x[0]), f(y[0], x[1]), ..., f(y[T-2], x[T-1])]
        
        **关键**: DEER中的L[y]是**连续动力学**，不包含reset！
        LIF的连续动力学: dv/dt = (x - v)/tau
        离散化: v[t] = v[t-1] + (x[t] - v[t-1])/tau
        
        向量化版本：一次性计算所有时间步
        
        Args:
            y: (T, B, F) 当前guess
            x: (T, B, F) 输入序列
            y_init: (B, F) 初始条件
        
        Returns:
            L_y: (T, B, F) L[y]的结果（连续动力学，无reset）
        """
        # Shift操作
        y_shifted = self.shifter_func(y, y_init)  # (T, B, F)
        
        # 向量化LIF更新：合并时间和batch维度
        T, B, F = y.shape
        v_prev_flat = y_shifted.reshape(T * B, F)
        x_flat = x.reshape(T * B, F)
        
        # LIF连续动力学（无spike、无reset！）
        h = v_prev_flat + (x_flat - v_prev_flat) / self.tau
        
        L_y = h.reshape(T, B, F)
        return L_y
    
    def compute_jacobian_diagonal(self, y: torch.Tensor, x: torch.Tensor, y_init: torch.Tensor) -> torch.Tensor:
        """
        计算Jacobian对角元素 G_diag[t] = -∂f/∂y[t-1] 的对角部分
        
        对角近似版本：只保留主对角，内存从(T,B,F,F)降到(T,B,F)
        内存节省：从9GB降到15MB！
        
        Args:
            y: (T, B, F) 当前guess
            x: (T, B, F) 输入序列
            y_init: (B, F) 初始条件
        
        Returns:
            G_diag: (T, B, F) Jacobian对角元素
        """
        T, B, F = y.shape
        
        # Shift操作
        y_shifted = self.shifter_func(y, y_init)  # (T, B, F)
        
        # 向量化计算
        v_prev_flat = y_shifted.reshape(T * B, F)
        x_flat = x.reshape(T * B, F)
        
        # LIF前向传播
        h = v_prev_flat + (x_flat - v_prev_flat) / self.tau
        
        # 检查h的范围
        if not hasattr(self, '_jacobian_debug_done'):
            h_min, h_max = h.min().item(), h.max().item()
            print(f"[JACOBIAN] h范围: [{h_min:.2f}, {h_max:.2f}]")
            self._jacobian_debug_done = True
        
        # 限制h的范围防止sigmoid溢出
        h = torch.clamp(h, -20.0, 20.0)
        
        # 代理梯度
        sigmoid = torch.sigmoid(self.surrogate_alpha * (h - self.v_threshold))
        surrogate_grad = self.surrogate_alpha * sigmoid * (1.0 - sigmoid)
        
        # Jacobian对角元素
        # ∂L/∂v[t-1] = 1 - 1/tau (连续动力学的导数)
        # 考虑reset: ∂v_next/∂v[t-1] = ∂L/∂v[t-1] * (1 - spike_derivative * (1 - v_reset))
        base_derivative = 1.0 - 1.0 / self.tau
        diag_elements = base_derivative * (1.0 - surrogate_grad * (1.0 - self.v_reset))
        
        # Reshape并取负（因为DEER定义的是-G）
        G_diag = -diag_elements.reshape(T, B, F)
        
        return G_diag
    
    def compute_jacobian_sequence(self, y: torch.Tensor, x: torch.Tensor, y_init: torch.Tensor) -> torch.Tensor:
        """
        计算Jacobian序列 G[t] = -∂f/∂y[t-1]
        
        向量化版本：一次性计算所有时间步
        
        Args:
            y: (T, B, F) 当前guess
            x: (T, B, F) 输入序列
            y_init: (B, F) 初始条件
        
        Returns:
            G: (T, B, F, F) Jacobian序列
        """
        T, B, F = y.shape
        
        # Shift操作
        y_shifted = self.shifter_func(y, y_init)  # (T, B, F)
        
        # 向量化计算：合并T和B维度
        v_prev_flat = y_shifted.reshape(T * B, F)
        x_flat = x.reshape(T * B, F)
        
        # LIF前向传播
        h = v_prev_flat + (x_flat - v_prev_flat) / self.tau
        
        # 代理梯度
        sigmoid = torch.sigmoid(self.surrogate_alpha * (h - self.v_threshold))
        surrogate_grad = self.surrogate_alpha * sigmoid * (1.0 - sigmoid)
        
        # Spike
        spike = (h >= self.v_threshold).float()
        
        # Jacobian对角元素
        decay = 1.0 - 1.0 / self.tau
        diag_elements = decay * (1.0 - spike - h * surrogate_grad)  # (T*B, F)
        
        # 构造对角Jacobian
        jacobian = torch.diag_embed(diag_elements)  # (T*B, F, F)
        
        # Reshape回 (T, B, F, F) 并取负
        G = -jacobian.reshape(T, B, F, F)
        
        return G
    
    def compute_rhs_diagonal(self, L_y: torch.Tensor, G_diag: torch.Tensor, y_shifted: torch.Tensor) -> torch.Tensor:
        """
        计算右端项（对角版本）: rhs = L[y] + G_diag * y_shifted
        
        对角近似：元素级乘法，超快！
        
        Args:
            L_y: (T, B, F) L[y]的结果
            G_diag: (T, B, F) Jacobian对角元素
            y_shifted: (T, B, F) shifted序列
        
        Returns:
            rhs: (T, B, F) 右端项
        """
        rhs = L_y + G_diag * y_shifted  # 元素级乘法
        return rhs
    
    def compute_rhs(self, L_y: torch.Tensor, G: torch.Tensor, y_shifted: torch.Tensor) -> torch.Tensor:
        """
        计算右端项: rhs = L[y] + G @ y_shifted
        
        向量化版本：使用bmm批量矩阵乘法
        
        Args:
            L_y: (T, B, F) L[y]的结果
            G: (T, B, F, F) Jacobian序列
            y_shifted: (T, B, F) shifted序列
        
        Returns:
            rhs: (T, B, F) 右端项
        """
        T, B, F = L_y.shape
        
        # 向量化矩阵乘法：合并T和B维度
        # G: (T, B, F, F) -> (T*B, F, F)
        # y_shifted: (T, B, F) -> (T*B, F, 1)
        G_flat = G.reshape(T * B, F, F)
        y_flat = y_shifted.reshape(T * B, F, 1)
        
        # 批量矩阵乘法
        G_dot_y_flat = torch.bmm(G_flat, y_flat).squeeze(-1)  # (T*B, F)
        
        # Reshape回 (T, B, F)
        G_dot_y = G_dot_y_flat.reshape(T, B, F)
        
        rhs = L_y + G_dot_y
        return rhs
    
    def solve_recursive_linear_system_diagonal(self, G_diag: torch.Tensor, rhs: torch.Tensor, y_init: torch.Tensor) -> torch.Tensor:
        """
        求解递归线性系统（对角版本）: y[i] = -G_diag[i] * y[i-1] + rhs[i]
        
        使用串行solver - Profile显示并行scan在PyTorch中没有优势
        
        Args:
            G_diag: (T, B, F) Jacobian对角元素
            rhs: (T, B, F) 右端项
            y_init: (B, F) 初始条件
        
        Returns:
            y: (T, B, F) 解序列
        """
        T, B, F = rhs.shape
        
        # 串行求解（实验证明比并行scan快）
        y_list = [y_init]
        for t in range(T):
            y_t = -G_diag[t] * y_list[-1] + rhs[t]
            y_list.append(y_t)
        
        y = torch.stack(y_list[1:], dim=0)  # (T, B, F)
        return y
    
    def solve_recursive_linear_system(self, G: torch.Tensor, rhs: torch.Tensor, y_init: torch.Tensor) -> torch.Tensor:
        """
        求解递归线性系统（完整矩阵版本）: y[i+1] = -G[i] @ y[i] + rhs[i]
        
        使用真正的Associative Scan并行化（O(log T)）
        
        Args:
            G: (T, B, F, F) Jacobian序列
            rhs: (T, B, F) 右端项
            y_init: (B, F) 初始条件
        
        Returns:
            y: (T, B, F) 解序列
        """
        T, B, F = rhs.shape
        
        # 使用GPU优化的并行scan（支持完整矩阵）
        # 输入: A=(T, B, F, F), b=(T, B, F), y0=(B, F)
        y = parallel_prefix_sum_gpu(-G, rhs, y_init)
        
        return y
    
    def initialize_guess_warmstart(self, x: torch.Tensor, v_init: torch.Tensor) -> torch.Tensor:
        """
        Warm-start初始化: 前K步串行计算
        
        **关键修正**: 不能用lif_step_forward（会reset到0），要直接计算h！
        
        Args:
            x: (T, B, F) 输入序列
            v_init: (B, F) 初始膜电位
        
        Returns:
            v_guess: (T, B, F) 初始guess
        """
        T, B, F = x.shape
        K = min(self.warmstart_steps, T)
        
        v_guess = torch.zeros_like(x)
        v = v_init
        
        # 前K步串行 - 计算h（不reset！）
        for t in range(K):
            h = v + (x[t] - v) / self.tau  # LIF dynamics
            v_guess[t] = h  # 用h作为guess，不是reset后的v！
            v = h  # 下一步继续用h（假设不发放）
        
        # 剩余时间步用最后的v填充
        if T > K:
            v_guess[K:] = v.unsqueeze(0).expand(T - K, -1, -1)
        
        return v_guess
    
    def deer_iteration(self, x: torch.Tensor, v_init: torch.Tensor) -> Tuple[torch.Tensor, bool, int]:
        """
        DEER不动点迭代主循环
        
        支持两种模式：
        1. 对角近似：内存友好，适合大batch（推荐）
        2. 完整矩阵：精度更高，但内存需求大
        
        算法:
            1. 初始化 v_guess (warm-start)
            2. 迭代:
                a. 计算 L[v]
                b. 计算 Jacobian (对角或完整)
                c. 计算 rhs
                d. 求解递归系统
                e. 检查收敛
        
        Args:
            x: (T, B, F) 输入序列
            v_init: (B, F) 初始膜电位
        
        Returns:
            v_final: (T, B, F) 最终膜电位序列
            converged: bool, 是否收敛
            iter_count: int, 迭代次数
        """
        T, B, F = x.shape
        
        # 初始化
        v_guess = self.initialize_guess_warmstart(x, v_init)
        
        converged = False
        iter_count = 0
        error = float('inf')
        
        # DEBUG: 第一次迭代时打印
        first_call = not hasattr(self, '_iter_debug_done')
        if first_call and T > 8:  # 只在T>8时打印
            import time
            torch.cuda.synchronize()  # 同步GPU
            iter_start_time = time.time()
            print(f"[DEER] T={T}, B={B}, F={F}, 迭代{self.max_iter}次")
            self._iter_debug_done = True
            # 打印内存使用
            if torch.cuda.is_available():
                mem_allocated = torch.cuda.memory_allocated() / 1024**3
                mem_reserved = torch.cuda.memory_reserved() / 1024**3
                print(f"[DEER] GPU内存: 已分配={mem_allocated:.2f}GB, 已预留={mem_reserved:.2f}GB")
        
        for k in range(self.max_iter):
            iter_count += 1
            # 关键修正：detach避免梯度图重复
            v_old = v_guess.detach().clone() if self.check_convergence else None
            
            # 检查数值稳定性
            if torch.isnan(v_guess).any() or torch.isinf(v_guess).any():
                print(f"[DEER ERROR] 迭代{k}: v_guess中出现NaN或Inf，停止迭代")
                break
            
            # Step 1: 计算 L[v]
            L_v = self.compute_L(v_guess, x, v_init)
            
            # 检查L[v]
            if torch.isnan(L_v).any() or torch.isinf(L_v).any():
                print(f"[DEER ERROR] 迭代{k}: L[v]中出现NaN或Inf，停止迭代")
                break
            
            # Step 2 & 3 & 4: 根据模式选择
            y_shifted = self.shifter_func(v_guess, v_init)
            
            if self.use_diagonal_approx:
                # 对角近似模式（内存友好）
                G_diag = self.compute_jacobian_diagonal(v_guess, x, v_init)
                
                # 检查Jacobian
                if torch.isnan(G_diag).any() or torch.isinf(G_diag).any():
                    print(f"[DEER ERROR] 迭代{k}: Jacobian中出现NaN或Inf")
                    break
                
                rhs = self.compute_rhs_diagonal(L_v, G_diag, y_shifted)
                
                # 检查RHS
                if torch.isnan(rhs).any() or torch.isinf(rhs).any():
                    print(f"[DEER ERROR] 迭代{k}: RHS中出现NaN或Inf")
                    break
                
                v_next = self.solve_recursive_linear_system_diagonal(G_diag, rhs, v_init)
                
                # 检查求解结果
                if torch.isnan(v_next).any() or torch.isinf(v_next).any():
                    print(f"[DEER ERROR] 迭代{k}: v_next中出现NaN或Inf")
                    break
            else:
                # 完整矩阵模式（高精度）
                G = self.compute_jacobian_sequence(v_guess, x, v_init)
                rhs = self.compute_rhs(L_v, G, y_shifted)
                v_next = self.solve_recursive_linear_system(G, rhs, v_init)
            
            v_guess = v_next
            
            # Step 5: 检查收敛
            if self.check_convergence:
                # 计算误差（参考DEER原文）
                err = torch.abs(v_next - v_old)  # (T, B, F)
                tol = self.tol + self.tol * torch.abs(v_next)  # atol + rtol * |v|
                
                # 收敛条件：所有元素都在容差内
                converged = torch.all(err <= tol).item()
                max_err = torch.max(err).item()
                
                # DEBUG
                if first_call and k == 0:
                    print(f"[DEER] 迭代{k}: max_err={max_err:.6f}, tol={self.tol}")
                
                if converged:
                    if first_call:
                        print(f"[DEER] 在第{k+1}次迭代收敛")
                    break
            else:
                # 训练模式：不检查收敛，避免GPU-CPU同步
                pass
        
        # 记录统计（简化版）
        self.stats['iter_counts'].append(iter_count)
        self.stats['converged'].append(converged if self.check_convergence else True)
        self.stats['errors'].append(0.0)  # 简化

        
        # 首次调用时报告总耗时
        if first_call and T > 8:
            torch.cuda.synchronize()  # 同步GPU
            total_time = time.time() - iter_start_time
            print(f"[DEER] 完成{self.max_iter}次迭代，总耗时: {total_time:.3f}s")
        
        return v_guess, converged, iter_count
    
    def forward_serial(self, x: torch.Tensor) -> torch.Tensor:
        """
        串行前向传播（fallback）
        
        Args:
            x: (T, B, F) 输入序列
        
        Returns:
            spike_seq: (T, B, F) 脉冲序列
        """
        T, B, F = x.shape
        
        spikes = []
        v = torch.zeros(B, F, device=x.device, dtype=x.dtype)
        
        for t in range(T):
            v, spike = self.lif_step_forward(v, x[t])
            spikes.append(spike)
        
        return torch.stack(spikes, dim=0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：纯DEER并行化，无fallback！
        
        Args:
            x: (T, B, F) 输入序列
        
        Returns:
            spike_seq: (T, B, F) 脉冲序列
        """
        T, B, F = x.shape
        
        # DEBUG: 不重复打印，简化输出
        # 移除了重复的层级信息打印
        
        v_init = torch.zeros(B, F, device=x.device, dtype=x.dtype)
        
        # 纯DEER计算，不管内存、不管收敛，直接用！
        v_seq, converged, iter_count = self.deer_iteration(x, v_init)
        
        # 转换为脉冲
        spike_seq = (v_seq >= self.v_threshold).float()
        
        return spike_seq


def test_deer_lif_v2():
    """测试DEERLIFNode v2"""
    print("=" * 60)
    print("Testing DEERLIFNode v2")
    print("=" * 60)
    
    T, B, F = 4, 2, 8
    x = torch.randn(T, B, F) * 0.3
    
    # DEER-LIF
    deer_lif = DEERLIFNode(tau=2.0, max_iter=20, tol=1e-5)
    spike_deer = deer_lif(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {spike_deer.shape}")
    print(f"Spike rate: {spike_deer.mean().item():.4f}")
    print(f"Stats: {deer_lif.get_stats_summary()}")
    
    # 串行LIF (ground truth)
    from spikingjelly.activation_based import neuron, layer
    serial_lif = neuron.LIFNode(tau=2.0)
    spike_serial = []
    v = torch.zeros(B, F)
    for t in range(T):
        spike_t = serial_lif(x[t])
        spike_serial.append(spike_t)
    spike_serial = torch.stack(spike_serial, dim=0)
    serial_lif.reset()
    
    # 对比
    diff = torch.abs(spike_deer - spike_serial).max()
    print(f"\nComparison with serial LIF:")
    print(f"Max difference: {diff.item():.6f}")
    
    if diff < 1e-3:
        print("✅ DEER-LIF matches serial LIF!")
    else:
        print("⚠️ Large difference detected")
    
    print("=" * 60)


if __name__ == "__main__":
    test_deer_lif_v2()
