"""
PyTorch实现的Associative Scan
严格按照DEER的maths.py实现，逐行翻译JAX代码
"""
import torch
from typing import Callable, Tuple, List


def associative_scan(
    fn: Callable[[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]], Tuple[torch.Tensor, ...]],
    elems: Tuple[torch.Tensor, ...],
    axis: int = 0
) -> Tuple[torch.Tensor, ...]:
    """
    PyTorch版本的associative scan，严格按照JAX实现
    
    参考：deer/deer/maths.py 中的 associative_scan 函数
    
    算法原理（Blelloch 1990）：
    对于结合律操作 ⊗，计算前缀积：
    result[i] = elem[0] ⊗ elem[1] ⊗ ... ⊗ elem[i]
    
    通过递归分治实现O(log n)并行复杂度：
    1. 合并相邻元素对：[e0, e1, e2, e3, ...] -> [e0⊗e1, e2⊗e3, ...]
    2. 递归处理reduced序列
    3. 重建完整结果
    
    Args:
        fn: 结合律操作 (elem_i, elem_j) -> elem_combined
        elems: 输入元素tuple，每个shape为(T, ...)
        axis: 扫描维度（默认0）
    
    Returns:
        前缀积结果，shape与输入相同
    """
    if not elems:
        return elems
    
    num_elems = elems[0].shape[axis]
    
    def get_idxs(elem, slc):
        """构造沿axis维度的切片索引"""
        lst = [slice(None)] * elem.ndim
        lst[axis] = slc
        return tuple(lst)
    
    def _scan(elems):
        """递归扫描函数（严格按照JAX实现）"""
        num_elems = elems[0].shape[axis]
        
        if num_elems < 2:
            return elems
        
        # 步骤1: 合并相邻元素对
        # JAX代码：slice(0, -1, 2) 表示 [0:n-1:2]，即偶数位置（除了可能的最后一个）
        # JAX代码：slice(1, None, 2) 表示 [1::2]，即奇数位置
        left = tuple(elem[get_idxs(elem, slice(0, -1, 2))] for elem in elems)
        right = tuple(elem[get_idxs(elem, slice(1, None, 2))] for elem in elems)
        
        # 合并：reduced[i] = left[i] ⊗ right[i] = elem[2i] ⊗ elem[2i+1]
        reduced_elems = fn(left, right)
        
        # 步骤2: 递归处理reduced序列
        odd_elems = _scan(reduced_elems)
        
        # 步骤3: 重建even位置元素
        if num_elems % 2 == 0:
            # 偶数个元素：even = odd[:-1] ⊗ elem[2::2]
            even_elems = fn(
                tuple(e[get_idxs(e, slice(0, -1))] for e in odd_elems),
                tuple(e[get_idxs(e, slice(2, None, 2))] for e in elems)
            )
        else:
            # 奇数个元素：even = odd ⊗ elem[2::2]
            even_elems = fn(
                odd_elems,
                tuple(e[get_idxs(e, slice(2, None, 2))] for e in elems)
            )
        
        # 步骤4: 第一个元素保持不变（elem[0]不需要combine）
        even_elems = tuple(
            torch.cat([elem[get_idxs(elem, slice(0, 1))], result], dim=axis)
            for elem, result in zip(elems, even_elems)
        )
        
        # 步骤5: 交错合并even和odd
        return tuple(_interleave(even, odd, axis) for even, odd in zip(even_elems, odd_elems))
    
    return tuple(_scan(elems))



def _interleave(a: torch.Tensor, b: torch.Tensor, axis: int) -> torch.Tensor:
    """
    交错合并两个张量（严格按照JAX实现）
    
    JAX实现使用padding：
    - a在偶数位置(0, 2, 4, ...)：pad为 (0, 1 if len==len else 0, 1)
    - b在奇数位置(1, 3, 5, ...)：pad为 (1, 0 if len==len else 1, 1)
    
    Args:
        a: shape[axis] = n 或 n+1
        b: shape[axis] = n
        axis: 交错的维度
    
    Returns:
        交错后的张量，shape[axis] = a.shape[axis] + b.shape[axis]
    """
    assert a.shape[axis] == b.shape[axis] or a.shape[axis] == b.shape[axis] + 1
    
    # PyTorch的pad格式：从最后一个维度开始，每对表示(left, right)
    # 需要将axis转换为从后往前的索引
    ndim = a.ndim
    axis_from_end = ndim - axis - 1  # axis=0 -> ndim-1, axis=ndim-1 -> 0
    
    # 构造padding：所有维度都是(0,0)，除了axis维度
    a_pad = [0] * (2 * ndim)
    b_pad = [0] * (2 * ndim)
    
    # axis维度的padding（从后往前的第axis_from_end个维度）
    pad_idx = 2 * axis_from_end  # padding list中对应的起始位置
    
    if a.shape[axis] == b.shape[axis]:
        # 长度相等：a需要在后面pad 1，stride为2
        a_pad[pad_idx + 1] = 1  # right pad
        # b需要在前面pad 1，stride为2
        b_pad[pad_idx] = 1      # left pad
    else:
        # a比b长1：a后面pad 0，b前后各pad 1
        a_pad[pad_idx + 1] = 0
        b_pad[pad_idx] = 1      # left pad
        b_pad[pad_idx + 1] = 1  # right pad
    
    # PyTorch的pad只支持最后几个维度，需要先permute
    # 将axis移到最后
    perm = list(range(ndim))
    perm[axis], perm[-1] = perm[-1], perm[axis]
    
    a_perm = a.permute(perm)
    b_perm = b.permute(perm)
    
    # 现在在最后一个维度上pad（每隔一个位置）
    # 但PyTorch的pad不支持stride，需要手动实现交错
    
    # 简化实现：直接构造输出张量（保持梯度）
    total_len = a.shape[axis] + b.shape[axis]
    out_shape = list(a.shape)
    out_shape[axis] = total_len
    
    # 使用torch.empty代替zeros，然后填充（避免破坏计算图）
    out = torch.empty(out_shape, dtype=a.dtype, device=a.device)
    
    # 填充偶数位置（a）
    idx_a = [slice(None)] * ndim
    idx_a[axis] = slice(0, None, 2)
    out[tuple(idx_a)] = a
    
    # 填充奇数位置（b）
    idx_b = [slice(None)] * ndim
    idx_b[axis] = slice(1, None, 2)
    out[tuple(idx_b)] = b
    
    return out


def matmul_recursive(
    mats: torch.Tensor,
    vecs: torch.Tensor,
    y0: torch.Tensor
) -> torch.Tensor:
    """
    并行求解递归矩阵乘法: y[i+1] = mats[i] @ y[i] + vecs[i]
    
    使用associative scan实现O(log T)并行化
    
    这是DEER线性求解器的核心！
    
    Args:
        mats: (T, B, F, F) 或 (T, B, F) 如果是对角
        vecs: (T, B, F) 右端项
        y0: (B, F) 初始条件
    
    Returns:
        y: (T+1, B, F) 解序列，包含y0在开头
    
    Example:
        # 求解 y[i+1] = -G[i] * y[i] + rhs[i]
        G = torch.randn(T, B, F)  # 对角Jacobian
        rhs = torch.randn(T, B, F)
        y0 = torch.zeros(B, F)
        y = matmul_recursive(-G, rhs, y0)  # (T+1, B, F)
    """
    T, B = vecs.shape[:2]
    F = y0.shape[-1]
    
    # 检查是否是对角矩阵（3D）还是完整矩阵（4D）
    is_diagonal = (mats.ndim == 3)
    
    if is_diagonal:
        # 对角情况：元素级乘法
        def scan_fn(elem_i, elem_j):
            a_i, b_i = elem_i  # a: (*, F), b: (*, F)
            a_j, b_j = elem_j
            a_new = a_j * a_i  # 元素级乘法
            b_new = a_j * b_i + b_j
            return (a_new, b_new)
    else:
        # 完整矩阵情况：矩阵乘法
        def scan_fn(elem_i, elem_j):
            a_i, b_i = elem_i  # a: (*, F, F), b: (*, F)
            a_j, b_j = elem_j
            a_new = torch.matmul(a_j, a_i)  # 矩阵乘法
            b_new = torch.matmul(a_j, b_i.unsqueeze(-1)).squeeze(-1) + b_j
            return (a_new, b_new)
    
    # 添加单位初始元素
    if is_diagonal:
        eye = torch.ones(1, B, F, dtype=mats.dtype, device=mats.device)
    else:
        eye = torch.eye(F, dtype=mats.dtype, device=mats.device).unsqueeze(0).unsqueeze(0)
        eye = eye.expand(1, B, F, F)
    
    first_a = torch.cat([eye, mats], dim=0)  # (T+1, B, F[, F])
    first_b = torch.cat([y0.unsqueeze(0), vecs], dim=0)  # (T+1, B, F)
    
    # 执行associative scan
    _, result = associative_scan(scan_fn, (first_a, first_b), axis=0)
    
    return result  # (T+1, B, F)


# 测试代码
if __name__ == "__main__":
    print("测试PyTorch Associative Scan")
    
    # 测试1: 简单的累加
    print("\n测试1: 累加扫描")
    def add_fn(a, b):
        return (a[0] + b[0],)
    
    x = torch.arange(10).unsqueeze(-1).float()  # (10, 1)
    result, = associative_scan(add_fn, (x,))
    expected = torch.cumsum(x, dim=0)
    print(f"输入: {x.squeeze().tolist()}")
    print(f"Associative scan结果: {result.squeeze().tolist()}")
    print(f"期望结果(cumsum): {expected.squeeze().tolist()}")
    print(f"误差: {(result - expected).abs().max().item()}")
    
    # 测试2: 递归线性系统（对角）
    print("\n测试2: 递归线性系统 y[i] = a[i] * y[i-1] + b[i]")
    T, B, F = 8, 2, 3
    a = torch.randn(T, B, F) * 0.5  # 系数
    b = torch.randn(T, B, F)  # 常数项
    y0 = torch.randn(B, F)  # 初始值
    
    # 使用associative scan
    result = matmul_recursive(a, b, y0)
    
    # 串行验证
    y_serial = torch.zeros(T + 1, B, F)
    y_serial[0] = y0
    for i in range(T):
        y_serial[i + 1] = a[i] * y_serial[i] + b[i]
    
    print(f"Associative scan结果shape: {result.shape}")
    print(f"串行计算结果shape: {y_serial.shape}")
    print(f"最大误差: {(result - y_serial).abs().max().item():.6e}")
    print(f"相对误差: {((result - y_serial).abs() / (y_serial.abs() + 1e-8)).max().item():.6e}")
    
    if (result - y_serial).abs().max().item() < 1e-4:
        print("✅ 测试通过！")
    else:
        print("❌ 测试失败！")
        print("差异最大的位置:")
        diff = (result - y_serial).abs()
        max_idx = diff.reshape(-1).argmax()
        t, b, f = max_idx // (B * F), (max_idx // F) % B, max_idx % F
        print(f"  位置: t={t}, b={b}, f={f}")
        print(f"  Scan: {result[t, b, f].item()}")
        print(f"  Serial: {y_serial[t, b, f].item()}")
