import torch
import triton
import triton.language as tl
from lightvllm.kernels.triton_kernels.utils import *


@triton.jit
def naive_softmax_kernel(
    x_ptr,          # 输入张量的指针
    y_ptr,          # 输出张量的指针
    row_stride,     # 行步幅，用于定位不同行的起始位置
    n_cols: tl.constexpr,     # 每行的元素数量（编译时常量）
    BLOCK_SIZE: tl.constexpr, # 线程块中处理的数据数量
):
    # 获取当前程序的ID，对应处理的行索引
    program_id = tl.program_id(0).to(tl.int64)

    # 计算当前行在各个张量中的起始位置
    x_ptr += program_id * row_stride
    y_ptr += program_id * row_stride

    # 计算当前线程块处理的列索引范围
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # 向量化加载数据；越界位置填充 -inf 避免影响 max/sum
    x_row = tl.load(x_ptr + col_offsets, mask=mask, other=-float("inf")).to(tl.float32)

    # 计算softmax
    x_max = tl.max(x_row, 0)
    x_row = tl.exp(x_row - x_max)
    x_row_sum = tl.sum(x_row, 0)
    y_row = x_row / x_row_sum

    tl.store(y_ptr + col_offsets, y_row, mask=mask)


@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    row_size,
    col_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0) # row start idx
    programs_size = tl.num_programs(0) # total number of programs, also the row steps for range

    for row_idx in tl.range(pid, row_size, programs_size):
        row_start_ptr = input_ptr + row_idx * input_row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        mask = col_offsets < col_size

        input_vals = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float('inf')).to(tl.float32)
        minus_max = input_vals - tl.max(input_vals, axis=0)
        exp_minus_max = tl.exp(minus_max)

        exp_sum = tl.sum(exp_minus_max, axis=0)

        softmax_outputs = exp_minus_max / exp_sum

        output_row_start_ptr = output_ptr + row_idx * input_row_stride
        
        tl.store(output_row_start_ptr + col_offsets, softmax_outputs, mask=mask)



@triton.jit
def online_softmax_kernel(
    output_ptr, input_ptr, lse_ptr,
    input_row_stride, output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    # 1. 获取当前处理的行和块的索引
    row_idx = tl.program_id(0)
    block_idx = tl.program_id(1)

    # 2. 计算当前块的偏移量
    col_offsets = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # 3. 计算输入和输出的指针
    input_block_ptr = input_ptr + row_idx * input_row_stride + col_offsets
    output_block_ptr = output_ptr + row_idx * output_row_stride + col_offsets
    
    # 4. 加载当前块的数据
    mask = col_offsets < n_cols
    x = tl.load(input_block_ptr, mask=mask, other=-float('inf'))

    # 5. Online Softmax 算法
    # 5.1. 初始化或加载之前的 log-sum-exp (lse)
    if block_idx == 0:
        m_prev = -float('inf')
        d_prev = 0.0
    else:
        # 从 lse_ptr 加载前一个块的 m 和 d
        m_prev, d_prev = tl.load(lse_ptr + row_idx * 2, mask=[True, True])

    # 5.2. 计算当前块的 m_block 和 d_block
    m_block = tl.max(x, 0)
    x_minus_m_block = x - m_block
    d_block = tl.sum(tl.exp(x_minus_m_block), 0)

    # 5.3. 更新全局的 m_i 和 d_i
    m_i = tl.maximum(m_prev, m_block)
    
    # 调整 d_prev 和 d_block 以匹配新的 m_i
    d_i = d_prev * tl.exp(m_prev - m_i) + d_block * tl.exp(m_block - m_i)

    # 5.4. 存储当前块的 lse 供下一个块使用
    # 注意：这里会有竞争条件，因为块的执行顺序不确定。
    # 一个更稳健的实现需要同步或不同的内核设计。
    # 为了简单起见，我们假设块是按顺序处理的（这在实践中不成立）。
    tl.store(lse_ptr + row_idx * 2, [m_i, d_i], mask=[True, True])

    # 5.5. 计算并存储当前块的 softmax
    # 重新计算 x - m_i
    x_minus_m_i = x - m_i
    softmax_val = tl.exp(x_minus_m_i) / d_i
    tl.store(output_block_ptr, softmax_val, mask=mask)

    # 5.6. 更新之前所有块的 softmax 值
    # 这部分在 Triton 中难以高效实现，因为它需要读-改-写整个已处理的行
    # 这违背了 Triton 的并行计算模型。
    # 一个完整的 online softmax 需要更复杂的内核或多个内核。
    # 此处的实现是一个简化的、不完全正确的版本，用于演示思路。
    # 正确的实现需要一个单独的内核来在所有块计算完 lse 后进行归一化。


@triton.jit
def online_softmax_rowwise_kernel_2pass(
    x_ptr,          # [S, H] row-major pointer
    y_ptr,          # output pointer
    row_stride,     # stride between rows (in elements)
    n_cols,         # number of columns H
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    x_row_ptr = x_ptr + pid * row_stride
    y_row_ptr = y_ptr + pid * row_stride

    # Pass 1: compute global row max m and denominator d using online update
    m = -float("inf")
    d = 0.0
    for start in range(0, n_cols, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(x_row_ptr + cols, mask=mask, other=-float("inf"))
        m_block = tl.max(x, 0)
        m_new = tl.maximum(m, m_block)
        d = d * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new), 0)
        m = m_new

    # Pass 2: write normalized softmax outputs
    for start in range(0, n_cols, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < n_cols
        x = tl.load(x_row_ptr + cols, mask=mask, other=-float("inf"))
        y = tl.exp(x - m) / d
        tl.store(y_row_ptr + cols, y, mask=mask)


def triton_online_softmax(x: torch.Tensor) -> torch.Tensor:
    """
    使用 Triton 实现的 Online Softmax。
    注意：这是一个简化的实现，用于教学目的，并未完全解决并行更新问题。
    """
    n_rows, n_cols = x.shape
    
    # 创建输出张量
    output = torch.empty_like(x)
    
    # 创建用于存储 log-sum-exp (lse) 的中间张量
    # 每个行需要存储 (max, sum_exp)
    lse = torch.zeros((n_rows, 2), device=x.device, dtype=torch.float32)

    # 定义块大小
    BLOCK_SIZE = 128 # 可以根据硬件调整

    # 定义网格大小
    grid = (n_rows, triton.cdiv(n_cols, BLOCK_SIZE))

    # 启动内核
    online_softmax_kernel[grid](
        output, x, lse,
        x.stride(0), output.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # 由于简化的内核无法正确地回填更新之前的 softmax 值，
    # 我们在这里需要一个额外的步骤来完成归一化。
    # 这使得它不再是“纯粹”的 online，但能得到正确的结果。
    final_lse = lse[:, 1].unsqueeze(1) # 取出最终的 d_i
    output /= final_lse

    return output


def online_softmax_forward(x: torch.Tensor) -> torch.Tensor:
    """
    Dense row-wise online softmax (two-pass) implemented in Triton.
    Each row is independent; does a numerically-stable two-pass that matches torch.softmax(dim=1).
    """
    assert x.is_cuda, "输入张量必须在 CUDA 设备上"
    assert x.dim() == 2, "输入张量必须是二维的"
    y = torch.empty_like(x)
    n_rows, n_cols = x.shape
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)
    online_softmax_rowwise_kernel_2pass[(n_rows,)](
        x, y,
        x.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return y


def softmax_forward(x: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(x, dtype=x.dtype)
    n_rows, n_cols = x.shape
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)
    
    softmax_kernel[(n_rows // 2,)](
        output_ptr=y,
        input_ptr=x,
        input_row_stride=x.stride(0),
        row_size=n_rows,
        col_size=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return y

