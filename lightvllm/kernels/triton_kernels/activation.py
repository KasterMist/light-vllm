import torch
import triton
import triton.language as tl
from lightvllm.kernels.triton_kernels.utils import *

# reference: https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/swiglu.py


@triton.jit
def silu(x):
    """
    Swish/SiLU激活函数的Triton实现
    
    SiLU (Sigmoid Linear Unit) 又称为Swish激活函数：
    SiLU(x) = x * sigmoid(x) = x * (1 / (1 + exp(-x)))
    
    特点：
    - 平滑且可微：相比ReLU，在x=0处平滑过渡
    - 自门控：函数本身具有门控特性
    - 有界下界：当x→-∞时，SiLU(x)→0
    - 无界上界：当x→+∞时，SiLU(x)→x
    
    在SwiGLU中的作用：
    作为门控函数，控制信息流的通过程度
    """
    return x * tl.sigmoid(x)


@triton.jit
def _swiglu_forward_kernel(
    a_ptr,              # 第一个输入张量的指针
    b_ptr,              # 第二个输入张量的指针
    c_ptr,              # 输出张量的指针
    row_stride,         # 行步幅，用于定位不同行的起始位置
    n_cols: tl.constexpr,     # 每行的元素数量（编译时常量）
    BLOCK_SIZE: tl.constexpr, # 线程块大小（编译时常量）
):
    """
    SwiGLU前向传播的Triton内核实现
    
    计算公式：c = SiLU(a) * b = (a * sigmoid(a)) * b
    
    内核设计：
    1. 每个程序（线程块）处理一行数据
    2. 在行内使用向量化操作，提高计算效率
    3. 使用掩码处理边界情况，确保正确性
    
    性能优化：
    - 向量化加载和存储操作
    - 合并内存访问模式
    - 最小化数据类型转换开销
    """
    # 获取当前程序的ID，对应处理的行索引
    program_id = tl.program_id(0).to(tl.int64)

    # 计算当前行在各个张量中的起始位置
    a_ptr += program_id * row_stride
    b_ptr += program_id * row_stride
    c_ptr += program_id * row_stride

    # 计算当前线程块处理的列索引范围
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols  # 掩码，防止越界

    # 向量化加载数据
    # a_row转换为float32确保sigmoid计算的数值精度
    a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)

    # 计算SwiGLU: SiLU(a) * b
    c_row = silu(a_row) * b_row
    
    output_dtype = a_ptr.dtype.element_ty # 获取 tl.bfloat16
    c_row = c_row.to(output_dtype)

    # 向量化存储结果
    tl.store(c_ptr + col_offsets, c_row, mask=mask)


def swiglu_forward(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    SwiGLU激活函数的前向传播主函数
    
    实现SwiGLU(a, b) = SiLU(a) * b = (a * sigmoid(a)) * b
    
    参数：
        a (torch.Tensor): 第一个输入张量，用于门控计算
            形状: (..., hidden_size)
        b (torch.Tensor): 第二个输入张量，被门控的数据
            形状: (..., hidden_size)
            
    返回：
        torch.Tensor: SwiGLU的输出结果，形状与输入相同
        
    使用场景：
        在Transformer的FFN层中，通常的使用方式为：
        1. 输入x通过两个不同的线性层得到a和b
        2. 计算SwiGLU(a, b)作为激活后的结果
        
    实现策略：
        1. 将多维张量重塑为2D，便于并行处理
        2. 每行启动一个Triton程序进行计算
        3. 使用向量化操作提高计算效率
        4. 恢复原始形状并返回结果
        
    性能优势：
        - GPU内核融合：避免多次内存访问
        - 向量化计算：充分利用GPU的并行计算能力
        - 内存访问优化：连续内存访问模式
    """
    original_shape = a.shape

    assert a.shape == b.shape, "输入张量a和b必须形状相同"

    # 确保输入张量在内存中是连续的
    a = a.contiguous()
    b = b.contiguous()

    # 获取最后一个维度的大小(通常是hidden_size)
    n_cols = a.shape[-1]

    # 将张量重塑为2D：(total_elements // n_cols, n_cols)
    # 这样可以将问题简化为对多行数据的并行处理
    a = a.view(-1, n_cols)
    b = b.view(-1, n_cols)

    # 创建输出张量，形状与重塑后的输入相同
    c = torch.empty_like(a)
    n_rows = a.shape[0]

    # 根据列数计算最优的内核配置参数
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)

    # 启动Triton内核进行计算
    # 网格配置：(n_rows,) - 每行启动一个程序
    _swiglu_forward_kernel[(n_rows,)](
        a,                      # 输入张量a
        b,                      # 输入张量b
        c,                      # 输出张量c
        c.stride(-2),       # 输出张量c的行步幅
        n_cols=n_cols,          # 每行的元素数量
        BLOCK_SIZE=BLOCK_SIZE,  # 线程块处理的元素数量
        num_warps=num_warps,    # 每个线程块的warp数量
    )

    return c.view(*original_shape)  # 恢复原始形状并返回结果


def swiglu(x: torch.Tensor):
    """
    SwiGLU激活函数的简化接口
    
    该函数将输入张量切分为两部分，分别作为SwiGLU的两个输入。

    参数：
        x (torch.Tensor): 输入张量，形状通常为 (..., 2 * hidden_size)
    """
    x1, x2 = torch.chunk(x, 2, dim=-1)
    return swiglu_forward(x1, x2)