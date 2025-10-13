import torch
from torch.utils.cpp_extension import load
import os

os.environ["TORCH_CUDA_ARCH_LIST"] = "Ada"  # 设置支持的CUDA架构列表

# 获取当前文件所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 加载 CUDA 内核
# JIT (Just-In-Time) 编译: PyTorch 会在第一次调用时自动编译这个 CUDA 源码文件。
# name: 编译后生成的 Python 模块名。
# sources: CUDA 源文件路径列表。
# verbose=True: 在编译时打印详细信息，便于调试。
cuda_kernels = load(
    name="lightvllm_cuda_kernels",
    sources=[os.path.join(current_dir, "activation.cu")],
    verbose=True
)

def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """
    调用 CUDA 内核执行 SiLU and Multiply (SwiGLU) 操作。

    这个函数是 CUDA C++ 实现的 Python 包装器。
    它接收一个 PyTorch 张量，调用 JIT 编译好的 C++ 函数，
    然后返回结果张量。

    Args:
        x (torch.Tensor): 输入张量，维度通常是 [..., 2 * intermediate_size]。
                          它被视为 gate 和 up 两部分的拼接。

    Returns:
        torch.Tensor: 激活后的输出张量，维度是 [..., intermediate_size]。
    """
    # 调用编译好的 C++ 模块中的 "silu_and_mul" 函数
    return cuda_kernels.silu_and_mul(x)

# 我们可以导出一个更符合 Python 风格的名字
swiglu = silu_and_mul
