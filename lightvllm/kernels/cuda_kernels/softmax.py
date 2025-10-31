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
    sources=[os.path.join(current_dir, "softmax.cu")],
    verbose=True
)

def softmax_per_token(x: torch.Tensor) -> torch.Tensor:
    return cuda_kernels.softmax_per_token(x)

def online_softmax_per_token(x: torch.Tensor) -> torch.Tensor:
    return cuda_kernels.online_softmax_per_token(x)
