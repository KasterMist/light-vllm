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
    name="lightvllm_layernorm_kernels",
    sources=[os.path.join(current_dir, "layernorm.cu")],
    verbose=True
)

def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    执行 RMS 归一化。
    
    Args:
        x (torch.Tensor): 输入张量，维度为 [batch_size, hidden_size]
        weight (torch.Tensor): 权重参数，维度为 [hidden_size]
        eps (float): 防止除零的小数值，默认为 1e-6
        
    Returns:
        torch.Tensor: 归一化后的张量，维度与输入相同
    """
    return cuda_kernels.rms_norm(x, weight, eps)

def add_rms_norm(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """
    执行带有残差连接的 RMS 归一化 (Add & Norm)。
    
    先执行残差连接 (x + residual)，然后对结果进行 RMS 归一化。
    
    Args:
        x (torch.Tensor): 输入张量，维度为 [batch_size, hidden_size]
        residual (torch.Tensor): 残差张量，维度与 x 相同
        weight (torch.Tensor): 权重参数，维度为 [hidden_size]
        eps (float): 防止除零的小数值，默认为 1e-6
        
    Returns:
        tuple[torch.Tensor, torch.Tensor]: 
            - 第一个元素：归一化后的张量
            - 第二个元素：新的残差 (x + residual 的结果)
    """
    return cuda_kernels.add_rms_norm(x, residual, weight, eps)