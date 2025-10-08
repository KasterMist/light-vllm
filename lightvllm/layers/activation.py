import torch
from torch import nn

class SiluAndMul(nn.Module):
    """
    实现 SwiGLU 激活函数: SiLU(gate) * up
    
    在 Transformer 模型（如 Llama, Qwen）的 MLP 层中，通常会将一个线性层的输出切分为两部分，
    一部分（gate）通过 SiLU (Sigmoid-weighted Linear Unit) 激活函数，另一部分（up）保持不变，
    然后将两者逐元素相乘。
    这个类将切分、激活和相乘的操作封装在一起，以便在底层使用更高效的 CUDA kernel。
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        执行前向传播。
        
        Args:
            x (torch.Tensor): 输入张量。
                               维度通常是 [num_tokens, 2 * intermediate_size]，
                               其中 intermediate_size 是 MLP 中间层的维度。
                               这个张量被视为 gate 和 up 两部分的拼接。
        
        Returns:
            torch.Tensor: 激活后的输出张量。
                          维度是 [num_tokens, intermediate_size]。
        """
        # 1. 切分张量
        # torch.chunk(x, 2, dim=-1) 会将输入张量 x 在最后一个维度上平均切分成 2 块。
        # x1 将是前半部分 (gate)，x2 将是后半部分 (up)。
        # x: [num_tokens, 2 * intermediate_size] -> x1: [num_tokens, intermediate_size], x2: [num_tokens, intermediate_size]
        x1, x2 = torch.chunk(x, 2, dim=-1)
        
        # 2. 计算 SiLU(x1) * x2
        # F.silu(x1) 计算 Sigmoid-weighted Linear Unit，即 x1 * sigmoid(x1)。
        # 然后与 x2 逐元素相乘。
        return torch.nn.functional.silu(x1) * x2