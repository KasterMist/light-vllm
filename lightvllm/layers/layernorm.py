import torch
from torch import nn

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm) 的实现。

    RMSNorm 是一种 Layer Normalization 的变体，它通过重新缩放输入张量的均方根来对其进行归一化。
    与标准的 LayerNorm 相比，它移除了均值减法，只通过 RMS 进行缩放，计算上更高效。
    公式为: y = x / sqrt(mean(x^2) + eps) * weight
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        """
        初始化 RMSNorm 层。

        Args:
            hidden_size (int): 隐藏层的大小，即输入张量最后一个维度的尺寸。
            eps (float): 一个小的浮点数，用于防止除以零。
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        # weight 是一个可学习的参数，用于缩放归一化后的输出。
        # 维度: [hidden_size]
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        执行标准的 RMS 归一化。

        Args:
            x (torch.Tensor): 输入张量，维度通常为 [..., hidden_size]。

        Returns:
            torch.Tensor: 归一化后的张量，维度与输入 x 相同。
        """
        origin_dtype = x.dtype
        # 为了数值稳定性，将计算切换到 float32。
        x = x.to(torch.float32)
        # 计算 x 的平方的均值（方差）。-1 表示在最后一个维度上计算。True 保持维度。
        # var 维度: [..., 1]
        var = x.pow(2).mean(-1, True)
        # 计算 1 / sqrt(var + eps)，即 rsqrt。
        # 然后将 x 与其相乘，完成归一化。
        x.mul_(torch.rsqrt(var + self.eps))
        # 将数据类型转换回原始类型，并乘以可学习的权重。
        x = x.to(origin_dtype).mul_(self.weight)
        return x

    @torch.compile
    def add_rms_forward(self, x: torch.Tensor, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        执行带有残差连接的 RMS 归一化 (Add & Norm)。

        这个函数首先将输入 x 与残差 residual 相加，然后对结果进行 RMS 归一化。
        这在 Transformer 的 Decoder 层中很常见。

        Args:
            x (torch.Tensor): 输入张量，通常是某个子层（如 Attention 或 FFN）的输出。维度为 [..., hidden_size]。
            residual (torch.Tensor): 残差连接的张量，即子层的输入。维度与 x 相同。

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - 第一个元素是归一化后的张量，维度与输入 x 相同。
                - 第二个元素是新的残差（即 x + residual 的结果），将作为下一层的输入残差。
        """
        origin_dtype = x.dtype
        # 为了数值稳定性，将 x 和 residual 都转换为 float32 进行相加。
        x = x.to(torch.float32).add_(residual.to(torch.float32))
        # 将相加后的结果保存为新的残差，它将作为下一层的输入残差。
        residual = x.to(origin_dtype)
        # 计算 x 的平方的均值（方差）。
        var = x.pow(2).mean(-1, True)
        # 执行 RMS 归一化。
        x.mul_(torch.rsqrt(var + self.eps))
        # 将数据类型转换回原始类型，并乘以可学习的权重。
        x = x.to(origin_dtype).mul_(self.weight)
        return x, residual

    def forward(self, 
                x: torch.Tensor, 
                residual: torch.Tensor | None = None
                ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        RMSNorm 层的前向传播。

        根据是否提供了 residual 张量，选择执行标准归一化还是“Add & Norm”。

        Args:
            x (torch.Tensor): 输入张量。
            residual (torch.Tensor | None, optional): 残差张量。如果为 None，则执行标准 RMSNorm。
                                                     否则，执行 Add & Norm。默认为 None。

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
                - 如果 residual 为 None，返回归一化后的张量。
                - 如果提供了 residual，返回一个元组，包含归一化后的张量和新的残差。
        """
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)

