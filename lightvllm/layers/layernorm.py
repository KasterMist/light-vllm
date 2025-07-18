import torch
from torch import nn

class RMSNorm(nn.Module):

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(self, x: torch.Tensor) -> torch.Tensor:
        origin_dtype = x.dtype
        x = x.to(torch.float32)
        var = x.pow(2).mean(-1, True) # 计算方差，-1表示最后一个维度，True表示保持维度
        x = x.to(origin_dtype).mul_(self.weight)
        return x

    @torch.compile
    def add_rms_forward(self, x: torch.Tensor, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        origin_dtype = x.dtype
        x = x.to(torch.float32).add_(residual.to(torch.float32))
        residual = x.to(origin_dtype) # 将x的当前值（残差连接后的结果）保存为新的residual, 这个residual将作为下一层的输入
        var = x.pow(2).mean(-1, True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(origin_dtype).mul_(self.weight)
        return x, residual

    def forward(self, 
                x: torch.Tensor, 
                residual: torch.Tensor | None = None
                ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)

