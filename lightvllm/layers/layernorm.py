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
        var = x.pow(2).mean(-1, True) # Calculate variance, -1 means last dimension, True means keep dimensions
        x.mul_(torch.rsqrt(var + self.eps))  # Apply RMS normalization
        x = x.to(origin_dtype).mul_(self.weight)
        return x

    @torch.compile
    def add_rms_forward(self, x: torch.Tensor, residual: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        origin_dtype = x.dtype
        x = x.to(torch.float32).add_(residual.to(torch.float32))
        residual = x.to(origin_dtype) # Save current value of x (result after residual connection) as new residual, this residual will be input to next layer
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

