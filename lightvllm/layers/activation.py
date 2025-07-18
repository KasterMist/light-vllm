import torch
from torch import nn
import torch.nn.functional as F

class SiluAndMul(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, x: torch.Tensor):
        # split x into two tensors along the last dimension
        # for instance, if x is [1, 2, 3, 4], x.chunk(2, -1) will return [1, 2] and [3, 4]
        x, y = x.chunk(2, -1) 
        # silu is a activation function that is defined as x * sigmoid(x)
        return F.silu(x) * y