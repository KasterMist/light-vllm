from functools import lru_cache
import torch
from torch import nn

def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor) -> torch.Tensor:
    """
    应用旋转位置编码到输入张量
    
    Args:
        x: 输入张量，形状为 [..., head_size]
        cos: 余弦值，形状为 [seq_len, rotary_dim//2]
        sin: 正弦值，形状为 [seq_len, rotary_dim//2]
    
    Returns:
        应用了旋转位置编码的张量
    """
    # 扩展维度以便广播
    cos = cos.unsqueeze(2)  # [seq_len, rotary_dim//2, 1]
    sin = sin.unsqueeze(-2)  # [seq_len, 1, rotary_dim//2]
    
    # 将输入转换为float32以确保精度，然后分成两半
    x1, x2 = torch.chunk(x.to(torch.float32), 2, -1)
    
    # 应用旋转变换: [x1, x2] -> [x1*cos - x2*sin, x2*cos + x1*sin]
    # cos 和 sin 会自行广播成 [seq_len, head_size, rotary_dim//2]
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    
    # 将结果拼接并转换回原始数据类型
    return torch.cat((y1, y2), -1).to(x.dtype)


class RotaryEmbedding(nn.Module):
    """
    旋转位置编码模块
    
    这个模块实现了RoPE (Rotary Position Embedding)，
    通过将token的位置信息编码到注意力机制的query和key中
    """

    def __init__(self, head_size: int,
                rotary_dim: int,
                max_position_embeddings: int,
                base: float):
        """
        初始化旋转位置编码
        
        Args:
            head_size: 注意力头的维度
            rotary_dim: 旋转编码的维度，通常等于head_size
            max_position_embeddings: 最大位置数量
            base: 频率基数，用于计算频率
        """
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size  # 确保旋转维度等于头维度
        
        # 计算逆频率: 1 / (base^(2i/d)) for i in [0, d/2)
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        
        # 位置序列: [0, 1, 2, ..., max_position_embeddings-1]
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        
        # 计算频率: outer product of positions and inverse frequencies
        # 结果形状: [max_position_embeddings, rotary_dim//2]
        freqs = torch.einsum("i, j -> ij", t, inv_freq)
        
        # 计算余弦和正弦值
        cos = freqs.cos()  # [max_position_embeddings, rotary_dim//2]
        sin = freqs.sin()  # [max_position_embeddings, rotary_dim//2]
        
        # 将cos和sin拼接并缓存
        cache = torch.cat((cos, sin), -1)  # [max_position_embeddings, rotary_dim]
        self.register_buffer("cos_sin_cache", cache, False)

    @torch.compile
    def forward(self, positions: torch.Tensor,
                query: torch.Tensor,
                key: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:   
        """
        前向传播，对query和key应用旋转位置编码
        
        Args:
            positions: 位置索引，形状为 [num_tokens]
            query: 查询张量，形状为 [num_tokens, num_heads, head_size]
            key: 键张量，形状为 [num_tokens, num_heads, head_size]
        
        Returns:
            应用了旋转位置编码的query和key
        """
        num_tokens = positions.size(0)
        
        # 根据位置索引获取对应的cos和sin值
        cos_sin = self.cos_sin_cache[positions]  # [num_tokens, rotary_dim]
        cos, sin = cos_sin.chunk(2, -1)  # 分别获取cos和sin部分
        
        # 对query应用旋转位置编码
        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size)  # 重塑为 [num_tokens, num_heads, head_size]
        query = apply_rotary_emb(query, cos, sin).view(query_shape)  # 应用RoPE并恢复原始形状
        
        # 对key应用旋转位置编码
        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_size)  # 重塑为 [num_tokens, num_heads, head_size]
        key = apply_rotary_emb(key, cos, sin).view(key_shape)  # 应用RoPE并恢复原始形状
        
        return query, key


@lru_cache(maxsize=1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None
    ):
    """
    获取旋转位置编码实例的工厂函数
    
    使用LRU缓存来避免重复创建相同的RoPE实例
    
    Args:
        head_size: 注意力头的维度
        rotary_dim: 旋转编码的维度
        max_position: 最大位置数量
        base: 频率基数
        rope_scaling: 旋转缩放参数（当前未使用）
    
    Returns:
        RotaryEmbedding实例
    """
    assert rope_scaling is None  # 当前实现不支持rope_scaling
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb