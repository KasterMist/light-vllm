from functools import lru_cache
import torch
from torch import nn

def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor) -> torch.Tensor:
    """
    对输入张量应用旋转位置编码 (RoPE)。

    RoPE 的核心思想是将位置信息通过旋转矩阵作用于 Query 和 Key 向量。
    对于一个二维向量 [x1, x2]，旋转 theta 角度后变为 [x1*cos - x2*sin, x2*cos + x1*sin]。
    RoPE 将 head_size 维的向量两两配对，视为 head_size/2 个二维向量，然后对每个二维向量进行不同角度的旋转。

    Args:
        x (torch.Tensor): 输入张量，通常是 Query 或 Key。维度: [..., head_size]。
        cos (torch.Tensor): 预先计算好的余弦值。维度: [num_tokens, rotary_dim//2]。
        sin (torch.Tensor): 预先计算好的正弦值。维度: [num_tokens, rotary_dim//2]。

    Returns:
        torch.Tensor: 应用了 RoPE 后的张量，维度与输入 x 相同。
    """
    # 为了广播，扩展 cos 和 sin 的维度。
    # cos 维度: [num_tokens, rotary_dim//2] -> [num_tokens, 1, rotary_dim//2]
    # sin 维度: [num_tokens, rotary_dim//2] -> [num_tokens, 1, rotary_dim//2]
    # 这里的 unsqueeze(-2) 是为了匹配 x reshape 后的维度 [num_tokens, num_heads, head_size]
    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)
    
    # 为了计算精度，将输入转换为 float32，然后沿最后一个维度切分成两半。
    # x1, x2 维度: [..., head_size//2]
    x1, x2 = torch.chunk(x.to(torch.float32), 2, -1)
    
    # 应用旋转变换: [x1, x2] -> [x1*cos - x2*sin, x2*cos + x1*sin]
    # cos 和 sin 会通过广播机制匹配 x1 和 x2 的维度。
    # y1, y2 维度: [..., head_size//2]
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    
    # 将结果拼接起来，并转换回原始数据类型。
    return torch.cat((y1, y2), -1).to(x.dtype)


class RotaryEmbedding(nn.Module):
    """
    旋转位置编码 (RoPE) 模块。

    该模块实现了 RoPE，用于将 token 的位置信息编码到注意力机制的 Query 和 Key 中。
    它会预先计算并缓存所有可能位置的正弦和余弦值，以便在 `forward` 中高效查找和应用。
    """

    def __init__(self, head_size: int,
                rotary_dim: int,
                max_position_embeddings: int,
                base: float):
        """
        初始化旋转位置编码。

        Args:
            head_size (int): 注意力头的维度。
            rotary_dim (int): 旋转编码的维度，通常等于 head_size。
            max_position_embeddings (int): 模型支持的最大序列长度。
            base (float): 用于计算旋转频率的基数，通常是 10000。
        """
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size, "rotary_dim 必须等于 head_size"
        
        # 计算频率的倒数: 1 / (base^(2i/d))，其中 i in [0, d/2)
        # inv_freq 维度: [rotary_dim//2]
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        
        # 创建位置序列: [0, 1, 2, ..., max_position_embeddings-1]
        # t 维度: [max_position_embeddings]
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        
        # 计算频率: 位置序列 t 和频率倒数 inv_freq 的外积。
        # freqs (theta) = m * theta_i，其中 m 是位置，theta_i 是频率。
        # freqs 维度: [max_position_embeddings, rotary_dim//2]
        freqs = torch.einsum("i, j -> ij", t, inv_freq)
        
        # 计算所有位置的余弦和正弦值
        cos = freqs.cos()  # 维度: [max_position_embeddings, rotary_dim//2]
        sin = freqs.sin()  # 维度: [max_position_embeddings, rotary_dim//2]
        
        # 将 cos 和 sin 拼接并缓存。
        # cache 维度: [max_position_embeddings, rotary_dim]
        cache = torch.cat((cos, sin), -1)
        # register_buffer 将 cache 注册为模型的 buffer，它会被保存但不会被视为模型参数（即不参与梯度更新）。
        self.register_buffer("cos_sin_cache", cache, False)

    @torch.compile
    def forward(self, positions: torch.Tensor,
                query: torch.Tensor,
                key: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:   
        """
        前向传播，对 query 和 key 应用旋转位置编码。

        Args:
            positions (torch.Tensor): token 的位置索引。维度: [num_tokens]。
            query (torch.Tensor): Query 张量。维度: [num_tokens, num_heads, head_size]。
            key (torch.Tensor): Key 张量。维度: [num_tokens, num_kv_heads, head_size]。
        
        Returns:
            tuple[torch.Tensor, torch.Tensor]: 应用了 RoPE 后的 Query 和 Key。
        """
        num_tokens = positions.size(0)
        
        # 根据位置索引从缓存中获取对应的 cos 和 sin 值。
        # cos_sin 维度: [num_tokens, rotary_dim]
        cos_sin = self.cos_sin_cache[positions]
        # 将其拆分为 cos 和 sin 两部分。
        # cos, sin 维度: [num_tokens, rotary_dim//2]
        cos, sin = cos_sin.chunk(2, -1)
        
        # 对 query 应用旋转位置编码
        query_shape = query.shape
        # 为了与 apply_rotary_emb 的输入兼容，可能需要 reshape。
        # view 的 -1 会自动推断 num_heads。
        # query reshape 后维度: [num_tokens, num_heads, head_size]
        query = query.view(num_tokens, -1, self.head_size)
        query = apply_rotary_emb(query, cos, sin).view(query_shape)  # 应用 RoPE 并恢复原始形状
        
        # 对 key 应用旋转位置编码
        key_shape = key.shape
        # key reshape 后维度: [num_tokens, num_kv_heads, head_size]
        key = key.view(num_tokens, -1, self.head_size)
        key = apply_rotary_emb(key, cos, sin).view(key_shape)  # 应用 RoPE 并恢复原始形状
        
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
    获取旋转位置编码实例的工厂函数。

    使用 LRU 缓存来避免重复创建相同的 RoPE 实例。
    如果使用相同的参数多次调用此函数，将直接返回缓存的实例，提高效率。

    Args:
        head_size (int): 注意力头的维度。
        rotary_dim (int): 旋转编码的维度。
        max_position (int): 最大序列长度。
        base (float): 频率基数。
        rope_scaling (dict | None, optional): 旋转缩放参数 (当前未使用)。

    Returns:
        RotaryEmbedding: RotaryEmbedding 的实例。
    """
    assert rope_scaling is None, "当前实现不支持 rope_scaling"
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
