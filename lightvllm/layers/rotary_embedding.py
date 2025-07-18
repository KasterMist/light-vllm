from functools import lru_cache
import torch
from torch import nn

def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary position encoding to input tensor
    
    Args:
        x: Input tensor, shape [..., head_size]
        cos: Cosine values, shape [seq_len, rotary_dim//2]
        sin: Sine values, shape [seq_len, rotary_dim//2]
    
    Returns:
        Tensor with rotary position encoding applied
    """
    # Expand dimensions for broadcasting
    cos = cos.unsqueeze(-2)  # [seq_len, rotary_dim//2, 1]
    sin = sin.unsqueeze(-2)  # [seq_len, rotary_dim//2, 1]
    
    # Convert input to float32 for precision, then split into two halves
    x1, x2 = torch.chunk(x.to(torch.float32), 2, -1)
    
    # Apply rotation transform: [x1, x2] -> [x1*cos - x2*sin, x2*cos + x1*sin]
    # cos and sin will broadcast to [seq_len, head_size, rotary_dim//2]
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    
    # Concatenate results and convert back to original data type
    return torch.cat((y1, y2), -1).to(x.dtype)


class RotaryEmbedding(nn.Module):
    """
    Rotary position encoding module
    
    This module implements RoPE (Rotary Position Embedding),
    encoding token position information into the query and key of attention mechanism
    """

    def __init__(self, head_size: int,
                rotary_dim: int,
                max_position_embeddings: int,
                base: float):
        """
        Initialize rotary position encoding
        
        Args:
            head_size: Attention head dimension
            rotary_dim: Rotary encoding dimension, usually equals head_size
            max_position_embeddings: Maximum number of positions
            base: Frequency base for calculating frequencies
        """
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size  # Ensure rotary dimension equals head dimension
        
        # Calculate inverse frequencies: 1 / (base^(2i/d)) for i in [0, d/2)
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        
        # Position sequence: [0, 1, 2, ..., max_position_embeddings-1]
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        
        # Calculate frequencies: outer product of positions and inverse frequencies
        # Result shape: [max_position_embeddings, rotary_dim//2]
        freqs = torch.einsum("i, j -> ij", t, inv_freq)
        
        # Calculate cosine and sine values
        cos = freqs.cos()  # [max_position_embeddings, rotary_dim//2]
        sin = freqs.sin()  # [max_position_embeddings, rotary_dim//2]
        
        # Concatenate cos and sin and cache
        cache = torch.cat((cos, sin), -1)  # [max_position_embeddings, rotary_dim]
        self.register_buffer("cos_sin_cache", cache, False)

    @torch.compile
    def forward(self, positions: torch.Tensor,
                query: torch.Tensor,
                key: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:   
        """
        Forward pass, apply rotary position encoding to query and key
        
        Args:
            positions: Position indices, shape [num_tokens]
            query: Query tensor, shape [num_tokens, num_heads, head_size]
            key: Key tensor, shape [num_tokens, num_heads, head_size]
        
        Returns:
            Query and key with rotary position encoding applied
        """
        num_tokens = positions.size(0)
        
        # Get corresponding cos and sin values based on position indices
        cos_sin = self.cos_sin_cache[positions]  # [num_tokens, rotary_dim]
        cos, sin = cos_sin.chunk(2, -1)  # Get cos and sin parts separately
        
        # Apply rotary position encoding to query
        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size)  # Reshape to [num_tokens, num_heads, head_size]
        query = apply_rotary_emb(query, cos, sin).view(query_shape)  # Apply RoPE and restore original shape
        
        # Apply rotary position encoding to key
        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_size)  # Reshape to [num_tokens, num_heads, head_size]
        key = apply_rotary_emb(key, cos, sin).view(key_shape)  # Apply RoPE and restore original shape
        
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
    Factory function to get rotary position encoding instance
    
    Uses LRU cache to avoid repeatedly creating the same RoPE instance
    
    Args:
        head_size: Attention head dimension
        rotary_dim: Rotary encoding dimension
        max_position: Maximum number of positions
        base: Frequency base
        rope_scaling: Rotary scaling parameters (currently unused)
    
    Returns:
        RotaryEmbedding instance
    """
    assert rope_scaling is None  # Current implementation doesn't support rope_scaling
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb