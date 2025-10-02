import torch
from torch import nn
import torch.distributed as dist
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from lightvllm.layers.activation import SiluAndMul
from lightvllm.layers.attention import Attention
from lightvllm.layers.layernorm import RMSNorm
from lightvllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from lightvllm.layers.rotary_embedding import get_rope
from lightvllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead 


class Qwen3Attention(nn.Module):
    """
    Multi-head attention module for Qwen3 model with support for grouped-query attention (GQA).
    Implements parallel attention computation with tensor parallelism support.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: tuple | None = None,
    ) -> None:
        super().__init__()
        # Tensor parallelism setup
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5

        # Parallel linear layers for QKV projection
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        # Output projection layer
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        # Rotary positional embeddings
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        # Core attention computation
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        # Query and key normalization layers
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # Project input to QKV
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        
        # Apply normalization to query and key
        q_by_head = q.view(-1, self.num_heads, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.view(-1, self.num_kv_heads, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)
        
        # Apply rotary positional embeddings
        q, k = self.rotary_emb(positions, q, k)
        
        # Compute attention
        o = self.attn(q, k, v)
        output = self.o_proj(o)
        return output


class Qwen3MLP(nn.Module):
    """
    Multi-layer perceptron (MLP) module for Qwen3 model.
    Implements SwiGLU activation with parallel computation support.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        # Gate and up projection layers (merged for efficiency)
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        # Down projection layer
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu"
        # SwiGLU activation function
        self.act_fn = SiluAndMul()

    def forward(self, x):
        # Apply gate and up projections
        gate_up = self.gate_up_proj(x)
        # Apply SwiGLU activation
        x = self.act_fn(gate_up)
        # Apply down projection
        x = self.down_proj(x)
        return x


class Qwen3DecoderLayer(nn.Module):
    """
    Single decoder layer for Qwen3 model.
    Contains self-attention and MLP blocks with residual connections and layer normalization.
    """

    def __init__(
        self,
        config: Qwen3Config
    ):
        super().__init__()
        # Self-attention block
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None)
        )
        # MLP block
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act
        )
        # Layer normalization layers
        self.input_layernorm = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None
    ):
        # Pre-attention normalization and residual connection
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        
        # Self-attention
        hidden_states = self.self_attn(positions, hidden_states)
        
        # Post-attention normalization and MLP
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3Model(nn.Module):
    """
    Core Qwen3 model without language modeling head.
    Contains token embeddings, decoder layers, and final normalization.
    """

    def __init__(
        self,
        config: Qwen3Config
    ):
        super().__init__()
        # Token embedding layer with vocabulary parallelism
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        # Stack of decoder layers
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)],
        )
        # Final layer normalization
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor
    ):
        # Token embedding
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        
        # Process through decoder layers
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        
        # Final normalization
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    """
    Complete Qwen3 model for causal language modeling.
    Includes the core model and language modeling head for text generation.
    """
    
    # Mapping for packed modules (used for model loading/saving)
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config
    ):
        super().__init__()
        # Core Qwen3 model
        self.model = Qwen3Model(config)
        # Language modeling head with vocabulary parallelism
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        
        # Weight tying between embedding and output layers
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data
        
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor
    ):
        # Forward pass through the model
        hidden_states = self.model(input_ids, positions)
        return hidden_states
    
    def compute_logits(
        self,
        hidden_states: torch.Tensor
    ):
        # Compute logits for language modeling
        logits = self.lm_head(hidden_states)
        return logits

