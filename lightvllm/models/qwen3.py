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
    Qwen3 模型的注意力模块。
    
    实现了分组查询注意力（GQA）并支持张量并行。
    与标准注意力机制的主要区别在于，它在应用旋转位置编码（RoPE）之前，
    对 Query 和 Key 向量进行了 RMS 归一化。
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
        # --- 张量并行设置 ---
        tp_size = dist.get_world_size()  # 获取张量并行的大小（即GPU数量）
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size  # 计算当前GPU应处理的查询头（Q）数量

        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size  # 计算当前GPU应处理的键/值（KV）头数量

        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim  # 当前GPU上所有Q头的总维度
        self.kv_size = self.num_kv_heads * self.head_dim # 当前GPU上所有KV头的总维度
        self.scaling = self.head_dim ** -0.5  # Attention-is-all-you-need中的缩放因子

        # --- 层定义 ---
        # QKV的并行线性层。它将Q、K、V的投影矩阵合并，并沿列（hidden_size）切分，分发到不同GPU上。
        self.qkv_proj = QKVParallelLinear(
            hidden_size, # 1024
            self.head_dim, # 128
            self.total_num_heads, # 16
            self.total_num_kv_heads, # 8
            bias=qkv_bias,
        )
        # 输出投影层。它将多头注意力的结果合并，并进行线性变换。
        # 它的权重矩阵是按行切分的，接收来自所有GPU的部分结果，在All-Reduce后得到最终输出。
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        # 旋转位置编码（Rotary Positional Embeddings）
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        # 核心的注意力计算层（PagedAttention实现）
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        # Qwen3特有的：对Query和Key进行RMSNorm
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # hidden_states: [num_tokens, hidden_size]
        
        # 1. QKV投影，根据词嵌入生成的token对应的向量，生成q k v
        qkv = self.qkv_proj(hidden_states)  # -> [num_tokens, q_size + 2 * kv_size]
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # q: [num_tokens, q_size], k: [num_tokens, kv_size], v: [num_tokens, kv_size]
        
        # 2. Q/K 归一化 (Qwen3 特有)
        # 为了对每个头进行归一化，需要先改变形状
        q_by_head = q.view(-1, self.num_heads, self.head_dim)    # -> [num_tokens, num_heads, head_dim]
        q_by_head = self.q_norm(q_by_head)                      # 形状不变
        q = q_by_head.view(q.shape)                             # -> [num_tokens, q_size]
        
        k_by_head = k.view(-1, self.num_kv_heads, self.head_dim) # -> [num_tokens, num_kv_heads, head_dim]
        k_by_head = self.k_norm(k_by_head)                      # 形状不变
        k = k_by_head.view(k.shape)                             # -> [num_tokens, kv_size]
        
        # 3. 应用旋转位置编码
        q, k = self.rotary_emb(positions, q, k) # 形状不变
        
        # 4. 计算注意力
        # self.attn 内部会处理KV缓存的存取
        o = self.attn(q, k, v)  # -> [num_tokens, num_heads * head_dim]
        
        # 5. 输出投影
        output = self.o_proj(o) # -> [num_tokens, hidden_size]
        return output


class Qwen3MLP(nn.Module):
    """
    Qwen3模型中的多层感知机（MLP）模块。
    它使用了SwiGLU激活函数，并通过合并gate和up投影来优化计算效率。
    """

    def __init__(
        self,
        config: Qwen3Config,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        # MergedColumnParallelLinear将两个并行的列并行线性层合并为一次矩阵乘法。
        # 这里，它同时计算gate和up的投影。
        # 输入: [num_tokens, hidden_size]
        # 输出: [num_tokens, 2 * intermediate_size]
        self.config = config
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        # RowParallelLinear用于将结果投影回hidden_size
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        assert hidden_act == "silu", "Qwen3 MLP只支持silu激活函数"
        # SiluAndMul实现了SwiGLU激活函数: SiLU(gate) * up
        self.act_fn = SiluAndMul(config.kernel_backend)

    def forward(self, x):
        # x: [num_tokens, hidden_size]
        
        # 1. Gate和Up投影
        gate_up = self.gate_up_proj(x) # -> [num_tokens, 2 * intermediate_size]
        
        # 2. SwiGLU激活
        # act_fn会将输入张量在最后一个维度上劈开，一半做SiLU，一半保持原样，然后相乘。
        x = self.act_fn(gate_up)       # -> [num_tokens, intermediate_size]
        
        # 3. Down投影
        x = self.down_proj(x)          # -> [num_tokens, hidden_size]
        return x


class Qwen3DecoderLayer(nn.Module):
    """
    Qwen3模型的单个解码器层。
    它遵循标准的Pre-Norm结构，包含一个自注意力块和一个MLP块。
    """

    def __init__(
        self,
        config: Qwen3Config
    ):
        super().__init__()
        self.config = config
        self.kernel_backend = config.kernel_backend
        # 自注意力模块
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
        # MLP模块
        self.mlp = Qwen3MLP(
            config=self.config,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act
        )
        # 输入层归一化
        self.input_layernorm = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)
        # 注意力后的层归一化
        self.post_attention_layernorm = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None
    ):
        # hidden_states: [num_tokens, hidden_size]
        # residual: [num_tokens, hidden_size] or None
        
        # 1. 输入层归一化 (Pre-Norm) 和残差连接
        # light-vllm中的RMSNorm实现可以同时返回归一化后的输出和新的残差，以优化计算
        if residual is None:
            # 第一次迭代，残差就是输入本身
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            # 后续迭代，将上一层的输出与更早的残差相加
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        
        # 2. 自注意力计算
        hidden_states = self.self_attn(positions, hidden_states)
        
        # 3. 注意力后的层归一化和残差连接
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        
        # 4. MLP计算
        hidden_states = self.mlp(hidden_states)
        
        # 返回当前层的输出和用于下一层计算的残差
        return hidden_states, residual


class Qwen3Model(nn.Module):
    """
    Qwen3模型的核心部分，不包含语言模型头部（LM Head）。
    它由词嵌入层、多个解码器层和最后的归一化层组成。
    """

    def __init__(
        self,
        config: Qwen3Config
    ):
        super().__init__()
        self.kernel_backend = config.kernel_backend
        # 词嵌入层，支持词汇表并行（当tp_size > 1时，词汇表会被切分）
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        # 堆叠的解码器层
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)],
        )
        # 最后的归一化层
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor
    ):
        # input_ids: [num_tokens]
        # positions: [num_tokens]
        
        # 1. 词嵌入, 将token值扩展到隐藏维度 input_ids: [num_tokens] -> hidden_states: [num_tokens, hidden_size]
        hidden_states = self.embed_tokens(input_ids) # -> [num_tokens, hidden_size]
        residual = None
        
        # 2. 逐层通过解码器
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        
        # 3. 最后的归一化
        hidden_states, _ = self.norm(hidden_states, residual) # -> [num_tokens, hidden_size]
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    """
    用于因果语言建模的完整Qwen3模型。
    它在Qwen3Model的基础上增加了用于生成下一个token的语言模型头部。
    """
    
    # 这个映射表告诉模型加载器（loader.py）如何处理权重名称。
    # 例如，它指示加载器将文件中名为 'q_proj' 的权重加载到模型中 'qkv_proj' 参数的 'q' 部分。
    # 'gate_proj' 和 'up_proj' 也是类似地被合并到 'gate_up_proj' 中。
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config,
    ):
        super().__init__()
        # 选择不同的算子后端（native, triton, cuda）
        assert config.kernel_backend in ["native", "triton", "cuda"], f"不支持的kernel_backend: {kernel_backend}"
        self.kernel_backend = config.kernel_backend
        # 核心模型
        self.model = Qwen3Model(config)
        # 语言模型头部，用于将模型输出映射到词汇表上
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        
        # 如果配置要求，则绑定词嵌入和输出层的权重（权重共享）
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data
        
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor
    ):
        # 前向传播，获取最后一层的隐藏状态
        hidden_states = self.model(input_ids, positions)
        return hidden_states
    
    def compute_logits(
        self,
        hidden_states: torch.Tensor
    ):
        # hidden_states: [num_tokens, hidden_size]
        # 计算logits
        logits = self.lm_head(hidden_states) # -> [num_tokens, vocab_size]
        return logits

