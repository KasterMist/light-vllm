from dataclasses import dataclass
import torch

"""
该模块定义了一个全局上下文（Context），用于在模型的前向传播过程中传递和管理各种关键信息。
它像一个“信息总线”，让模型中的不同层（如 Attention、Embedding）可以方便地获取到当前批次处理的元数据。

这个模块主要用于:
1. 批处理推理 (Batching): 管理多个序列的长度和位置信息。
2. 注意力机制 (Attention): 为 FlashAttention 等优化过的注意力计算提供所需的参数。
3. 内存管理 (KV Cache): 通过 slot_mapping 和 block_tables 动态管理 GPU 显存。
4. 阶段切换 (Prefill/Decode): 通过 is_prefill 标志来区分是处理提示词（Prefill）阶段还是生成新词（Decode）阶段。
"""

@dataclass
class Context:
    """
    一个数据类，用于存储单次前向传播的所有上下文信息。
    """
    # --- 阶段标志 ---
    # 是否处于 Prefill 阶段。True 表示正在处理完整的 Prompt，False 表示正在进行逐个 Token 的生成。
    is_prefill: bool = False

    # --- Attention 相关参数 (主要为 FlashAttention 使用) ---
    # Query 的累积序列长度。一个一维张量，例如 [0, 5, 12]，表示第一个序列长度为5，第二个为7。
    # 用于在将 (batch, seq_len, dim) 展平为 (total_tokens, dim) 后，快速定位每个序列的边界。
    cu_seqlens_q: torch.Tensor | None = None
    # Key 的累积序列长度。在 Prefill 阶段，它与 cu_seqlens_q 相同。
    # 在 Decode 阶段，它也包含了 KV Cache 中历史 Token 的长度。
    cu_seqlens_k: torch.Tensor | None = None
    # 当前批次中 Query 序列的最大长度。
    max_seqlen_q: int = 0
    # 当前批次中 Key 序列的最大长度。
    max_seqlen_k: int = 0

    # --- KV Cache 内存管理相关参数 ---
    # Slot 映射表。一个一维张量，告诉每个 Token 应该去哪个物理内存块（Slot）中读写 KV Cache。
    # 维度: [total_tokens]
    slot_mapping: torch.Tensor | None = None
    # 每个序列的有效上下文长度。用于在 Attention 计算中生成正确的注意力掩码 (attention mask)，
    # 确保 Token 不会注意到它之后的 Token 或 Padding Token。
    context_lens : torch.Tensor | None = None
    # 块表 (Block Tables)。一个二维张量，记录了每个序列分配了哪些物理内存块 (Block)。
    # 维度: [batch_size, max_num_blocks_per_seq]
    block_tables: torch.Tensor | None = None

# 创建一个全局的、模块级别的 Context 实例。
_CONTEXT = Context()

def get_context() -> Context:
    """
    获取当前的全局上下文。
    模型中的任何部分都可以通过调用此函数来获取当前批次的信息。
    """
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None,
                max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, 
                context_lens=None, block_tables=None):
    """
    设置或更新全局上下文。
    通常在每次模型迭代（`model.forward`）开始之前，由调度器（Scheduler）或引擎（Engine）调用，
    传入当前批次的所有元数据。
    """
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
                         max_seqlen_k, slot_mapping, context_lens, block_tables)

def reset_context():
    """
    重置全局上下文为其初始状态。
    通常在一次完整的推理请求结束后调用，以清理状态。
    """
    global _CONTEXT
    _CONTEXT = Context()
