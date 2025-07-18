from dataclasses import dataclass
import torch

"""
translate to english:
This module is mainly used for:
1. Batch processing inference: managing the length and position information of multiple sequences
2. Attention mechanism: providing parameters required for optimized attention calculations like FlashAttention
3. Memory management: dynamically allocating GPU memory through slot_mapping and block_tables
4. Phase switching: identifying the prefill and generation phases through the is_prefill flag
"""

@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None    # slot mapping for dynamic memory management to determine token storage location
    context_lens : torch.Tensor | None = None   # valid context length for each sequence (for padding short context length use i.e. contest_lens[i] = 5, max_len = 10, then padding 5 tokens to the end of the sequence)
    block_tables: torch.Tensor | None = None    # block tables for dynamic memory management to determine token storage location

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None,
                max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, 
                context_lens=None, block_tables=None):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q,
                         max_seqlen_k, slot_mapping, context_lens, block_tables)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()