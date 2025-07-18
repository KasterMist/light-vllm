import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from lightvllm.utils.context import get_context

@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr, # slot_mapping记录每个序列的token存储在物理内存的哪个slot
    D,
):
    idx = tl.program_id(0)
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    slot = tl.load(slot_mapping_ptr + idx) # 获取当前序列的token存储在物理内存的哪个slot
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)

def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, 
                    v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    # key和value的shape是[N, num_heads, head_dim], N是该批次中token的数量
    # k_cache和v_cache的shape是[max_slots, D], max_slots是最大可用的存储slot数量
    # slot_mapping的shape是[N]
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0),
                                k_cache, v_cache, slot_mapping, D)

# GQA: Group Query Attention
class Attention(nn.Module):
    def __init__(self, num_heads, head_dim, scale, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = torch.tensor([])
        self.v_cache = torch.tensor([])

        def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
            o: torch.Tensor
            # torch.view usage: x = torch.randn(4, 4), x.view(16) means x.reshape(16). 
            # z = x.view(-1, 8) means x.reshape(2, 8)
            q = q.view(-1, self.num_heads, self.head_dim)
            k = k.view(-1, self.num_kv_heads, self.head_dim)
            v = v.view(-1, self.num_kv_heads, self.head_dim)
            context = get_context()
            k_cache, v_cache = self.k_cache, self.v_cache
            if k_cache.numel() and v_cache.numel():
                store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
            if context.is_prefill:
                if context.block_tables is not None: # 如果有前缀缓存，则使用前缀缓存中的k和v
                    k, v = k_cache, v_cache
                o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
            else:   # decode
                # 只处理单个新的token，q.unsqueeze(1)将Q从(batch, heads, dim)变为(batch, 1, heads, dim)
                o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                    context.context_lens, context.block_tables,
                                    self.scale, True)
            o = o.view(-1, self.num_heads * self.head_dim)
            return o
