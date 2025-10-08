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
    slot_mapping_ptr, # slot_mapping 记录了每个序列的token存储在物理内存中的哪个slot
    D: tl.constexpr,
):
    """
    Triton JIT (Just-In-Time) 内核，用于将计算出的K和V向量高效地存入KV缓存中。
    这个内核是为GPU并行计算设计的。
    
    Args:
        key_ptr: 输入的K向量张量的指针。
        key_stride: K向量张量在序列维度上的步长。
        value_ptr: 输入的V向量张量的指针。
        value_stride: V向量张量在序列维度上的步长。
        k_cache_ptr: 物理KV缓存中K部分的指针。
        v_cache_ptr: 物理KV缓存中V部分的指针。
        slot_mapping_ptr: 一个映射表，告诉我们当前批次中每个token应该被存到物理缓存的哪个“槽位”（slot）。
        D (tl.constexpr): 每个头的总维度 (num_heads * head_dim)，作为编译时常量以优化性能。
    """
    # tl.program_id(0) 获取当前并行任务的唯一ID，这里对应批次中的一个token。
    idx = tl.program_id(0)
    
    # --- 加载输入的K和V ---
    # 计算当前token的K和V向量在输入张量中的内存地址偏移
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    # 从内存中加载K和V向量
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    
    # --- 存入KV缓存 ---
    # 从slot_mapping中加载当前token对应的物理缓存槽位索引
    slot = tl.load(slot_mapping_ptr + idx)
    # 计算该槽位在物理缓存张量中的内存地址偏移
    cache_offsets = slot * D + tl.arange(0, D)
    # 将K和V向量存入计算出的物理缓存位置
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)

def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, 
                    v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    """
    一个Python包装函数，用于调用上面的Triton内核。
    它负责准备参数并启动内核。
    
    Args:
        key (torch.Tensor): 当前批次计算出的K向量，维度 [num_tokens, num_kv_heads, head_dim]。
        value (torch.Tensor): 当前批次计算出的V向量，维度 [num_tokens, num_kv_heads, head_dim]。
        k_cache (torch.Tensor): 整个物理K缓存，维度 [max_slots, num_kv_heads * head_dim]。
        v_cache (torch.Tensor): 整个物理V缓存，维度 [max_slots, num_kv_heads * head_dim]。
        slot_mapping (torch.Tensor): 槽位映射表，维度 [num_tokens]。
    """
    # key/value 维度: [N, num_heads, head_dim], N是批次中的token总数
    # k_cache/v_cache 维度: [max_slots, D], max_slots是最大可用存储槽位数
    # slot_mapping 维度: [N]
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    
    # 一些断言，确保张量的内存布局是连续的，以获得最佳性能
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    
    # 启动Triton内核。grid参数 (N,) 表示启动N个并行实例，每个实例处理一个token。
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0),
                                k_cache, v_cache, slot_mapping, D)

# GQA: Group Query Attention
class Attention(nn.Module):
    """
    核心注意力模块，封装了PagedAttention的实现。
    它利用flash-attn库来高效地计算prefill和decode阶段的注意力。
    """
    def __init__(self, num_heads, head_dim, scale, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        # 初始化空的KV缓存张量，它们将在模型运行时被实际的物理缓存所替换
        self.k_cache = torch.tensor([])
        self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        """
        执行前向传播，根据当前是prefill还是decode阶段，调用不同的flash-attn函数。
        
        Args:
            q (torch.Tensor): Query向量, 维度 [num_tokens, num_heads * head_dim]。
            k (torch.Tensor): Key向量, 维度 [num_tokens, num_kv_heads * head_dim]。
            v (torch.Tensor): Value向量, 维度 [num_tokens, num_kv_heads * head_dim]。
        
        Returns:
            torch.Tensor: 注意力计算的输出，维度 [num_tokens, num_heads * head_dim]。
        """
        o: torch.Tensor
        # 将输入的Q,K,V调整为flash-attn期望的[num_tokens, num_heads, head_dim]格式
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        
        # 从全局上下文中获取调度信息
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        
        # 如果KV缓存已经分配，则将当前计算出的K和V存入缓存
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        
        if context.is_prefill:
            # --- Prefill阶段 ---
            # 使用 flash_attn_varlen_func，它专门为处理一批长度不一的序列（可变长度）而设计。
            if context.block_tables is not None:
                # 如果存在前缀缓存（prefix cache），意味着部分历史KV已经在了，
                # 我们直接从物理缓存中读取K和V来参与计算。
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                    cu_seqlens_q=context.cu_seqlens_q,
                                    cu_seqlens_k=context.cu_seqlens_k,
                                    max_seqlen_q=context.max_seqlen_q,
                                    max_seqlen_k=context.max_seqlen_k,
                                    softmax_scale=self.scale, causal=True, 
                                    block_table=context.block_tables)
        else:
            # --- Decode阶段 ---
            # 使用 flash_attn_with_kvcache，它为自回归生成（每次只生成一个token）做了优化。
            # 它直接从KV缓存中读取历史的K和V，只接受新的Q。
            # q.unsqueeze(1) 将Q的维度从 [batch_size, num_heads, head_dim] 变为 [batch_size, 1, num_heads, head_dim]，
            # 以符合函数对单个新token的输入要求。
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                cache_seqlens=context.context_lens, 
                                block_table=context.block_tables,
                                softmax_scale=self.scale, causal=True)
                                
        # 将输出调整回 [num_tokens, num_heads * head_dim] 的二维形状
        o = o.view(-1, self.num_heads * self.head_dim)
        return o
