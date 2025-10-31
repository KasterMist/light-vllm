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

def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    """
    将“本步新产生（未缓存）的 token”的 K/V 写入到全局 KV 缓存（PagedAttention 的物理缓存）中。

    核心要点：
    - 写入哪些 KV：仅写入当前批次需要新增的 token（未命中前缀缓存的部分）的 K/V；历史（已缓存）的 KV 不会在本函数中重复写入。
    - 写到哪里：通过 slot_mapping 将“逻辑上的第 i 个 token”映射到“物理缓存的第 slot 行”。
      每个 slot 对应一个扁平槽位，覆盖 num_kv_heads * head_dim 个连续元素（记作 D）。
      实际写入地址范围是 [slot * D, slot * D + D)。
    - 如何计算 slot_mapping：由调度/准备阶段（如 prepare_prefill）依据 block_table（每个逻辑块映射到哪个物理块）
      和块内偏移（offset）生成。公式上可理解为 slot = block_id * block_size + offset。

    张量约定：
    - key/value: [N, num_kv_heads, head_dim]，N 是“本步需要写入的 token 总数”（所有序列合并后）。
    - k_cache/v_cache: 视图为 [max_slots, D] 的二维张量，其中 D = num_kv_heads * head_dim；
      在本项目中，注意力层通常会将其内存布局保证为该视图可用（或通过 stride 满足第二维跨度为 D）。
    - slot_mapping: [N] 的一维整型张量，slot_mapping[i] 给出第 i 个 token 的目标物理槽位。

    性能与并行：
    - 我们使用 Triton JIT 内核为每个 token 启动一个并行程序（grid = (N,)），实现逐 token 的并行写入。

    Args:
        key (torch.Tensor): 当前批次“未缓存 token”的 K，形状 [N, num_kv_heads, head_dim]。
        value (torch.Tensor): 当前批次“未缓存 token”的 V，形状 [N, num_kv_heads, head_dim]。
        k_cache (torch.Tensor): 全局物理 K 缓存（按 slot 扁平化视图）[max_slots, D]。
        v_cache (torch.Tensor): 全局物理 V 缓存（按 slot 扁平化视图）[max_slots, D]。
        slot_mapping (torch.Tensor): 逻辑 token → 物理槽位的映射，形状 [N]。
    """
    # 形状速记：key/value 为 [N, num_kv_heads, head_dim]，N 是“本步要写入”的 token 数；
    # k_cache/v_cache 为 [max_slots, D] 的视图，其中 D = num_kv_heads * head_dim；slot_mapping 为 [N]。
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim

    # 断言：确保内存布局满足内核假设（提高带宽利用效率）。
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N

    # 启动 Triton 内核：grid=(N,) 表示每个 token 启动一个并行程序。
    # 对于第 idx 个 token：
    #   - 读取 key[idx, :, :] / value[idx, :, :]
    #   - 计算 slot = slot_mapping[idx]
    #   - 将该 token 的 K/V 写入到 k_cache[slot, :] / v_cache[slot, :] （长度为 D 的连续内存）
    store_kvcache_kernel[(N,)](
        key, key.stride(0),
        value, value.stride(0),
        k_cache, v_cache, slot_mapping, D
    )

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
        
        # 如果KV缓存已经分配，则将当前计算出的K和V存入缓存。这一步会用slot_mapping把“本次需要写入的 token”对应的 K/V 值写进物理缓存的正确槽位
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
