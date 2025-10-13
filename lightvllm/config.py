import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    """
    存储推理引擎所有配置项的数据类。
    使用@dataclass可以自动生成__init__、__repr__等方法，使代码更简洁。
    """
    # --- 模型和并行相关配置 ---
    model: str                                  # 模型路径，必须是一个包含模型权重和配置的本地目录
    tensor_parallel_size: int = 1               # 张量并行的大小，即使用的GPU数量。例如，设置为2表示使用2张GPU进行张量并行。

    # --- 批处理（Batching）相关配置 ---
    max_num_batched_tokens: int = 16384         # 一个批次中允许处理的最大token总数。这是控制批次大小的关键参数，主要用于限制prefill阶段的计算量。
    max_num_seqs: int = 512                     # 一个批次中允许的最大序列（请求）数量。
    max_model_len: int = 4096                   # 模型支持的最大序列长度。实际值会与模型自身配置中的max_position_embeddings取最小值。

    # --- 内存和KV缓存相关配置 ---
    gpu_memory_utilization: float = 0.9         # 目标GPU显存使用率。用于自动计算KV缓存能分配的总块数。例如，0.9表示使用90%的空闲显存。
    kvcache_block_size: int = 256               # PagedAttention中每个KV缓存块（block）的大小（以token数量计）。
    num_kvcache_blocks: int = -1                # KV缓存的总块数。如果为-1（默认），则会根据gpu_memory_utilization自动计算。如果设置为具体数值，则会使用该值。

    # --- 杂项配置 ---
    enforce_eager: bool = True                 # 是否强制使用Eager模式，禁用CUDA Graphs。CUDA Graphs可以优化decode阶段的性能，但不利于调试。开启此选项主要用于调试目的。
    hf_config: AutoConfig | None = None         # HuggingFace的模型配置对象（transformers.AutoConfig），将在__post_init__中根据模型路径自动加载。
    eos: int = -1                               # 序列结束符（End-of-Sequence）的token ID。-1表示尚未设置，后续将由Tokenizer自动设置。

    # 算子选择
    kernel_backend: str = "native"              # 使用的算子后端。可选值包括 "native"（默认）, "triton" 和 "cuda"。不同后端在性能和兼容性上有所不同。

    def __post_init__(self):
        """
        在dataclass的__init__方法执行完毕后自动调用的函数。
        用于进行配置的验证、后处理和动态计算。
        """
        # 验证模型路径是否存在且为一个目录
        assert os.path.isdir(self.model), f"模型路径 '{self.model}' 不是一个有效的目录。"
        
        # 验证KV缓存块大小的合法性。通常需要是某个值的倍数以优化内存对齐和计算效率。
        assert self.kvcache_block_size % 256 == 0, "kvcache_block_size 必须是 256 的倍数。"
        
        # 验证张量并行的大小在合理范围内
        assert 1 <= self.tensor_parallel_size <= 8, "tensor_parallel_size 必须在 1 到 8 之间。"
        
        # 从给定的模型路径加载HuggingFace的官方模型配置
        self.hf_config = AutoConfig.from_pretrained(self.model)
        
        # 确保我们设定的模型最大长度不超过模型本身支持的最大位置编码（max_position_embeddings）
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        
        # 验证批处理的最大token数至少要能容纳一个达到最大长度的序列，否则prefill阶段可能无法处理长序列。
        assert self.max_num_batched_tokens >= self.max_model_len, "max_num_batched_tokens 必须大于或等于 max_model_len。"
