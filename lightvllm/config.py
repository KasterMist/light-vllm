import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384         # max number of tokens in a batch
    max_num_seqs: int = 512                     # max number of sequences in a batch
    max_model_len: int = 4096                   # max length of a sequence
    gpu_memory_utilization: float = 0.9         # gpu memory utilization
    tensor_parallel_size: int = 1               # tensor parallel size
    enforce_eager: bool = False                 # whether to disable CUDA graphs and use the eager mode
    hf_config: AutoConfig | None = None         # huggingface config
    eos: int = -1                               # end of sequence token ID (default -1 means not set)
    kvcache_block_size: int = 256               # kvcache block size
    num_kvcache_blocks: int = -1                # number of kvcache blocks

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
