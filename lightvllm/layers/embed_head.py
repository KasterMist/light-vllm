"""
Embedding layer and language model head module - supporting vocabulary parallel embedding and output layers

This module provides two main classes:
1. VocabParallelEmbedding: Vocabulary parallel embedding layer that partitions vocabulary across different GPUs
2. ParallelLMHead: Parallel language model head for generating logits for the next token

"""

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from lightvllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):
    """
    Vocabulary parallel embedding layer
    
    Partitions the vocabulary across different GPUs, where each GPU only stores embedding vectors for a portion of the vocabulary.
    This design can:
    - Reduce memory usage on each GPU
    - Support larger vocabularies
    - Reduce computation during inference
    
    Features:
    - Each GPU only stores embeddings for a portion of the vocabulary
    - Forward pass requires handling vocabulary ID mapping
    - Uses all-reduce operations to merge results from different GPUs
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        # Get current process rank and total size in distributed group, tp -> tensor parallelism
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        
        # Ensure vocabulary size is divisible by number of GPUs
        assert num_embeddings % self.tp_size == 0
        
        self.num_embeddings = num_embeddings
        # Number of vocabulary items each GPU is responsible for
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        
        # Vocabulary ID range that current GPU is responsible for
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        
        # Create embedding weight matrix, only containing vocabulary items current GPU is responsible for
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        # Set weight loader (for loading weights from checkpoints)
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """
        Weight loader: Extract the portion corresponding to current GPU from complete vocabulary table
        
        Args:
            param: Parameter to update (current GPU's embedding weights)
            loaded_weight: Complete vocabulary table weights loaded from checkpoint
        """
        param_data = param.data
        # Get vocabulary count for current partition
        shard_size = param_data.size(0)
        # Calculate starting position of current partition in complete vocabulary table
        start_idx = self.tp_rank * shard_size
        # Extract corresponding portion from complete weights
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size) # narrow crops tensor along specified dimension torch.narrow(dim, start, length)
        # Ensure sizes match
        assert param_data.size() == loaded_weight.size()

        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        """
        Forward pass: Convert token IDs to embedding vectors
        
        Args:
            x: Input token ID tensor, shape [batch_size, seq_len]
            
        Returns:
            Embedding vector tensor, shape [batch_size, seq_len, embedding_dim]
        """
        if self.tp_size > 1:
            # Create mask to identify which tokens belong to vocabulary range current GPU is responsible for
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            # Map vocabulary IDs to current GPU's local indices
            x = mask * (x - self.vocab_start_idx)
        
        # Perform embedding lookup
        y = F.embedding(x, self.weight)
        
        if self.tp_size > 1:
            # Extend mask to embedding dimensions
            y = mask.unsqueeze(1) * y
            # Use all-reduce operation to merge results from all GPUs
            dist.all_reduce(y)
        
        return y


class ParallelLMHead(VocabParallelEmbedding):
    """
    Parallel language model head
    
    Inherits from VocabParallelEmbedding, specifically for generating logits for the next token.
    During inference, usually only logits for the last token are needed, so optimization is supported.
    
    Features:
    - Supports bias terms
    - Only computes logits for the last token during prefill phase
    - Uses gather operation to collect logits from all GPUs
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        super().__init__(num_embeddings, embedding_dim)
        if bias:
            # Create bias parameter, only containing vocabulary items current GPU is responsible for
            self.bias = nn.Parameter(torch.empty(self.num_embeddings_per_partition))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor):
        """
        Forward pass: Compute logits for next token prediction
        
        Args:
            x: Input hidden state tensor, shape [batch_size, seq_len, hidden_dim]
            
        Returns:
            Logits tensor, shape [batch_size, vocab_size] or [batch_size, seq_len, vocab_size]
        """
        # Get current context information
        context = get_context()
        
        if context.is_prefill:
            # During prefill phase, only compute logits for the last token of each sequence
            # cu_seqlens_q[1:] - 1 gives the last position of each sequence
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        
        # Compute logits: x @ weight.T + bias
        logits = F.linear(x, self.weight, self.bias)
        
        if self.tp_size > 1:
            # Gather logits from all GPUs
            if self.tp_rank == 0:
                # rank 0 creates receive buffer
                all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)]
            else:
                all_logits = None
            # all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            # Use gather operation to collect logits from all GPUs to rank 0
            dist.gather(logits, all_logits, 0)
            
            # logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None

            if self.tp_rank == 0:
                # Concatenate logits from all GPUs on rank 0
                logits = torch.cat(all_logits, -1)
            else:
                # Other GPUs return None
                logits = None
        
        return logits
