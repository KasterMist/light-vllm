"""
Linear layer module - supporting tensor parallelized linear layer implementations

This module provides various linear layer implementations supporting different tensor parallelization strategies:
1. ReplicatedLinear: Replicated linear layer, each GPU has complete weight copies
2. ColumnParallelLinear: Column parallel linear layer, weights split by columns
3. MergedColumnParallelLinear: Merged column parallel linear layer, supporting multiple outputs
4. QKVParallelLinear: QKV parallel linear layer, specifically for attention mechanisms
5. RowParallelLinear: Row parallel linear layer, weights split by rows

Author: Based on nano-vllm project
"""

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from typing_extensions import override


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):
    """
    Linear layer base class
    
    Provides basic functionality for tensor parallelization, including:
    - Tensor parallelization dimension settings
    - Current process parallelization rank
    - Total parallelization size
    
    Attributes:
        input_size: Input feature dimension
        output_size: Output feature dimension
        tp_dim: Tensor parallelization dimension (0=column parallel, 1=row parallel, None=no parallel)
        tp_rank: Current process rank in tensor parallel group
        tp_size: Total size of tensor parallel group
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        tp_dim: int | None = None,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.tp_dim = tp_dim
        # Get current process rank in distributed group
        self.tp_rank = dist.get_rank()
        # Get total size of distributed group (number of GPUs)
        self.tp_size = dist.get_world_size()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass, needs to be implemented by subclasses"""
        raise NotImplementedError


class ReplicatedLinear(LinearBase):
    """
    Replicated linear layer
    
    Each GPU saves complete weight copies, suitable for:
    - Output layers (such as language model vocabulary projection layers)
    - Linear layers that don't need parallelization
    
    Features:
    - Each GPU has complete weight matrix
    - Each GPU computes independently during forward pass
    - Higher memory usage but simpler computation
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size)
        # Create complete weight matrix, each GPU has a copy
        self.weight = nn.Parameter(torch.empty(self.output_size, self.input_size))
        # Set weight loader
        self.weight.weight_loader = self.weight_loader
        if bias:
            # Create bias parameter
            self.bias = nn.Parameter(torch.empty(self.output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """
        Weight loader: Directly copy loaded weights to parameter
        
        Args:
            param: Parameter to update
            loaded_weight: Weights loaded from checkpoint
        """
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Standard linear transformation
        
        Args:
            x: Input tensor, shape [batch_size, seq_len, input_size]
            
        Returns:
            Output tensor, shape [batch_size, seq_len, output_size]
        """
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):
    """
    Column parallel linear layer
    
    Split weight matrix by columns across different GPUs, suitable for:
    - Hidden layer to hidden layer transformations
    - Scenarios requiring reduced memory usage
    
    Features:
    - Each GPU only saves a portion of the weight matrix columns
    - Input data is the same on all GPUs
    - Output needs concatenation or further processing
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        # tp_dim=0 means split by columns (dimension 0)
        super().__init__(input_size, output_size, 0)
        # Input size per partition (unchanged)
        self.input_size_per_partition = input_size
        # Output size per partition (split by number of GPUs)
        self.output_size_per_partition = divide(output_size, self.tp_size)

        # Create split weight matrix
        self.weight = nn.Parameter(torch.empty(self.output_size_per_partition, self.input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            # Create split bias
            self.bias = nn.Parameter(torch.empty(self.output_size_per_partition))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """
        Weight loader: Extract corresponding partition portion from complete weights
        
        Args:
            param: Parameter to update
            loaded_weight: Complete weights loaded from checkpoint
        """
        param_data = param.data
        # Get current partition size
        shard_size = param_data.size(self.tp_dim)
        # Calculate starting index of current partition
        start_idx = self.tp_rank * shard_size
        # Extract corresponding portion from complete weights
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Use split weights for linear transformation
        
        Args:
            x: Input tensor, shape [batch_size, seq_len, input_size]
            
        Returns:
            Output tensor, shape [batch_size, seq_len, output_size_per_partition]
        """
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):
    """
    Merged column parallel linear layer
    
    Used for computing multiple linear transformations simultaneously, such as computing Q, K, V projections at the same time.
    Merges weights from multiple linear layers into one matrix, then splits by columns.
    
    Features:
    - Supports multiple outputs of different sizes
    - Weight matrix is concatenation of multiple linear layer weights
    - Requires special weight loading logic to handle different sub-matrices
    """

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        # Total output size is sum of all sub-output sizes
        super().__init__(input_size, sum(output_sizes), bias=bias)

    @override
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        """
        Weight loader: Handle loading of specific sub-matrix in merged matrix
        
        Args:
            param: Parameter to update
            loaded_weight: Weights loaded from checkpoint (corresponding to specific sub-matrix)
            loaded_shard_id: Sub-matrix ID, used to determine position in merged matrix
        """
        param_data = param.data
        # Calculate offset of current sub-matrix in merged matrix
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        # Calculate size of current sub-matrix
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        # Extract corresponding portion from merged matrix
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        # Split loaded weights by number of GPUs, take portion corresponding to current GPU
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):
    """
    QKV parallel linear layer
    
    Specifically for Q, K, V projections in attention mechanisms, supports:
    - Multi-head attention mechanisms
    - Grouped query attention (GQA)
    - Tensor parallelization
    
    Features:
    - Computes Q, K, V three projections simultaneously
    - Supports different numbers of heads (e.g., K, V heads may be fewer than Q in GQA)
    - Weight matrix is concatenation of Q, K, V weights
    """

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        # If KV head count not specified, default to same as Q head count
        self.total_num_kv_heads = total_num_kv_heads or total_num_heads
        tp_size = dist.get_world_size()
        # Calculate number of heads on each GPU
        self.num_heads = divide(self.total_num_heads, tp_size)
        self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
        input_size = hidden_size
        # Total output size = Q heads * head size + K heads * head size + V heads * head size
        output_size = (self.total_num_heads + 2 * self.total_num_kv_heads) * self.head_size
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        """
        Weight loader: Handle loading of Q, K, V weights
        
        Args:
            param: Parameter to update
            loaded_weight: Weights loaded from checkpoint
            loaded_shard_id: Identifies whether it's Q, K, or V weights ("q", "k", "v")
        """
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        
        if loaded_shard_id == "q":
            # Q weights: from start to Q heads * head size
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            # K weights: start after Q weights
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:  # loaded_shard_id == "v"
            # V weights: start after K weights
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
            
        # Extract corresponding portion from merged matrix
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        # Split loaded weights by number of GPUs, take portion corresponding to current GPU
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):
    """
    Row parallel linear layer
    
    Split weight matrix by rows across different GPUs, usually used together with ColumnParallelLinear.
    Suitable for:
    - Hidden layer to hidden layer transformations
    - Scenarios requiring reduced computation
    
    Features:
    - Each GPU only saves a portion of the weight matrix rows
    - Input data needs to be split by rows
    - Output requires all-reduce operation to merge results
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        # tp_dim=1 means split by rows (dimension 1)
        super().__init__(input_size, output_size, 1)
        # Input size per partition (split by number of GPUs)
        self.input_size_per_partition = divide(input_size, self.tp_size)
        # Output size per partition (unchanged)
        self.output_size_per_partition = output_size

        # Create split weight matrix
        self.weight = nn.Parameter(torch.empty(self.output_size, self.input_size_per_partition))
        self.weight.weight_loader = self.weight_loader
        if bias:
            # Bias is only saved on rank 0 to avoid duplication
            self.bias = nn.Parameter(torch.empty(self.output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """
        Weight loader: Extract corresponding partition portion from complete weights
        
        Args:
            param: Parameter to update
            loaded_weight: Complete weights loaded from checkpoint
        """
        param_data = param.data
        # Get current partition size
        shard_size = param_data.size(self.tp_dim)
        # Calculate starting index of current partition
        start_idx = self.tp_rank * shard_size
        # Extract corresponding portion from complete weights
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Use split weights for linear transformation, then merge results
        
        Args:
            x: Input tensor, shape [batch_size, seq_len, input_size_per_partition]
            
        Returns:
            Output tensor, shape [batch_size, seq_len, output_size]
        """
        # Perform linear transformation, bias is only added on rank 0
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        # If there are multiple GPUs, use all-reduce to merge results
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
