"""
线性层模块 - 支持张量并行化的线性层实现

这个模块提供了多种线性层实现，支持不同的张量并行化策略：
1. ReplicatedLinear: 复制式线性层，每个GPU都有完整的权重副本
2. ColumnParallelLinear: 列并行线性层，权重按列分割
3. MergedColumnParallelLinear: 合并的列并行线性层，支持多个输出
4. QKVParallelLinear: QKV并行线性层，专门用于注意力机制
5. RowParallelLinear: 行并行线性层，权重按行分割

作者: 基于nano-vllm项目
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
    线性层基类
    
    提供张量并行化的基础功能，包括：
    - 张量并行化维度的设置
    - 当前进程的并行化排名
    - 总并行化大小
    
    Attributes:
        input_size: 输入特征维度
        output_size: 输出特征维度
        tp_dim: 张量并行化维度 (0=列并行, 1=行并行, None=无并行)
        tp_rank: 当前进程在张量并行组中的排名
        tp_size: 张量并行组的总大小
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
        # 获取当前进程在分布式组中的排名
        self.tp_rank = dist.get_rank()
        # 获取分布式组的总大小（GPU数量）
        self.tp_size = dist.get_world_size()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播，需要子类实现"""
        raise NotImplementedError


class ReplicatedLinear(LinearBase):
    """
    复制式线性层
    
    每个GPU都保存完整的权重副本，适用于：
    - 输出层（如语言模型的词表投影层）
    - 不需要并行化的线性层
    
    特点：
    - 每个GPU都有完整的权重矩阵
    - 前向传播时每个GPU独立计算
    - 内存使用量较大，但计算简单
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size)
        # 创建完整的权重矩阵，每个GPU都有副本
        self.weight = nn.Parameter(torch.empty(self.output_size, self.input_size))
        # 设置权重加载器
        self.weight.weight_loader = self.weight_loader
        if bias:
            # 创建偏置参数
            self.bias = nn.Parameter(torch.empty(self.output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """
        权重加载器：直接将加载的权重复制到参数中
        
        Args:
            param: 要更新的参数
            loaded_weight: 从检查点加载的权重
        """
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：标准的线性变换
        
        Args:
            x: 输入张量，形状为 [batch_size, seq_len, input_size]
            
        Returns:
            输出张量，形状为 [batch_size, seq_len, output_size]
        """
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):
    """
    列并行线性层
    
    将权重矩阵按列分割到不同的GPU上，适用于：
    - 隐藏层到隐藏层的变换
    - 需要减少内存使用的场景
    
    特点：
    - 每个GPU只保存权重矩阵的一部分列
    - 输入数据在所有GPU上相同
    - 输出需要拼接或进一步处理
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        # tp_dim=0 表示按列（第0维）分割
        super().__init__(input_size, output_size, 0)
        # 每个分区的输入大小（保持不变）
        self.input_size_per_partition = input_size
        # 每个分区的输出大小（按GPU数量分割）
        self.output_size_per_partition = divide(output_size, self.tp_size)

        # 创建分割后的权重矩阵
        self.weight = nn.Parameter(torch.empty(self.output_size_per_partition, self.input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            # 创建分割后的偏置
            self.bias = nn.Parameter(torch.empty(self.output_size_per_partition))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """
        权重加载器：从完整权重中提取对应分区的部分
        
        Args:
            param: 要更新的参数
            loaded_weight: 从检查点加载的完整权重
        """
        param_data = param.data
        # 获取当前分区的大小
        shard_size = param_data.size(self.tp_dim)
        # 计算当前分区的起始索引
        start_idx = self.tp_rank * shard_size
        # 从完整权重中提取对应部分
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：使用分割的权重进行线性变换
        
        Args:
            x: 输入张量，形状为 [batch_size, seq_len, input_size]
            
        Returns:
            输出张量，形状为 [batch_size, seq_len, output_size_per_partition]
        """
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):
    """
    合并的列并行线性层
    
    用于同时计算多个线性变换，比如同时计算Q、K、V投影。
    将多个线性层的权重合并到一个矩阵中，然后按列分割。
    
    特点：
    - 支持多个不同大小的输出
    - 权重矩阵是多个线性层权重的拼接
    - 需要特殊的权重加载逻辑来处理不同的子矩阵
    """

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        # 总输出大小是所有子输出大小的和
        super().__init__(input_size, sum(output_sizes), bias=bias)

    @override
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        """
        权重加载器：处理合并矩阵中特定子矩阵的加载
        
        Args:
            param: 要更新的参数
            loaded_weight: 从检查点加载的权重（对应特定子矩阵）
            loaded_shard_id: 子矩阵的ID，用于确定在合并矩阵中的位置
        """
        param_data = param.data
        # 计算当前子矩阵在合并矩阵中的偏移量
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        # 计算当前子矩阵的大小
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        # 从合并矩阵中提取对应部分
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        # 将加载的权重按GPU数量分割，取当前GPU对应的部分
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):
    """
    QKV并行线性层
    
    专门用于注意力机制中的Q、K、V投影，支持：
    - 多头注意力机制
    - 分组查询注意力（GQA）
    - 张量并行化
    
    特点：
    - 同时计算Q、K、V三个投影
    - 支持不同的头数（如GQA中K、V的头数可能少于Q）
    - 权重矩阵是Q、K、V权重的拼接
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
        # 如果没有指定KV头数，默认与Q头数相同
        self.total_num_kv_heads = total_num_kv_heads or total_num_heads
        tp_size = dist.get_world_size()
        # 计算每个GPU上的头数
        self.num_heads = divide(self.total_num_heads, tp_size)
        self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
        input_size = hidden_size
        # 总输出大小 = Q头数*头大小 + K头数*头大小 + V头数*头大小
        output_size = (self.total_num_heads + 2 * self.total_num_kv_heads) * self.head_size
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        """
        权重加载器：处理Q、K、V权重的加载
        
        Args:
            param: 要更新的参数
            loaded_weight: 从检查点加载的权重
            loaded_shard_id: 标识是Q、K还是V权重 ("q", "k", "v")
        """
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        
        if loaded_shard_id == "q":
            # Q权重：从开始到Q头数*头大小
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            # K权重：从Q权重之后开始
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:  # loaded_shard_id == "v"
            # V权重：从K权重之后开始
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
            
        # 从合并矩阵中提取对应部分
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        # 将加载的权重按GPU数量分割，取当前GPU对应的部分
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):
    """
    行并行线性层
    
    将权重矩阵按行分割到不同的GPU上，通常与ColumnParallelLinear配合使用。
    适用于：
    - 隐藏层到隐藏层的变换
    - 需要减少计算量的场景
    
    特点：
    - 每个GPU只保存权重矩阵的一部分行
    - 输入数据需要按行分割
    - 输出需要all-reduce操作来合并结果
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        # tp_dim=1 表示按行（第1维）分割
        super().__init__(input_size, output_size, 1)
        # 每个分区的输入大小（按GPU数量分割）
        self.input_size_per_partition = divide(input_size, self.tp_size)
        # 每个分区的输出大小（保持不变）
        self.output_size_per_partition = output_size

        # 创建分割后的权重矩阵
        self.weight = nn.Parameter(torch.empty(self.output_size, self.input_size_per_partition))
        self.weight.weight_loader = self.weight_loader
        if bias:
            # 偏置只在rank 0上保存，避免重复
            self.bias = nn.Parameter(torch.empty(self.output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """
        权重加载器：从完整权重中提取对应分区的部分
        
        Args:
            param: 要更新的参数
            loaded_weight: 从检查点加载的完整权重
        """
        param_data = param.data
        # 获取当前分区的大小
        shard_size = param_data.size(self.tp_dim)
        # 计算当前分区的起始索引
        start_idx = self.tp_rank * shard_size
        # 从完整权重中提取对应部分
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：使用分割的权重进行线性变换，然后合并结果
        
        Args:
            x: 输入张量，形状为 [batch_size, seq_len, input_size_per_partition]
            
        Returns:
            输出张量，形状为 [batch_size, seq_len, output_size]
        """
        # 进行线性变换，偏置只在rank 0上添加
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        # 如果有多个GPU，需要all-reduce来合并结果
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
