"""
线性层模块 - 支持张量并行化的线性层实现

该模块提供了多种支持不同张量并行策略的线性层实现：
1. ReplicatedLinear: 复制线性层，每个GPU都拥有完整的权重副本。
2. ColumnParallelLinear: 列并行线性层，权重按列切分。
3. MergedColumnParallelLinear: 合并的列并行线性层，支持一次性计算多个输出。
4. QKVParallelLinear: QKV并行线性层，专用于注意力机制。
5. RowParallelLinear: 行并行线性层，权重按行切分。

作者: 基于 nano-vllm 项目
"""

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from typing_extensions import override


def divide(numerator, denominator):
    """
    一个简单的整数除法函数，确保可以整除。
    """
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):
    """
    线性层基类
    
    提供了张量并行的基础功能，包括：
    - 张量并行维度设置
    - 当前进程的并行rank
    - 总的并行大小
    
    属性:
        input_size: 输入特征维度
        output_size: 输出特征维度
        tp_dim: 张量并行维度 (0=列并行, 1=行并行, None=无并行)
        tp_rank: 当前进程在张量并行组中的rank
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
        # 获取当前进程在分布式组中的rank
        self.tp_rank = dist.get_rank()
        # 获取分布式组的总大小 (GPU数量)
        self.tp_size = dist.get_world_size()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播，需要由子类实现"""
        raise NotImplementedError


class ReplicatedLinear(LinearBase):
    """
    复制线性层 (非并行)
    
    每个GPU都保存完整的权重副本，适用于：
    - 不需要并行化的线性层
    
    特性:
    - 每个GPU都有完整的权重矩阵
    - 前向传播时每个GPU独立计算
    - 内存占用较高，但计算简单
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size)
        # 创建完整的权重矩阵，每个GPU都有一份副本
        # 维度: [output_size, input_size]
        self.weight = nn.Parameter(torch.empty(self.output_size, self.input_size))
        # 设置权重加载器
        self.weight.weight_loader = self.weight_loader
        if bias:
            # 创建偏置参数
            # 维度: [output_size]
            self.bias = nn.Parameter(torch.empty(self.output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """
        权重加载器: 直接将加载的权重复制到参数中
        
        Args:
            param: 需要更新的参数
            loaded_weight: 从checkpoint加载的权重
        """
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播: 标准的线性变换
        
        Args:
            x: 输入张量, 维度: [..., input_size]
            
        Returns:
            输出张量, 维度: [..., output_size]
        """
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):
    """
    列并行线性层
    
    将权重矩阵 W 按列（输出维度）切分到不同的GPU上。
    W = [W_1, W_2, ..., W_p]
    Y = XA  =>  Y_i = X * W_i
    
    特性:
    - 每个GPU只保存权重矩阵的一部分列
    - 所有GPU上的输入数据 X 是相同的
    - 每个GPU的输出 Y_i 是部分结果，需要后续处理（如拼接或直接用于下一层）
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        # tp_dim=0 表示按列（维度0）切分
        super().__init__(input_size, output_size, 0)
        # 每个分区的输入大小（不变）
        self.input_size_per_partition = input_size
        # 每个分区的输出大小（按GPU数量切分）
        self.output_size_per_partition = divide(output_size, self.tp_size)

        # 创建切分后的权重矩阵
        # 维度: [output_size_per_partition, input_size]
        self.weight = nn.Parameter(torch.empty(self.output_size_per_partition, self.input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            # 创建切分后的偏置
            # 维度: [output_size_per_partition]
            self.bias = nn.Parameter(torch.empty(self.output_size_per_partition))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """
        权重加载器: 从完整权重中抽取出当前GPU对应的分区部分
        
        Args:
            param: 需要更新的参数
            loaded_weight: 从checkpoint加载的完整权重
        """
        param_data = param.data
        # 获取当前分区的大小
        shard_size = param_data.size(self.tp_dim)
        # 计算当前分区的起始索引
        start_idx = self.tp_rank * shard_size
        # 从完整权重中抽取出对应的部分
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播: 使用切分后的权重进行线性变换
        
        Args:
            x: 输入张量, 维度: [..., input_size]
            
        Returns:
            输出张量 (部分结果), 维度: [..., output_size_per_partition]
        """
        # 输入 x 在所有GPU上是相同的
        # 输出是部分结果，维度为 [..., output_size_per_partition]
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):
    """
    合并的列并行线性层
    
    用于一次性计算多个线性变换，例如在FFN层中同时计算gate_proj和up_proj。
    它将多个线性层的权重合并成一个大矩阵，然后按列进行切分。
    
    特性:
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
        # 总输出大小是所有子输出大小之和
        super().__init__(input_size, sum(output_sizes), bias=bias)

    @override
    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        """
        权重加载器: 处理合并矩阵中特定子矩阵的加载
        
        Args:
            param: 需要更新的参数 (整个合并后的权重)
            loaded_weight: 从checkpoint加载的权重 (对应特定的子矩阵，如gate_proj的权重)
            loaded_shard_id: 子矩阵的ID，用于确定在合并矩阵中的位置
        """
        param_data = param.data
        # 计算当前子矩阵在合并矩阵中的偏移量
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        # 计算当前子矩阵的大小
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        # 从合并矩阵中抽取出对应部分
        param_data_shard = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        # 将加载的权重按GPU数量切分，取当前GPU对应的部分
        loaded_weight_shard = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        
        assert param_data_shard.shape == loaded_weight_shard.shape
        param_data_shard.copy_(loaded_weight_shard)


class QKVParallelLinear(ColumnParallelLinear):
    """
    QKV并行线性层
    
    专用于注意力机制中的Q, K, V投影，支持：
    - 多头注意力 (MHA)
    - 分组查询注意力 (GQA)
    - 张量并行
    
    特性:
    - 一次性计算Q, K, V三个投影
    - 支持不同的头数 (例如GQA中K, V的头数可能少于Q)
    - 权重矩阵是Q, K, V权重的拼接
    """

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        self.head_size = head_size  # 128
        self.total_num_heads = total_num_heads # 16
        # 如果未指定KV头数，则默认为与Q头数相同 (MHA)
        self.total_num_kv_heads = total_num_kv_heads or total_num_heads # 8
        tp_size = dist.get_world_size()
        # 计算每个GPU上的头数
        self.num_heads = divide(self.total_num_heads, tp_size)
        self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
        input_size = hidden_size
        # 总输出大小 = (Q总头数 + K总头数 + V总头数) * head_size = (16 + 8 + 8) * 128 = 4096
        output_size = (self.total_num_heads + 2 * self.total_num_kv_heads) * self.head_size
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        """
        权重加载器: 处理Q, K, V权重的加载
        
        Args:
            param: 需要更新的参数 (Q,K,V合并后的权重)
            loaded_weight: 从checkpoint加载的权重 (q_proj, k_proj, 或 v_proj)
            loaded_shard_id: 标识是Q, K, 还是V的权重 ("q", "k", "v")
        """
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        
        # 计算每个GPU上Q, K, V的切片大小和偏移量
        q_shard_size = self.num_heads * self.head_size
        k_shard_size = self.num_kv_heads * self.head_size
        v_shard_size = self.num_kv_heads * self.head_size
        
        q_shard_offset = 0
        k_shard_offset = q_shard_size
        v_shard_offset = q_shard_size + k_shard_size

        if loaded_shard_id == "q":
            shard_size, shard_offset = q_shard_size, q_shard_offset
        elif loaded_shard_id == "k":
            shard_size, shard_offset = k_shard_size, k_shard_offset
        else:  # loaded_shard_id == "v"
            shard_size, shard_offset = v_shard_size, v_shard_offset
            
        # 从合并矩阵中抽取出对应部分
        param_data_shard = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        # 将加载的权重按GPU数量切分，取当前GPU对应的部分
        loaded_weight_shard = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        
        assert param_data_shard.shape == loaded_weight_shard.shape
        param_data_shard.copy_(loaded_weight_shard)


class RowParallelLinear(LinearBase):
    """
    行并行线性层
    
    将权重矩阵 W 按行（输入维度）切分到不同的GPU上。通常与列并行线性层配合使用。
    W = [W_1; W_2; ...; W_p] (分号表示按行堆叠)
    Y = XA  =>  X = [X_1, X_2, ..., X_p], Y = sum(X_i * W_i)
    
    特性:
    - 每个GPU只保存权重矩阵的一部分行
    - 输入数据 X 需要按列切分 (因为 W 是按行切分的)
    - 输出需要在所有GPU上进行 all-reduce 操作来合并结果
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        # tp_dim=1 表示按行（维度1）切分
        super().__init__(input_size, output_size, 1)
        # 每个分区的输入大小（按GPU数量切分）
        self.input_size_per_partition = divide(input_size, self.tp_size)
        # 每个分区的输出大小（不变）
        self.output_size_per_partition = output_size

        # 创建切分后的权重矩阵
        # 维度: [output_size, input_size_per_partition]
        self.weight = nn.Parameter(torch.empty(self.output_size, self.input_size_per_partition))
        self.weight.weight_loader = self.weight_loader
        if bias:
            # 偏置只在 rank 0 上保存，以避免重复计算和存储
            self.bias = nn.Parameter(torch.empty(self.output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """
        权重加载器: 从完整权重中抽取出当前GPU对应的分区部分
        
        Args:
            param: 需要更新的参数
            loaded_weight: 从checkpoint加载的完整权重
        """
        param_data = param.data
        # 如果是偏置项，并且当前rank不是0，则不需要加载
        if param is self.bias and self.tp_rank != 0:
            return
            
        # 获取当前分区的大小
        shard_size = param_data.size(self.tp_dim) if param is self.weight else param_data.size(0)
        # 计算当前分区的起始索引
        start_idx = self.tp_rank * shard_size
        # 从完整权重中抽取出对应的部分
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播: 使用切分后的权重进行线性变换，然后合并结果
        
        Args:
            x: 输入张量 (部分输入), 维度: [..., input_size_per_partition]
            
        Returns:
            输出张量 (完整结果), 维度: [..., output_size]
        """
        # 输入 x 是按特征维度切分后的部分输入
        # 线性变换后得到部分输出
        # y_partial 维度: [..., output_size]
        y_partial = F.linear(x, self.weight)
        
        # 如果有多个GPU，使用 all-reduce 将所有GPU上的部分输出相加，得到最终结果
        if self.tp_size > 1:
            dist.all_reduce(y_partial)
            
        # 只有 rank 0 添加偏置项
        if self.bias is not None:
            return y_partial + self.bias
        
        return y_partial
