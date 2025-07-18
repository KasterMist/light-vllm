"""
嵌入层和语言模型头部模块 - 支持词汇并行的嵌入和输出层

这个模块提供了两种主要的类：
1. VocabParallelEmbedding: 词汇并行嵌入层，将词汇表分割到不同GPU上
2. ParallelLMHead: 并行语言模型头部，用于生成下一个token的logits

"""

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from lightvllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):
    """
    词汇并行嵌入层
    
    将词汇表分割到不同的GPU上，每个GPU只保存部分词汇的嵌入向量。
    这种设计可以：
    - 减少每个GPU的内存使用
    - 支持更大的词汇表
    - 在推理时减少计算量
    
    特点：
    - 每个GPU只保存部分词汇的嵌入
    - 前向传播时需要处理词汇ID的映射
    - 使用all-reduce操作合并不同GPU的结果
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        # 获取当前进程在分布式组中的排名和总大小, tp -> tensor parallelism
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        
        # 确保词汇表大小能被GPU数量整除
        assert num_embeddings % self.tp_size == 0
        
        self.num_embeddings = num_embeddings
        # 每个GPU负责的词汇数量
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        
        # 当前GPU负责的词汇ID范围
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        
        # 创建嵌入权重矩阵，只包含当前GPU负责的词汇
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        # 设置权重加载器（用于从检查点加载权重）
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """
        权重加载器：从完整词汇表中提取当前GPU负责的部分
        
        Args:
            param: 要更新的参数（当前GPU的嵌入权重）
            loaded_weight: 从检查点加载的完整词汇表权重
        """
        param_data = param.data
        # 获取当前分区的词汇数量
        shard_size = param_data.size(0)
        # 计算当前分区在完整词汇表中的起始位置
        start_idx = self.tp_rank * shard_size
        # 从完整权重中提取对应部分
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size) # narrow 是将tensor在指定维度上进行裁剪torch.narrow(dim, start, length)
        # 确保大小匹配
        assert param_data.size() == loaded_weight.size()

        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        """
        前向传播：将token ID转换为嵌入向量
        
        Args:
            x: 输入的token ID张量，形状为 [batch_size, seq_len]
            
        Returns:
            嵌入向量张量，形状为 [batch_size, seq_len, embedding_dim]
        """
        if self.tp_size > 1:
            # 创建掩码，标识哪些token属于当前GPU负责的词汇范围
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            # 将词汇ID映射到当前GPU的局部索引
            x = mask * (x - self.vocab_start_idx)
        
        # 执行嵌入查找
        y = F.embedding(x, self.weight)
        
        if self.tp_size > 1:
            # 将掩码扩展到嵌入维度
            y = mask.unsqueeze(1) * y
            # 使用all-reduce操作合并所有GPU的结果
            dist.all_reduce(y)
        
        return y


class ParallelLMHead(VocabParallelEmbedding):
    """
    并行语言模型头部
    
    继承自VocabParallelEmbedding，专门用于生成下一个token的logits。
    在推理时，通常只需要最后一个token的logits，因此支持优化。
    
    特点：
    - 支持偏置项
    - 在prefill阶段只计算最后一个token的logits
    - 使用gather操作收集所有GPU的logits
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        super().__init__(num_embeddings, embedding_dim)
        if bias:
            # 创建偏置参数，只包含当前GPU负责的词汇
            self.bias = nn.Parameter(torch.empty(self.num_embeddings_per_partition))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor):
        """
        前向传播：计算logits用于下一个token预测
        
        Args:
            x: 输入的隐藏状态张量，形状为 [batch_size, seq_len, hidden_dim]
            
        Returns:
            logits张量，形状为 [batch_size, vocab_size] 或 [batch_size, seq_len, vocab_size]
        """
        # 获取当前上下文信息
        context = get_context()
        
        if context.is_prefill:
            # 在prefill阶段，只计算每个序列最后一个token的logits
            # cu_seqlens_q[1:] - 1 给出每个序列的最后一个位置
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        
        # 计算logits：x @ weight.T + bias
        logits = F.linear(x, self.weight, self.bias)
        
        if self.tp_size > 1:
            # 收集所有GPU的logits
            if self.tp_rank == 0:
                # rank 0 创建接收缓冲区
                all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)]
            else:
                all_logits = None
            
            # 使用gather操作收集所有GPU的logits到rank 0
            dist.gather(logits, all_logits, 0)
            
            if self.tp_rank == 0:
                # 在rank 0上拼接所有GPU的logits
                logits = torch.cat(all_logits, -1)
            else:
                # 其他GPU返回None
                logits = None
        
        return logits
