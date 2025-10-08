"""
Embedding层和语言模型头模块 - 支持词汇并行嵌入和输出层

该模块提供了两个主要类:
1. VocabParallelEmbedding: 词汇并行嵌入层，将词汇表分区到不同的GPU上
2. ParallelLMHead: 并行的语言模型头，用于生成下一个token的logits

"""

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from lightvllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):
    """
    词汇并行嵌入层
    
    将词汇表分区到不同的GPU上，每个GPU只存储词汇表一部分的嵌入向量。
    这种设计可以:
    - 减少每个GPU上的内存使用
    - 支持更大的词汇表
    - 减少推理时的计算量
    
    特性:
    - 每个GPU只存储一部分词汇表的嵌入
    - 前向传播需要处理词汇ID的映射
    - 使用all-reduce操作合并不同GPU的结果
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        # 获取当前进程在分布式组中的rank和总大小, tp -> tensor parallelism (张量并行)
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        
        # 确保词汇表大小可以被GPU数量整除
        assert num_embeddings % self.tp_size == 0
        
        self.num_embeddings = num_embeddings
        # 每个GPU负责的词汇数量
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        
        # 当前GPU负责的词汇ID范围
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        
        # 创建嵌入权重矩阵，只包含当前GPU负责的词汇
        # 维度: [num_embeddings_per_partition, embedding_dim]
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        # 设置权重加载器 (用于从checkpoint加载权重)
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        """
        权重加载器: 从完整的词汇表中抽取出当前GPU对应的部分
        
        Args:
            param: 需要更新的参数 (当前GPU的嵌入权重)
            loaded_weight: 从checkpoint加载的完整词汇表权重
        """
        param_data = param.data
        # 获取当前分区的词汇数量
        shard_size = param_data.size(0)
        # 计算当前分区在完整词汇表中的起始位置
        start_idx = self.tp_rank * shard_size
        # 从完整权重中抽取出对应的部分
        # narrow: 沿着指定维度裁剪张量 torch.narrow(dim, start, length)
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        # 确保尺寸匹配
        assert param_data.size() == loaded_weight.size()

        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        """
        前向传播: 将token ID转换为嵌入向量
        
        Args:
            x: 输入的token ID张量, 维度: [batch_size, seq_len]
            
        Returns:
            嵌入向量张量, 维度: [batch_size, seq_len, embedding_dim]
        """
        if self.tp_size > 1:
            # 创建掩码，用于识别哪些token属于当前GPU负责的词汇范围
            # mask 维度: [batch_size, seq_len], bool类型
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            # 将词汇ID映射到当前GPU的本地索引
            # 不在范围内的token ID会被置为0，但由于mask的存在，它们在后续计算中会被忽略
            x = mask * (x - self.vocab_start_idx)
        
        # 执行嵌入查找
        # y 维度: [batch_size, seq_len, embedding_dim]
        y = F.embedding(x, self.weight)
        
        if self.tp_size > 1:
            # 将掩码扩展到嵌入维度
            # mask.unsqueeze(-1) 维度: [batch_size, seq_len, 1]
            # y 维度: [batch_size, seq_len, embedding_dim]
            # 通过广播，将不在当前GPU范围内的token的嵌入向量置为0
            y = y * mask.unsqueeze(-1)
            # 使用all-reduce操作合并所有GPU的结果
            # 每个GPU上的y张量（部分结果）会按元素相加，最终所有GPU都得到完整的、合并后的y
            dist.all_reduce(y)
        
        return y


class ParallelLMHead(VocabParallelEmbedding):
    """
    并行的语言模型头
    
    继承自VocabParallelEmbedding，专门用于生成下一个token的logits。
    在推理时，通常只需要最后一个token的logits，因此支持优化。
    
    特性:
    - 支持偏置项 (bias)
    - 在prefill阶段只计算最后一个token的logits
    - 使用gather操作从所有GPU收集logits
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
            # 维度: [num_embeddings_per_partition]
            self.bias = nn.Parameter(torch.empty(self.num_embeddings_per_partition))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor):
        """
        前向传播: 计算用于下一个token预测的logits
        
        Args:
            x: 输入的隐藏状态张量, 维度: [num_tokens, hidden_dim] or [batch_size, seq_len, hidden_dim]
            
        Returns:
            Logits张量。
            如果 tp_size > 1 且 tp_rank != 0, 返回 None。
            否则, 返回维度为 [num_tokens, vocab_size] 的张量。
        """
        # 获取当前上下文信息
        context = get_context()
        
        # x 初始维度: [total_tokens, hidden_dim]
        if context.is_prefill:
            # 在prefill阶段，只计算每个序列最后一个token的logits
            # cu_seqlens_q 是一维累积序列长度，如 [0, 5, 12], 表示第一个序列长度为5，第二个为7
            # cu_seqlens_q[1:] - 1 得到每个序列最后一个token在 total_tokens 维度上的索引
            last_indices = context.cu_seqlens_q[1:] - 1
            # x 维度变为: [batch_size, hidden_dim]
            x = x[last_indices].contiguous()
        # 在decode阶段, x的维度已经是 [batch_size, hidden_dim] (因为每个请求只解码一个token), 所以无需操作
        
        # 计算logits: x @ weight.T + bias
        # x 维度: [N, hidden_dim], self.weight 维度: [vocab_partition_size, hidden_dim]
        # logits 维度: [N, vocab_partition_size]
        logits = F.linear(x, self.weight, self.bias)
        
        if self.tp_size > 1:
            # 从所有GPU收集logits
            if self.tp_rank == 0:
                # rank 0 创建接收缓冲区, 列表长度为tp_size
                # 每个元素的维度与logits相同: [N, vocab_partition_size]
                all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)]
            else:
                all_logits = None
            
            # 使用gather操作将所有GPU的logits收集到rank 0
            # dist.gather(tensor_to_send, list_to_receive_on_rank_0, destination_rank)
            dist.gather(logits, all_logits, 0)
            
            if self.tp_rank == 0:
                # 在rank 0上，将所有GPU的logits沿词汇表维度拼接起来
                # all_logits 是一个list of tensor, 每个tensor维度是 [N, vocab_partition_size]
                # 拼接后 logits 维度: [N, vocab_partition_size * tp_size] = [N, vocab_size]
                logits = torch.cat(all_logits, -1)
            else:
                # 其他GPU返回None
                logits = None
        
        return logits
