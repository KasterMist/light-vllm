import torch
from torch import nn

# 混合采样策略，结合了贪婪解码和温度采样
class Sampler(nn.Module):

    def __init__(self):
        super().__init__()
    
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # logits: 模型输出的原始分数，shape: (batch_size, vocab_size)
        # temperatures: 每个样本的温度参数，shape: (batch_size,)
        logits = logits.to(torch.float)
        greedy_tokens = logits.argmax(-1) # 选择概率最高的token，shape: (batch_size,)

        # temperatures.unsqueeze(1): 从(batch_size,)变为(batch_size, 1)
        # logits.div_(): 原地除法，logits /= temperature
        # 温度的作用：
        # temperature = 0: 除以0会得到inf，softmax后变成one-hot分布
        # temperature = 1: 不改变logits分布
        # temperature > 1: 使分布更平坦（增加随机性）
        # temperature < 1: 使分布更尖锐（减少随机性）
        logits.div_(temperatures.unsqueeze(1)) 
        # 将logits转换为概率分布
        # 当temperature=0时，probs接近one-hot分布
        # 当temperature>0时，probs是平滑的概率分布
        probs = torch.softmax(logits, -1, dtype=torch.float) # 将logits转换为概率，shape: (batch_size, vocab_size)

        # 使用指数分布采样，epsilon防止除以0
        epsilon = 1e-10
        # 1. torch.empty_like(probs).exponential_(1)  # 生成随机噪声
        # 2. probs.div_(random_noise + epsilon)       # 概率除以随机数
        # 3. .argmax(-1)                              # 选择最大值
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1) + epsilon).argmax(-1)

        # 如果温度为0，则使用贪婪解码，否则使用温度采样
        return torch.where(temperatures == 0, greedy_tokens, sample_tokens)
