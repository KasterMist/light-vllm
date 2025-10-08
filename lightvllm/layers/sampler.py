import torch
from torch import nn

class Sampler(nn.Module):
    """
    混合采样器，结合了贪心解码和温度采样。
    
    这个采样器能够在一个批次(batch)中为不同的请求应用不同的采样策略。
    - 如果请求的温度(temperature)为 0，则使用贪心解码 (argmax)。
    - 如果请求的温度大于 0，则使用温度采样。
    """

    def __init__(self):
        super().__init__()
    
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
        """
        对模型的输出 logits 进行采样，生成下一个 token。

        Args:
            logits (torch.Tensor): 模型输出的原始分数。维度: (batch_size, vocab_size)。
            temperatures (torch.Tensor): 每个样本的温度参数。维度: (batch_size,)。

        Returns:
            torch.Tensor: 采样得到的 token ID。维度: (batch_size,)。
        """
        # 为了数值稳定性，将 logits 转换为 float 类型
        logits = logits.to(torch.float)
        
        # --- 贪心解码部分 ---
        # 对于温度为0的情况，我们总是选择概率最高的 token。
        # 这里预先计算好，后面通过 torch.where 选择。
        greedy_tokens = logits.argmax(-1) # 维度: (batch_size,)

        # --- 温度采样部分 ---
        # 1. 应用温度缩放
        # temperatures.unsqueeze(1) 将维度从 (batch_size,) 变为 (batch_size, 1)，以便进行广播。
        # logits.div_() 是一个原地除法操作，即 logits /= temperature。
        # 温度的影响:
        # - temperature = 0: 会导致除以0，得到inf。softmax后会变成 one-hot 分布。代码通过最后的 where 操作处理这种情况。
        # - temperature = 1: 不改变 logits 分布。
        # - temperature > 1: 使分布更平坦 (增加随机性)。
        # - temperature < 1: 使分布更尖锐 (降低随机性)。
        logits.div_(temperatures.unsqueeze(1))
        
        # 2. 将 logits 转换为概率分布
        # 当 temperature > 0 时，probs 是一个平滑的概率分布。
        probs = torch.softmax(logits, -1, dtype=torch.float) # 维度: (batch_size, vocab_size)

        # 3. Gumbel-Max 技巧进行采样
        # 这种方法在数学上等价于 torch.multinomial(probs, 1)，但在某些硬件上可能更高效。
        # 原理: argmax(log(p_i) + G_i) 等价于从 Categorical(p) 分布中采样，其中 G_i 是独立的 Gumbel(0,1) 分布的随机变量。
        # G_i 可以通过 -log(-log(U_i)) 生成，其中 U_i ~ Uniform(0,1)。
        # 这里的实现 `probs.div(Exponential(1))` 是一个等价的简化形式。
        epsilon = 1e-10 # 防止除以零
        # a. torch.empty_like(probs).exponential_(1): 生成服从指数分布(1)的随机噪声，维度与 probs 相同。
        # b. probs.div_(...): 将概率除以这个随机噪声。
        # c. .argmax(-1): 取最大值的索引作为采样结果。
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1) + epsilon).argmax(-1)

        # --- 混合选择 ---
        # 根据每个请求的温度值，决定使用贪心解码的结果还是温度采样的结果。
        # torch.where(condition, x, y)
        return torch.where(temperatures == 0, greedy_tokens, sample_tokens)
