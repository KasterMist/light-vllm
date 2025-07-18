import torch
from torch import nn

# Mixed sampling strategy, combining greedy decoding and temperature sampling
class Sampler(nn.Module):

    def __init__(self):
        super().__init__()
    
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # logits: Raw scores output by model, shape: (batch_size, vocab_size)
        # temperatures: Temperature parameter for each sample, shape: (batch_size,)
        logits = logits.to(torch.float)
        greedy_tokens = logits.argmax(-1) # Select token with highest probability, shape: (batch_size,)

        # temperatures.unsqueeze(1): Change from (batch_size,) to (batch_size, 1)
        # logits.div_(): In-place division, logits /= temperature
        # Temperature effects:
        # temperature = 0: Division by 0 gives inf, after softmax becomes one-hot distribution
        # temperature = 1: Doesn't change logits distribution
        # temperature > 1: Makes distribution flatter (increases randomness)
        # temperature < 1: Makes distribution sharper (decreases randomness)
        logits.div_(temperatures.unsqueeze(1)) 
        # Convert logits to probability distribution
        # When temperature=0, probs approaches one-hot distribution
        # When temperature>0, probs is smooth probability distribution
        probs = torch.softmax(logits, -1, dtype=torch.float) # Convert logits to probabilities, shape: (batch_size, vocab_size)

        # Use exponential distribution sampling, epsilon prevents division by zero
        epsilon = 1e-10
        # 1. torch.empty_like(probs).exponential_(1)  # Generate random noise
        # 2. probs.div_(random_noise + epsilon)       # Divide probabilities by random number
        # 3. .argmax(-1)                              # Select maximum value
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1) + epsilon).argmax(-1)

        # If temperature is 0, use greedy decoding, otherwise use temperature sampling
        return torch.where(temperatures == 0, greedy_tokens, sample_tokens)
