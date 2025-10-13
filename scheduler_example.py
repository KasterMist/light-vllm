import os
from dataclasses import fields
from transformers import AutoTokenizer
from lightvllm.llm import LLM
from lightvllm.sampling_params import SamplingParams
from lightvllm.engine.sequence import Sequence
from lightvllm.engine.scheduler import Scheduler
from lightvllm.config import Config
from lightvllm.utils.loader import load_model


if __name__ == "__main__":
    path = os.path.expanduser("~/.cache/huggingface/hub/Qwen3-0.6B")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)
    sampling_params = SamplingParams(temperature=0.8, max_tokens=256)

    prompts = [
        "introduce yourself",
        "who are you?",
        "list all prime numbers within 100",
    ]
    # prompts[0]: '<|im_start|>user\nintroduce yourself<|im_end|>\n<|im_start|>assistant\n'
    #   <|im_start|>  151644  
    #     user        872    
    #     \n           198
    #    int          396
    #    roduce       47845
    #     yourself    6133
    #    <|im_end|>   151634
    #    \n           198
    #    <|im_start|>  151644
    #    assistant    77091
    #    \n           198

    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        for prompt in prompts
    ]

    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")

