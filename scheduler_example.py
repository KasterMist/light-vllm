import os
from transformers import AutoTokenizer
from lightvllm.sampling_params import SamplingParams
from lightvllm.engine.sequence import Sequence
from lightvllm.engine.scheduler import Scheduler
from lightvllm.config import Config
from dataclasses import fields
from lightvllm.utils.loader import load_model

if __name__ == "__main__":
    path = os.path.expanduser("~/.cache/huggingface/hub/Qwen3-0.6B")
    tokenizer = AutoTokenizer.from_pretrained(path)

    sampling_params = SamplingParams(temperature=0.8, max_tokens=128)

    prompts = [
        "introduce yourself",
        "who are you?",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        for prompt in prompts
    ]
    kwargs = {
        "enforce_eager": True,
        "tensor_parallel_size": 1
    }

    config_fields = {field.name for field in fields(Config)}
    config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
    config = Config(path, **config_kwargs)
    print("config:", config)
    # print(config_fields)
    # print(config_kwargs)
    scheduler = Scheduler(config)
    print(prompts)
    prompt_encode = tokenizer.encode(prompts[0])
    seq_1 = Sequence(prompt_encode, sampling_params)
    print(seq_1.token_ids)
    scheduler.add(seq_1)
    print(scheduler.waiting)

