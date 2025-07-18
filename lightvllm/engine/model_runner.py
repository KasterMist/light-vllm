import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from lightvllm.config import Config
from lightvllm.engine.sequence import Sequence
from lightvllm.models.qwen3 import Qwen3ForCausalLM