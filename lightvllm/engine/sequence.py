from copy import copy
from enum import Enum, auto
from itertools import count
from lightvllm.sampling_params import SamplingParams

# 定义序列的三种可能状态
class SequenceStatus(Enum):
    """
    描述一个生成序列（请求）当前所处的生命周期阶段。
    - WAITING: 序列正在等待被处理，通常在等待队列中。
    - RUNNING: 序列正在被模型处理，处于prefill或decode阶段。
    - FINISHED: 序列已完成生成（达到最大长度、遇到EOS token等）。
    """
    WAITING = auto()  # 等待状态
    RUNNING = auto()  # 运行状态
    FINISHED = auto() # 完成状态

class Sequence:
    """
    表示一个独立的生成序列（即一个请求）。
    
    这个类是推理引擎中管理请求的核心数据结构。它不仅包含了token ID，
    还追踪了与调度、KV缓存管理（特别是PagedAttention）相关的状态。
    
    每个Sequence对象代表一个从prompt开始的文本生成任务。
    """
    # KV缓存块的大小，这是一个固定的类属性，与调度器和模型运行器中的设置保持一致。
    block_size = 256
    # 一个全局计数器，用于为每个新创建的Sequence实例分配一个唯一的ID。
    # 这使得我们可以在整个系统中唯一地标识和追踪每个请求。
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        """
        初始化一个Sequence对象。
        
        Args:
            token_ids (list[int]): 输入的prompt对应的token ID列表。
            sampling_params (SamplingParams): 该序列的采样参数，如温度、最大生成长度等。
        """
        # 分配一个唯一的序列ID
        self.seq_id = next(Sequence.counter)
        # 初始状态为等待处理
        self.status = SequenceStatus.WAITING
        # 存储完整的token ID列表（prompt + 生成的token）
        self.token_ids = copy(token_ids)
        # 序列中最后一个token的ID，在decode阶段作为模型的输入
        self.last_token = token_ids[-1]
        # 序列当前的token总数
        self.num_tokens = len(self.token_ids)
        # prompt部分的token数量，这个值在序列生命周期内是固定的
        self.num_prompt_tokens = len(token_ids)
        # 已被计算并存入KV缓存的token数量。用于前缀缓存（prefix caching）。
        self.num_cached_tokens = 0
        # 核心数据结构：PagedAttention的块表。
        # 这是一个整数列表，存储了分配给该序列的物理KV缓存块的索引。
        # 例如：block_table = [12, 5, 23] 表示该序列的数据存储在物理缓存的第12、5、23号块中。
        # 它的维度是 [num_blocks]。
        self.block_table: list[int] = []
        # 从采样参数中提取的配置
        self.temperature = sampling_params.temperature
        # TODO: 这个好像是在prompt之后生成的最大token数量, 默认是128, 可后续修改，需要确认
        self.max_tokens = sampling_params.max_tokens 
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        """使得可以直接对Sequence实例使用 `len()` 函数，返回序列的总token数。"""
        return self.num_tokens
    
    def __getitem__(self, key):
        """使得Sequence实例可以像列表一样通过索引或切片访问其token_ids。"""
        return self.token_ids[key]

    @property
    def is_finished(self):
        """检查序列是否已完成。"""
        return self.status == SequenceStatus.FINISHED
    
    @property
    def is_running(self):
        """检查序列是否正在运行。"""
        return self.status == SequenceStatus.RUNNING
    
    @property
    def is_waiting(self):
        """检查序列是否正在等待。"""
        return self.status == SequenceStatus.WAITING

    @property
    def num_completion_tokens(self):
        """计算并返回已生成的token（completion部分）的数量。"""
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        """返回prompt部分的token ID列表。"""
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        """返回completion部分的token ID列表（即模型生成的部分）。"""
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        """计算并返回已缓存的token占用的块数。"""
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        """
        计算并返回当前序列总共需要的KV缓存块数。
        
        例如：如果 block_size=256, num_tokens=513,
        则 num_blocks = (513 + 255) // 256 = 3。
        """
        return (self.num_tokens + self.block_size - 1) // self.block_size
    
    @property
    def last_block_num_tokens(self):
        """
        计算并返回最后一个KV缓存块中已使用的token数量。
        
        这对于在PagedAttention中精确定位下一个要写入的slot至关重要。
        例如：如果 num_tokens=513, block_size=256, num_blocks=3,
        则 last_block_num_tokens = 513 - (3 - 1) * 256 = 1。
        """
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def get_token_ids_from_block(self, i):
        """获取逻辑上第i个块对应的token ID列表。"""
        assert 0 <= i < self.num_blocks
        return self.token_ids[i * self.block_size: (i + 1) * self.block_size]

    def append_token(self, token_id):
        """
        向序列追加一个新生成的token。
        
        这是decode阶段的核心操作之一，在模型生成一个token后被调用。
        """
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1
    
    def __getstate__(self):
        """
        定义序列化（pickle）时要保存的状态。
        
        用于在多进程之间传递Sequence对象。为了效率，这里做了优化：
        - 如果序列还没有开始生成（completion_tokens为0），则保存完整的token_ids。
        - 如果已经开始生成，只保存last_token，因为其他进程不需要完整的历史token。
        """
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens,
                self.block_table, self.token_ids if self.num_completion_tokens == 0 else self.last_token)
    
    def __setstate__(self, state):
        """
        定义反序列化（unpickle）时如何恢复对象状态。
        
        与 `__getstate__` 对应，根据保存的状态恢复Sequence对象。
        """
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        # 根据 __getstate__ 的逻辑，如果 completion_tokens 为 0，则 state 的最后一个元素是完整的 token_ids 列表
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            # 否则，最后一个元素是 last_token
            # 注意：这里只恢复了 last_token，token_ids 列表在工作进程中是不完整的。
            # 这是设计上的选择，因为工作进程通常只需要最后一个 token 来进行下一步计算。
            self.last_token = state[-1]
            # 为了保持数据一致性，虽然 token_ids 不完整，但我们至少可以确保 last_token 是对的。
            # 在实际的 light-vllm 实现中，工作进程可能不需要完整的 token_ids 历史。
            # 如果需要，则 __getstate__ 和 __setstate__ 的逻辑需要改变，以始终传递完整的列表。
            self.token_ids = [self.last_token] # 作为一个占位符或简化表示
        