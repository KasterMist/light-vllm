from collections import deque

from lightvllm.config import Config
from lightvllm.engine.sequence import Sequence, SequenceStatus
from lightvllm.engine.block_manager import BlockManager

class Scheduler:
    """
    调度器类，负责管理待处理和正在处理的序列，并决定在每个推理步骤中运行哪些序列。
    这是实现连续批处理（Continuous Batching）的核心组件。
    """
    def __init__(self, config: Config):
        """
        初始化调度器。
        
        Args:
            config (Config): 包含调度器和块管理器所需配置的对象。
        """
        # 一个批次中允许的最大序列数
        self.max_num_seqs = config.max_num_seqs
        # 一个批次中所有序列的token总数的最大值
        self.max_num_batched_tokens = config.max_num_batched_tokens
        # 序列结束符的token ID
        self.eos = config.eos
        # 块管理器，用于分配和释放KV缓存块
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        # 等待队列，使用双端队列（deque）存储等待处理的序列
        self.waiting: deque[Sequence] = deque()
        # 运行队列，存储正在处理（已prefill，正在decode）的序列
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        """检查是否所有序列都已处理完毕。"""
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        """将新的序列添加到等待队列中，等待调度。"""
        self.waiting.append(seq)
    
    def preempt(self, seq: Sequence):
        """
        抢占一个序列。当需要为新序列或更高优先级的序列腾出空间时调用。
        被抢占的序列会从运行状态转换回等待状态，并释放其占用的KV缓存块。
        
        Args:
            seq (Sequence): 要抢占的序列。
        """
        # 将序列状态设置回等待
        seq.status = SequenceStatus.WAITING
        # 释放该序列占用的所有KV缓存块
        self.block_manager.deallocate(seq)
        # 将被抢占的序列放回等待队列的头部，以便下次优先调度
        self.waiting.appendleft(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """
        执行一次调度，决定当前推理步骤要处理哪些序列。
        调度逻辑分为两个主要阶段：prefill 和 decode。
        
        Returns:
            tuple[list[Sequence], bool]: 
                - 一个包含本次调度选中的序列的列表。
                - 一个布尔值，如果本次是prefill阶段则为True，decode阶段则为False。
        """
        # 阶段一：Prefill
        # 尝试从等待队列中取出序列进行prefill。
        # prefill阶段的序列是那些首次被处理的序列。
        scheduled_seqs = []  # 存储本次调度选中的序列
        num_seqs = 0         # 当前批次中的序列数量
        num_batched_tokens = 0 # 当前批次中的token总数

        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]  # 查看等待队列的第一个序列，但不立即取出
            
            # 检查加入这个序列后是否会超出批次token限制，或者KV缓存是否足够分配
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break  # 如果不满足条件，则停止向批次中添加新的prefill序列
            
            # 如果满足条件，则正式将该序列加入调度列表
            num_seqs += 1
            self.block_manager.allocate(seq)  # 为序列分配KV缓存块
            num_batched_tokens += len(seq) - seq.num_cached_tokens # 累加token数（考虑前缀缓存,当前seq的num_cached_tokens不会被记录在内）
            seq.status = SequenceStatus.RUNNING  # 更新序列状态为运行中
            self.waiting.popleft()  # 从等待队列中移除
            self.running.append(seq)  # 添加到运行队列
            scheduled_seqs.append(seq) # 添加到本次调度的序列列表

        if scheduled_seqs:
            # 如果有序列被调度进行prefill，则本次调度结束，返回prefill批次
            return scheduled_seqs, True
        
        # 阶段二：Decode
        # 如果没有新的序列需要prefill，则从运行队列中选择序列进行decode。
        # decode阶段的序列是那些已经完成prefill，正在生成新token的序列。
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft() # 从运行队列头部取出一个序列
            
            # 检查是否可以为该序列追加一个新的token（即是否还有可用的KV缓存块）
            while not self.block_manager.can_append(seq):
                # 如果无法追加，说明KV缓存已满，需要抢占一个或多个序列来释放空间
                if self.running:
                    # 优先抢占运行队列中优先级较低的序列（队尾的序列）
                    self.preempt(self.running.pop())
                else:
                    # 如果运行队列已空，只能抢占当前序列自身
                    self.preempt(seq)
                    break # 跳出内层while循环
            else: # 正常结束while循环才会执行else
                # 如果可以追加（或抢占后可以追加）
                num_seqs += 1
                self.block_manager.may_append(seq) # 预留一个块的空间
                scheduled_seqs.append(seq) # 将序列加入调度列表
        
        assert scheduled_seqs, "在decode阶段，必须至少调度一个序列"
        # 将本次调度的序列重新放回运行队列的前面，保持它们的优先级
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]):
        """
        在模型推理步骤完成后，对序列进行后处理。
        
        Args:
            seqs (list[Sequence]): 刚刚被模型处理过的序列列表。
            token_ids (list[int]): 模型为每个序列生成的新token ID。
        """
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id) # 将新生成的token追加到序列中
            
            # 检查序列是否完成
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED # 更新状态为完成
                self.block_manager.deallocate(seq) # 释放其占用的KV缓存块
                self.running.remove(seq) # 从运行队列中移除
                

            