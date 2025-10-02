from collections import deque

from lightvllm.config import Config
from lightvllm.engine.sequence import Sequence, SequenceStatus
from lightvllm.engine.block_manager import BlockManager

class Scheduler:
    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos   # end of sequence token
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()  # sequences will be allocated to the waiting at first
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    # add sequence to waiting queue for later allocation
    def add(self, seq: Sequence):
        self.waiting.append(seq)
    
    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        # prefill
        # pop sequences from waiting queue. Make sure the number of sequences is less than max_num_seqs.
        # when the num_seqs (number of poped seqs in waiting queue) is enough, break the loop
        scheduled_seqs = [] # sequences scheduled in the current step
        num_seqs = 0    # number of sequences in the current batch
        num_batched_tokens = 0 # total number of tokens in the current batch
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0] # get the first sequence in the waiting queue
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq) # these sequences are executed in prefilling work
        if scheduled_seqs:
            return scheduled_seqs, True
        
        # decode
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]):
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if(not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
                

            