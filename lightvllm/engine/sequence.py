from copy import copy
from enum import Enum, auto
from itertools import count
from lightvllm.sampling_params import SamplingParams

# create a enum, which WAITING means 0, RUNNING means 1, FINISHED means 2
class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()

class Sequence:
    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        # next(Sequence.counter) is a generator that returns a new value each time it is called.
        # This is used to assign a unique ID to each sequence.
        self.seq_id = next(Sequence.counter)            # assign a unique ID to each sequence
        self.status = SequenceStatus.WAITING            # set the status to waiting
        self.token_ids = copy(token_ids)              # copy the token_ids to the sequence
        self.last_token = token_ids[-1]                 # set the last token to the last token in the token_ids
        self.num_tokens = len(self.token_ids)           # set the number of tokens to the length of the token_ids
        self.num_prompt_tokens = len(token_ids)         # set the number of prompt tokens to the length of the token_ids
        self.num_cached_tokens = 0                      # set the number of cached tokens to 0
        self.block_table: list[int] = []                # set the block table to an empty list. It is used to store the block ids of the sequence.
        self.temperature = sampling_params.temperature  # set the temperature to the temperature in the sampling_params
        self.max_tokens = sampling_params.max_tokens    # set the max tokens to the max tokens in the sampling_params
        self.ignore_eos = sampling_params.ignore_eos    # set the ignore eos to the ignore eos in the sampling_params, if True, the sequence will not stop at the end of the sequence

    # this function is called when the sequence is indexed, like len(sequence)
    def __len__(self):
        return self.num_tokens
    
    # this function is called when the sequence is indexed, like sequence[0]
    def __getitem__(self, key): 
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED
    
    @property
    def is_running(self):
        return self.status == SequenceStatus.RUNNING
    
    @property
    def is_waiting(self):
        return self.status == SequenceStatus.WAITING

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    # get all the prompt token ids to a list
    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    # get all the token ids (after the prompt token ids) to a list
    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size
    
    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    # get the token ids of the block i
    def get_token_ids_from_block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i * self.block_size: (i + 1) * self.block_size]

    def append_token(self, token_id):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1
    
    # this function is called when the sequence is pickled
    def __getstate__(self):
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens,
                self.block_table, self.token_ids if self.num_completion_tokens == 0 else self.last_token)
    
    # this function is called when the sequence is unpickled
    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[: -1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.last_token = state
        