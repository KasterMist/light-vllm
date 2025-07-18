from collections import deque
import xxhash
import numpy as np

from lightvllm.engine.sequence import Sequence

# Block is a container for a set of token ids.
# It is used to store the token ids of a sequence (several sequences can share the same block with ref_count > 1).
class Block:
    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0      # how many sequences are using this block
        self.hash = -1          # hash of the block, this hash is used to identify the block
        self.token_ids = []     # token ids in the block

    # update the block with new token ids (with its hash)
    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []
    

class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        assert num_blocks > 0
        self.block_size = block_size                                                # the size of each block
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]   # stores all the blocks
        self.hash_to_block_id: dict[int, int] = dict()                              # hash -> block_id
        self.free_block_ids: deque[int] = deque(range(num_blocks))                  # stores the free block ids. All the blocks are free at the begining.
        self.used_block_ids: set[int] = set()                                       # stores the used block ids

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1: # the prefix is used to identify the same token ids but with different prefix token ids
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    # allocate a new block -> remove the block(block_id) from free_block_ids and add it to used_block_ids
    # the new allocated block is reset and ref_count is set to 1
    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    # deallocate a block -> add the block(block_id) to free_block_ids and remove it from used_block_ids
    # the block is not reset, so the ref_count is not changed
    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    # can_allocate is used to check if the sequence can allocate a new block
    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks # means there are enough free blocks(ids) to allocate

    def allocate(self, seq: Sequence):
        assert not seq.block_table # make sure the sequence has no block_table (not allocated yet)
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.get_token_ids_from_block(i)
            h = self.compute_hash(token_ids) if len(token_ids) == self.block_size else -1  # -1 means the block is not full
            block_id = self.hash_to_block_id.get(h, -1)
            # if the block is not found or the token ids are different (which means cache miss), we need to allocate a new block
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            
            if cache_miss:
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                block = self._allocate_block(block_id)
            
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id # update the hash to block_id mapping
            seq.block_table.append(block_id) # link this block info to the sequence's block_table

    # deallocate all the blocks of the sequence
    def deallocate(self, seq: Sequence):
        for block_id in seq.block_table:
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0: # this block is not used by any sequence anymore
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0   # reset the number of cached tokens
        seq.block_table.clear()     # clear the block_table

    # can_allocate is used to check if the sequence can allocate a new block
    # can_allocate is based on new seq
    # can_append is used to check if the sequence can append a new token to the last block
    # can_append is based on current seq
    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)
    
    # this may be called when the sequence adds new tokens(then the sequence's info such as num blocks may be changed)
    # cannot use may_append without allocate this seq first
    def may_append(self, seq: Sequence):
        block_table = seq.block_table 
        last_block = self.blocks[block_table[-1]]
        
        if len(seq) % self.block_size == 1: # current the last block is full, the new added token will be in a new block
            assert last_block.hash != -1   # assert the last block's hash is calculated
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0: # current the last block is not full, the new added token will be in the last block
            assert last_block.hash == -1      # assert the last block's hash is not calculated
            token_ids = seq.get_token_ids_from_block(seq.num_blocks - 1) # get the token ids of the last block
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1 # get the prefix of the last last block (if there exists)
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            assert last_block.hash == -1











