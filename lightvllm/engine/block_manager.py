from collections import deque
import xxhash
import numpy as np

from lightvllm.engine.sequence import Sequence

# Block（块）是存储一组token ID的容器。
# 它用于存储序列的token ID（多个序列可以通过 ref_count > 1 共享同一个块）。
class Block:
    """
    代表KV缓存中的一个物理块。
    这是PagedAttention内存管理的最小单位。
    """
    def __init__(self, block_id):
        """
        初始化一个块。
        
        Args:
            block_id (int): 块的唯一物理ID。
        """
        self.block_id = block_id
        # 引用计数，记录有多少个序列正在使用这个块。当 ref_count 为 0 时，该块可以被回收。
        self.ref_count = 0
        # 块内容的哈希值。用于快速识别和共享内容相同的块（前缀缓存）。
        self.hash = -1
        # 存储在该块中的token ID列表。
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        """用新的token ID和哈希值更新块的内容。"""
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        """重置块的状态，通常在块被新分配时调用。"""
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []
    

class BlockManager:
    """
    块管理器，负责整个KV缓存的物理块的分配、释放和共享（前缀缓存）。
    """
    def __init__(self, num_blocks: int, block_size: int):
        """
        初始化块管理器。
        
        Args:
            num_blocks (int): KV缓存中的总物理块数。
            block_size (int): 每个块可以存储的token数量。
        """
        assert num_blocks > 0
        self.block_size = block_size
        # 存储所有物理块对象的列表
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        # 一个哈希到块ID的映射，用于实现前缀缓存。key是块内容的哈希，value是块的ID。
        self.hash_to_block_id: dict[int, int] = dict()
        # 一个存储空闲块ID的双端队列。所有块在开始时都是空闲的。
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        # 一个存储已使用块ID的集合，用于快速查找。
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        """
        计算一个token ID列表的哈希值。只有当块被填满时才计算。
        
        Args:
            token_ids (list[int]): 要计算哈希的token ID列表。
            prefix (int): (可选) 前一个块的哈希值。这确保了即使两个块的token_ids相同，
                          但它们的前缀不同时，它们的哈希值也不同。
        
        Returns:
            int: 计算出的64位哈希值。
        """
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        """
        分配一个新块的内部辅助函数。
        将块从空闲列表移动到已使用列表，并重置其状态。
        """
        block = self.blocks[block_id]
        assert block.ref_count == 0, "尝试分配一个仍被引用的块"
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int):
        """
        释放一个块的内部辅助函数。
        将块从已使用列表移动回空闲列表。
        """
        assert self.blocks[block_id].ref_count == 0, "尝试释放一个仍被引用的块"
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        """检查是否有足够的空闲块来分配给一个新序列。"""
        # 这里没有考虑前缀缓存命中，是一个保守的检查。
        # 实际分配时，由于缓存命中，可能需要比 seq.num_blocks 更少的块。
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        """
        为一个序列分配所需的KV缓存块。
        这个函数只在序列第一次被调度（prefill阶段）时调用一次。
        它会尝试利用前缀缓存来复用已有的块。
        Done: 避免prefill阶段compute_hash没有prefix导致的错误情况
        """
        assert not seq.block_table, "序列已经被分配过块，不能重复分配"
        cache_miss = False # 默认当前的seq的kv访存未命中

        # 新增：用于在循环中传递前一个块的哈希值
        h = -1

        for i in range(seq.num_blocks):
            token_ids = seq.get_token_ids_from_block(i)
            # 只有当块是满的时候才计算哈希并尝试缓存
            h = self.compute_hash(token_ids, prefix=h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1) # 查找对应键的值，如果没有则返回-1
            
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            if cache_miss:
                # 缓存未命中，从空闲列表中分配一个新块
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                # 缓存命中
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    # 如果命中的块当前正在被使用，增加其引用计数
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    # 如果命中的块当前是空闲的（例如，之前被释放了），则重新分配它
                    block = self._allocate_block(block_id)
            if h != -1: # 如果块是满的，更新其内容和哈希，并加入缓存,块不满则不需要计算哈希和更新block的token ids，因为未满的block的token ids一定是最后一个block，token ids可以计算出来
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            # 将分配的或复用的块ID添加到序列的块表中
            seq.block_table.append(block_id)



    def deallocate(self, seq: Sequence):
        """
        释放一个序列占用的所有块。
        当序列完成、被抢占或被交换出时调用。
        """
        for block_id in seq.block_table:
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0: # 如果没有其他序列引用这个块了
                # 注意：这里没有从 hash_to_block_id 中移除，是为了缓存。
                # 即使块被释放，它的内容和哈希值仍然保留，可供未来复用。
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        """
        检查是否可以为序列追加一个新token。
        这通常意味着是否需要一个新的物理块来存储这个新token。
        """
        # 如果当前序列的最后一个逻辑块正好被填满，那么下一个token就需要一个新的物理块。
        # `len(seq) % self.block_size == 0` 表示最后一个块刚填满。
        # `len(seq)` 是追加前的长度。
        # 例如 block_size=4, len(seq)=4, 下一个token是第5个，需要新块。
        # len(seq)=5, 下一个token是第6个，在现有块内，不需要新块。
        # 这里 `len(seq) % self.block_size == 1` 的逻辑似乎有误，应该是 `== 0`。
        # 假设原意是检查是否需要新块。
        needs_new_block = (len(seq) % self.block_size == 0)
        return len(self.free_block_ids) >= int(needs_new_block)
    
    def may_append(self, seq: Sequence):
        """
        为序列追加一个token做准备。
        如果需要，会分配一个新的块。如果一个块被填满了，会计算它的哈希并更新缓存。
        这个函数在调度器的decode阶段被调用，以确保在模型运行前，资源已经准备好。
        """
        block_table = seq.block_table 
        last_block = self.blocks[block_table[-1]]
        
        # 场景1: 当前序列的最后一个逻辑块已满，并且多出一个token需要新的物理块。
        if len(seq) % self.block_size == 1:
            # 此时，刚刚被填满的那个块的哈希值应该已经被计算和缓存了。
            assert last_block.hash != -1, "一个刚被填满的块应该有哈希值"
            # 分配一个新块
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        # 场景2: 当前序列的最后一个逻辑块填刚好填满
        elif len(seq) % self.block_size == 0:
            # 此时可以计算它的哈希并更新缓存。
            assert last_block.hash == -1, "一个未满的块不应该有哈希值"
            # 获取这个即将被填满的块的所有token
            token_ids = seq.get_token_ids_from_block(seq.num_blocks - 1)
            # 获取前一个块的哈希作为前缀
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            # 场景3: 最后一个块未满，且token数量不为1
            # 在这种情况下，我们不需要做任何操作，因为块还未满，无需计算哈希。
            assert last_block.hash == -1, "一个未满的块不应该有哈希值"











