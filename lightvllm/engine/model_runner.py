# 导入必要的库
import pickle  # 用于序列化和反序列化Python对象，以便在进程间传递
import torch
import torch.distributed as dist  # PyTorch分布式计算库
from multiprocessing.synchronize import Event  # 多进程同步事件
from multiprocessing.shared_memory import SharedMemory  # 多进程共享内存

# 导入项目内部模块
from lightvllm.config import Config  # 配置类
from lightvllm.engine.sequence import Sequence  # 序列管理类
from lightvllm.models.qwen3 import Qwen3ForCausalLM  # 具体的模型实现
from lightvllm.layers.sampler import Sampler  # Token采样器
from lightvllm.utils.context import set_context, get_context, reset_context  # 上下文管理工具
from lightvllm.utils.loader import load_model  # 模型加载工具


class ModelRunner:
    """
    模型运行器，负责在单个进程（GPU）上执行模型的前向传播。
    
    这个类是张量并行（Tensor Parallelism）的核心组件。每个ModelRunner实例
    对应一个GPU，并管理该GPU上的模型副本、KV缓存和计算逻辑。
    
    主要职责：
    1. 初始化分布式环境和模型。
    2. 管理KV缓存的分配和使用。
    3. 准备模型输入（prefill和decode两种模式）。
    4. 执行模型前向计算（支持CUDA Graph以提高性能）。
    5. 与主进程（rank 0）通过共享内存进行通信。
    """

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        """
        初始化ModelRunner实例。
        
        Args:
            config (Config): 全局配置对象。
            rank (int): 当前进程在分布式环境中的排名（rank）。
            event (Event | list[Event]): 用于进程同步的事件。
                                         - rank 0: 接收一个事件列表，用于通知所有工作进程。
                                         - rank > 0: 接收单个事件，用于接收主进程的通知。
        """
        self.config = config
        self.kernel_backend = config.kernel_backend  # 使用的算子后端
        hf_config = config.hf_config
        hf_config.kernel_backend = self.kernel_backend
        self.block_size = config.kvcache_block_size     # PagedAttention中每个KV缓存块（block）的大小（以token数量计）。
        self.enforce_eager = config.enforce_eager       # 是否强制使用Eager模式，禁用CUDA Graphs
        self.world_size = config.tensor_parallel_size   # 张量并行的大小，即使用的GPU数量。例如，设置为2表示使用2张GPU进行张量并行。
        self.rank = rank    # 当前进程在分布式环境中的rank值
        self.event = event

        # 1. 初始化分布式环境
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        
        # 2. 设置PyTorch默认环境（数据类型、设备）
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        
        # 3. 加载模型和采样器
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)  # 从HuggingFace Hub加载预训练权重
        self.sampler = Sampler()
        
        # 4. 模型预热、分配KV缓存、捕获CUDA Graph
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            print("Capturing CUDA Graphs...")
            self.capture_cudagraph()
            
        # 5. 恢复默认设备和数据类型
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        # 6. 设置多进程通信（共享内存）
        if self.world_size > 1:
            if rank == 0:
                # 主进程（rank 0）创建共享内存
                self.shm = SharedMemory(name="lightvllm", create=True, size=2**20) # 1MB
            else:
                # 工作进程等待主进程创建共享内存
                pass
            dist.barrier()  # 同步所有进程，确保共享内存已创建
            if rank == 0:
                pass
            else:
                # 工作进程连接到已创建的共享内存，并进入循环等待指令
                self.shm = SharedMemory(name="lightvllm")
                self.loop()

    def exit(self):
        """
        清理资源并退出进程。
        
        这个方法负责：
        1. 关闭和释放共享内存。
        2. 释放CUDA Graph占用的资源。
        3. 同步GPU操作。
        4. 销毁分布式进程组。
        """
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                # 主进程负责删除共享内存文件
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        """
        工作进程（rank > 0）的主循环。
        
        在这个循环中，工作进程会持续等待主进程通过共享内存发送的指令，
        然后执行相应的操作。当收到 "exit" 指令时，循环结束。
        """
        while True:
            # 从共享内存读取指令
            method_name, args = self.read_shm()
            # 调用相应的方法
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        """
        从共享内存中读取指令和参数（仅工作进程使用）。
        
        实现细节：
        1. 等待主进程设置的事件，表示有新数据可读。
        2. 从共享内存的开头读取4个字节，获取数据长度。
        3. 根据长度读取序列化的数据。
        4. 使用pickle反序列化数据，得到方法名和参数。
        5. 清除事件，准备下一次读取。
        """
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()  # 阻塞等待，直到主进程发出信号
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        """
        将指令和参数写入共享内存（仅主进程使用）。
        
        实现细节：
        1. 使用pickle序列化方法名和参数。
        2. 将序列化后的数据长度写入共享内存的前4个字节。
        3. 将数据本身写入共享内存。
        4. 设置所有工作进程的事件，通知它们有新数据。
        """
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()  # 触发所有工作进程的事件

    def call(self, method_name, *args):
        """
        动态调用本对象的指定方法。
        
        - 如果是主进程（rank 0），在本地调用前，会先将指令写入共享内存，
          以广播给所有工作进程。
        - 如果是工作进程，它只是在本地执行从共享内存收到的指令。
        """
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        
        method = getattr(self, method_name, None)
        if method:
            return method(*args)

    def warmup_model(self):
        """
        模型预热。
        
        通过运行一次最大尺寸的推理来预热模型。这有助于：
        1. 确保所有CUDA核心和库都已初始化。
        2. 减少首次实际推理的延迟。
        3. 准确测量模型加载后的峰值内存使用量。
        """
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens = self.config.max_num_batched_tokens
        max_model_len = self.config.max_model_len
        # 计算一个合理的预热批次大小
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)  # 以prefill模式运行一次
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        """
        动态计算并分配KV缓存。
        
        根据可用的GPU显存和配置中的内存利用率，计算可以分配多少个KV缓存块。
        
        计算逻辑：
        1. 获取当前GPU的总显存和已用显存。
        2. 获取模型预热后的峰值显存使用量。
        3. 计算可用显存 = (总显存 * 利用率) - 已用显存 - (峰值 - 当前)
        4. 计算每个KV缓存块所需字节数。
        5. 可用显存 / 每个块的字节数 = 可分配的块数。
        
        分配后，将KV缓存张量直接关联到模型各层的`k_cache`和`v_cache`属性。
        """
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        
        # 计算每个GPU上的KV头数量
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        # 计算一个KV缓存块的大小（字节）,包含了所有的层占用的大小
        block_bytes = (2 * hf_config.num_hidden_layers * self.block_size * 
                       num_kv_heads * hf_config.head_dim * hf_config.torch_dtype.itemsize)
                       
        # 计算可分配的块数
        # 计算可用于KV缓存的显存。
        # 公式: (总预算) - (当前已用) - (推理时激活值等临时显存占用)
        # 其中，(推理时临时显存) 通过 (预热峰值 - 预热后稳定值) 来估算。
        # 这种方式可以精确地为KV缓存预留出最大空间，同时保证运行时不会因激活值而OOM。
        available_memory = total * config.gpu_memory_utilization - used - peak + current
        config.num_kvcache_blocks = int(available_memory) // block_bytes
        assert config.num_kvcache_blocks > 0, "Not enough memory for KV cache"
        
        # 创建KV缓存张量
        self.kv_cache = torch.zeros(
            2,  # 0 for K, 1 for V
            hf_config.num_hidden_layers,
            config.num_kvcache_blocks,
            self.block_size,
            num_kv_heads,
            hf_config.head_dim
        )
        
        # 将KV缓存分配给模型的每一层
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        """
        准备block_tables张量。
        
        block_tables是一个二维张量，每一行对应一个序列，行中的每个元素是
        该序列使用的KV缓存块的索引。
        
        例如，一个序列使用了第5、10、3个块，其block_table就是[5, 10, 3]。
        为了能批处理，所有行的长度会填充到与最长序列一致，填充值为-1。
        """
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        """
        为prefill（预填充）阶段准备模型输入。
        
        Prefill阶段处理每个请求的初始prompt。由于每个prompt长度不同，
        需要将它们拼接成一个大的批次，并提供相应的元数据。
        
        返回：
            - input_ids: 所有序列的待处理token拼接成的一维张量。
            - positions: 对应的位置ID。
        
        同时，通过`set_context`设置全局上下文，包含：
            - cu_seqlens_q/k: 累积序列长度，用于FlashAttention。
            - max_seqlen_q/k: Q和K的最大序列长度。
            - slot_mapping: 将每个token映射到其在KV缓存中的具体位置（slot）。
            - block_tables: 如果有前缀缓存，则提供块表。
        """
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        
        for seq in seqs:
            seqlen = len(seq)
            # 仅处理未缓存的token
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            
            if not seq.block_table:
                continue
            
            # 计算新token的slot mapping
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens 
                slot_mapping.extend(list(range(start, end)))
                
        # 如果存在前缀缓存（即K的长度大于Q的长度），则需要提供block_tables
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            block_tables = self.prepare_block_tables(seqs)
            
        # 将列表转换为PyTorch张量并移到GPU
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        
        # 设置全局上下文，供模型内部使用
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        """
        为decode（解码）阶段准备模型输入。
        
        Decode阶段为批次中的每个序列生成一个新token。此时，每个序列只输入
        其最后一个token。
        
        返回：
            - input_ids: 每个序列的最后一个token组成的一维张量。
            - positions: 每个token在其序列中的位置。
            
        同时，通过`set_context`设置全局上下文，包含：
            - slot_mapping: 每个新token要写入的KV缓存位置。
            - context_lens: 每个序列的当前总长度。
            - block_tables: 所有序列的KV缓存块表。
        """
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq))
            context_lens.append(len(seq))
            # 计算新token的KV缓存槽位
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
            
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        
        # 设置全局上下文
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        """
        准备采样所需的参数，如温度。
        """
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        """
        执行模型前向计算，获取logits。
        
        - 如果是prefill模式、强制eager模式或批次大小超过阈值，则正常执行模型。
        - 否则，使用预先捕获的CUDA Graph来执行，以减少CPU开销和提高性能。
        """
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            # 正常（Eager）模式执行
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            # CUDA Graph模式执行
            bs = input_ids.size(0)
            context = get_context()
            # 选择一个大小合适的graph
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            
            # 更新graph的输入张量
            for k, v in graph_vars.items():
                if k != "outputs":
                    v.zero_()
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            
            # 重放CUDA Graph
            graph.replay()
            
            # 返回计算结果
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int] | None:
        """
        完整的单步推理流程。
        
        Args:
            seqs (list[Sequence]): 当前批次要处理的序列。
            is_prefill (bool): 是否为prefill模式。
            
        Returns:
            - rank 0: 返回采样后的token ID列表。
            - rank > 0: 返回None。
        
        流程：
        1. 根据模式（prefill/decode）准备输入数据。
        2. 主进程准备采样参数。
        3. 所有进程执行模型前向计算，得到logits。
        4. 主进程根据logits和采样参数进行采样，得到新token。
        5. 清理上下文。
        """
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        """
        捕获用于decode阶段的CUDA Graph。
        
        为了优化小批量的decode步骤，预先为不同批次大小（batch size）
        捕获CUDA计算图。在实际推理时，只需更新输入并重放图，
        可以显著减少CPU开销。
        
        实现：
        1. 定义一系列要捕获的批次大小（如1, 2, 4, 8, 16...）。
        2. 为每个批次大小创建一个静态的输入张量。
        3. 使用`torch.cuda.graph`上下文管理器来捕获模型的前向传播过程。
        4. 将捕获的图和相关的输入/输出张量存储起来，以备后续使用。
        """
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        
        # 创建静态的输入/输出张量
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        
        # 定义要捕获的批次大小
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        # 从大到小捕获，以便复用内存池
        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            
            # 预热一次
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
            
            # 捕获图
            with torch.cuda.graph(graph, pool=self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])
                
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        # 保存静态张量，以便在重放时更新
        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
