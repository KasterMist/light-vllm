# 导入必要的库和模块
from typing import Any

import atexit  # 用于注册退出时的清理函数
from dataclasses import fields  # 用于获取数据类的字段信息
from time import perf_counter  # 高精度计时器，用于性能测量
from tqdm.auto import tqdm  # 进度条显示库
from transformers import AutoTokenizer  # HuggingFace的自动分词器
import torch.multiprocessing as mp  # PyTorch的多进程库

# 导入本项目的自定义模块
from lightvllm.config import Config  # 配置管理类
from lightvllm.sampling_params import SamplingParams  # 采样参数类
from lightvllm.engine.sequence import Sequence  # 序列管理类
from lightvllm.engine.scheduler import Scheduler  # 任务调度器
from lightvllm.engine.model_runner import ModelRunner  # 模型执行器


class LLMEngine:
    """
    LLM推理引擎的主类，负责协调模型加载、多进程管理、任务调度和生成文本
    
    该类实现了一个高性能的LLM推理框架，支持：
    1. 张量并行（tensor parallelism）- 将模型分布到多个GPU上
    2. 批处理推理 - 同时处理多个请求以提高吞吐量
    3. 连续批处理 - 动态调度新请求和完成的请求
    4. 预填充和解码阶段的区分处理
    """
    
    def __init__(self, model, **kwargs):
        """
        初始化LLM引擎
        
        Args:
            model: 模型名称或路径，例如 "Qwen/Qwen2.5-7B"
            **kwargs: 其他配置参数，会被自动筛选并传递给Config类
        
        实现步骤：
        1. 提取并构建配置对象
        2. 启动多进程工作节点（用于张量并行）
        3. 初始化主模型运行器、分词器和调度器
        4. 注册退出清理函数
        """
        # 从Config类的字段定义中提取有效的配置参数名
        config_fields = {field.name for field in fields(Config)}
        # 筛选出kwargs中属于Config类的参数
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        # 创建配置对象
        config = Config(model, **config_kwargs)
        
        # 初始化进程和事件列表，用于管理多进程通信
        self.ps = []  # 存储子进程对象
        self.events = []  # 存储进程间同步事件
        
        # 获取"spawn"上下文，确保子进程完全独立启动
        ctx = mp.get_context("spawn")

        # 启动张量并行的工作进程
        # 从rank 1开始，因为rank 0是主进程
        for i in range(1, config.tensor_parallel_size):
            # 创建进程间同步事件
            event = ctx.Event()
            # 创建ModelRunner子进程，传入配置、进程编号和同步事件
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()  # 启动子进程
            self.ps.append(process)  # 记录进程对象
            self.events.append(event)  # 记录同步事件
        
        # 初始化主进程的模型运行器（rank 0）
        self.model_runner = ModelRunner(config, 0, self.events)
        
        # 加载分词器，use_fast=True使用快速tokenizer提高性能
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        
        # 设置结束标记ID到配置中
        config.eos = self.tokenizer.eos_token_id
        
        # 初始化调度器，负责管理请求队列和批处理逻辑
        self.scheduler = Scheduler(config)
        
        # 注册退出时的清理函数，确保资源正确释放
        atexit.register(self.exit)
    
    def exit(self):
        """
        清理资源和退出函数
        
        该函数确保所有进程正确退出，避免资源泄漏：
        1. 向主模型运行器发送退出信号
        2. 删除模型运行器对象释放GPU内存
        3. 等待所有子进程正确终止
        
        注：该函数会在程序退出时由atexit自动调用
        """
        # 向主模型运行器发送退出命令，这会通知所有子进程退出
        self.model_runner.call("exit")
        # 删除模型运行器对象，释放相关资源
        del self.model_runner
        # 等待所有子进程正确终止，避免僵尸进程
        for p in self.ps:
            p.join()
    
    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        """
        添加生成请求到调度队列
        
        Args:
            prompt: 输入提示，可以是字符串或已编码的token ID列表
                   例如："你好，请介绍一下人工智能" 或 [101, 2123, 1045, ...]
            sampling_params: 采样参数，包括温度、top_p、最大长度等设置
        
        实现过程：
        1. 如果输入是字符串，使用分词器将其编码为token ID序列
        2. 创建Sequence对象来管理这个请求的状态和参数
        3. 将序列添加到调度器的等待队列中
        
        例如：engine.add_request("生成一首诗", SamplingParams(temperature=0.7))
        """
        # 如果输入是字符串，需要先编码为token IDs
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        
        # 创建序列对象，封装输入tokens和采样参数
        seq = Sequence(prompt, sampling_params)
        
        # 将序列添加到调度器的等待队列
        self.scheduler.add(seq)
    
    def step(self):
        """
        执行一次推理步骤
        
        这是引擎的核心方法，实现了连续批处理的单次迭代：
        
        Returns:
            tuple: (完成的输出列表, token处理数量)
                  - outputs: [(seq_id, token_ids), ...] 已完成序列的ID和生成的tokens
                  - num_tokens: 正数表示prefill阶段处理的tokens，负数表示decode阶段的序列数
        
        执行流程：
        1. 调度器决定这一步要处理哪些序列（可能是prefill或decode）
        2. 模型运行器并行执行推理计算
        3. 调度器根据生成结果更新序列状态
        4. 收集已完成的序列输出
        5. 返回结果和性能统计信息
        
        例如：
        - prefill阶段：处理新请求的输入tokens，num_tokens=256（处理了256个输入tokens）
        - decode阶段：为4个序列各生成1个token，num_tokens=-4（负数表示decode模式）
        """
        # 调度器决定本轮要处理的序列和执行模式（prefill或decode）
        seqs, is_prefill = self.scheduler.schedule()
        
        # 调用模型运行器执行推理，返回生成的token IDs
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        
        # 调度器根据生成结果更新序列状态（添加新tokens、检查完成条件等）
        self.scheduler.postprocess(seqs, token_ids)
        
        # 收集已完成序列的输出（序列ID和生成的完整token序列）
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        
        # 计算处理的token数量用于性能统计
        # prefill模式：返回处理的input tokens总数（正数）
        # decode模式：返回序列数的负数（用于区分两种模式）
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        
        return outputs, num_tokens

    def is_finished(self):
        """
        检查是否所有请求都已完成
        
        Returns:
            bool: True表示没有待处理的请求，False表示还有请求在处理中
        
        该方法通过调度器检查：
        - 等待队列是否为空
        - 正在运行的序列是否都已完成
        - 是否还有需要继续生成的序列
        """
        return self.scheduler.is_finished()
    
    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True
    ) -> list[dict[Any, Any]]:
        """
        批量生成文本的主要接口
        
        Args:
            prompts: 输入提示列表，每个元素可以是字符串或token ID列表
                    例如：["写一首诗", "介绍Python"] 或 [[101,234], [105,567]]
            sampling_params: 采样参数，可以是单个参数对象或参数列表
                           如果是单个对象，会复制给所有prompts使用
            use_tqdm: 是否显示进度条和吞吐量统计
        
        Returns:
            list[dict]: 生成结果列表，每个元素包含：
                       - "text": 解码后的文本字符串
                       - "token_ids": 生成的token ID序列
        
        实现流程：
        1. 初始化进度条（如果启用）
        2. 标准化采样参数格式
        3. 将所有请求添加到调度队列
        4. 循环执行推理步骤直到所有请求完成
        5. 实时更新吞吐量统计和进度
        6. 整理并返回最终结果
        
        性能特点：
        - 支持连续批处理，新请求可以与现有请求并行处理
        - 自动区分prefill（输入处理）和decode（生成）阶段
        - 实时显示两个阶段的吞吐量性能
        
        使用示例：
        results = engine.generate(
            prompts=["你好", "再见"], 
            sampling_params=SamplingParams(max_tokens=100, temperature=0.8)
        )
        """
        # 初始化进度条，用于显示生成进度
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        
        # 标准化采样参数：如果不是列表，则复制给每个prompt使用
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        
        # 将所有输入请求添加到调度队列
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)  # 编码字符串并添加到等待队列
        
        # 初始化输出收集和性能统计变量
        outputs = {}  # 用seq_id作为key收集完成的输出
        prefill_throughput = decode_throughput = 0.0  # 两阶段的吞吐量统计
        
        # 主循环：持续执行推理步骤直到所有请求完成
        while not self.is_finished():
            # 记录步骤开始时间，用于计算吞吐量
            t = perf_counter()
            
            # 执行一次推理步骤
            output, num_tokens = self.step()
            
            # 更新吞吐量统计并显示进度
            if use_tqdm:
                if num_tokens > 0:
                    # 正数表示prefill阶段，计算输入token处理速度
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    # 负数表示decode阶段，计算生成token的速度
                    decode_throughput = -num_tokens / (perf_counter() - t)
                
                # 更新进度条显示当前吞吐量
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)} tokens/s",
                    "Decode": f"{int(decode_throughput)} tokens/s"
                })
            
            # 收集本轮完成的序列输出
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                # 每完成一个序列，进度条增加1
                if use_tqdm:
                    pbar.update(1)
        
        # 按序列ID排序，保证输出顺序与输入顺序一致
        outputs = [outputs[seq_id] for seq_id in sorted(outputs)]
        
        # 将token IDs解码为文本，构建最终输出格式
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]

        # 关闭进度条
        if use_tqdm:
            pbar.close()
            
        return outputs

