# model_runner.py 详细介绍

## 1. 文件定位与作用

`model_runner.py` 位于 `lightvllm/engine/` 目录下，是 LightVLLM 推理引擎的核心组件之一。其主要职责是**管理模型的加载、推理、分布式通信、KV Cache 分配、CUDA 图捕获等流程**，为高效的多卡推理提供底层支持。

---

## 2. 主要类与成员

### 2.1 ModelRunner

#### 2.1.1 初始化流程（`__init__`）

- **参数说明**：
  - `config`: 配置对象，类型为 `Config`，包含模型、推理、分布式等参数。
  - `rank`: 当前进程的分布式 rank。
  - `event`: 进程间同步事件。

- **主要步骤**：
  1. 读取配置，设置 block size、enforce_eager、world_size、rank、event 等。
  2. 初始化分布式进程组（NCCL，TCP 通信）。
  3. 设置 CUDA 设备与默认数据类型。
  4. 加载模型（`Qwen3ForCausalLM`），并通过 `load_model` 加载权重。
  5. 初始化采样器（`Sampler`）。
  6. 预热模型（`warmup_model`）。
  7. 分配 KV Cache（`allocate_kv_cache`）。
  8. 如果不是 eager 模式，捕获 CUDA 图（`capture_cudagraph`）。
  9. 多卡场景下，rank 0 创建共享内存，其他 rank 等待并进入主循环（`loop`）。

#### 2.1.2 主要方法

- `exit()`: 释放资源，关闭共享内存，销毁分布式进程组。
- `loop()`: 非 rank 0 进程的主循环，等待主进程指令并执行。
- `read_shm()` / `write_shm()`: 进程间通过共享内存通信，传递方法名和参数。
- `call()`: 根据方法名调用对应成员方法。
- `warmup_model()`: 预热模型，减少首次推理延迟。
- `allocate_kv_cache()`: 动态分配 KV Cache，绑定到模型各层。
- `prepare_block_tables()`: 构建 block table，用于 KV Cache 管理。
- `prepare_prefill()` / `prepare_decode()`: 构建输入张量，设置上下文，支持 prefill 和 decode 两种推理阶段。
- `prepare_sample()`: 构建采样温度张量。
- `run_model()`: 执行模型推理，支持 eager 和 CUDA Graph 两种模式。
- `run()`: 推理主入口，串联输入准备、模型推理、采样等步骤。
- `capture_cudagraph()`: 捕获 CUDA Graph，提高 decode 阶段推理效率。

---

## 3. 关键流程详解

### 3.1 分布式与多进程通信

- 采用 PyTorch 的 `torch.distributed` 实现多卡通信。
- 通过 `multiprocessing.shared_memory` 和 `Event` 实现主进程与子进程间的高效通信。
- rank 0 负责写入共享内存并通知其他进程，其他进程读取并执行指令。

### 3.2 KV Cache 分配与管理

- 根据 GPU 显存动态计算可分配的 KV Cache block 数量。
- 将分配好的 KV Cache 绑定到模型每一层的 `k_cache` 和 `v_cache` 属性上，实现高效的缓存复用。

### 3.3 CUDA Graph 捕获

- 对 decode 阶段的推理流程进行 CUDA Graph 捕获，极大提升小 batch size 下的推理吞吐。
- 支持不同 batch size 的 Graph 复用。

### 3.4 输入准备与上下文管理

- `prepare_prefill` 和 `prepare_decode` 分别针对 prefill 和 decode 阶段准备输入张量。
- 通过 `set_context`/`reset_context` 管理全局上下文，便于模型内部访问。

### 3.5 推理与采样

- `run_model` 负责实际的模型前向推理。
- `Sampler` 用于根据 logits 和温度采样下一个 token。

---

## 4. 相关外部类与工具

### 4.1 Sequence

- 来源：`lightvllm.engine.sequence`
- 作用：表示单条推理序列，管理 token、block_table、缓存状态等。
- 主要属性：`block_table`（KV Cache block 索引）、`num_cached_tokens`、`num_blocks`、`last_block_num_tokens`、`last_token`、`temperature` 等。

### 4.2 Sampler

- 来源：`lightvllm.layers.sampler`
- 作用：根据 logits 和温度参数进行采样，输出下一个 token 的 id。
- 支持多种采样策略（如 top-k、top-p、温度采样等）。

### 4.3 Qwen3ForCausalLM

- 来源：`lightvllm.models.qwen3`
- 作用：Qwen3 结构的因果语言模型，支持高效的推理和 KV Cache 管理。

### 4.4 set_context / get_context / reset_context

- 来源：`lightvllm.utils.context`
- 作用：管理全局推理上下文，便于模型和 CUDA Graph 访问输入相关信息。

### 4.5 load_model

- 来源：`lightvllm.utils.loader`
- 作用：加载模型权重到指定的模型实例。

---

## 5. 总结

`ModelRunner` 是 LightVLLM 推理引擎的核心调度与执行单元，负责模型加载、分布式通信、KV Cache 管理、输入准备、推理执行、采样输出等全流程。其设计充分利用了 PyTorch 的分布式与 CUDA Graph 能力，结合高效的内存与上下文管理，实现了大模型在多卡环境下的高效推理。

---

如需进一步了解每个方法的具体实现细节，可结合源码和注释逐步阅读。 