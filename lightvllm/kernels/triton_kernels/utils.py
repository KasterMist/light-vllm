import torch
import triton


def is_hip() -> bool:
    """
    检查当前是否运行在AMD HIP环境上
    
    HIP (Heterogeneous-Compute Interface for Portability) 是AMD的GPU计算平台。
    不同的GPU平台在Triton内核配置上可能有差异，需要特殊处理。
    """
    return torch.version.hip is not None


def calculate_settings(n):
    """
    根据问题规模计算最优的Triton内核配置
    
    参数：
        n (int): 需要处理的元素数量（通常是特征维度大小）
        
    返回：
        tuple: (BLOCK_SIZE, num_warps)
            - BLOCK_SIZE: 每个线程块处理的元素数量
            - num_warps: 每个线程块使用的warp数量
            
    优化策略：
        1. BLOCK_SIZE设为2的幂次，提高内存访问效率
        2. 根据BLOCK_SIZE调整warp数量，平衡并行度和资源利用率
        3. 限制最大BLOCK_SIZE，避免寄存器压力过大
    """
    # reference: https://github.com/unslothai/unsloth/blob/fd753fed99ed5f10ef8a9b7139588d9de9ddecfb/unsloth/kernels/utils.py#L43

    MAX_FUSED_SIZE = 65536  # Triton推荐的最大块大小
    BLOCK_SIZE = triton.next_power_of_2(n)  # 向上取整到最近的2的幂次
    
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(
            f"Cannot launch Triton kernel since n = {n} exceeds "
            f"the recommended Triton blocksize = {MAX_FUSED_SIZE}."
        )

    # 根据BLOCK_SIZE大小调整warp数量，实现最佳性能
    num_warps = 4  # 默认值
    if BLOCK_SIZE >= 32768:
        num_warps = 32 if not is_hip() else 16  # AMD GPU需要较少的warp
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
        
    return BLOCK_SIZE, num_warps