import torch
from lightvllm.kernels.triton_kernels import triton_online_softmax, triton_softmax
from lightvllm.kernels.cuda_kernels import cuda_softmax, cuda_online_softmax
import pytest

@pytest.mark.parametrize("S, H", [(4096, 256), (2048, 512), (1024, 1024)])
def test_triton_online_softmax(S, H):
    """
    测试 Triton online softmax 实现的正确性。

    Args:
        S (int): 序列长度 (行数).
        H (int): 隐藏层维度 (列数).
    """
    # 1. 创建随机输入张量
    input_tensor = torch.randn((S, H), device="cuda", dtype=torch.float32)

    # 2. 使用 torch.softmax 计算预期结果
    #    torch.softmax 默认在最后一维计算，但我们的 online_softmax 是逐行处理的，
    #    因此我们需要在 dim=1 上计算。
    expected_output = torch.softmax(input_tensor, dim=1)

    # 3. 使用 Triton 内核计算实际结果
    triton_output = triton_online_softmax(input_tensor)

    # 4. 比较结果
    #    使用 torch.allclose 来检查两个张量是否在数值上足够接近
    assert torch.allclose(triton_output, expected_output, atol=1e-5, rtol=1e-5), \
        f"Triton softmax output does not match torch.softmax for shape ({S}, {H})"

    print(f"\nTest passed for shape ({S}, {H})")
    print(f"Triton output (first 3): {triton_output.flatten().tolist()[:3]}")
    print(f"PyTorch output (first 3): {expected_output.flatten().tolist()[:3]}")


@pytest.mark.parametrize("S, H", [(4096, 256), (2048, 512), (1024, 1024)])
def test_triton_naive_softmax(S, H):
    """
    测试 Triton naive softmax 实现的正确性。

    Args:
        S (int): 序列长度 (行数).
        H (int): 隐藏层维度 (列数).
    """
    # 1. 创建随机输入张量
    input_tensor = torch.randn((S, H), device="cuda", dtype=torch.float32)

    # 2. 使用 torch.softmax 计算预期结果
    expected_output = torch.softmax(input_tensor, dim=1)

    # 3. 使用 Triton 内核计算实际结果
    triton_output = triton_softmax(input_tensor)

    # 4. 比较结果
    assert torch.allclose(triton_output, expected_output, atol=1e-5, rtol=1e-5), \
        f"Triton softmax output does not match torch.softmax for shape ({S}, {H})"

@pytest.mark.parametrize("S, H", [(4096, 256), (2048, 512), (1024, 1024)])
def test_cuda_softmax(S, H):
    """
    测试 CUDA softmax 实现的正确性。

    Args:
        S (int): 序列长度 (行数).
        H (int): 隐藏层维度 (列数).
    """
    # 1. 创建随机输入张量
    input_tensor = torch.randn((S, H), device="cuda", dtype=torch.float32)

    # 2. 使用 torch.softmax 计算预期结果
    expected_output = torch.softmax(input_tensor, dim=1)

    # 3. 使用 CUDA 内核计算实际结果
    cuda_output = cuda_softmax(input_tensor)

    # 4. 比较结果
    assert torch.allclose(cuda_output, expected_output, atol=1e-5, rtol=1e-5), \
        f"CUDA softmax output does not match torch.softmax for shape ({S}, {H})"

@pytest.mark.parametrize("S, H", [(4096, 256), (2048, 512), (1024, 1024)])
def test_cuda_online_softmax(S, H):
    """
    测试 CUDA online softmax 实现的正确性。

    Args:
        S (int): 序列长度 (行数).
        H (int): 隐藏层维度 (列数).
    """
    # 1. 创建随机输入张量
    input_tensor = torch.randn((S, H), device="cuda", dtype=torch.float32)

    # 2. 使用 torch.softmax 计算预期结果
    expected_output = torch.softmax(input_tensor, dim=1)

    # 3. 使用 CUDA 内核计算实际结果
    cuda_output = cuda_online_softmax(input_tensor)

    # 4. 比较结果
    assert torch.allclose(cuda_output, expected_output, atol=1e-5, rtol=1e-5), \
        f"CUDA online softmax output does not match torch.softmax for shape ({S}, {H})"


