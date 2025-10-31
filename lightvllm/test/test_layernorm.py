import torch
from lightvllm.kernels.cuda_kernels.layernorm import rms_norm, add_rms_norm
from lightvllm.layers.layernorm import RMSNorm
import pytest

@pytest.mark.parametrize("batch_size, hidden_size", [(32, 128), (64, 256), (128, 512), (256, 1024)])
def test_cuda_rms_norm(batch_size, hidden_size):
    """
    测试 CUDA RMS Norm 实现的正确性。

    Args:
        batch_size (int): 批次大小 (行数).
        hidden_size (int): 隐藏层维度 (列数).
    """
    # 1. 创建随机输入张量和权重
    input_tensor = torch.randn((batch_size, hidden_size), device="cuda", dtype=torch.float32)
    weight = torch.randn(hidden_size, device="cuda", dtype=torch.float32)
    eps = 1e-6

    # 2. 使用 PyTorch RMSNorm 计算预期结果（使用副本避免原地修改）
    rms_norm_layer = RMSNorm(hidden_size, eps=eps).to("cuda")
    rms_norm_layer.weight.data.copy_(weight)
    
    with torch.no_grad():
        expected_output = rms_norm_layer.rms_forward(input_tensor.clone())

    # 3. 使用 CUDA 内核计算实际结果（使用原始的input_tensor）
    cuda_output = rms_norm(input_tensor, weight, eps)

    # 4. 比较结果
    assert torch.allclose(cuda_output, expected_output, atol=1e-4, rtol=1e-4), \
        f"CUDA RMS Norm output does not match PyTorch RMS Norm for shape ({batch_size}, {hidden_size})"

    print(f"\nRMS Norm test passed for shape ({batch_size}, {hidden_size})")
    print(f"CUDA output (first 3): {cuda_output.flatten().tolist()[:3]}")
    print(f"PyTorch output (first 3): {expected_output.flatten().tolist()[:3]}")
    print(f"Max absolute difference: {torch.max(torch.abs(cuda_output - expected_output)).item():.8f}")


@pytest.mark.parametrize("batch_size, hidden_size", [(32, 128), (64, 256), (128, 512), (256, 1024)])
def test_cuda_add_rms_norm(batch_size, hidden_size):
    """
    测试 CUDA Add & RMS Norm 实现的正确性。

    Args:
        batch_size (int): 批次大小 (行数).
        hidden_size (int): 隐藏层维度 (列数).
    """
    # 1. 创建随机输入张量、残差张量和权重
    input_tensor = torch.randn((batch_size, hidden_size), device="cuda", dtype=torch.float32)
    residual_tensor = torch.randn((batch_size, hidden_size), device="cuda", dtype=torch.float32)
    weight = torch.randn(hidden_size, device="cuda", dtype=torch.float32)
    eps = 1e-6

    # 2. 使用 PyTorch RMSNorm 计算预期结果（使用副本避免原地修改）
    rms_norm_layer = RMSNorm(hidden_size, eps=eps).to("cuda")
    rms_norm_layer.weight.data.copy_(weight)
    
    with torch.no_grad():
        expected_output, expected_residual = rms_norm_layer.add_rms_forward(input_tensor.clone(), residual_tensor.clone())

    # 3. 使用 CUDA 内核计算实际结果（使用原始张量）
    cuda_output, cuda_residual = add_rms_norm(input_tensor, residual_tensor, weight, eps)

    # 4. 比较归一化输出结果
    assert torch.allclose(cuda_output, expected_output, atol=1e-4, rtol=1e-4), \
        f"CUDA Add & RMS Norm output does not match PyTorch for shape ({batch_size}, {hidden_size})"

    # 5. 比较新残差结果
    assert torch.allclose(cuda_residual, expected_residual, atol=1e-4, rtol=1e-4), \
        f"CUDA Add & RMS Norm residual does not match PyTorch for shape ({batch_size}, {hidden_size})"

    print(f"\nAdd & RMS Norm test passed for shape ({batch_size}, {hidden_size})")
    print(f"CUDA output (first 3): {cuda_output.flatten().tolist()[:3]}")
    print(f"PyTorch output (first 3): {expected_output.flatten().tolist()[:3]}")
    print(f"CUDA residual (first 3): {cuda_residual.flatten().tolist()[:3]}")
    print(f"PyTorch residual (first 3): {expected_residual.flatten().tolist()[:3]}")
    print(f"Max output difference: {torch.max(torch.abs(cuda_output - expected_output)).item():.8f}")
    print(f"Max residual difference: {torch.max(torch.abs(cuda_residual - expected_residual)).item():.8f}")


# def test_rms_norm_correctness():
#     """
#     验证 RMS Norm 的数学正确性。
#     """
#     batch_size, hidden_size = 2, 4
#     eps = 1e-6
    
#     # 手动创建简单的测试数据
#     x = torch.tensor([[1.0, 2.0, 3.0, 4.0],
#                       [0.5, 1.5, 2.5, 3.5]], device="cuda", dtype=torch.float32)
#     weight = torch.ones(hidden_size, device="cuda", dtype=torch.float32)
    
#     # 手动计算预期结果
#     # RMS Norm公式: y = x / sqrt(mean(x^2) + eps) * weight
#     x_squared = x.pow(2)
#     mean_x_squared = x_squared.mean(dim=1, keepdim=True)
#     rsqrt_var = torch.rsqrt(mean_x_squared + eps)
#     expected = x * rsqrt_var * weight
    
#     # 使用 CUDA 内核计算
#     cuda_output = rms_norm(x, weight, eps)
    
#     # 比较结果
#     assert torch.allclose(cuda_output, expected, atol=1e-6, rtol=1e-6), \
#         "Manual calculation does not match CUDA RMS Norm"
    
#     print(f"\nCorrectness test passed")
#     print(f"Manual calculation: {expected}")
#     print(f"CUDA output: {cuda_output}")


# def test_add_rms_norm_correctness():
#     """
#     验证 Add & RMS Norm 的数学正确性。
#     """
#     batch_size, hidden_size = 2, 4
#     eps = 1e-6
    
#     # 手动创建简单的测试数据
#     x = torch.tensor([[1.0, 2.0, 3.0, 4.0],
#                       [0.5, 1.5, 2.5, 3.5]], device="cuda", dtype=torch.float32)
#     residual = torch.tensor([[0.1, 0.2, 0.3, 0.4],
#                             [0.05, 0.15, 0.25, 0.35]], device="cuda", dtype=torch.float32)
#     weight = torch.ones(hidden_size, device="cuda", dtype=torch.float32)
    
#     # 手动计算预期结果
#     # 1. 残差连接
#     added = x + residual
#     expected_residual = added.clone()
    
#     # 2. RMS Norm
#     x_squared = added.pow(2)
#     mean_x_squared = x_squared.mean(dim=1, keepdim=True)
#     rsqrt_var = torch.rsqrt(mean_x_squared + eps)
#     expected_output = added * rsqrt_var * weight
    
#     # 使用 CUDA 内核计算
#     cuda_output, cuda_residual = add_rms_norm(x, residual, weight, eps)
    
#     # 比较结果
#     assert torch.allclose(cuda_output, expected_output, atol=1e-6, rtol=1e-6), \
#         "Manual calculation does not match CUDA Add & RMS Norm output"
    
#     assert torch.allclose(cuda_residual, expected_residual, atol=1e-6, rtol=1e-6), \
#         "Manual calculation does not match CUDA Add & RMS Norm residual"
    
#     print(f"\nAdd & RMS Norm correctness test passed")
#     print(f"Expected output: {expected_output}")
#     print(f"CUDA output: {cuda_output}")
#     print(f"Expected residual: {expected_residual}")
#     print(f"CUDA residual: {cuda_residual}")

