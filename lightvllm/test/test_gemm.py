import torch
from lightvllm.kernels.cuda_kernels.gemm import gemm_f32
import pytest

@pytest.mark.parametrize("M, N, K", [(64, 64, 64), (128, 128, 128), (256, 256, 256), (512, 256, 128)])
def test_gemm_f32(M, N, K):
    """
    测试 CUDA GEMM F32 实现的正确性。

    Args:
        M (int): 矩阵A的行数和输出矩阵C的行数.
        N (int): 矩阵B的列数和输出矩阵C的列数.
        K (int): 矩阵A的列数和矩阵B的行数.
    """
    # 1. 创建随机输入张量
    #    矩阵A: M x K
    #    矩阵B: K x N  
    #    矩阵C: M x N (输出)
    a = torch.randn((M, K), device="cuda", dtype=torch.float32)
    b = torch.randn((K, N), device="cuda", dtype=torch.float32)
    c = torch.zeros((M, N), device="cuda", dtype=torch.float32)

    # 2. 使用 torch.matmul 计算预期结果
    expected_output = torch.matmul(a, b)

    # 3. 使用 CUDA GEMM 内核计算实际结果
    gemm_f32(a, b, c)

    # 4. 比较结果
    #    使用 torch.allclose 来检查两个张量是否在数值上足够接近
    assert torch.allclose(c, expected_output, atol=1e-4, rtol=1e-4), \
        f"CUDA GEMM output does not match torch.matmul for shape A({M}, {K}) x B({K}, {N})"

    print(f"\nTest passed for GEMM shape A({M}, {K}) x B({K}, {N}) = C({M}, {N})")
    print(f"CUDA output (first 3): {c.flatten().tolist()[:3]}")
    print(f"PyTorch output (first 3): {expected_output.flatten().tolist()[:3]}")
    print(f"Max absolute difference: {torch.max(torch.abs(c - expected_output)).item():.8f}")


@pytest.mark.parametrize("M, N, K", [(32, 32, 32), (64, 128, 256)])  
def test_gemm_f32_small_matrices(M, N, K):
    """
    测试小矩阵的 CUDA GEMM F32 实现。

    Args:
        M (int): 矩阵A的行数和输出矩阵C的行数.
        N (int): 矩阵B的列数和输出矩阵C的列数.
        K (int): 矩阵A的列数和矩阵B的行数.
    """
    # 1. 创建测试数据
    a = torch.randn((M, K), device="cuda", dtype=torch.float32)
    b = torch.randn((K, N), device="cuda", dtype=torch.float32)
    c = torch.zeros((M, N), device="cuda", dtype=torch.float32)

    # 2. 计算预期结果
    expected_output = torch.matmul(a, b)

    # 3. 使用 CUDA GEMM 内核
    gemm_f32(a, b, c)

    # 4. 验证结果
    assert torch.allclose(c, expected_output, atol=1e-4, rtol=1e-4), \
        f"Small matrix GEMM failed for shape A({M}, {K}) x B({K}, {N})"

    print(f"Small matrix test passed for A({M}, {K}) x B({K}, {N})")
