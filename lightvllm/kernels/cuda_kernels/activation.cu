#include <torch/extension.h>
#include <torch/types.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "utils.h"

// 为不支持 bfloat16 的旧架构（< SM80）提供一个简单的 sigmoid 实现
// 注意：这只是一个示例，实际生产中可能需要更精确的实现
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
__device__ __forceinline__ float bfloat16_to_float(const __nv_bfloat16 b) {
    float f;
    unsigned int* f_ptr = reinterpret_cast<unsigned int*>(&f);
    *f_ptr = static_cast<unsigned int>(b.x) << 16;
    return f;
}
#endif

template <typename T>
__device__ __forceinline__ float silu_op(T val) {
    // 将输入转换为 float 进行计算，保证精度
    float val_float = static_cast<float>(val);
    // 计算 sigmoid(x) = 1 / (1 + exp(-x))
    const float sigmoid_val = 1.0f / (1.0f + expf(-val_float));
    // 返回 x * sigmoid(x)
    return val_float * sigmoid_val;
}

template <typename T>
__global__ void silu_and_mul_kernel(
    const T* __restrict__ gate_ptr, // gate 张量的指针
    const T* __restrict__ up_ptr,   // up 张量的指针
    T* __restrict__ output_ptr,     // 输出张量的指针
    const int num_elements         // 需要处理的元素总数
) {
    // 计算当前线程的全局唯一索引
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 确保线程索引在有效范围内
    if (idx < num_elements) {
        // 1. 从全局内存加载 gate 和 up 的值
        const T gate_val = gate_ptr[idx];
        const T up_val = up_ptr[idx];

        // 2. 在 float32 精度下执行 SiLU(gate) 操作
        const float silu_result = silu_op(gate_val);

        // 3. 与 up 值相乘
        const float final_result_float = silu_result * static_cast<float>(up_val);

        // 4. 将 float32 结果转换回原始类型并存入全局内存
        output_ptr[idx] = static_cast<T>(final_result_float);
    }
}

torch::Tensor silu_and_mul_cuda(torch::Tensor x) {
    // 1. 输入校验
    TORCH_CHECK(x.is_cuda(), "输入张量必须在 CUDA 设备上");
    TORCH_CHECK(x.dim() >= 1, "输入张量至少需要一维");
    TORCH_CHECK(x.size(-1) % 2 == 0, "最后一维的大小必须是偶数");

    // 2. 获取维度信息
    const auto original_shape = x.sizes();
    auto output_shape = original_shape.vec();
    output_shape.back() /= 2; // 输出的最后一维是输入的一半
    
    const int64_t intermediate_size = output_shape.back();
    const int64_t num_tokens = x.numel() / (2 * intermediate_size);
    const int64_t num_elements = num_tokens * intermediate_size;

    // 3. 切分输入张量
    // torch.chunk(x, 2, dim=-1) 的 CUDA 实现
    // gate 是前半部分，up 是后半部分
    auto tensors = x.chunk(2, -1);
    auto gate = tensors[0].contiguous();
    auto up = tensors[1].contiguous();

    // 4. 创建输出张量
    auto output = torch::empty(output_shape, x.options());

    // 5. 配置 CUDA 内核启动参数
    const int threads_per_block = 256;
    const int blocks_per_grid = (num_elements + threads_per_block - 1) / threads_per_block;

    // 6. 根据输入类型分发到对应的模板化内核
    AT_DISPATCH_FLOATING_TYPES_AND_HALF_AND_BF16(x.scalar_type(), "silu_and_mul_cuda", ([&] {
        silu_and_mul_kernel<scalar_t><<<blocks_per_grid, threads_per_block>>>(
            gate.data_ptr<scalar_t>(),
            up.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            num_elements
        );
    }));

    return output;
}


// 使用 Pybind11 将 C++ 函数绑定到 Python 模块
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("silu_and_mul", &silu_and_mul_cuda, "SiLU and Multiply CUDA Kernel");
}
