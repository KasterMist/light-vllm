#include <torch/extension.h>
#include <torch/types.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "reduce.cu"
#include "utils.h"


/*
 * RMS Norm CUDA Kernel实现
 * 
 * RMS Norm公式: y = x / sqrt(mean(x^2) + eps) * weight
 * 
 * 算法流程:
 * 1. 每个block处理一行数据 (一个token的所有hidden_size维度)
 * 2. 每个线程处理一个元素
 * 3. 使用block级别的reduce计算均值
 * 4. 应用归一化和权重缩放
 */

// rms norm kernel NUM_THREADS is the number of threads in one block
template <const int NUM_THREADS = 256>
__global__ void rms_norm_kernel(float* x, float* weight, float* output, int hidden_size, float eps){
    const int tid = threadIdx.x;
    const int row_idx = blockIdx.x;
    const int idx = row_idx * hidden_size + tid;
    
    // 1. 加载数据，超出范围的线程加载0
    float val = (tid < hidden_size) ? x[idx] : 0.0f;
    
    // 2. 计算x^2并使用block reduce求和
    float x_squared = val * val;
    float sum_x_squared = block_reduce_sum_f32<NUM_THREADS>(x_squared);
    
    // 3. 计算mean(x^2) + eps，然后计算rsqrt
    float mean_x_squared = sum_x_squared / hidden_size;
    float rsqrt_var = rsqrtf(mean_x_squared + eps);
    
    // 4. 使用共享内存在线程间广播rsqrt结果
    __shared__ float shared_rsqrt;
    if (tid == 0) {
        shared_rsqrt = rsqrt_var;
    }
    __syncthreads();
    
    // 5. 应用归一化和权重缩放
    if (tid < hidden_size) {
        float weight_val = weight[tid];
        output[idx] = val * shared_rsqrt * weight_val;
    }
}

/*
 * Add & RMS Norm CUDA Kernel实现
 * 
 * 对应Python中的add_rms_forward函数
 * 先执行残差连接 (x = x + residual)，然后执行RMS Norm
 * 
 * 返回两个结果:
 * 1. 归一化后的输出
 * 2. 新的残差 (x + residual的结果)
 */
template <const int NUM_THREADS = 256>
__global__ void add_rms_norm_kernel(float* x, float* residual, float* weight, 
                                   float* output, float* new_residual, 
                                   int hidden_size, float eps){
    const int tid = threadIdx.x;
    const int row_idx = blockIdx.x;
    const int idx = row_idx * hidden_size + tid;
    
    // 1. 加载数据并执行残差连接
    float x_val = 0.0f, residual_val = 0.0f;
    if (tid < hidden_size) {
        x_val = x[idx];
        residual_val = residual[idx];
    }
    
    // 执行残差连接: x = x + residual
    float added_val = x_val + residual_val;
    
    // 2. 保存新的残差 (用于下一层)
    if (tid < hidden_size) {
        new_residual[idx] = added_val;
    }
    
    // 3. 计算x^2并使用block reduce求和
    float x_squared = added_val * added_val;
    float sum_x_squared = block_reduce_sum_f32<NUM_THREADS>(x_squared);
    
    // 4. 计算mean(x^2) + eps，然后计算rsqrt
    float mean_x_squared = sum_x_squared / hidden_size;
    float rsqrt_var = rsqrtf(mean_x_squared + eps);
    
    // 5. 使用共享内存在线程间广播rsqrt结果
    __shared__ float shared_rsqrt;
    if (tid == 0) {
        shared_rsqrt = rsqrt_var;
    }
    __syncthreads();
    
    // 6. 应用归一化和权重缩放
    if (tid < hidden_size) {
        float weight_val = weight[tid];
        output[idx] = added_val * shared_rsqrt * weight_val;
    }
}

// kernel launchers

torch::Tensor rms_norm(torch::Tensor x, torch::Tensor weight, float eps = 1e-6) {
    TORCH_CHECK(x.is_cuda(), "输入张量必须在 CUDA 设备上");
    TORCH_CHECK(weight.is_cuda(), "权重张量必须在 CUDA 设备上");
    TORCH_CHECK(x.dim() >= 2, "输入张量至少需要二维");
    
    const int batch_size = x.size(0);
    const int hidden_size = x.size(1);
    
    torch::Tensor output = torch::empty_like(x);
    
    // 配置CUDA内核启动参数
    const int threads_per_block = std::min(hidden_size, 1024);
    const int blocks_per_grid = batch_size;
    
    // 启动kernel
    if (threads_per_block == 32) {
        rms_norm_kernel<32><<<blocks_per_grid, threads_per_block>>>(
            x.data_ptr<float>(), 
            weight.data_ptr<float>(), 
            output.data_ptr<float>(), 
            hidden_size, 
            eps
        );
    } else if (threads_per_block == 64) {
        rms_norm_kernel<64><<<blocks_per_grid, threads_per_block>>>(
            x.data_ptr<float>(), 
            weight.data_ptr<float>(), 
            output.data_ptr<float>(), 
            hidden_size, 
            eps
        );
    } else if (threads_per_block == 128) {
        rms_norm_kernel<128><<<blocks_per_grid, threads_per_block>>>(
            x.data_ptr<float>(), 
            weight.data_ptr<float>(), 
            output.data_ptr<float>(), 
            hidden_size, 
            eps
        );
    } else if (threads_per_block == 256) {
        rms_norm_kernel<256><<<blocks_per_grid, threads_per_block>>>(
            x.data_ptr<float>(), 
            weight.data_ptr<float>(), 
            output.data_ptr<float>(), 
            hidden_size, 
            eps
        );
    } else if (threads_per_block == 512) {
        rms_norm_kernel<512><<<blocks_per_grid, threads_per_block>>>(
            x.data_ptr<float>(), 
            weight.data_ptr<float>(), 
            output.data_ptr<float>(), 
            hidden_size, 
            eps
        );
    } else if(threads_per_block == 1024){
        rms_norm_kernel<1024><<<blocks_per_grid, threads_per_block>>>(
            x.data_ptr<float>(), 
            weight.data_ptr<float>(), 
            output.data_ptr<float>(), 
            hidden_size, 
            eps
        );
    } else {
        throw std::runtime_error("Unsupported head size for softmax kernel. Supported sizes: 32, 64, 128, 256, 512, 1024."); 
    }
    
    return output;
}

std::tuple<torch::Tensor, torch::Tensor> add_rms_norm(torch::Tensor x, torch::Tensor residual, 
                                                      torch::Tensor weight, float eps = 1e-6) {
    TORCH_CHECK(x.is_cuda(), "输入张量x必须在 CUDA 设备上");
    TORCH_CHECK(residual.is_cuda(), "残差张量必须在 CUDA 设备上");
    TORCH_CHECK(weight.is_cuda(), "权重张量必须在 CUDA 设备上");
    TORCH_CHECK(x.dim() >= 2, "输入张量至少需要二维");
    TORCH_CHECK(x.sizes() == residual.sizes(), "x和residual的尺寸必须相同");
    
    const int batch_size = x.size(0);
    const int hidden_size = x.size(1);
    
    torch::Tensor output = torch::empty_like(x);
    torch::Tensor new_residual = torch::empty_like(x);
    
    // 配置CUDA内核启动参数
    const int threads_per_block = std::min(hidden_size, 1024);
    const int blocks_per_grid = batch_size;
    
    // 启动kernel
    if (threads_per_block == 32) {
        add_rms_norm_kernel<32><<<blocks_per_grid, threads_per_block>>>(
            x.data_ptr<float>(), 
            residual.data_ptr<float>(), 
            weight.data_ptr<float>(), 
            output.data_ptr<float>(), 
            new_residual.data_ptr<float>(), 
            hidden_size, 
            eps
        );
    } else if (threads_per_block == 64) {
        add_rms_norm_kernel<64><<<blocks_per_grid, threads_per_block>>>(
            x.data_ptr<float>(), 
            residual.data_ptr<float>(), 
            weight.data_ptr<float>(), 
            output.data_ptr<float>(), 
            new_residual.data_ptr<float>(), 
            hidden_size, 
            eps
        );
    } else if (threads_per_block == 128) {
        add_rms_norm_kernel<128><<<blocks_per_grid, threads_per_block>>>(
            x.data_ptr<float>(), 
            residual.data_ptr<float>(), 
            weight.data_ptr<float>(), 
            output.data_ptr<float>(), 
            new_residual.data_ptr<float>(), 
            hidden_size, 
            eps
        );
    } else if (threads_per_block == 256) {
        add_rms_norm_kernel<256><<<blocks_per_grid, threads_per_block>>>(
            x.data_ptr<float>(), 
            residual.data_ptr<float>(), 
            weight.data_ptr<float>(), 
            output.data_ptr<float>(), 
            new_residual.data_ptr<float>(), 
            hidden_size, 
            eps
        );
    } else if (threads_per_block == 512) {
        add_rms_norm_kernel<512><<<blocks_per_grid, threads_per_block>>>(
            x.data_ptr<float>(), 
            residual.data_ptr<float>(), 
            weight.data_ptr<float>(), 
            output.data_ptr<float>(), 
            new_residual.data_ptr<float>(), 
            hidden_size, 
            eps
        );
    } else if (threads_per_block == 1024){
        add_rms_norm_kernel<1024><<<blocks_per_grid, threads_per_block>>>(
            x.data_ptr<float>(), 
            residual.data_ptr<float>(), 
            weight.data_ptr<float>(), 
            output.data_ptr<float>(), 
            new_residual.data_ptr<float>(), 
            hidden_size, 
            eps
        );
    } else {
        throw std::runtime_error("Unsupported head size for softmax kernel. Supported sizes: 32, 64, 128, 256, 512, 1024."); 
    }
    
    return std::make_tuple(output, new_residual);
}

// 使用 Pybind11 将 C++ 函数绑定到 Python 模块
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rms_norm", &rms_norm, "RMS Normalization CUDA Kernel", 
          py::arg("x"), py::arg("weight"), py::arg("eps") = 1e-6);
    m.def("add_rms_norm", &add_rms_norm, "Add & RMS Normalization CUDA Kernel",
          py::arg("x"), py::arg("residual"), py::arg("weight"), py::arg("eps") = 1e-6);
}