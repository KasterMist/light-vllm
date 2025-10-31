#include <torch/extension.h>
#include <torch/types.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "reduce.cu"
#include "utils.h"

// DS required for Online Softmax
struct __align__(8) MD {float m; float d;};

/*
 * Online Softmax Warp Reduce算法 - 在Warp内进行高效的Softmax计算
 * 
 * 算法原理：
 * 1. Online Softmax通过维护两个状态量来避免两遍扫描：
 *    - m: 当前已处理元素的最大值 (max value)
 *    - d: 当前已处理元素的指数和 (exponential sum)
 * 
 * 2. 当合并两个状态(value, other)时，使用以下数学公式：
 *    - 新的最大值: max_new = max(value.m, other.m)
 *    - 新的指数和: d_new = d_bigger + d_smaller * exp(m_smaller - m_bigger)
 *    其中bigger/smaller指的是具有更大/更小最大值的状态
 * 
 * 3. Warp Reduce过程：
 *    - 使用__shfl_xor_sync在warp内的线程间交换数据
 *    - 每次迭代将参与计算的线程数减半(stride >>= 1)
 *    - 最终warp内的第0号线程将得到整个warp的softmax结果
 * 
 * 4. 数值稳定性：通过减去最大值避免指数运算溢出
 */
template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ MD warp_reduce_md_op(MD value){
    unsigned int mask = 0xffffffff;
    #pragma unroll
    for(int stride = kWarpSize >> 1; stride > 0; stride >>= 1){
        MD other;
        other.m = __shfl_xor_sync(mask, value.m, stride);
        other.d = __shfl_xor_sync(mask, value.d, stride);

        bool value_bigger = (value.m > other.m);
        MD bigger_m = value_bigger ? value : other;
        MD smaller_m = value_bigger ? other : value;
        
        value.d = bigger_m.d + smaller_m.d * __expf(smaller_m.m - bigger_m.m);
        value.m = bigger_m.m;
    }
    return value;
}


// Softmax Kernel per token --> each thread block handles one token
// NUM_THREADS is the number of threads per block
template<const int NUM_THREADS = 256>
__global__ void softmax_per_token_kernel(float* x, float* y, int N){
    const int tid = threadIdx.x;
    const int idx = tid + blockIdx.x * blockDim.x;

    float val = (idx < N) ? x[idx] : -FLT_MAX;
    float max_val = block_reduce_max_f32<NUM_THREADS>(val);
    float exp_val = (idx < N) ? expf(val - max_val) : 0.0f;
    float exp_sum = block_reduce_sum_f32<NUM_THREADS>(exp_val);
    
    if(idx < N){
        y[idx] = exp_val / exp_sum;
    }
}


// reference: https://arxiv.org/pdf/1805.02867 (Online normalizer calculation for softmax)
template<const int NUM_THREADS = 256 >
__global__ void online_softmax_per_token_kernel(float* x, float* y, int N){
    int local_tid = threadIdx.x;
    int global_tid = blockIdx.x * NUM_THREADS + threadIdx.x; // 每个block处理一个token
    const int WARP_NUM = NUM_THREADS / WARP_SIZE;

    int warp_id = local_tid / WARP_SIZE;
    int lane_id = local_tid % WARP_SIZE;

    // 初始化数据 - 每个线程处理一个元素
    MD val;
    val.m = global_tid < N ? x[global_tid] : -FLT_MAX;
    val.d = global_tid < N ? 1.0f : 0.0f;

    // 每个warp计算online softmax
    __shared__ MD shared[WARP_NUM];
    MD res = warp_reduce_md_op<WARP_SIZE>(val);

    if(lane_id == 0){
        shared[warp_id] = res;
    }
    __syncthreads();

    // 只让warp1的线程处理上一步每个warp计算的中间值
    if(local_tid < WARP_SIZE){
        MD block_res = shared[local_tid];
        block_res = warp_reduce_md_op<WARP_NUM>(block_res); 
        if (local_tid == 0) {
            shared[0] = block_res; 
        }
    }
    __syncthreads();

    MD final_res = shared[0];
    float d_total_inverse = __fdividef(1.0f, final_res.d); // 计算1/exp(sum)

    if(global_tid < N){
        y[global_tid] = __expf(x[global_tid] - final_res.m) * d_total_inverse;
    }

}

// kernel launcher

torch::Tensor softmax_per_token(torch::Tensor x){
    TORCH_CHECK(x.is_cuda(), "输入张量必须在 CUDA 设备上");
    TORCH_CHECK(x.dim() >= 1, "输入张量至少需要一维");

    const int S = x.size(0); // seqlens
    const int H = x.size(1); // head size / kv_len
    const int N = S * H;

    torch::Tensor y = torch::empty({S, H}, x.options());

    // 5. 配置 CUDA 内核启动参数
    const int threads_per_block = H;
    const int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    // 6. 启动 CUDA 内核
    if(H == 32){
        softmax_per_token_kernel<32><<<blocks_per_grid, threads_per_block>>>(x.data_ptr<float>(), y.data_ptr<float>(), N);
    }
    else if(H == 64){
        softmax_per_token_kernel<64><<<blocks_per_grid, threads_per_block>>>(x.data_ptr<float>(), y.data_ptr<float>(), N);
    }
    else if(H == 128){
        softmax_per_token_kernel<128><<<blocks_per_grid, threads_per_block>>>(x.data_ptr<float>(), y.data_ptr<float>(), N);
    }
    else if(H == 256){
        softmax_per_token_kernel<256><<<blocks_per_grid, threads_per_block>>>(x.data_ptr<float>(), y.data_ptr<float>(), N);
    }
    else if(H == 512){
        softmax_per_token_kernel<512><<<blocks_per_grid, threads_per_block>>>(x.data_ptr<float>(), y.data_ptr<float>(), N);
    }
    else if(H == 1024){
        softmax_per_token_kernel<1024><<<blocks_per_grid, threads_per_block>>>(x.data_ptr<float>(), y.data_ptr<float>(), N);
    }
    else{
        throw std::runtime_error("Unsupported head size for softmax kernel. Supported sizes: 32, 64, 128, 256, 512, 1024.");
    }
    
    return y;
}


torch::Tensor online_softmax_per_token(torch::Tensor x){
    TORCH_CHECK(x.is_cuda(), "输入张量必须在 CUDA 设备上");
    TORCH_CHECK(x.dim() >= 1, "输入张量至少需要一维");

    const int S = x.size(0); // seqlens
    const int H = x.size(1); // head size / kv_len
    const int N = S * H;

    torch::Tensor y = torch::empty({S, H}, x.options());

    // 5. 配置 CUDA 内核启动参数
    const int threads_per_block = H;
    const int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    // 启动 Online Softmax CUDA 内核
    // 6. 启动 CUDA 内核
    if(H == 32){
        online_softmax_per_token_kernel<32><<<blocks_per_grid, threads_per_block>>>(x.data_ptr<float>(), y.data_ptr<float>(), N);
    }
    else if(H == 64){
        online_softmax_per_token_kernel<64><<<blocks_per_grid, threads_per_block>>>(x.data_ptr<float>(), y.data_ptr<float>(), N);
    }
    else if(H == 128){
        online_softmax_per_token_kernel<128><<<blocks_per_grid, threads_per_block>>>(x.data_ptr<float>(), y.data_ptr<float>(), N);
    }
    else if(H == 256){
        online_softmax_per_token_kernel<256><<<blocks_per_grid, threads_per_block>>>(x.data_ptr<float>(), y.data_ptr<float>(), N);
    }
    else if(H == 512){
        online_softmax_per_token_kernel<512><<<blocks_per_grid, threads_per_block>>>(x.data_ptr<float>(), y.data_ptr<float>(), N);
    }
    else if(H == 1024){
        online_softmax_per_token_kernel<1024><<<blocks_per_grid, threads_per_block>>>(x.data_ptr<float>(), y.data_ptr<float>(), N);
    }
    else{
        throw std::runtime_error("Unsupported head size for softmax kernel. Supported sizes: 32, 64, 128, 256, 512, 1024.");
    }
    
    return y;
}




// 使用 Pybind11 将 C++ 函数绑定到 Python 模块
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("softmax_per_token", &softmax_per_token, "Softmax per token CUDA Kernel");
    m.def("online_softmax_per_token", &online_softmax_per_token, "Online Softmax per token CUDA Kernel");
}