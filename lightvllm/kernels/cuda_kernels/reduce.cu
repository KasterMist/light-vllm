#include <torch/extension.h>
#include <torch/types.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "utils.h"

// Warp Reduce Sum
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val){
    # pragma unroll
    for(int mask = kWarpSize >> 1; mask > 0; mask >>= 1){
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

// Warp Reduce Max
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_max_f32(float val){
    # pragma unroll
    for(int mask = kWarpSize >> 1; mask > 0; mask >>= 1){
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

// Block Reduce Sum
// i.e. grid 1D block 1D, grid(N / 256), block(256)
template <const int NUM_THREADS = 256>
__device__ __forceinline__ float block_reduce_sum_f32(float val){
    // always <= 32 warps per bock (limited by 1024 threads per block)
    constexpr int NUM_WARPS = CEIL(NUM_THREADS, WARP_SIZE);
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    static __shared__ float shared[NUM_WARPS]; // shared mem for 32 partial sums
    
    float value = warp_reduce_sum_f32<WARP_SIZE>(val);
    // Let first thread of each warp writes the shared memory
    if(lane_id == 0){
        shared[warp_id] = value;
    }
    __syncthreads();
    value = (lane_id < WARP_SIZE) ? shared[lane_id] : 0.0f;
    value = warp_reduce_sum_f32<NUM_WARPS>(value);

    // Need to broadcast value to all threads within warps
    value = __shfl_sync(0xffffffff, value, 0, 32);
    return value;

}


template <const int NUM_THREADS = 256>
__device__ __forceinline__ float block_reduce_max_f32(float val){
    // always <= 32 warps per bock (limited by 1024 threads per block)
    constexpr int NUM_WARPS = CEIL(NUM_THREADS, WARP_SIZE);
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    static __shared__ float shared[NUM_WARPS]; // shared mem for 32 partial sums
    
    float value = warp_reduce_max_f32<WARP_SIZE>(val);
    // Let first thread of each warp writes the shared memory
    if(lane_id == 0){
        shared[warp_id] = value;
    }
    __syncthreads();
    value = (lane_id < WARP_SIZE) ? shared[lane_id] : -FLT_MAX;
    value = warp_reduce_max_f32<NUM_WARPS>(value);

    // Need to broadcast value to all threads within warps
    value = __shfl_sync(0xffffffff, value, 0, 32);
    return value;
}