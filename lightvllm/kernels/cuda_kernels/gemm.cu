#include <torch/extension.h>
#include <torch/types.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "reduce.cu"
#include "utils.h"

// SGEMM naive: compute one c[i,j] element per threads, all row major
__global__ void sgemm_naive_f32_kernel(float* a, float* b, float* c, int M, int N, int K){
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;

    if(m < M && n < N){
        float psum = 0.0;
        #pragma unroll
        for(int k = 0; k < K; k++){
            psum += a[m * K + k] * b[k * N + n];
        }
        c[m * N + n] = psum; // c[m, n]
    }
}

// SGEMM: Block Tile + K Tile, with smem
// Block Tile (BM, BN) + K Tile (BK=32)
// grid((N + BN - 1) / BN, (M + BM - 1) / BM), block(BN, BM)
// a: MxK, b: KxN, c: MxN, compute: c = a * b, all row major  
template<const int BM = 32, const int BN = 32, const int BK = 32>
__global__ void sgemm_sliced_k_f32_kernel(float* a, float* b, float* c, int M, int N, int K){
    // Block tile: 32x32的block处理c的一块32x32的元素计算
    // K tile: 使用共享内存，并将K分块为BK大小的块
    __shared__ float s_a[BM][BK], s_b[BK][BN]; 
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = threadIdx.y * blockDim.x + threadIdx.x; // tid within the block

    // load values to shared memory, 32x32 threads working together 
    // to fetch data along the row direction of a and b both for s_a 
    // and s_b 32x32x4x2=8KB, we use 32x32 threads within block to 
    // load 32x32 elements from global memory to shared memory, namely, 
    // each thread will load 1 element.
    int load_smem_a_m = tid / 32; // load s_a idx in row
    int load_smem_a_k = tid % 32; // load s_a idx in col
    int load_smem_b_k = tid / 32; // load s_b idx in row
    int load_smem_b_n = tid % 32; // load s_b idx in col
    // the global idx of a/c's row and global idx of b/c's col is same. So need to determine below
    int load_gmem_a_m = by * BM + load_smem_a_m; // global idx in row of a and c
    int load_gmem_b_n = bx * BN + load_smem_b_n; // global idx in col of b and c

    float sum = 0.0f;
    // outer loop for each thread block to calculate BK in K
    for(int bk = 0; bk < (K + BK - 1) / BK; bk++){
        int load_gmem_a_k = bk * BK + load_smem_a_k; // global idx in col of a and c
        int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k; // idx of a
        s_a[load_smem_a_m][load_smem_a_k] = a[load_gmem_a_addr];
        int load_gmem_b_k = bk * BK + load_smem_b_k; // global idx in row of b and c
        int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n; // idx of b
        s_b[load_smem_b_k][load_smem_b_n] = b[load_gmem_b_addr];
        __syncthreads();

        // inner loop for each block to calculate k in BK
        #pragma unroll
        for(int k = 0; k < BK; k++){
            int comp_smem_a_m = load_smem_a_m;
            int comp_smem_b_n = load_smem_b_n;
            sum += s_a[comp_smem_a_m][k] * s_b[k][comp_smem_b_n];
        }
        __syncthreads();
    }

    // store data in c
    int store_gmem_c_m = load_gmem_a_m;
    int store_gmem_c_n = load_gmem_b_n;
    int store_gmem_c_addr = load_gmem_a_m * N + store_gmem_c_n;
    c[store_gmem_c_addr] = sum;
    
}


void sgemm_sliced_k_f32(torch::Tensor a, torch::Tensor b, torch::Tensor c){
    TORCH_CHECK(a.is_cuda(), "输入张量必须在 CUDA 设备上");
    const int M = a.size(0);
    const int K = a.size(1);
    const int N = b.size(1);

    constexpr int BM = 32;
    constexpr int BN = 32;
    constexpr int BK = 32;

    dim3 block(BN, BM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    sgemm_sliced_k_f32_kernel<BM, BN, BK><<<grid, block>>>(
        reinterpret_cast<float*>(a.data_ptr()),
        reinterpret_cast<float*>(b.data_ptr()),
        reinterpret_cast<float*>(c.data_ptr()),
        M, N, K
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sgemm_sliced_k_f32", &sgemm_sliced_k_f32, "sgemm_sliced_k_f32 CUDA Kernel");
}
