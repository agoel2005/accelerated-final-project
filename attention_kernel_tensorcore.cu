/*
 * TENSOR CORE OPTIMIZED ATTENTION KERNEL
 *
 * Specialized for large hidden dimensions: 2048, 4096, 8192
 * Uses mma.sync.aligned.m16n8k8 tensor core instructions for Q@K^T
 *
 * Expected speedup: 2-4x over baseline due to tensor core acceleration
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace nvcuda;

#define CUDA_CHECK(x) \
    do { \
        cudaError_t err = (x); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error " << static_cast<int>(err) << " (" \
                      << cudaGetErrorString(err) << ") at " << __FILE__ << ":" \
                      << __LINE__ << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

// Warp-level reduction primitives
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float block_reduce_max(float val, float* shared) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warp_reduce_max(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    if (wid == 0) {
        val = (threadIdx.x < (blockDim.x + 31) / 32) ? shared[lane] : -INFINITY;
        val = warp_reduce_max(val);
    }
    __syncthreads();

    return val;
}

__device__ __forceinline__ float block_reduce_sum(float val, float* shared) {
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    val = warp_reduce_sum(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    if (wid == 0) {
        val = (threadIdx.x < (blockDim.x + 31) / 32) ? shared[lane] : 0.0f;
        val = warp_reduce_sum(val);
    }
    __syncthreads();

    return val;
}

// Tensor core kernel for computing attention scores using mma.sync
// Processes seq_k keys against 1 query using tensor cores
template<int BLOCK_K, int BLOCK_N>
__global__ void __launch_bounds__(256) attention_tensorcore_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ out,
    int bs, int nh, int seq_q, int seq_k, int hdim,
    float scale_factor
) {
    // Each block processes one query position
    const int batch_head_idx = blockIdx.y;
    const int b_idx = batch_head_idx / nh;
    const int h_idx = batch_head_idx % nh;
    const int q_idx = blockIdx.x;

    if (q_idx >= seq_q) return;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    const int qkv_base = (b_idx * nh + h_idx) * seq_q * hdim;
    const int kv_base = (b_idx * nh + h_idx) * seq_k * hdim;
    const int q_offset = qkv_base + q_idx * hdim;

    // Tensor core constants
    constexpr int MMA_M = 16;  // Output rows per mma
    constexpr int MMA_N = 8;   // Output cols per mma
    constexpr int MMA_K = 8;   // K dimension per mma

    // Shared memory layout with padding
    constexpr int PADDING = 8;
    extern __shared__ float smem[];

    // s_Q: [MMA_M x hdim + PADDING] - we'll treat query as MMA_M copies for tensor core
    // s_K: [BLOCK_N x hdim + PADDING] - keys
    // s_scores: [BLOCK_N] - attention scores
    float* s_Q = smem;
    float* s_K = s_Q + MMA_M * (hdim + PADDING);
    float* s_scores = s_K + BLOCK_N * (hdim + PADDING);
    float* s_reduce = s_scores + BLOCK_N;

    // Load Q into shared memory, replicated MMA_M times for tensor core format
    for (int m = 0; m < MMA_M; m++) {
        for (int d = tid; d < hdim; d += blockDim.x) {
            s_Q[m * (hdim + PADDING) + d] = Q[q_offset + d];
        }
    }
    __syncthreads();

    // Online softmax accumulators
    float m_max = -INFINITY;
    float l_sum = 0.0f;

    // Register accumulators for output
    constexpr int REG_TILE = 16;
    float O_reg[REG_TILE];
    #pragma unroll
    for (int r = 0; r < REG_TILE; r++) {
        O_reg[r] = 0.0f;
    }

    // Process keys in blocks
    for (int k_start = 0; k_start < seq_k; k_start += BLOCK_N) {
        const int k_end = min(k_start + BLOCK_N, seq_k);
        const int num_k = k_end - k_start;

        // Load K block into shared memory (transposed for better access)
        for (int i = tid; i < num_k * hdim; i += blockDim.x) {
            int k_local = i / hdim;
            int d = i % hdim;
            int k_global = k_start + k_local;
            s_K[k_local * (hdim + PADDING) + d] = K[kv_base + k_global * hdim + d];
        }
        __syncthreads();

        // Compute Q @ K^T using tensor cores
        // Each warp computes a portion of the scores
        const int warps_per_block = blockDim.x / 32;
        const int k_per_warp = (num_k + warps_per_block - 1) / warps_per_block;
        const int k_warp_start = warp_id * k_per_warp;
        const int k_warp_end = min(k_warp_start + k_per_warp, num_k);

        for (int k_local = k_warp_start; k_local < k_warp_end; k_local++) {
            float score = 0.0f;

            // Use tensor cores for dot product
            // Process hdim in chunks of MMA_K=8
            for (int k_tile = 0; k_tile < hdim; k_tile += MMA_K) {
                // Load fragments for mma.sync
                uint32_t frag_q[4];  // 16x8 matrix for Q
                uint32_t frag_k[2];  // 8x8 matrix for K^T
                uint32_t frag_c[4] = {0, 0, 0, 0};  // 16x8 accumulator

                // Load Q fragment (treat as 16x8 tile from row 0-15, cols k_tile:k_tile+8)
                int q_row = lane_id / 4;     // 0-7
                int q_col = (lane_id % 4);   // 0-3

                if (k_tile + q_col < hdim) {
                    frag_q[0] = __float_as_uint(s_Q[q_row * (hdim + PADDING) + k_tile + q_col]);
                    frag_q[1] = __float_as_uint(s_Q[(q_row + 8) * (hdim + PADDING) + k_tile + q_col]);
                    frag_q[2] = __float_as_uint(s_Q[q_row * (hdim + PADDING) + k_tile + q_col + 4]);
                    frag_q[3] = __float_as_uint(s_Q[(q_row + 8) * (hdim + PADDING) + k_tile + q_col + 4]);
                }

                // Load K fragment (8x8 tile)
                int k_row = lane_id % 4;     // 0-3
                int k_col = lane_id / 4;     // 0-7

                if (k_tile + k_row < hdim) {
                    frag_k[0] = __float_as_uint(s_K[k_local * (hdim + PADDING) + k_tile + k_row]);
                    frag_k[1] = __float_as_uint(s_K[k_local * (hdim + PADDING) + k_tile + k_row + 4]);
                }

                // Perform mma.sync: C = A * B + C
                asm volatile(
                    "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
                    "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
                    : "=r"(frag_c[0]), "=r"(frag_c[1]), "=r"(frag_c[2]), "=r"(frag_c[3])
                    : "r"(frag_q[0]), "r"(frag_q[1]), "r"(frag_q[2]), "r"(frag_q[3]),
                      "r"(frag_k[0]), "r"(frag_k[1]),
                      "r"(frag_c[0]), "r"(frag_c[1]), "r"(frag_c[2]), "r"(frag_c[3])
                );

                // Accumulate from first row of output (we only care about 1 query)
                if (q_row == 0) {
                    score += __uint_as_float(frag_c[0]);
                }
            }

            // Warp-level reduction to get final score
            #pragma unroll
            for (int offset = 4; offset > 0; offset >>= 1) {
                score += __shfl_xor_sync(0xffffffff, score, offset);
            }

            // Lane 0 of each warp writes the score
            if ((lane_id % 8) == 0) {
                score *= scale_factor;
                s_scores[k_local] = score;
            }
        }
        __syncthreads();

        // Find max score
        float block_max = -INFINITY;
        for (int k_local = tid; k_local < num_k; k_local += blockDim.x) {
            block_max = fmaxf(block_max, s_scores[k_local]);
        }
        block_max = block_reduce_max(block_max, s_reduce);

        // Online softmax update
        float m_new = fmaxf(m_max, block_max);
        float correction = expf(m_max - m_new);

        // Compute softmax
        float block_sum = 0.0f;
        for (int k_local = tid; k_local < num_k; k_local += blockDim.x) {
            float exp_val = expf(s_scores[k_local] - m_new);
            s_scores[k_local] = exp_val;
            block_sum += exp_val;
        }
        block_sum = block_reduce_sum(block_sum, s_reduce);

        l_sum = correction * l_sum + block_sum;

        // Update output with register tiling
        #pragma unroll
        for (int r = 0; r < REG_TILE; r++) {
            int d = tid + r * blockDim.x;
            if (d < hdim) {
                O_reg[r] *= correction;

                float v_acc = 0.0f;
                #pragma unroll 4
                for (int k_local = 0; k_local < num_k; k_local++) {
                    int k_global = k_start + k_local;
                    v_acc += s_scores[k_local] * V[kv_base + k_global * hdim + d];
                }

                O_reg[r] += v_acc;
            }
        }

        m_max = m_new;
        __syncthreads();
    }

    // Write output
    #pragma unroll
    for (int r = 0; r < REG_TILE; r++) {
        int d = tid + r * blockDim.x;
        if (d < hdim) {
            out[q_offset + d] = O_reg[r] / l_sum;
        }
    }
}

// Host function
void attention_forward_tensorcore(
    const float* Q,
    const float* K,
    const float* V,
    float* out,
    int batch_sz,
    int num_heads,
    int len_q,
    int len_k,
    int head_d
) {
    float scale = 1.0f / sqrtf(static_cast<float>(head_d));

    // Tensor core config
    constexpr int BLOCK_K = 32;  // Not used but kept for template consistency
    constexpr int BLOCK_N = 64;  // Process 64 keys at a time
    constexpr int MMA_M = 16;
    constexpr int PADDING = 8;

    dim3 grid(len_q, batch_sz * num_heads);
    const int nthreads = 256;

    // Shared memory: Q (16 x hdim), K (64 x hdim), scores (64), reduce (32)
    const int shmem_sz = (MMA_M * (head_d + PADDING) +
                          BLOCK_N * (head_d + PADDING) +
                          BLOCK_N +
                          32) * sizeof(float);

    attention_tensorcore_kernel<BLOCK_K, BLOCK_N><<<grid, nthreads, shmem_sz>>>(
        Q, K, V, out, batch_sz, num_heads, len_q, len_k, head_d, scale
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

//---------------------------------------------------------------------------------------------------------------------
// CPU reference and testing
//---------------------------------------------------------------------------------------------------------------------

void attention_cpu(
    const float* Q,
    const float* K,
    const float* V,
    float* out,
    int batch_sz,
    int num_heads,
    int len_q,
    int len_k,
    int head_d
) {
    float scale = 1.0f / sqrtf(static_cast<float>(head_d));

    for (int b = 0; b < batch_sz; b++) {
        for (int h = 0; h < num_heads; h++) {
            for (int i = 0; i < len_q; i++) {
                int qoff = ((b * num_heads + h) * len_q + i) * head_d;

                std::vector<float> attn_scores(len_k);
                float max_s = -INFINITY;

                for (int j = 0; j < len_k; j++) {
                    int koff = ((b * num_heads + h) * len_k + j) * head_d;
                    float s = 0.0f;

                    for (int d = 0; d < head_d; d++) {
                        s += Q[qoff + d] * K[koff + d];
                    }

                    s *= scale;
                    attn_scores[j] = s;
                    max_s = fmaxf(max_s, s);
                }

                float exp_total = 0.0f;
                for (int j = 0; j < len_k; j++) {
                    attn_scores[j] = expf(attn_scores[j] - max_s);
                    exp_total += attn_scores[j];
                }

                for (int j = 0; j < len_k; j++) {
                    attn_scores[j] /= exp_total;
                }

                for (int d = 0; d < head_d; d++) {
                    float result = 0.0f;
                    for (int j = 0; j < len_k; j++) {
                        int voff = ((b * num_heads + h) * len_k + j) * head_d;
                        result += attn_scores[j] * V[voff + d];
                    }
                    out[qoff + d] = result;
                }
            }
        }
    }
}

void init_random(float* data, int size, float min_val = -1.0f, float max_val = 1.0f) {
    for (int i = 0; i < size; i++) {
        data[i] = min_val + (max_val - min_val) * (rand() / (float)RAND_MAX);
    }
}

bool check_results(const float* gpu_out, const float* cpu_out, int size, float tol = 5e-2f) {
    float max_diff = 0.0f;
    int nerrors = 0;
    const int max_print = 10;

    for (int i = 0; i < size; i++) {
        float diff = fabsf(gpu_out[i] - cpu_out[i]);
        max_diff = fmaxf(max_diff, diff);

        if (diff > tol) {
            if (nerrors < max_print) {
                printf("  Mismatch at index %d: GPU=%.6f, CPU=%.6f, diff=%.6f\n",
                       i, gpu_out[i], cpu_out[i], diff);
            }
            nerrors++;
        }
    }

    if (nerrors > 0) {
        printf("  Total mismatches: %d / %d\n", nerrors, size);
        printf("  Max difference: %.6f\n", max_diff);
        return false;
    }

    printf("  Max difference: %.6f (within tolerance %.6f)\n", max_diff, tol);
    return true;
}

void test_attention_tensorcore(int bs, int nh, int seqlen, int hdim) {
    printf("\n========================================\n");
    printf("Testing TENSOR CORE attention kernel:\n");
    printf("  Batch size: %d\n", bs);
    printf("  Num heads: %d\n", nh);
    printf("  Sequence length: %d\n", seqlen);
    printf("  Head dimension: %d\n", hdim);
    printf("========================================\n\n");

    int sz_q = bs * nh * seqlen * hdim;
    int sz_kv = bs * nh * seqlen * hdim;

    float *h_Q = new float[sz_q];
    float *h_K = new float[sz_kv];
    float *h_V = new float[sz_kv];
    float *h_out_gpu = new float[sz_q];
    float *h_out_cpu = new float[sz_q];

    init_random(h_Q, sz_q);
    init_random(h_K, sz_kv);
    init_random(h_V, sz_kv);

    float *d_Q, *d_K, *d_V, *d_out;
    CUDA_CHECK(cudaMalloc(&d_Q, sz_q * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, sz_kv * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V, sz_kv * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, sz_q * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_Q, h_Q, sz_q * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K, sz_kv * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V, sz_kv * sizeof(float), cudaMemcpyHostToDevice));

    // Warmup
    attention_forward_tensorcore(d_Q, d_K, d_V, d_out, bs, nh, seqlen, seqlen, hdim);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    printf("Running tensor core kernel...\n");
    const int num_iters = 100;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iters; i++) {
        attention_forward_tensorcore(d_Q, d_K, d_V, d_out, bs, nh, seqlen, seqlen, hdim);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float kernel_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));
    kernel_ms /= num_iters;

    CUDA_CHECK(cudaMemcpy(h_out_gpu, d_out, sz_q * sizeof(float), cudaMemcpyDeviceToHost));

    printf("Running CPU reference...\n");
    attention_cpu(h_Q, h_K, h_V, h_out_cpu, bs, nh, seqlen, seqlen, hdim);

    printf("\n--- Performance Results ---\n");
    printf("Tensor core kernel time: %.3f ms\n", kernel_ms);

    // Calculate FLOPS
    long long flops = 2LL * bs * nh * seqlen * seqlen * hdim; // Q@K^T
    flops += 2LL * bs * nh * seqlen * seqlen * hdim; // softmax@V
    float tflops = (flops / (kernel_ms / 1000.0f)) / 1e12f;
    printf("Performance: %.2f TFLOPS\n", tflops);

    // Memory bandwidth
    long long bytes = (long long)bs * nh * seqlen * hdim * sizeof(float) * 4;
    float bandwidth_gb_s = (bytes / (kernel_ms / 1000.0f)) / 1e9f;
    printf("Memory bandwidth: %.2f GB/s\n", bandwidth_gb_s);

    printf("\nChecking results vs CPU...\n");
    bool passed = check_results(h_out_gpu, h_out_cpu, sz_q);

    if (passed) {
        printf("\n✓ TEST PASSED\n");
    } else {
        printf("\n✗ TEST FAILED\n");
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    delete[] h_Q;
    delete[] h_K;
    delete[] h_V;
    delete[] h_out_gpu;
    delete[] h_out_cpu;

    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_out));
}

int main() {
    printf("=================================================\n");
    printf("Tensor Core Accelerated Attention Kernel\n");
    printf("Target: hdim = 2048, 4096, 8192\n");
    printf("=================================================\n");

    srand(42);

    // Test target dimensions
    printf("\n*** Testing hdim=2048 ***\n");
    test_attention_tensorcore(1, 4, 64, 2048);

    printf("\n*** Testing hdim=4096 ***\n");
    test_attention_tensorcore(1, 2, 32, 4096);

    printf("\n*** Testing hdim=8192 ***\n");
    test_attention_tensorcore(1, 1, 16, 8192);

    printf("\n=================================================\n");
    printf("All tests completed!\n");
    printf("=================================================\n");

    return 0;
}
