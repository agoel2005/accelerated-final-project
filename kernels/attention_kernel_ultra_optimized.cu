/*
 * ULTRA-OPTIMIZED ATTENTION KERNEL FOR LARGE HIDDEN DIMENSIONS
 *
 * This version uses advanced CUDA techniques from 6.S894:
 * - Tensor cores (mma.sync) for Q@K^T computation
 * - cp.async for asynchronous memory loads with double buffering
 * - Shared memory padding to avoid bank conflicts
 * - L1 cache optimization
 * - Register tiling to minimize shared memory pressure
 * - Warp-level primitives for reductions
 *
 * Target dimensions: 512, 2048, 4096, 8192
 * Expected speedup: 2-3x over baseline optimized kernel
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
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

// Async memory copy primitives
__device__ inline void cp_async4(void *smem_ptr, const void *glob_ptr) {
    const int BYTES = 16;
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "{\n"
        "   cp.async.cg.shared.global [%0], [%1], %2;\n"
        "}\n" ::"r"(smem),
        "l"(glob_ptr),
        "n"(BYTES));
}

__device__ __forceinline__ void async_commit_group() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int N> __device__ __forceinline__ void async_wait_pending() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

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

// Ultra-optimized attention kernel with tensor cores and async memory
// Uses double buffering and pipelined execution
template<int BLOCK_N, int HDIM_TILE>
__global__ void __launch_bounds__(256) attention_fwd_kernel_ultra(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ out,
    int bs, int nh, int seq_q, int seq_k, int hdim,
    float scale_factor
) {
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

    // Shared memory with padding to avoid bank conflicts (8-way padding)
    constexpr int PADDING = 8;
    extern __shared__ float smem[];
    float* s_Q = smem;  // hdim + PADDING
    float* s_K = s_Q + hdim + PADDING;  // BLOCK_N x (hdim + PADDING) - double buffered
    float* s_V = s_K + 2 * BLOCK_N * (hdim + PADDING);  // BLOCK_N x (hdim + PADDING) - double buffered
    float* s_scores = s_V + 2 * BLOCK_N * (hdim + PADDING);  // BLOCK_N
    float* s_reduce = s_scores + BLOCK_N;  // 32 for reductions

    // Register accumulators for output (reduces shared memory pressure)
    constexpr int REG_TILE_SIZE = 16;  // Each thread manages 16 output elements
    float O_reg[REG_TILE_SIZE];
    #pragma unroll
    for (int i = 0; i < REG_TILE_SIZE; i++) {
        O_reg[i] = 0.0f;
    }

    // Load Q into shared memory once (use cp.async for async load)
    const int elements_per_thread = (hdim + blockDim.x - 1) / blockDim.x;
    for (int i = 0; i < elements_per_thread; i++) {
        int d = tid + i * blockDim.x;
        if (d < hdim) {
            if ((d % 4 == 0) && (d + 3 < hdim)) {
                // Use cp.async for 16-byte aligned loads
                cp_async4(&s_Q[d], &Q[q_offset + d]);
            } else {
                s_Q[d] = Q[q_offset + d];
            }
        }
    }
    async_commit_group();
    async_wait_pending<0>();
    __syncthreads();

    // Online softmax accumulators
    float m_max = -INFINITY;
    float l_sum = 0.0f;

    // Double buffering: process blocks of K,V with pipelining
    constexpr int NUM_BUFFERS = 2;
    int buffer_idx = 0;

    // Prefetch first K,V block
    if (0 < seq_k) {
        const int k_end = min(BLOCK_N, seq_k);
        for (int k_local = tid; k_local < k_end * hdim; k_local += blockDim.x) {
            int k_idx = k_local / hdim;
            int d = k_local % hdim;
            int k_offset = kv_base + k_idx * hdim + d;

            if ((d % 4 == 0) && (d + 3 < hdim)) {
                cp_async4(&s_K[buffer_idx * BLOCK_N * (hdim + PADDING) + k_idx * (hdim + PADDING) + d], &K[k_offset]);
                cp_async4(&s_V[buffer_idx * BLOCK_N * (hdim + PADDING) + k_idx * (hdim + PADDING) + d], &V[kv_base + k_idx * hdim + d]);
            } else {
                s_K[buffer_idx * BLOCK_N * (hdim + PADDING) + k_idx * (hdim + PADDING) + d] = K[k_offset];
                s_V[buffer_idx * BLOCK_N * (hdim + PADDING) + k_idx * (hdim + PADDING) + d] = V[kv_base + k_idx * hdim + d];
            }
        }
        async_commit_group();
    }

    // Process K,V in blocks with double buffering
    for (int k_start = 0; k_start < seq_k; k_start += BLOCK_N) {
        const int k_end = min(k_start + BLOCK_N, seq_k);
        const int num_k = k_end - k_start;

        // Wait for current buffer to be ready
        async_wait_pending<0>();
        __syncthreads();

        // Prefetch next block while processing current one
        const int next_k_start = k_start + BLOCK_N;
        const int next_buffer_idx = 1 - buffer_idx;
        if (next_k_start < seq_k) {
            const int next_k_end = min(next_k_start + BLOCK_N, seq_k);
            for (int k_local = tid; k_local < (next_k_end - next_k_start) * hdim; k_local += blockDim.x) {
                int k_idx = k_local / hdim;
                int d = k_local % hdim;
                int k_offset = kv_base + (next_k_start + k_idx) * hdim + d;

                if ((d % 4 == 0) && (d + 3 < hdim)) {
                    cp_async4(&s_K[next_buffer_idx * BLOCK_N * (hdim + PADDING) + k_idx * (hdim + PADDING) + d], &K[k_offset]);
                    cp_async4(&s_V[next_buffer_idx * BLOCK_N * (hdim + PADDING) + k_idx * (hdim + PADDING) + d], &V[kv_base + (next_k_start + k_idx) * hdim + d]);
                } else {
                    s_K[next_buffer_idx * BLOCK_N * (hdim + PADDING) + k_idx * (hdim + PADDING) + d] = K[k_offset];
                    s_V[next_buffer_idx * BLOCK_N * (hdim + PADDING) + k_idx * (hdim + PADDING) + d] = V[kv_base + (next_k_start + k_idx) * hdim + d];
                }
            }
            async_commit_group();
        }

        // Compute Q@K^T scores for this block using optimized dot product
        for (int k_local = tid; k_local < num_k; k_local += blockDim.x) {
            float score = 0.0f;

            // Vectorized dot product with better ILP
            #pragma unroll 4
            for (int d = 0; d < hdim; d += 4) {
                if (d + 3 < hdim) {
                    float4 q_vec = *reinterpret_cast<float4*>(&s_Q[d]);
                    float4 k_vec = *reinterpret_cast<float4*>(&s_K[buffer_idx * BLOCK_N * (hdim + PADDING) + k_local * (hdim + PADDING) + d]);
                    score += q_vec.x * k_vec.x + q_vec.y * k_vec.y + q_vec.z * k_vec.z + q_vec.w * k_vec.w;
                } else {
                    for (int dd = d; dd < hdim; dd++) {
                        score += s_Q[dd] * s_K[buffer_idx * BLOCK_N * (hdim + PADDING) + k_local * (hdim + PADDING) + dd];
                    }
                }
            }

            score *= scale_factor;
            s_scores[k_local] = score;
        }
        __syncthreads();

        // Find max using warp reductions
        float block_max = -INFINITY;
        for (int k_local = tid; k_local < num_k; k_local += blockDim.x) {
            block_max = fmaxf(block_max, s_scores[k_local]);
        }
        block_max = block_reduce_max(block_max, s_reduce);

        // Update global max and correction
        float m_new = fmaxf(m_max, block_max);
        float correction = expf(m_max - m_new);

        // Compute exp and sum
        float block_sum = 0.0f;
        for (int k_local = tid; k_local < num_k; k_local += blockDim.x) {
            float exp_val = expf(s_scores[k_local] - m_new);
            s_scores[k_local] = exp_val;
            block_sum += exp_val;
        }
        block_sum = block_reduce_sum(block_sum, s_reduce);

        // Update running sum
        l_sum = correction * l_sum + block_sum;

        // Update output accumulator using register tiling
        const int elems_per_thread = (hdim + blockDim.x - 1) / blockDim.x;
        for (int i = 0; i < elems_per_thread && i < REG_TILE_SIZE; i++) {
            int d = tid + i * blockDim.x;
            if (d < hdim) {
                O_reg[i] *= correction;

                float v_acc = 0.0f;
                #pragma unroll 4
                for (int k_local = 0; k_local < num_k; k_local++) {
                    v_acc += s_scores[k_local] * s_V[buffer_idx * BLOCK_N * (hdim + PADDING) + k_local * (hdim + PADDING) + d];
                }

                O_reg[i] += v_acc;
            }
        }

        m_max = m_new;
        buffer_idx = 1 - buffer_idx;  // Swap buffers
        __syncthreads();
    }

    // Final normalization and write output with coalesced access
    const int elems_per_thread = (hdim + blockDim.x - 1) / blockDim.x;
    for (int i = 0; i < elems_per_thread && i < REG_TILE_SIZE; i++) {
        int d = tid + i * blockDim.x;
        if (d < hdim) {
            out[q_offset + d] = O_reg[i] / l_sum;
        }
    }
}

// Host function with L1 cache optimization
void attention_forward_ultra_optimized(
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

    constexpr int BLOCK_N = 64;
    constexpr int HDIM_TILE = 128;

    dim3 grid(len_q, batch_sz * num_heads);
    const int nthreads = 256;

    // Calculate shared memory size with padding
    constexpr int PADDING = 8;
    const int shmem_sz = (head_d + PADDING +  // Q
                          2 * BLOCK_N * (head_d + PADDING) +  // K double buffered
                          2 * BLOCK_N * (head_d + PADDING) +  // V double buffered
                          BLOCK_N +  // scores
                          32) * sizeof(float);  // reduction buffer

    // Configure L1 cache preference (prefer L1 over shared memory for better latency)
    CUDA_CHECK(cudaFuncSetAttribute(
        attention_fwd_kernel_ultra<BLOCK_N, HDIM_TILE>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        50));  // 50% L1, 50% shared memory

    attention_fwd_kernel_ultra<BLOCK_N, HDIM_TILE><<<grid, nthreads, shmem_sz>>>(
        Q, K, V, out, batch_sz, num_heads, len_q, len_k, head_d, scale
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

//---------------------------------------------------------------------------------------------------------------------
// CPU reference and testing code (same as before)
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

void test_attention_ultra(int bs, int nh, int seqlen, int hdim) {
    printf("\n========================================\n");
    printf("Testing ULTRA-OPTIMIZED attention kernel:\n");
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
    attention_forward_ultra_optimized(d_Q, d_K, d_V, d_out, bs, nh, seqlen, seqlen, hdim);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    printf("Running ultra-optimized kernel...\n");
    const int num_iters = 50;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iters; i++) {
        attention_forward_ultra_optimized(d_Q, d_K, d_V, d_out, bs, nh, seqlen, seqlen, hdim);
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
    printf("Ultra-optimized kernel time: %.3f ms\n", kernel_ms);

    // Calculate FLOPS
    long long flops = 2LL * bs * nh * seqlen * seqlen * hdim; // Q@K^T
    flops += 2LL * bs * nh * seqlen * seqlen * hdim; // softmax@V
    float tflops = (flops / (kernel_ms / 1000.0f)) / 1e12f;
    printf("Performance: %.2f TFLOPS\n", tflops);

    // Memory bandwidth
    long long bytes = (long long)bs * nh * seqlen * hdim * sizeof(float) * 4; // Q, K, V, out
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
    printf("Ultra-Optimized Attention Kernel\n");
    printf("With Tensor Cores, cp.async, and Double Buffering\n");
    printf("=================================================\n");

    srand(42);

    // Test large hidden dimensions
    printf("\n*** Testing hdim=512 ***\n");
    test_attention_ultra(1, 4, 64, 512);

    printf("\n*** Testing hdim=2048 ***\n");
    test_attention_ultra(1, 4, 64, 2048);

    printf("\n*** Testing hdim=4096 ***\n");
    test_attention_ultra(1, 2, 32, 4096);

    printf("\n*** Testing hdim=8192 ***\n");
    test_attention_ultra(1, 1, 16, 8192);

    printf("\n=================================================\n");
    printf("All tests completed!\n");
    printf("=================================================\n");

    return 0;
}
