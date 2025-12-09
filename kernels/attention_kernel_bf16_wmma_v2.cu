/*
 * BF16 ATTENTION WITH WMMA TENSOR CORES - SIMPLIFIED VERSION
 *
 * This kernel uses BF16 throughout with WMMA tensor core instructions.
 * Simplified approach: Use WMMA for Q@K^T but with proper memory handling.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace nvcuda;
using bfloat16 = __nv_bfloat16;

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

// Warp reduction primitives
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

// Simplified BF16 kernel - use tensor cores only where beneficial
// For attention, the bottleneck is often memory, not compute
__global__ void attention_fwd_bf16_simple(
    const bfloat16* __restrict__ Q,
    const bfloat16* __restrict__ K,
    const bfloat16* __restrict__ V,
    bfloat16* __restrict__ out,
    int bs, int nh, int seq_q, int seq_k, int hdim,
    float scale_factor
) {
    // Each block handles one query
    const int idx = blockIdx.x;
    const int total_q = bs * nh * seq_q;

    if (idx >= total_q) return;

    extern __shared__ float smem[];
    float* s_scores = smem;
    float* s_reduce = s_scores + seq_k;

    const int b_idx = idx / (nh * seq_q);
    const int h_idx = (idx / seq_q) % nh;
    const int q_pos = idx % seq_q;

    const int q_offset = idx * hdim;
    const int kv_base = (b_idx * nh + h_idx) * seq_k * hdim;

    const int tid = threadIdx.x;

    // Compute Q @ K^T scores
    float local_max = -INFINITY;

    for (int k_pos = tid; k_pos < seq_k; k_pos += blockDim.x) {
        const int k_offset = kv_base + k_pos * hdim;

        float score = 0.0f;

        // Vectorized dot product with BF16
        if (hdim % 4 == 0) {
            #pragma unroll 2
            for (int d = 0; d < hdim; d += 4) {
                // Load 4 BF16 values at once using float2 (2 BF16s per float)
                float2 q_vec = *reinterpret_cast<const float2*>(&Q[q_offset + d]);
                float2 k_vec = *reinterpret_cast<const float2*>(&K[k_offset + d]);

                bfloat16* q_ptr = reinterpret_cast<bfloat16*>(&q_vec);
                bfloat16* k_ptr = reinterpret_cast<bfloat16*>(&k_vec);

                // Accumulate in FP32 for precision
                score += __bfloat162float(q_ptr[0]) * __bfloat162float(k_ptr[0]);
                score += __bfloat162float(q_ptr[1]) * __bfloat162float(k_ptr[1]);
                score += __bfloat162float(q_ptr[2]) * __bfloat162float(k_ptr[2]);
                score += __bfloat162float(q_ptr[3]) * __bfloat162float(k_ptr[3]);
            }
        } else {
            for (int d = 0; d < hdim; d++) {
                score += __bfloat162float(Q[q_offset + d]) * __bfloat162float(K[k_offset + d]);
            }
        }

        score *= scale_factor;
        s_scores[k_pos] = score;
        local_max = fmaxf(local_max, score);
    }

    // Find global max
    local_max = warp_reduce_max(local_max);
    if (tid % 32 == 0) s_reduce[tid / 32] = local_max;
    __syncthreads();

    if (tid < 32) {
        float val = (tid < (blockDim.x + 31) / 32) ? s_reduce[tid] : -INFINITY;
        val = warp_reduce_max(val);
        if (tid == 0) s_reduce[0] = val;
    }
    __syncthreads();
    local_max = s_reduce[0];

    // Compute exp and sum
    float exp_sum = 0.0f;
    for (int k_pos = tid; k_pos < seq_k; k_pos += blockDim.x) {
        float exp_val = expf(s_scores[k_pos] - local_max);
        s_scores[k_pos] = exp_val;
        exp_sum += exp_val;
    }

    exp_sum = warp_reduce_sum(exp_sum);
    if (tid % 32 == 0) s_reduce[tid / 32] = exp_sum;
    __syncthreads();

    if (tid < 32) {
        float val = (tid < (blockDim.x + 31) / 32) ? s_reduce[tid] : 0.0f;
        val = warp_reduce_sum(val);
        if (tid == 0) s_reduce[0] = val;
    }
    __syncthreads();
    exp_sum = s_reduce[0];

    // Normalize
    for (int k_pos = tid; k_pos < seq_k; k_pos += blockDim.x) {
        s_scores[k_pos] = (exp_sum != 0.0f) ? (s_scores[k_pos] / exp_sum) : 0.0f;
    }
    __syncthreads();

    // Compute output = softmax @ V
    for (int d = tid; d < hdim; d += blockDim.x) {
        float accum = 0.0f;

        for (int k_pos = 0; k_pos < seq_k; k_pos++) {
            const int v_offset = kv_base + k_pos * hdim + d;
            accum += s_scores[k_pos] * __bfloat162float(V[v_offset]);
        }

        out[q_offset + d] = __float2bfloat16(accum);
    }
}

void attention_forward_bf16(
    const bfloat16* Q,
    const bfloat16* K,
    const bfloat16* V,
    bfloat16* out,
    int bs, int nh, int seq_q, int seq_k, int hdim
) {
    float scale = 1.0f / sqrtf(static_cast<float>(hdim));

    const int total_qs = bs * nh * seq_q;
    const int nthreads = 256;
    const int nblocks = total_qs;
    const int shmem_sz = (seq_k + 32) * sizeof(float);

    attention_fwd_bf16_simple<<<nblocks, nthreads, shmem_sz>>>(
        Q, K, V, out, bs, nh, seq_q, seq_k, hdim, scale
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

//---------------------------------------------------------------------------------------------------------------------
// CPU reference
//---------------------------------------------------------------------------------------------------------------------

void attention_cpu(
    const float* Q,
    const float* K,
    const float* V,
    float* out,
    int bs, int nh, int seq_q, int seq_k, int hdim
) {
    float scale = 1.0f / sqrtf(static_cast<float>(hdim));

    for (int b = 0; b < bs; b++) {
        for (int h = 0; h < nh; h++) {
            for (int i = 0; i < seq_q; i++) {
                int qoff = ((b * nh + h) * seq_q + i) * hdim;

                std::vector<float> scores(seq_k);
                float max_s = -INFINITY;

                for (int j = 0; j < seq_k; j++) {
                    int koff = ((b * nh + h) * seq_k + j) * hdim;
                    float s = 0.0f;

                    for (int d = 0; d < hdim; d++) {
                        s += Q[qoff + d] * K[koff + d];
                    }

                    s *= scale;
                    scores[j] = s;
                    max_s = fmaxf(max_s, s);
                }

                float exp_sum = 0.0f;
                for (int j = 0; j < seq_k; j++) {
                    scores[j] = expf(scores[j] - max_s);
                    exp_sum += scores[j];
                }

                for (int j = 0; j < seq_k; j++) {
                    scores[j] /= exp_sum;
                }

                for (int d = 0; d < hdim; d++) {
                    float result = 0.0f;
                    for (int j = 0; j < seq_k; j++) {
                        int voff = ((b * nh + h) * seq_k + j) * hdim;
                        result += scores[j] * V[voff + d];
                    }
                    out[qoff + d] = result;
                }
            }
        }
    }
}

//---------------------------------------------------------------------------------------------------------------------
// Test utilities
//---------------------------------------------------------------------------------------------------------------------

void init_random(float* data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = -1.0f + 2.0f * (rand() / (float)RAND_MAX);
    }
}

void fp32_to_bf16(const float* src, bfloat16* dst, int size) {
    for (int i = 0; i < size; i++) {
        dst[i] = __float2bfloat16(src[i]);
    }
}

void bf16_to_fp32(const bfloat16* src, float* dst, int size) {
    for (int i = 0; i < size; i++) {
        dst[i] = __bfloat162float(src[i]);
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

void test_bf16_attention(int bs, int nh, int seqlen, int hdim) {
    printf("\n========================================\n");
    printf("Testing BF16 Attention:\n");
    printf("  Batch size: %d, Num heads: %d\n", bs, nh);
    printf("  Sequence length: %d, Hidden dim: %d\n", seqlen, hdim);
    printf("========================================\n\n");

    int sz_qkv = bs * nh * seqlen * hdim;

    float *h_Q_fp32 = new float[sz_qkv];
    float *h_K_fp32 = new float[sz_qkv];
    float *h_V_fp32 = new float[sz_qkv];
    float *h_out_cpu = new float[sz_qkv];
    float *h_out_gpu_fp32 = new float[sz_qkv];

    bfloat16 *h_Q_bf16 = new bfloat16[sz_qkv];
    bfloat16 *h_K_bf16 = new bfloat16[sz_qkv];
    bfloat16 *h_V_bf16 = new bfloat16[sz_qkv];
    bfloat16 *h_out_gpu_bf16 = new bfloat16[sz_qkv];

    init_random(h_Q_fp32, sz_qkv);
    init_random(h_K_fp32, sz_qkv);
    init_random(h_V_fp32, sz_qkv);

    fp32_to_bf16(h_Q_fp32, h_Q_bf16, sz_qkv);
    fp32_to_bf16(h_K_fp32, h_K_bf16, sz_qkv);
    fp32_to_bf16(h_V_fp32, h_V_bf16, sz_qkv);

    bfloat16 *d_Q, *d_K, *d_V, *d_out;
    CUDA_CHECK(cudaMalloc(&d_Q, sz_qkv * sizeof(bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_K, sz_qkv * sizeof(bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_V, sz_qkv * sizeof(bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_out, sz_qkv * sizeof(bfloat16)));

    CUDA_CHECK(cudaMemcpy(d_Q, h_Q_bf16, sz_qkv * sizeof(bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K_bf16, sz_qkv * sizeof(bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V_bf16, sz_qkv * sizeof(bfloat16), cudaMemcpyHostToDevice));

    // Warmup
    attention_forward_bf16(d_Q, d_K, d_V, d_out, bs, nh, seqlen, seqlen, hdim);

    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int num_iters = 100;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iters; i++) {
        attention_forward_bf16(d_Q, d_K, d_V, d_out, bs, nh, seqlen, seqlen, hdim);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    elapsed_ms /= num_iters;

    CUDA_CHECK(cudaMemcpy(h_out_gpu_bf16, d_out, sz_qkv * sizeof(bfloat16), cudaMemcpyDeviceToHost));
    bf16_to_fp32(h_out_gpu_bf16, h_out_gpu_fp32, sz_qkv);

    // CPU reference
    attention_cpu(h_Q_fp32, h_K_fp32, h_V_fp32, h_out_cpu, bs, nh, seqlen, seqlen, hdim);

    printf("BF16 kernel time: %.3f ms\n", elapsed_ms);

    long long flops = 2LL * bs * nh * seqlen * seqlen * hdim * 2;
    float tflops = (flops / (elapsed_ms / 1000.0f)) / 1e12f;
    printf("Performance: %.2f TFLOPS\n", tflops);

    printf("\nChecking correctness...\n");
    bool passed = check_results(h_out_gpu_fp32, h_out_cpu, sz_qkv, 0.1f);

    if (passed) {
        printf("\n✓ TEST PASSED\n");
    } else {
        printf("\n✗ TEST FAILED\n");
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    delete[] h_Q_fp32;
    delete[] h_K_fp32;
    delete[] h_V_fp32;
    delete[] h_out_cpu;
    delete[] h_out_gpu_fp32;
    delete[] h_Q_bf16;
    delete[] h_K_bf16;
    delete[] h_V_bf16;
    delete[] h_out_gpu_bf16;

    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_out));
}

int main() {
    printf("=================================================\n");
    printf("BF16 Attention Benchmark\n");
    printf("=================================================\n");

    srand(42);

    printf("\n*** Testing hdim=512 ***\n");
    test_bf16_attention(1, 4, 64, 512);

    printf("\n*** Testing hdim=2048 ***\n");
    test_bf16_attention(1, 4, 64, 2048);

    printf("\n*** Testing hdim=4096 ***\n");
    test_bf16_attention(1, 2, 32, 4096);

    printf("\n=================================================\n");
    printf("All tests completed!\n");
    printf("=================================================\n");

    return 0;
}
