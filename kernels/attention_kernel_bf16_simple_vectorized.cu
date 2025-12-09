/*
 * SIMPLE BF16 ATTENTION - NO MMA, JUST VECTORIZED BF16
 *
 * Goal: Match or beat FP32 by using BF16 for memory bandwidth savings
 * Strategy: All threads participate, vectorized loads, no MMA overhead
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

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

// Simple BF16 attention - vectorized but no MMA
__global__ void attention_fwd_bf16_simple(
    const bfloat16* __restrict__ Q,
    const bfloat16* __restrict__ K,
    const bfloat16* __restrict__ V,
    bfloat16* __restrict__ out,
    int bs, int nh, int seq_q, int seq_k, int hdim,
    float scale_factor
) {
    const int idx = blockIdx.x;
    const int total_q = bs * nh * seq_q;

    if (idx >= total_q) return;

    const int b_idx = idx / (nh * seq_q);
    const int h_idx = (idx / seq_q) % nh;

    const int q_offset = idx * hdim;
    const int kv_base = (b_idx * nh + h_idx) * seq_k * hdim;

    const int tid = threadIdx.x;

    extern __shared__ float smem[];
    float* s_scores = smem;
    float* s_reduce = s_scores + seq_k;

    // Compute Q @ K^T with ALL threads participating
    float local_max = -INFINITY;
    for (int k = tid; k < seq_k; k += blockDim.x) {
        float score = 0.0f;

        // Vectorized BF16 dot product
        if (hdim % 8 == 0) {
            for (int d = 0; d < hdim; d += 8) {
                #pragma unroll 8
                for (int i = 0; i < 8; i++) {
                    score += __bfloat162float(Q[q_offset + d + i]) *
                             __bfloat162float(K[kv_base + k * hdim + d + i]);
                }
            }
        } else {
            for (int d = 0; d < hdim; d++) {
                score += __bfloat162float(Q[q_offset + d]) *
                         __bfloat162float(K[kv_base + k * hdim + d]);
            }
        }

        score *= scale_factor;
        s_scores[k] = score;
        local_max = fmaxf(local_max, score);
    }

    // Block-level max reduction
    local_max = block_reduce_max(local_max, s_reduce);
    if (tid == 0) s_reduce[0] = local_max;
    __syncthreads();
    local_max = s_reduce[0];

    // Compute exp and sum
    float exp_sum = 0.0f;
    for (int k = tid; k < seq_k; k += blockDim.x) {
        float exp_val = expf(s_scores[k] - local_max);
        s_scores[k] = exp_val;
        exp_sum += exp_val;
    }

    exp_sum = block_reduce_sum(exp_sum, s_reduce);
    if (tid == 0) s_reduce[0] = exp_sum;
    __syncthreads();
    exp_sum = s_reduce[0];

    // Normalize
    for (int k = tid; k < seq_k; k += blockDim.x) {
        s_scores[k] = (exp_sum != 0.0f) ? (s_scores[k] / exp_sum) : 0.0f;
    }
    __syncthreads();

    // Compute output
    for (int d = tid; d < hdim; d += blockDim.x) {
        float accum = 0.0f;
        for (int k = 0; k < seq_k; k++) {
            accum += s_scores[k] * __bfloat162float(V[kv_base + k * hdim + d]);
        }
        out[q_offset + d] = __float2bfloat16(accum);
    }
}

void attention_forward_bf16_simple(
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
// Testing utilities
//---------------------------------------------------------------------------------------------------------------------

void attention_cpu(const float* Q, const float* K, const float* V, float* out,
    int bs, int nh, int seq_q, int seq_k, int hdim) {
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

bool check_results(const float* gpu_out, const float* cpu_out, int size, float tol = 0.15f) {
    float max_diff = 0.0f;
    int nerrors = 0;
    for (int i = 0; i < size; i++) {
        float diff = fabsf(gpu_out[i] - cpu_out[i]);
        max_diff = fmaxf(max_diff, diff);
        if (diff > tol) nerrors++;
    }
    printf("  Max difference: %.6f (tolerance %.6f)\n", max_diff, tol);
    if (nerrors > 0) printf("  Errors: %d / %d\n", nerrors, size);
    return nerrors == 0;
}

void test_bf16_simple(int bs, int nh, int seqlen, int hdim) {
    printf("\n========== BF16 Simple Vectorized ==========\n");
    printf("  bs=%d nh=%d seq=%d hdim=%d\n", bs, nh, seqlen, hdim);

    int sz = bs * nh * seqlen * hdim;
    float *h_Q_fp32 = new float[sz];
    float *h_K_fp32 = new float[sz];
    float *h_V_fp32 = new float[sz];
    float *h_out_cpu = new float[sz];
    float *h_out_gpu = new float[sz];
    bfloat16 *h_Q = new bfloat16[sz];
    bfloat16 *h_K = new bfloat16[sz];
    bfloat16 *h_V = new bfloat16[sz];
    bfloat16 *h_out_bf16 = new bfloat16[sz];

    init_random(h_Q_fp32, sz);
    init_random(h_K_fp32, sz);
    init_random(h_V_fp32, sz);
    fp32_to_bf16(h_Q_fp32, h_Q, sz);
    fp32_to_bf16(h_K_fp32, h_K, sz);
    fp32_to_bf16(h_V_fp32, h_V, sz);

    bfloat16 *d_Q, *d_K, *d_V, *d_out;
    CUDA_CHECK(cudaMalloc(&d_Q, sz * sizeof(bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_K, sz * sizeof(bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_V, sz * sizeof(bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_out, sz * sizeof(bfloat16)));
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q, sz * sizeof(bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K, sz * sizeof(bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V, sz * sizeof(bfloat16), cudaMemcpyHostToDevice));

    // Warmup
    attention_forward_bf16_simple(d_Q, d_K, d_V, d_out, bs, nh, seqlen, seqlen, hdim);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < 100; i++) {
        attention_forward_bf16_simple(d_Q, d_K, d_V, d_out, bs, nh, seqlen, seqlen, hdim);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    ms /= 100.0f;

    CUDA_CHECK(cudaMemcpy(h_out_bf16, d_out, sz * sizeof(bfloat16), cudaMemcpyDeviceToHost));
    bf16_to_fp32(h_out_bf16, h_out_gpu, sz);
    attention_cpu(h_Q_fp32, h_K_fp32, h_V_fp32, h_out_cpu, bs, nh, seqlen, seqlen, hdim);

    printf("Time: %.3f ms\n", ms);
    long long flops = 2LL * bs * nh * seqlen * seqlen * hdim * 2;
    printf("Performance: %.2f TFLOPS\n", (flops / (ms / 1000.0f)) / 1e12f);

    bool passed = check_results(h_out_gpu, h_out_cpu, sz);
    printf(passed ? "✓ PASSED\n" : "✗ FAILED\n");

    delete[] h_Q_fp32; delete[] h_K_fp32; delete[] h_V_fp32;
    delete[] h_out_cpu; delete[] h_out_gpu;
    delete[] h_Q; delete[] h_K; delete[] h_V; delete[] h_out_bf16;
    CUDA_CHECK(cudaFree(d_Q)); CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V)); CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
}

int main() {
    printf("=================================================\n");
    printf("BF16 Simple Vectorized (No MMA) - Bandwidth Test\n");
    printf("=================================================\n");

    srand(42);

    test_bf16_simple(1, 4, 64, 512);
    test_bf16_simple(1, 4, 64, 2048);
    test_bf16_simple(1, 2, 32, 4096);
    test_bf16_simple(1, 1, 16, 8192);
    test_bf16_simple(1, 1, 8, 16384);

    return 0;
}
