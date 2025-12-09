/*
 * OPTIMIZED ATTENTION KERNEL FOR LARGE HIDDEN DIMENSIONS
 *
 * This file contains an optimized kernel specifically designed for
 * large hidden dimensions (512, 2048, 4096, 8192).
 *
 * Current optimizations:
 * - Warp-level reduction primitives (replaces slow atomicMax/atomicAdd)
 * - Vectorized memory access (float4) for Q@K^T computation
 * - Coalesced memory access patterns
 *
 * Baseline performance (with warp reductions + vectorization):
 * - hdim=512:  0.68 TFLOPS
 * - hdim=2048: 0.82 TFLOPS
 * - hdim=4096: 0.27 TFLOPS
 * - hdim=8192: 0.05 TFLOPS
 *
 * Main function to call: attention_forward_large_dims()
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

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

__device__ __forceinline__ float safe_div(float a, float b) {
    return b != 0.0f ? a / b : 0.0f;
}

// Warp-level reduction primitives for efficient max/sum reductions
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

// Optimized kernel for large dimensions with warp-level reductions and vectorization
__global__ void attention_fwd_kernel_large_hdim(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ out,
    int bs, int nh, int seq_q, int seq_k, int hdim,
    float scale_factor
) {
    int idx = blockIdx.x;
    int total_q = bs * nh * seq_q;

    if (idx >= total_q) return;

    extern __shared__ float smem[];
    float* scores = smem;
    float* s_reduce = scores + seq_k;

    int b_idx = idx / (nh * seq_q);
    int h_idx = (idx / seq_q) % nh;

    int q_off = idx * hdim;
    int kv_off = (b_idx * nh + h_idx) * seq_k * hdim;

    int tid = threadIdx.x;

    // Compute attention scores Q@K^T with aggressive ILP and vectorization
    float local_max = -INFINITY;
    for (int kpos = tid; kpos < seq_k; kpos += blockDim.x) {
        int k_off = kv_off + kpos * hdim;

        float score = 0.0f;

        // Simple float4 vectorization with good unrolling
        if (hdim % 4 == 0) {
            #pragma unroll 4
            for (int d = 0; d < hdim; d += 4) {
                float4 q_vec = *reinterpret_cast<const float4*>(&Q[q_off + d]);
                float4 k_vec = *reinterpret_cast<const float4*>(&K[k_off + d]);
                score += q_vec.x * k_vec.x + q_vec.y * k_vec.y +
                         q_vec.z * k_vec.z + q_vec.w * k_vec.w;
            }
        } else {
            #pragma unroll 8
            for (int d = 0; d < hdim; d++) {
                score += Q[q_off + d] * K[k_off + d];
            }
        }
        score *= scale_factor;
        scores[kpos] = score;
        local_max = fmaxf(local_max, score);
    }

    // Find global max using fast warp-level reduction (replaces slow atomicMax)
    local_max = block_reduce_max(local_max, s_reduce);
    if (tid == 0) s_reduce[0] = local_max;
    __syncthreads();
    local_max = s_reduce[0];

    // Compute exp and sum using fast warp-level reduction
    float exp_sum = 0.0f;
    for (int kpos = tid; kpos < seq_k; kpos += blockDim.x) {
        float exp_val = expf(scores[kpos] - local_max);
        scores[kpos] = exp_val;
        exp_sum += exp_val;
    }

    exp_sum = block_reduce_sum(exp_sum, s_reduce);
    if (tid == 0) s_reduce[0] = exp_sum;
    __syncthreads();
    exp_sum = s_reduce[0];

    // Normalize
    for (int kpos = tid; kpos < seq_k; kpos += blockDim.x) {
        scores[kpos] = safe_div(scores[kpos], exp_sum);
    }
    __syncthreads();

    // Compute output = softmax@V
    for (int d = tid; d < hdim; d += blockDim.x) {
        float accum = 0.0f;

        for (int kpos = 0; kpos < seq_k; kpos++) {
            int v_off = kv_off + kpos * hdim + d;
            accum += scores[kpos] * V[v_off];
        }

        out[q_off + d] = accum;
    }
}

// Attention forward for large hidden dimensions (512, 2048, 4096, 8192)
void attention_forward_large_dims(
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

    const int total_qs = batch_sz * num_heads * len_q;
    const int nthreads = 256;
    const int nblocks = total_qs;
    const int shmem_sz = (len_k + 32) * sizeof(float);  // scores + reduction buffer

    attention_fwd_kernel_large_hdim<<<nblocks, nthreads, shmem_sz>>>(
        Q, K, V, out, batch_sz, num_heads, len_q, len_k, head_d, scale
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

//---------------------------------------------------------------------------------------------------------------------
// CPU reference implementation for testing
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

//---------------------------------------------------------------------------------------------------------------------
// Testing utilities
//---------------------------------------------------------------------------------------------------------------------

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

void test_attention_large_dims(int bs, int nh, int seqlen, int hdim) {
    printf("\n========================================\n");
    printf("Testing LARGE DIM attention kernel:\n");
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
    attention_forward_large_dims(d_Q, d_K, d_V, d_out, bs, nh, seqlen, seqlen, hdim);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    printf("Running optimized kernel for large dimensions...\n");
    const int num_iters = 20;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iters; i++) {
        attention_forward_large_dims(d_Q, d_K, d_V, d_out, bs, nh, seqlen, seqlen, hdim);
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
    printf("Optimized kernel time: %.3f ms\n", kernel_ms);

    // Calculate FLOPS
    long long flops = 2LL * bs * nh * seqlen * seqlen * hdim; // Q@K^T
    flops += 2LL * bs * nh * seqlen * seqlen * hdim; // softmax@V
    float tflops = (flops / (kernel_ms / 1000.0f)) / 1e12f;
    printf("Performance: %.2f TFLOPS\n", tflops);

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
    printf("Optimized Attention Kernel for Large Dimensions\n");
    printf("Target: hdim = 2048, 4096, 8192\n");
    printf("=================================================\n");

    srand(42);

    // Test large hidden dimensions
    printf("\n*** Testing hdim=512 ***\n");
    test_attention_large_dims(1, 4, 64, 512);

    printf("\n*** Testing hdim=2048 ***\n");
    test_attention_large_dims(1, 4, 64, 2048);

    printf("\n*** Testing hdim=4096 ***\n");
    test_attention_large_dims(1, 2, 32, 4096);

    printf("\n*** Testing hdim=8192 ***\n");
    test_attention_large_dims(1, 1, 16, 8192);

    printf("\n*** Testing EXTREME hdim=16384 ***\n");
    test_attention_large_dims(1, 1, 8, 16384);

    printf("\n=================================================\n");
    printf("All tests completed!\n");
    printf("=================================================\n");

    return 0;
}
