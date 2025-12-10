/*
 * OPTIMIZED FP16 ATTENTION KERNEL WITH ONLINE SOFTMAX + K/V TILING
 *
 * This combines:
 * - Pure FP16 inputs/outputs (no conversion overhead)
 * - Online softmax with K/V tiling (memory-efficient streaming)
 * - half2 vectorization (2x throughput)
 * - FP32 accumulation for numerical stability
 *
 * Should be MUCH faster than naive FP16!
 */

#include <cuda.h>
#include <cuda_runtime.h>
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

// Warp-level reductions
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

// Optimized FP16 kernel with online softmax and K/V tiling
template<int BLOCK_N>
__global__ void attention_fwd_kernel_fp16_optimized(
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    half* __restrict__ out,
    int bs, int nh, int seq_q, int seq_k, int hdim,
    float scale_factor
) {
    const int batch_head_idx = blockIdx.y;
    const int b_idx = batch_head_idx / nh;
    const int h_idx = batch_head_idx % nh;
    const int q_idx = blockIdx.x;

    if (q_idx >= seq_q) return;

    const int tid = threadIdx.x;
    const int qkv_base = (b_idx * nh + h_idx) * seq_q * hdim;
    const int kv_base = (b_idx * nh + h_idx) * seq_k * hdim;

    extern __shared__ float smem[];
    float* s_scores = smem;
    float* s_reduce = s_scores + BLOCK_N;

    const int q_offset = qkv_base + q_idx * hdim;

    // Running stats for online softmax (use FP32 for stability)
    float m_max = -INFINITY;
    float l_sum = 0.0f;

    // Online softmax: process K/V in blocks
    for (int k_start = 0; k_start < seq_k; k_start += BLOCK_N) {
        const int k_end = min(k_start + BLOCK_N, seq_k);
        const int num_k = k_end - k_start;

        // Compute Q@K^T scores for this block with half2 vectorization
        for (int k_local = tid; k_local < num_k; k_local += blockDim.x) {
            int k_idx = k_start + k_local;
            int k_offset = kv_base + k_idx * hdim;

            // Accumulate in FP32 for numerical stability
            float score = 0.0f;

            // Use half2 for 2x throughput
            if (hdim % 2 == 0) {
                #pragma unroll 4
                for (int d = 0; d < hdim; d += 2) {
                    half2 q_pair = *reinterpret_cast<const half2*>(&Q[q_offset + d]);
                    half2 k_pair = *reinterpret_cast<const half2*>(&K[k_offset + d]);

                    float2 q_f = __half22float2(q_pair);
                    float2 k_f = __half22float2(k_pair);

                    score += q_f.x * k_f.x + q_f.y * k_f.y;
                }
            } else {
                for (int d = 0; d < hdim; d++) {
                    float q_val = __half2float(Q[q_offset + d]);
                    float k_val = __half2float(K[k_offset + d]);
                    score += q_val * k_val;
                }
            }

            score *= scale_factor;
            s_scores[k_local] = score;
        }
        __syncthreads();

        // Find max in this block
        float block_max = -INFINITY;
        for (int k_local = tid; k_local < num_k; k_local += blockDim.x) {
            block_max = fmaxf(block_max, s_scores[k_local]);
        }
        block_max = block_reduce_max(block_max, s_reduce);

        // Update global max with correction factor
        float m_new = fmaxf(m_max, block_max);
        float correction = expf(m_max - m_new);

        // Compute exp and sum for this block
        float block_sum = 0.0f;
        for (int k_local = tid; k_local < num_k; k_local += blockDim.x) {
            float exp_val = expf(s_scores[k_local] - m_new);
            s_scores[k_local] = exp_val;
            block_sum += exp_val;
        }
        block_sum = block_reduce_sum(block_sum, s_reduce);

        // Update running sum with correction
        l_sum = correction * l_sum + block_sum;

        // Update output incrementally with correction factor
        for (int d = tid; d < hdim; d += blockDim.x) {
            // Read previous output (FP16 -> FP32)
            float prev_out = (k_start > 0) ? __half2float(out[q_offset + d]) : 0.0f;
            float corrected_out = prev_out * correction;

            // Compute contribution from this K/V block
            float v_acc = 0.0f;
            for (int k_local = 0; k_local < num_k; k_local++) {
                int k_idx = k_start + k_local;
                int v_offset = kv_base + k_idx * hdim + d;
                float v_val = __half2float(V[v_offset]);
                v_acc += s_scores[k_local] * v_val;
            }

            // Write updated output (FP32 -> FP16)
            out[q_offset + d] = __float2half(corrected_out + v_acc);
        }

        m_max = m_new;
        __syncthreads();
    }

    // Final normalization
    for (int d = tid; d < hdim; d += blockDim.x) {
        float val = __half2float(out[q_offset + d]);
        out[q_offset + d] = __float2half(val / l_sum);
    }
}

// Optimized FP16 forward function
void attention_forward_fp16_optimized(
    const half* Q,
    const half* K,
    const half* V,
    half* out,
    int batch_sz,
    int num_heads,
    int len_q,
    int len_k,
    int head_d
) {
    float scale = 1.0f / sqrtf(static_cast<float>(head_d));

    constexpr int BLOCK_N = 64;
    dim3 grid(len_q, batch_sz * num_heads);
    const int nthreads = 256;
    const int shmem_sz = (BLOCK_N + 32) * sizeof(float);

    attention_fwd_kernel_fp16_optimized<BLOCK_N><<<grid, nthreads, shmem_sz>>>(
        Q, K, V, out,
        batch_sz, num_heads, len_q, len_k, head_d,
        scale
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

//---------------------------------------------------------------------------------------------------------------------
// CPU reference in FP32 for testing
void attention_cpu_fp32(
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

void init_random_fp32(float* data, int size, float min_val = -1.0f, float max_val = 1.0f) {
    for (int i = 0; i < size; i++) {
        data[i] = min_val + (max_val - min_val) * (rand() / (float)RAND_MAX);
    }
}

void fp32_to_fp16(const float* src, half* dst, int size) {
    for (int i = 0; i < size; i++) {
        dst[i] = __float2half(src[i]);
    }
}

void fp16_to_fp32(const half* src, float* dst, int size) {
    for (int i = 0; i < size; i++) {
        dst[i] = __half2float(src[i]);
    }
}

bool check_results(const float* gpu_out, const float* cpu_out, int size, float tol = 1e-1f) {
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

void test_attention_fp16_optimized(int bs, int nh, int seqlen, int hdim) {
    printf("\n========================================\n");
    printf("Testing OPTIMIZED FP16 attention kernel:\n");
    printf("  Batch size: %d\n", bs);
    printf("  Num heads: %d\n", nh);
    printf("  Sequence length: %d\n", seqlen);
    printf("  Head dimension: %d\n", hdim);
    printf("========================================\n\n");

    int sz_q = bs * nh * seqlen * hdim;
    int sz_kv = bs * nh * seqlen * hdim;

    // Allocate FP32 host memory for initialization
    float *h_Q_fp32 = new float[sz_q];
    float *h_K_fp32 = new float[sz_kv];
    float *h_V_fp32 = new float[sz_kv];
    float *h_out_cpu = new float[sz_q];
    float *h_out_gpu_fp32 = new float[sz_q];

    // Allocate FP16 host memory
    half *h_Q_fp16 = new half[sz_q];
    half *h_K_fp16 = new half[sz_kv];
    half *h_V_fp16 = new half[sz_kv];
    half *h_out_gpu_fp16 = new half[sz_q];

    // Initialize with random FP32 data
    init_random_fp32(h_Q_fp32, sz_q);
    init_random_fp32(h_K_fp32, sz_kv);
    init_random_fp32(h_V_fp32, sz_kv);

    // Convert to FP16
    fp32_to_fp16(h_Q_fp32, h_Q_fp16, sz_q);
    fp32_to_fp16(h_K_fp32, h_K_fp16, sz_kv);
    fp32_to_fp16(h_V_fp32, h_V_fp16, sz_kv);

    // Allocate GPU memory (FP16)
    half *d_Q, *d_K, *d_V, *d_out;
    CUDA_CHECK(cudaMalloc(&d_Q, sz_q * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_K, sz_kv * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_V, sz_kv * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_out, sz_q * sizeof(half)));

    // Copy FP16 data to GPU
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q_fp16, sz_q * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K_fp16, sz_kv * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V_fp16, sz_kv * sizeof(half), cudaMemcpyHostToDevice));

    // Warmup
    attention_forward_fp16_optimized(d_Q, d_K, d_V, d_out, bs, nh, seqlen, seqlen, hdim);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    printf("Running OPTIMIZED FP16 GPU kernel...\n");
    const int num_iters = 100;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iters; i++) {
        attention_forward_fp16_optimized(d_Q, d_K, d_V, d_out, bs, nh, seqlen, seqlen, hdim);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float fp16_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&fp16_ms, start, stop));
    fp16_ms /= num_iters;

    // Copy result back and convert to FP32
    CUDA_CHECK(cudaMemcpy(h_out_gpu_fp16, d_out, sz_q * sizeof(half), cudaMemcpyDeviceToHost));
    fp16_to_fp32(h_out_gpu_fp16, h_out_gpu_fp32, sz_q);

    printf("Running CPU reference (FP32)...\n");
    attention_cpu_fp32(h_Q_fp32, h_K_fp32, h_V_fp32, h_out_cpu, bs, nh, seqlen, seqlen, hdim);

    // Calculate TFLOPS
    long long flops = 2LL * bs * nh * seqlen * seqlen * hdim * 2;
    double tflops = (flops / (fp16_ms * 1e-3)) / 1e12;

    printf("\n--- Performance Results ---\n");
    printf("OPTIMIZED FP16 kernel time: %.3f ms\n", fp16_ms);
    printf("Performance: %.2f TFLOPS\n", tflops);

    printf("\nChecking OPTIMIZED FP16 results vs CPU (FP32)...\n");
    bool passed = check_results(h_out_gpu_fp32, h_out_cpu, sz_q);

    if (passed) {
        printf("\n✓ TEST PASSED\n");
    } else {
        printf("\n✗ TEST FAILED (Note: FP16 has less precision than FP32)\n");
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    delete[] h_Q_fp32;
    delete[] h_K_fp32;
    delete[] h_V_fp32;
    delete[] h_out_cpu;
    delete[] h_out_gpu_fp32;
    delete[] h_Q_fp16;
    delete[] h_K_fp16;
    delete[] h_V_fp16;
    delete[] h_out_gpu_fp16;

    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_out));
}

int main() {
    printf("=================================================\n");
    printf("OPTIMIZED FP16 Attention Kernel Test Suite\n");
    printf("With Online Softmax + K/V Tiling!\n");
    printf("=================================================\n");

    srand(42);

    // Test basic correctness
    printf("\n--- BASIC CORRECTNESS TESTS ---\n");
    test_attention_fp16_optimized(1, 1, 4, 8);
    test_attention_fp16_optimized(1, 2, 8, 16);
    test_attention_fp16_optimized(2, 4, 32, 32);

    // LARGE HEAD DIMENSION TESTS
    printf("\n--- LARGE HEAD DIMENSION TESTS ---\n");

    printf("\n*** Testing hdim=512 ***\n");
    test_attention_fp16_optimized(1, 4, 64, 512);

    printf("\n*** Testing hdim=2048 ***\n");
    test_attention_fp16_optimized(1, 4, 64, 2048);

    printf("\n*** Testing hdim=4096 ***\n");
    test_attention_fp16_optimized(1, 2, 32, 4096);

    printf("\n*** Testing hdim=8192 ***\n");
    test_attention_fp16_optimized(1, 1, 16, 8192);

    printf("\n=================================================\n");
    printf("All OPTIMIZED FP16 tests completed!\n");
    printf("Compare with previous results:\n");
    printf("  - FP32 optimized (hdim=2048): 0.214 ms, 2.51 TFLOPS\n");
    printf("  - FP16 naive (hdim=2048):     0.244 ms, 0.55 TFLOPS\n");
    printf("  - FP16 OPTIMIZED should beat both!\n");
    printf("=================================================\n");

    return 0;
}
