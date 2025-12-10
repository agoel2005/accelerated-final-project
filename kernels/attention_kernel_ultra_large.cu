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

__device__ __forceinline__ float safe_div(float a, float b) {
    return b != 0.0f ? a / b : 0.0f;
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

// Optimized kernel for ULTRA large hidden dimensions (16384, 32768)
// Uses online softmax and tiling to reduce memory traffic
template<int BLOCK_N>
__global__ void attention_fwd_kernel_ultra_large(
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
    const int qkv_base = (b_idx * nh + h_idx) * seq_q * hdim;
    const int kv_base = (b_idx * nh + h_idx) * seq_k * hdim;

    extern __shared__ float smem[];
    float* s_scores = smem;
    float* s_reduce = s_scores + BLOCK_N;

    const int q_offset = qkv_base + q_idx * hdim;

    // Running stats for online softmax
    float m_max = -INFINITY;
    float l_sum = 0.0f;

    // Online softmax: process K/V in blocks
    for (int k_start = 0; k_start < seq_k; k_start += BLOCK_N) {
        const int k_end = min(k_start + BLOCK_N, seq_k);
        const int num_k = k_end - k_start;

        // Compute Q@K^T scores for this block
        for (int k_local = tid; k_local < num_k; k_local += blockDim.x) {
            int k_idx = k_start + k_local;
            int k_offset = kv_base + k_idx * hdim;

            float score = 0.0f;

            // Vectorized dot product with float4
            if (hdim % 4 == 0) {
                #pragma unroll 4
                for (int d = 0; d < hdim; d += 4) {
                    float4 q_vec = *reinterpret_cast<const float4*>(&Q[q_offset + d]);
                    float4 k_vec = *reinterpret_cast<const float4*>(&K[k_offset + d]);
                    score += q_vec.x * k_vec.x + q_vec.y * k_vec.y +
                             q_vec.z * k_vec.z + q_vec.w * k_vec.w;
                }
            } else {
                for (int d = 0; d < hdim; d++) {
                    score += Q[q_offset + d] * K[k_offset + d];
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
            float prev_out = (k_start > 0) ? out[q_offset + d] : 0.0f;
            float corrected_out = prev_out * correction;

            float v_acc = 0.0f;
            for (int k_local = 0; k_local < num_k; k_local++) {
                int k_idx = k_start + k_local;
                int v_offset = kv_base + k_idx * hdim + d;
                v_acc += s_scores[k_local] * V[v_offset];
            }

            out[q_offset + d] = corrected_out + v_acc;
        }

        m_max = m_new;
        __syncthreads();
    }

    // Final normalization
    for (int d = tid; d < hdim; d += blockDim.x) {
        out[q_offset + d] /= l_sum;
    }
}

// Forward function for ultra large dimensions
void attention_forward_ultra_large(
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
    dim3 grid(len_q, batch_sz * num_heads);
    const int nthreads = 256;
    const int shmem_sz = (BLOCK_N + 32) * sizeof(float);

    attention_fwd_kernel_ultra_large<BLOCK_N><<<grid, nthreads, shmem_sz>>>(
        Q, K, V, out,
        batch_sz, num_heads, len_q, len_k, head_d,
        scale
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// CPU reference
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

void test_ultra_large_attention(int bs, int nh, int seqlen, int hdim) {
    printf("\n========================================\n");
    printf("Testing ULTRA LARGE attention kernel:\n");
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
    attention_forward_ultra_large(d_Q, d_K, d_V, d_out, bs, nh, seqlen, seqlen, hdim);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    printf("Running GPU kernel...\n");
    const int num_iters = 100;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iters; i++) {
        attention_forward_ultra_large(d_Q, d_K, d_V, d_out, bs, nh, seqlen, seqlen, hdim);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float gpu_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, start, stop));
    gpu_ms /= num_iters;

    CUDA_CHECK(cudaMemcpy(h_out_gpu, d_out, sz_q * sizeof(float), cudaMemcpyDeviceToHost));

    printf("Running CPU reference...\n");
    attention_cpu(h_Q, h_K, h_V, h_out_cpu, bs, nh, seqlen, seqlen, hdim);

    // Calculate TFLOPS
    long long flops = 2LL * bs * nh * seqlen * seqlen * hdim * 2;
    double tflops = (flops / (gpu_ms * 1e-3)) / 1e12;

    printf("\n--- Performance Results ---\n");
    printf("GPU kernel time: %.3f ms\n", gpu_ms);
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
    printf("ULTRA LARGE Hidden Dimension Attention Tests\n");
    printf("Testing hdim = 16384 and 32768\n");
    printf("=================================================\n");

    srand(42);

    // Test with reduced sequence lengths to fit in memory
    printf("\n*** Testing hdim=16384 (seq=8) ***\n");
    test_ultra_large_attention(1, 1, 8, 16384);

    printf("\n*** Testing hdim=32768 (seq=4) ***\n");
    test_ultra_large_attention(1, 1, 4, 32768);

    printf("\n=================================================\n");
    printf("ULTRA LARGE dimension tests completed!\n");
    printf("Compare with previous results:\n");
    printf("  - hdim=8192:  0.194 ms\n");
    printf("  - hdim=16384: ??? ms\n");
    printf("  - hdim=32768: ??? ms\n");
    printf("=================================================\n");

    return 0;
}
