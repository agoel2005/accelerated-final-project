/*
 * BF16 (BFLOAT16) ATTENTION KERNELS - ALL VARIANTS
 *
 * Tests three kernels in BF16:
 * 1. Basic attention with online softmax + K/V tiling
 * 2. RoPE-fused attention
 * 3. Sinusoidal-fused attention
 *
 * BF16 has better range than FP16, should be more numerically stable!
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
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

//==============================================================================
// KERNEL 1: BASIC ATTENTION WITH ONLINE SOFTMAX (BF16)
//==============================================================================

template<int BLOCK_N>
__global__ void attention_fwd_kernel_bf16(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ out,
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

    // Running stats for online softmax (FP32 for stability)
    float m_max = -INFINITY;
    float l_sum = 0.0f;

    // Online softmax: process K/V in blocks
    for (int k_start = 0; k_start < seq_k; k_start += BLOCK_N) {
        const int k_end = min(k_start + BLOCK_N, seq_k);
        const int num_k = k_end - k_start;

        // Compute Q@K^T scores for this block with nv_bfloat162 vectorization
        for (int k_local = tid; k_local < num_k; k_local += blockDim.x) {
            int k_idx = k_start + k_local;
            int k_offset = kv_base + k_idx * hdim;

            // Accumulate in FP32 for numerical stability
            float score = 0.0f;

            // Use nv_bfloat162 for 2x throughput
            if (hdim % 2 == 0) {
                #pragma unroll 4
                for (int d = 0; d < hdim; d += 2) {
                    __nv_bfloat162 q_pair = *reinterpret_cast<const __nv_bfloat162*>(&Q[q_offset + d]);
                    __nv_bfloat162 k_pair = *reinterpret_cast<const __nv_bfloat162*>(&K[k_offset + d]);

                    float2 q_f = __bfloat1622float2(q_pair);
                    float2 k_f = __bfloat1622float2(k_pair);

                    score += q_f.x * k_f.x + q_f.y * k_f.y;
                }
            } else {
                for (int d = 0; d < hdim; d++) {
                    float q_val = __bfloat162float(Q[q_offset + d]);
                    float k_val = __bfloat162float(K[k_offset + d]);
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
            float prev_out = (k_start > 0) ? __bfloat162float(out[q_offset + d]) : 0.0f;
            float corrected_out = prev_out * correction;

            float v_acc = 0.0f;
            for (int k_local = 0; k_local < num_k; k_local++) {
                int k_idx = k_start + k_local;
                int v_offset = kv_base + k_idx * hdim + d;
                float v_val = __bfloat162float(V[v_offset]);
                v_acc += s_scores[k_local] * v_val;
            }

            out[q_offset + d] = __float2bfloat16(corrected_out + v_acc);
        }

        m_max = m_new;
        __syncthreads();
    }

    // Final normalization
    for (int d = tid; d < hdim; d += blockDim.x) {
        float val = __bfloat162float(out[q_offset + d]);
        out[q_offset + d] = __float2bfloat16(val / l_sum);
    }
}

//==============================================================================
// KERNEL 2: ROPE-FUSED ATTENTION (BF16)
//==============================================================================

template<int BLOCK_N>
__global__ void attention_fwd_kernel_rope_fused_bf16(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ out,
    int bs, int nh, int seq_q, int seq_k, int hdim,
    float scale_factor,
    float rope_base
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

    float m_max = -INFINITY;
    float l_sum = 0.0f;

    for (int k_start = 0; k_start < seq_k; k_start += BLOCK_N) {
        const int k_end = min(k_start + BLOCK_N, seq_k);
        const int num_k = k_end - k_start;

        for (int k_local = tid; k_local < num_k; k_local += blockDim.x) {
            int k_idx = k_start + k_local;
            int k_offset = kv_base + k_idx * hdim;

            float score = 0.0f;

            // Apply RoPE rotation on-the-fly
            for (int pair = 0; pair < hdim / 2; pair++) {
                float freq = 1.0f / powf(rope_base, (2.0f * pair) / hdim);
                float theta_q = q_idx * freq;
                float theta_k = k_idx * freq;

                float cos_q = cosf(theta_q);
                float sin_q = sinf(theta_q);
                float cos_k = cosf(theta_k);
                float sin_k = sinf(theta_k);

                float q0 = __bfloat162float(Q[q_offset + 2 * pair]);
                float q1 = __bfloat162float(Q[q_offset + 2 * pair + 1]);
                float k0 = __bfloat162float(K[k_offset + 2 * pair]);
                float k1 = __bfloat162float(K[k_offset + 2 * pair + 1]);

                // RoPE rotation
                float q_rot0 = q0 * cos_q - q1 * sin_q;
                float q_rot1 = q0 * sin_q + q1 * cos_q;
                float k_rot0 = k0 * cos_k - k1 * sin_k;
                float k_rot1 = k0 * sin_k + k1 * cos_k;

                score += q_rot0 * k_rot0 + q_rot1 * k_rot1;
            }

            score *= scale_factor;
            s_scores[k_local] = score;
        }
        __syncthreads();

        float block_max = -INFINITY;
        for (int k_local = tid; k_local < num_k; k_local += blockDim.x) {
            block_max = fmaxf(block_max, s_scores[k_local]);
        }
        block_max = block_reduce_max(block_max, s_reduce);

        float m_new = fmaxf(m_max, block_max);
        float correction = expf(m_max - m_new);

        float block_sum = 0.0f;
        for (int k_local = tid; k_local < num_k; k_local += blockDim.x) {
            float exp_val = expf(s_scores[k_local] - m_new);
            s_scores[k_local] = exp_val;
            block_sum += exp_val;
        }
        block_sum = block_reduce_sum(block_sum, s_reduce);

        l_sum = correction * l_sum + block_sum;

        for (int d = tid; d < hdim; d += blockDim.x) {
            float prev_out = (k_start > 0) ? __bfloat162float(out[q_offset + d]) : 0.0f;
            float corrected_out = prev_out * correction;

            float v_acc = 0.0f;
            for (int k_local = 0; k_local < num_k; k_local++) {
                int k_idx = k_start + k_local;
                int v_offset = kv_base + k_idx * hdim + d;
                float v_val = __bfloat162float(V[v_offset]);
                v_acc += s_scores[k_local] * v_val;
            }

            out[q_offset + d] = __float2bfloat16(corrected_out + v_acc);
        }

        m_max = m_new;
        __syncthreads();
    }

    for (int d = tid; d < hdim; d += blockDim.x) {
        float val = __bfloat162float(out[q_offset + d]);
        out[q_offset + d] = __float2bfloat16(val / l_sum);
    }
}

//==============================================================================
// KERNEL 3: SINUSOIDAL-FUSED ATTENTION (BF16)
//==============================================================================

template<int BLOCK_N>
__global__ void attention_fwd_kernel_sinusoidal_fused_bf16(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ out,
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

    float m_max = -INFINITY;
    float l_sum = 0.0f;

    for (int k_start = 0; k_start < seq_k; k_start += BLOCK_N) {
        const int k_end = min(k_start + BLOCK_N, seq_k);
        const int num_k = k_end - k_start;

        for (int k_local = tid; k_local < num_k; k_local += blockDim.x) {
            int k_idx = k_start + k_local;
            int k_offset = kv_base + k_idx * hdim;

            float score = 0.0f;

            // Apply sinusoidal position encoding
            for (int d = 0; d < hdim; d++) {
                float freq = 1.0f / powf(10000.0f, (2.0f * (d / 2)) / (float)hdim);

                float pos_enc_q = (d % 2 == 0) ? sinf(q_idx * freq) : cosf(q_idx * freq);
                float pos_enc_k = (d % 2 == 0) ? sinf(k_idx * freq) : cosf(k_idx * freq);

                float q_val = __bfloat162float(Q[q_offset + d]) + pos_enc_q;
                float k_val = __bfloat162float(K[k_offset + d]) + pos_enc_k;

                score += q_val * k_val;
            }

            score *= scale_factor;
            s_scores[k_local] = score;
        }
        __syncthreads();

        float block_max = -INFINITY;
        for (int k_local = tid; k_local < num_k; k_local += blockDim.x) {
            block_max = fmaxf(block_max, s_scores[k_local]);
        }
        block_max = block_reduce_max(block_max, s_reduce);

        float m_new = fmaxf(m_max, block_max);
        float correction = expf(m_max - m_new);

        float block_sum = 0.0f;
        for (int k_local = tid; k_local < num_k; k_local += blockDim.x) {
            float exp_val = expf(s_scores[k_local] - m_new);
            s_scores[k_local] = exp_val;
            block_sum += exp_val;
        }
        block_sum = block_reduce_sum(block_sum, s_reduce);

        l_sum = correction * l_sum + block_sum;

        for (int d = tid; d < hdim; d += blockDim.x) {
            float prev_out = (k_start > 0) ? __bfloat162float(out[q_offset + d]) : 0.0f;
            float corrected_out = prev_out * correction;

            float v_acc = 0.0f;
            for (int k_local = 0; k_local < num_k; k_local++) {
                int k_idx = k_start + k_local;
                int v_offset = kv_base + k_idx * hdim + d;
                float v_val = __bfloat162float(V[v_offset]);
                v_acc += s_scores[k_local] * v_val;
            }

            out[q_offset + d] = __float2bfloat16(corrected_out + v_acc);
        }

        m_max = m_new;
        __syncthreads();
    }

    for (int d = tid; d < hdim; d += blockDim.x) {
        float val = __bfloat162float(out[q_offset + d]);
        out[q_offset + d] = __float2bfloat16(val / l_sum);
    }
}

//==============================================================================
// FORWARD FUNCTIONS
//==============================================================================

void attention_forward_bf16(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    __nv_bfloat16* out, int bs, int nh, int len_q, int len_k, int head_d
) {
    float scale = 1.0f / sqrtf(static_cast<float>(head_d));
    constexpr int BLOCK_N = 64;
    dim3 grid(len_q, bs * nh);
    const int nthreads = 256;
    const int shmem_sz = (BLOCK_N + 32) * sizeof(float);

    attention_fwd_kernel_bf16<BLOCK_N><<<grid, nthreads, shmem_sz>>>(
        Q, K, V, out, bs, nh, len_q, len_k, head_d, scale);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void attention_forward_rope_bf16(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    __nv_bfloat16* out, int bs, int nh, int len_q, int len_k, int head_d
) {
    float scale = 1.0f / sqrtf(static_cast<float>(head_d));
    constexpr int BLOCK_N = 64;
    dim3 grid(len_q, bs * nh);
    const int nthreads = 256;
    const int shmem_sz = (BLOCK_N + 32) * sizeof(float);

    attention_fwd_kernel_rope_fused_bf16<BLOCK_N><<<grid, nthreads, shmem_sz>>>(
        Q, K, V, out, bs, nh, len_q, len_k, head_d, scale, 10000.0f);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void attention_forward_sinusoidal_bf16(
    const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
    __nv_bfloat16* out, int bs, int nh, int len_q, int len_k, int head_d
) {
    float scale = 1.0f / sqrtf(static_cast<float>(head_d));
    constexpr int BLOCK_N = 64;
    dim3 grid(len_q, bs * nh);
    const int nthreads = 256;
    const int shmem_sz = (BLOCK_N + 32) * sizeof(float);

    attention_fwd_kernel_sinusoidal_fused_bf16<BLOCK_N><<<grid, nthreads, shmem_sz>>>(
        Q, K, V, out, bs, nh, len_q, len_k, head_d, scale);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

//==============================================================================
// TEST UTILITIES
//==============================================================================

void init_random_fp32(float* data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = -1.0f + 2.0f * (rand() / (float)RAND_MAX);
    }
}

void fp32_to_bf16(const float* src, __nv_bfloat16* dst, int size) {
    for (int i = 0; i < size; i++) {
        dst[i] = __float2bfloat16(src[i]);
    }
}

void bf16_to_fp32(const __nv_bfloat16* src, float* dst, int size) {
    for (int i = 0; i < size; i++) {
        dst[i] = __bfloat162float(src[i]);
    }
}

void test_kernel(
    const char* name,
    void (*kernel_fn)(const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, int, int, int, int, int),
    int bs, int nh, int seqlen, int hdim
) {
    printf("\n========================================\n");
    printf("Testing %s (BF16):\n", name);
    printf("  bs=%d, nh=%d, seq=%d, hdim=%d\n", bs, nh, seqlen, hdim);
    printf("========================================\n\n");

    int sz_q = bs * nh * seqlen * hdim;
    int sz_kv = bs * nh * seqlen * hdim;

    // Allocate host memory
    float *h_Q_fp32 = new float[sz_q];
    float *h_K_fp32 = new float[sz_kv];
    float *h_V_fp32 = new float[sz_kv];
    __nv_bfloat16 *h_Q_bf16 = new __nv_bfloat16[sz_q];
    __nv_bfloat16 *h_K_bf16 = new __nv_bfloat16[sz_kv];
    __nv_bfloat16 *h_V_bf16 = new __nv_bfloat16[sz_kv];
    __nv_bfloat16 *h_out_bf16 = new __nv_bfloat16[sz_q];

    // Initialize
    init_random_fp32(h_Q_fp32, sz_q);
    init_random_fp32(h_K_fp32, sz_kv);
    init_random_fp32(h_V_fp32, sz_kv);

    fp32_to_bf16(h_Q_fp32, h_Q_bf16, sz_q);
    fp32_to_bf16(h_K_fp32, h_K_bf16, sz_kv);
    fp32_to_bf16(h_V_fp32, h_V_bf16, sz_kv);

    // Allocate device memory
    __nv_bfloat16 *d_Q, *d_K, *d_V, *d_out;
    CUDA_CHECK(cudaMalloc(&d_Q, sz_q * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_K, sz_kv * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_V, sz_kv * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_out, sz_q * sizeof(__nv_bfloat16)));

    CUDA_CHECK(cudaMemcpy(d_Q, h_Q_bf16, sz_q * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K_bf16, sz_kv * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V_bf16, sz_kv * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    // Warmup
    kernel_fn(d_Q, d_K, d_V, d_out, bs, nh, seqlen, seqlen, hdim);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int num_iters = 100;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iters; i++) {
        kernel_fn(d_Q, d_K, d_V, d_out, bs, nh, seqlen, seqlen, hdim);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    elapsed_ms /= num_iters;

    // Calculate TFLOPS
    long long flops = 2LL * bs * nh * seqlen * seqlen * hdim * 2;
    double tflops = (flops / (elapsed_ms * 1e-3)) / 1e12;

    printf("Time: %.3f ms\n", elapsed_ms);
    printf("Performance: %.2f TFLOPS\n", tflops);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    delete[] h_Q_fp32;
    delete[] h_K_fp32;
    delete[] h_V_fp32;
    delete[] h_Q_bf16;
    delete[] h_K_bf16;
    delete[] h_V_bf16;
    delete[] h_out_bf16;

    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_out));
}

int main() {
    printf("=================================================\n");
    printf("BF16 ATTENTION KERNELS - COMPREHENSIVE TEST\n");
    printf("=================================================\n");

    srand(42);

    // Test configurations for different hidden dimensions
    const int test_configs[][3] = {
        {1, 4, 512},    // hdim=512
        {1, 4, 2048},   // hdim=2048
        {1, 2, 4096},   // hdim=4096
        {1, 1, 8192},   // hdim=8192
    };

    const int test_seqlens[] = {64, 64, 32, 16};

    for (int i = 0; i < 4; i++) {
        int bs = test_configs[i][0];
        int nh = test_configs[i][1];
        int hdim = test_configs[i][2];
        int seqlen = test_seqlens[i];

        printf("\n\n╔═══════════════════════════════════════════════════╗\n");
        printf("║  TESTING hdim=%d (seq=%d)                     ║\n", hdim, seqlen);
        printf("╚═══════════════════════════════════════════════════╝\n");

        test_kernel("BASIC ATTENTION", attention_forward_bf16, bs, nh, seqlen, hdim);
        test_kernel("ROPE-FUSED ATTENTION", attention_forward_rope_bf16, bs, nh, seqlen, hdim);
        test_kernel("SINUSOIDAL-FUSED ATTENTION", attention_forward_sinusoidal_bf16, bs, nh, seqlen, hdim);
    }

    printf("\n\n=================================================\n");
    printf("ALL BF16 TESTS COMPLETED!\n");
    printf("=================================================\n");
    printf("\nSummary:\n");
    printf("  - Tested 3 kernel variants in BF16\n");
    printf("  - Tested 4 hidden dimensions: 512, 2048, 4096, 8192\n");
    printf("  - BF16 should provide better stability than FP16\n");
    printf("  - BF16 has 2x less memory vs FP32\n");
    printf("=================================================\n");

    return 0;
}
