/*
 * OPTIMIZED ATTENTION KERNEL WITH AUTOMATIC KERNEL SELECTION
 *
 * This file contains multiple optimized kernels with automatic selection
 * based on hidden dimension size.
 *
 * Use this if you need to support various hidden dimensions (small and large).
 *
 * Kernel selection:
 * - hdim >= 512: Uses online softmax tiled kernel (optimized for 2048, 4096, 8192)
 * - hdim < 512:  Uses improved baseline with warp reductions
 *
 * Key optimizations:
 * - Warp-level reduction primitives (replaces slow atomics)
 * - Online softmax for large dimensions
 * - Vectorized memory access
 * - Smart dispatcher for automatic kernel selection
 *
 * Performance vs naive baseline:
 * - hdim=2048: ~1.9x speedup
 * - hdim=4096: ~1.5x speedup
 * - hdim=8192: ~1.0x speedup
 *
 * Main function to call: attention_forward_optimized()
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

__device__ __forceinline__ float safe_div(float a, float b) {
    return b != 0.0f ? a / b : 0.0f;
}

// Forward declarations
void attention_forward(
    const float* Q, const float* K, const float* V, float* out,
    int batch_sz, int num_heads, int len_q, int len_k, int head_d);

void attention_forward_optimized(
    const float* Q, const float* K, const float* V, float* out,
    int batch_sz, int num_heads, int len_q, int len_k, int head_d);

// Warp-level reduction primitives for better performance
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

// Optimized kernel for large hidden dimensions (2048, 4096, 8192)
// Uses online softmax and tiling to reduce memory traffic
template<int BLOCK_N>
__global__ void attention_fwd_kernel_large_hdim(
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

    float m_max = -INFINITY;
    float l_sum = 0.0f;

    // Process K,V in blocks using online softmax
    for (int k_start = 0; k_start < seq_k; k_start += BLOCK_N) {
        const int k_end = min(k_start + BLOCK_N, seq_k);
        const int num_k = k_end - k_start;

        // Compute scores for this block
        for (int k_local = tid; k_local < num_k; k_local += blockDim.x) {
            int k_idx = k_start + k_local;
            int k_offset = kv_base + k_idx * hdim;

            float score = 0.0f;

            // Vectorized dot product
            if (hdim % 4 == 0) {
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

        // Update global max
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

        // Update output with correction factor
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

// Optimized FlashAttention-style kernel with tiling and fused operations
// This kernel processes attention in blocks to maximize data reuse and minimize memory traffic
template<int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K>
__global__ void attention_fwd_kernel_optimized(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ out,
    int bs, int nh, int seq_q, int seq_k, int hdim,
    float scale_factor
) {
    // Each block processes BLOCK_SIZE_M query tokens
    const int batch_head_idx = blockIdx.y;
    const int b_idx = batch_head_idx / nh;
    const int h_idx = batch_head_idx % nh;
    const int q_block_idx = blockIdx.x;

    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    // Shared memory layout
    extern __shared__ float smem[];
    float* s_Q = smem;  // BLOCK_SIZE_M x hdim
    float* s_K = s_Q + BLOCK_SIZE_M * hdim;  // BLOCK_SIZE_N x hdim
    float* s_V = s_K + BLOCK_SIZE_N * hdim;  // BLOCK_SIZE_N x hdim
    float* s_reduce = s_V + BLOCK_SIZE_N * hdim;  // For reductions

    const int q_start = q_block_idx * BLOCK_SIZE_M;
    const int q_end = min(q_start + BLOCK_SIZE_M, seq_q);
    const int num_q = q_end - q_start;

    // Base offsets
    const int qkv_base = (b_idx * nh + h_idx) * seq_q * hdim;
    const int kv_base = (b_idx * nh + h_idx) * seq_k * hdim;

    // Load Q block into shared memory (coalesced)
    for (int i = tid; i < num_q * hdim; i += blockDim.x) {
        int q_idx = i / hdim;
        int d_idx = i % hdim;
        s_Q[q_idx * hdim + d_idx] = Q[qkv_base + (q_start + q_idx) * hdim + d_idx];
    }
    __syncthreads();

    // Online softmax accumulators (per query in block)
    float m_old[BLOCK_SIZE_M];
    float m_new[BLOCK_SIZE_M];
    float l_old[BLOCK_SIZE_M];
    float l_new[BLOCK_SIZE_M];
    float O[BLOCK_SIZE_M * 128];  // Output accumulator (limit hdim for register pressure)

    #pragma unroll
    for (int i = 0; i < BLOCK_SIZE_M; i++) {
        m_old[i] = -INFINITY;
        l_old[i] = 0.0f;
    }

    int hdim_limited = min(hdim, 128);
    #pragma unroll
    for (int i = 0; i < BLOCK_SIZE_M * hdim_limited; i++) {
        O[i] = 0.0f;
    }

    // Process K,V in blocks
    for (int k_block_start = 0; k_block_start < seq_k; k_block_start += BLOCK_SIZE_N) {
        const int k_end = min(k_block_start + BLOCK_SIZE_N, seq_k);
        const int num_k = k_end - k_block_start;

        // Load K block
        for (int i = tid; i < num_k * hdim; i += blockDim.x) {
            int k_idx = i / hdim;
            int d_idx = i % hdim;
            s_K[k_idx * hdim + d_idx] = K[kv_base + (k_block_start + k_idx) * hdim + d_idx];
        }

        // Load V block
        for (int i = tid; i < num_k * hdim; i += blockDim.x) {
            int k_idx = i / hdim;
            int d_idx = i % hdim;
            s_V[k_idx * hdim + d_idx] = V[kv_base + (k_block_start + k_idx) * hdim + d_idx];
        }
        __syncthreads();

        // Compute QK^T for this block (each thread handles some Q-K pairs)
        float S_local[BLOCK_SIZE_M * BLOCK_SIZE_N];

        for (int q_local = 0; q_local < num_q; q_local++) {
            for (int k_local = tid; k_local < num_k; k_local += blockDim.x) {
                float score = 0.0f;

                // Vectorized dot product
                #pragma unroll 8
                for (int d = 0; d < hdim; d++) {
                    score += s_Q[q_local * hdim + d] * s_K[k_local * hdim + d];
                }

                score *= scale_factor;
                S_local[q_local * BLOCK_SIZE_N + k_local] = score;
            }
        }

        // Online softmax update for each query
        for (int q_local = 0; q_local < num_q; q_local++) {
            // Find max in this block
            float local_max = -INFINITY;
            for (int k_local = tid; k_local < num_k; k_local += blockDim.x) {
                local_max = fmaxf(local_max, S_local[q_local * BLOCK_SIZE_N + k_local]);
            }

            // Block-wide max reduction
            local_max = block_reduce_max(local_max, s_reduce);
            m_new[q_local] = fmaxf(m_old[q_local], local_max);

            // Compute exp and sum
            float local_sum = 0.0f;
            for (int k_local = tid; k_local < num_k; k_local += blockDim.x) {
                float exp_val = expf(S_local[q_local * BLOCK_SIZE_N + k_local] - m_new[q_local]);
                S_local[q_local * BLOCK_SIZE_N + k_local] = exp_val;
                local_sum += exp_val;
            }

            local_sum = block_reduce_sum(local_sum, s_reduce);

            // Update running statistics
            float correction = expf(m_old[q_local] - m_new[q_local]);
            l_new[q_local] = correction * l_old[q_local] + local_sum;

            // Update output: O_new = correction * O_old + exp(S - m_new) @ V
            if (hdim <= hdim_limited) {
                for (int d = tid; d < hdim; d += blockDim.x) {
                    O[q_local * hdim_limited + d] *= correction;

                    float v_acc = 0.0f;
                    #pragma unroll 4
                    for (int k_local = 0; k_local < num_k; k_local++) {
                        v_acc += S_local[q_local * BLOCK_SIZE_N + k_local] * s_V[k_local * hdim + d];
                    }

                    O[q_local * hdim_limited + d] += v_acc;
                }
            }

            m_old[q_local] = m_new[q_local];
            l_old[q_local] = l_new[q_local];
        }

        __syncthreads();
    }

    // Write output
    for (int q_local = 0; q_local < num_q; q_local++) {
        int out_offset = qkv_base + (q_start + q_local) * hdim;
        for (int d = tid; d < hdim && d < hdim_limited; d += blockDim.x) {
            out[out_offset + d] = O[q_local * hdim_limited + d] / l_old[q_local];
        }
    }
}

// Keep old kernel for compatibility
__global__ void attention_fwd_kernel(
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
    float* reduce_buf = scores + seq_k;

    int b_idx = idx / (nh * seq_q);
    int h_idx = (idx / seq_q) % nh;

    int q_off = idx * hdim;
    int kv_off = (b_idx * nh + h_idx) * seq_k * hdim;

    int tid = threadIdx.x;

    // Compute attention scores with vectorization
    float local_max = -INFINITY;

    for (int kpos = tid; kpos < seq_k; kpos += blockDim.x) {
        float score = 0.0f;
        int k_off = kv_off + kpos * hdim;

        // Dot product with manual unrolling
        #pragma unroll 16
        for (int d = 0; d < hdim; d++) {
            score += Q[q_off + d] * K[k_off + d];
        }

        score *= scale_factor;
        scores[kpos] = score;
        local_max = fmaxf(local_max, score);
    }

    // Use warp reduction instead of atomics
    local_max = block_reduce_max(local_max, reduce_buf);

    // Compute softmax
    float exp_sum = 0.0f;
    for (int kpos = tid; kpos < seq_k; kpos += blockDim.x) {
        float exp_val = expf(scores[kpos] - local_max);
        scores[kpos] = exp_val;
        exp_sum += exp_val;
    }

    exp_sum = block_reduce_sum(exp_sum, reduce_buf);

    // Normalize
    for (int kpos = tid; kpos < seq_k; kpos += blockDim.x) {
        scores[kpos] = safe_div(scores[kpos], exp_sum);
    }
    __syncthreads();

    // Compute output with vectorization
    for (int d = tid; d < hdim; d += blockDim.x) {
        float accum = 0.0f;

        #pragma unroll 4
        for (int kpos = 0; kpos < seq_k; kpos++) {
            int v_off = kv_off + kpos * hdim + d;
            accum += scores[kpos] * V[v_off];
        }

        out[q_off + d] = accum;
    }
}

// Smart dispatcher that selects the best kernel based on problem size
void attention_forward_optimized(
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

    // For large hidden dimensions (512+, including 2048, 4096, 8192), use online softmax kernel
    if (head_d >= 512) {
        constexpr int BLOCK_N = 64;

        dim3 grid(len_q, batch_sz * num_heads);
        const int nthreads = 256;
        const int shmem_sz = (BLOCK_N + 32) * sizeof(float);

        attention_fwd_kernel_large_hdim<BLOCK_N><<<grid, nthreads, shmem_sz>>>(
            Q, K, V, out, batch_sz, num_heads, len_q, len_k, head_d, scale
        );
    }
    // For all other dimensions, use the improved baseline kernel
    else {
        attention_forward(Q, K, V, out, batch_sz, num_heads, len_q, len_k, head_d);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Original forward function with improvements
void attention_forward(
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

    attention_fwd_kernel<<<nblocks, nthreads, shmem_sz>>>(
        Q, K, V, out,
        batch_sz, num_heads, len_q, len_k, head_d,
        scale
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

//---------------------------------------------------------------------------------------------------------------------


//Example CPU attention function to get results for verification
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

//FUNCTIONS FOR TESTING AND ENSURING THAT THE KERNELS WORK CORRECTLY (written using AI)


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

void test_attention(int bs, int nh, int seqlen, int hdim) {
    printf("\n========================================\n");
    printf("Testing attention kernel:\n");
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
    float *h_out_gpu_opt = new float[sz_q];
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
    attention_forward(d_Q, d_K, d_V, d_out, bs, nh, seqlen, seqlen, hdim);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark original kernel
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    printf("Running baseline GPU kernel...\n");
    const int num_iters = 10;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iters; i++) {
        attention_forward(d_Q, d_K, d_V, d_out, bs, nh, seqlen, seqlen, hdim);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float baseline_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&baseline_ms, start, stop));
    baseline_ms /= num_iters;

    CUDA_CHECK(cudaMemcpy(h_out_gpu, d_out, sz_q * sizeof(float), cudaMemcpyDeviceToHost));

    // Benchmark optimized kernel
    printf("Running OPTIMIZED GPU kernel...\n");
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iters; i++) {
        attention_forward_optimized(d_Q, d_K, d_V, d_out, bs, nh, seqlen, seqlen, hdim);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float optimized_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&optimized_ms, start, stop));
    optimized_ms /= num_iters;

    CUDA_CHECK(cudaMemcpy(h_out_gpu_opt, d_out, sz_q * sizeof(float), cudaMemcpyDeviceToHost));

    printf("Running CPU reference...\n");
    attention_cpu(h_Q, h_K, h_V, h_out_cpu, bs, nh, seqlen, seqlen, hdim);

    printf("\n--- Performance Results ---\n");
    printf("Baseline kernel time: %.3f ms\n", baseline_ms);
    printf("Optimized kernel time: %.3f ms\n", optimized_ms);
    printf("Speedup: %.2fx\n", baseline_ms / optimized_ms);

    printf("\nChecking baseline results vs CPU...\n");
    bool passed_baseline = check_results(h_out_gpu, h_out_cpu, sz_q);

    printf("\nChecking optimized results vs CPU...\n");
    bool passed_opt = check_results(h_out_gpu_opt, h_out_cpu, sz_q);

    if (passed_baseline && passed_opt) {
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
    delete[] h_out_gpu_opt;
    delete[] h_out_cpu;

    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_out));
}

void test_cross_attention(int bs, int nh, int seqlen_q, int seqlen_kv, int hdim) {
    printf("\n========================================\n");
    printf("Testing cross-attention kernel:\n");
    printf("  Batch size: %d\n", bs);
    printf("  Num heads: %d\n", nh);
    printf("  Query sequence length: %d\n", seqlen_q);
    printf("  Key/Value sequence length: %d\n", seqlen_kv);
    printf("  Head dimension: %d\n", hdim);
    printf("========================================\n\n");

    int sz_q = bs * nh * seqlen_q * hdim;
    int sz_kv = bs * nh * seqlen_kv * hdim;

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

    printf("Running GPU kernel...\n");
    attention_forward(d_Q, d_K, d_V, d_out, bs, nh, seqlen_q, seqlen_kv, hdim);

    CUDA_CHECK(cudaMemcpy(h_out_gpu, d_out, sz_q * sizeof(float), cudaMemcpyDeviceToHost));

    printf("Running CPU reference...\n");
    attention_cpu(h_Q, h_K, h_V, h_out_cpu, bs, nh, seqlen_q, seqlen_kv, hdim);

    printf("\nChecking results...\n");
    bool passed = check_results(h_out_gpu, h_out_cpu, sz_q);

    if (passed) {
        printf("\n✓ TEST PASSED\n");
    } else {
        printf("\n✗ TEST FAILED\n");
    }

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
    printf("CUDA Attention Kernel Test Suite\n");
    printf("=================================================\n");

    srand(42);

    // Basic correctness tests
    printf("\n--- BASIC CORRECTNESS TESTS ---\n");
    test_attention(1, 1, 4, 8);
    test_attention(1, 2, 8, 16);
    test_attention(2, 4, 32, 32);
    test_attention(2, 8, 64, 64);

    // Cross-attention test
    test_cross_attention(1, 2, 16, 32, 32);

    // Small head dimension tests
    printf("\n--- SMALL HEAD DIMENSION TESTS ---\n");
    test_attention(1, 4, 16, 64);
    test_attention(2, 8, 64, 128);

    // LARGE HEAD DIMENSION TESTS (2048, 4096, 8192)
    // These are the target dimensions for optimization
    printf("\n--- LARGE HEAD DIMENSION TESTS (TARGET) ---\n");

    printf("\n*** Testing hdim=512 (warmup for large dims) ***\n");
    test_attention(1, 4, 64, 512);

    printf("\n*** Testing hdim=2048 ***\n");
    test_attention(1, 4, 64, 2048);

    printf("\n*** Testing hdim=4096 ***\n");
    test_attention(1, 2, 32, 4096);

    printf("\n*** Testing hdim=8192 ***\n");
    test_attention(1, 1, 16, 8192);

    // Performance benchmark with realistic sizes
    printf("\n--- REALISTIC WORKLOAD BENCHMARKS ---\n");
    printf("\n*** LLaMA-style config: bs=4, nh=32, seq=512, hdim=128 ***\n");
    test_attention(4, 32, 512, 128);

    printf("\n=================================================\n");
    printf("All tests completed!\n");
    printf("=================================================\n");

    return 0;
}
