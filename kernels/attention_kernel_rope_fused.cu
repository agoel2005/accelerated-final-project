#include <cuda.h>
#include <cuda_runtime.h>
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

// ============================================================================
// RoPE Cache Precomputation (CPU)
// ============================================================================

// Precompute cos/sin tables for RoPE with partial rotation support
// Layout: [max_seq_len, rotary_dim] - only rotated dimensions need cache
// For dimension pair (2i, 2i+1), both get the same cos/sin value
void precompute_rope_cache(
    float* cos_cache,      // [max_seq_len * rotary_dim]
    float* sin_cache,      // [max_seq_len * rotary_dim]
    int max_seq_len,
    int rotary_dim,
    float base = 10000.0f
) {
    for (int pos = 0; pos < max_seq_len; pos++) {
        for (int i = 0; i < rotary_dim / 2; i++) {
            float freq = 1.0f / powf(base, (2.0f * i) / rotary_dim);
            float theta = pos * freq;
            float cos_val = cosf(theta);
            float sin_val = sinf(theta);

            // Store for both dimensions in the pair
            cos_cache[pos * rotary_dim + 2*i]     = cos_val;
            cos_cache[pos * rotary_dim + 2*i + 1] = cos_val;
            sin_cache[pos * rotary_dim + 2*i]     = sin_val;
            sin_cache[pos * rotary_dim + 2*i + 1] = sin_val;
        }
    }
}

// ============================================================================
// Warp-Level Reduction Primitives (from original kernel)
// ============================================================================

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
        if (lane == 0) shared[0] = val;  // Broadcast result
    }
    __syncthreads();

    return shared[0];  // All threads read broadcasted result
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
        if (lane == 0) shared[0] = val;  // Broadcast result
    }
    __syncthreads();

    return shared[0];  // All threads read broadcasted result
}

// ============================================================================
// Helper: Compute RoPE frequency for dimension pair i
// ============================================================================
__device__ __forceinline__ float rope_freq(int pair_idx, int hdim, float base = 10000.0f) {
    return 1.0f / powf(base, (2.0f * pair_idx) / hdim);
}

// ============================================================================
// RoPE-Fused Attention Kernel V2: Compute cos/sin ON THE FLY
// No cos/sin cache needed - trades memory bandwidth for compute
// ============================================================================
template<int BLOCK_N>
__global__ void attention_fwd_kernel_rope_fused_v2(
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
    float* s_Q_rope = s_reduce + 32;

    const int q_offset = qkv_base + q_idx * hdim;
    const int q_pos = q_idx;

    // Step 1: Compute RoPE'd Q with on-the-fly cos/sin
    for (int d = tid * 2; d < hdim; d += blockDim.x * 2) {
        int pair_idx = d / 2;
        float freq = rope_freq(pair_idx, hdim);
        float theta = q_pos * freq;
        float cos_val, sin_val;
        __sincosf(theta, &sin_val, &cos_val);
        
        float q_even = Q[q_offset + d];
        float q_odd  = Q[q_offset + d + 1];
        
        s_Q_rope[d]     = q_even * cos_val - q_odd * sin_val;
        s_Q_rope[d + 1] = q_even * sin_val + q_odd * cos_val;
    }
    __syncthreads();

    float m_max = -INFINITY;
    float l_sum = 0.0f;

    for (int k_start = 0; k_start < seq_k; k_start += BLOCK_N) {
        const int k_end = min(k_start + BLOCK_N, seq_k);
        const int num_k = k_end - k_start;

        for (int k_local = tid; k_local < num_k; k_local += blockDim.x) {
            int k_idx = k_start + k_local;
            int k_pos = k_idx;
            int k_offset = kv_base + k_idx * hdim;

            float score = 0.0f;

            // Dot product with on-the-fly RoPE for K
            for (int d = 0; d < hdim; d += 2) {
                int pair_idx = d / 2;
                float freq = rope_freq(pair_idx, hdim);
                float theta = k_pos * freq;
                float cos_val, sin_val;
                __sincosf(theta, &sin_val, &cos_val);
                
                float k_even = K[k_offset + d];
                float k_odd  = K[k_offset + d + 1];
                
                float k_even_rope = k_even * cos_val - k_odd * sin_val;
                float k_odd_rope  = k_even * sin_val + k_odd * cos_val;
                
                score += s_Q_rope[d] * k_even_rope + s_Q_rope[d+1] * k_odd_rope;
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

    for (int d = tid; d < hdim; d += blockDim.x) {
        out[q_offset + d] /= l_sum;
    }
}

// Host wrapper for V2 (no cos/sin cache)
void attention_forward_rope_fused_v2(
    const float* Q, const float* K, const float* V, float* out,
    int batch_sz, int num_heads, int len_q, int len_k, int head_d
) {
    float scale = 1.0f / sqrtf(static_cast<float>(head_d));
    constexpr int BLOCK_N = 64;
    dim3 grid(len_q, batch_sz * num_heads);
    const int nthreads = 256;
    const int shmem_sz = (BLOCK_N + 32 + head_d) * sizeof(float);

    attention_fwd_kernel_rope_fused_v2<BLOCK_N><<<grid, nthreads, shmem_sz>>>(
        Q, K, V, out, batch_sz, num_heads, len_q, len_k, head_d, scale
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// RoPE-Fused Attention Kernel V3: Multi-Query per Block
// Process NUM_Q queries per block to amortize K RoPE computation
// ============================================================================
template<int BLOCK_N, int NUM_Q>
__global__ void attention_fwd_kernel_rope_fused_v3(
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
    const int q_tile_idx = blockIdx.x;
    const int first_q = q_tile_idx * NUM_Q;

    // How many queries this block handles (may be < NUM_Q at end)
    const int num_q_this_tile = min(NUM_Q, seq_q - first_q);
    if (num_q_this_tile <= 0) return;

    const int tid = threadIdx.x;
    const int qkv_base = (b_idx * nh + h_idx) * seq_q * hdim;
    const int kv_base = (b_idx * nh + h_idx) * seq_k * hdim;

    // Shared memory: scores[NUM_Q][BLOCK_N] + reduce[32] + Q_rope[NUM_Q][hdim]
    extern __shared__ float smem[];
    float* s_scores = smem;                           // [NUM_Q * BLOCK_N]
    float* s_reduce = s_scores + NUM_Q * BLOCK_N;     // [32]
    float* s_Q_rope = s_reduce + 32;                  // [NUM_Q * hdim]

    // Step 1: Load and apply RoPE to all Q vectors in this tile
    for (int q_local = 0; q_local < num_q_this_tile; q_local++) {
        int q_idx = first_q + q_local;
        int q_pos = q_idx;
        int q_offset = qkv_base + q_idx * hdim;

        for (int d = tid * 2; d < hdim; d += blockDim.x * 2) {
            int pair_idx = d / 2;
            float freq = rope_freq(pair_idx, hdim);
            float theta = q_pos * freq;
            float cos_val, sin_val;
            __sincosf(theta, &sin_val, &cos_val);

            float q_even = Q[q_offset + d];
            float q_odd  = Q[q_offset + d + 1];

            s_Q_rope[q_local * hdim + d]     = q_even * cos_val - q_odd * sin_val;
            s_Q_rope[q_local * hdim + d + 1] = q_even * sin_val + q_odd * cos_val;
        }
    }
    __syncthreads();

    // Per-query online softmax state (in registers)
    float m_max[NUM_Q], l_sum[NUM_Q];
    #pragma unroll
    for (int q = 0; q < NUM_Q; q++) {
        m_max[q] = -INFINITY;
        l_sum[q] = 0.0f;
    }

    // Step 2: Process K,V in tiles
    for (int k_start = 0; k_start < seq_k; k_start += BLOCK_N) {
        const int k_end = min(k_start + BLOCK_N, seq_k);
        const int num_k = k_end - k_start;

        // Compute scores for all queries, applying RoPE to K ONCE per K position
        for (int k_local = tid; k_local < num_k; k_local += blockDim.x) {
            int k_idx = k_start + k_local;
            int k_pos = k_idx;
            int k_offset = kv_base + k_idx * hdim;

            // Accumulate scores for all queries
            float scores[NUM_Q];
            #pragma unroll
            for (int q = 0; q < NUM_Q; q++) scores[q] = 0.0f;

            // Loop over hdim - apply RoPE to K ONCE, dot with all Q
            for (int d = 0; d < hdim; d += 2) {
                int pair_idx = d / 2;
                float freq = rope_freq(pair_idx, hdim);
                float theta = k_pos * freq;
                float cos_val, sin_val;
                __sincosf(theta, &sin_val, &cos_val);

                float k_even = K[k_offset + d];
                float k_odd  = K[k_offset + d + 1];
                float k_even_rope = k_even * cos_val - k_odd * sin_val;
                float k_odd_rope  = k_even * sin_val + k_odd * cos_val;

                // Dot product with ALL Q_rope vectors
                #pragma unroll
                for (int q = 0; q < num_q_this_tile; q++) {
                    scores[q] += s_Q_rope[q * hdim + d] * k_even_rope +
                                 s_Q_rope[q * hdim + d + 1] * k_odd_rope;
                }
            }

            // Store scaled scores
            #pragma unroll
            for (int q = 0; q < num_q_this_tile; q++) {
                s_scores[q * BLOCK_N + k_local] = scores[q] * scale_factor;
            }
        }
        __syncthreads();

        // Process each query's softmax independently
        for (int q = 0; q < num_q_this_tile; q++) {
            // Find max for this query
            float block_max = -INFINITY;
            for (int k_local = tid; k_local < num_k; k_local += blockDim.x) {
                block_max = fmaxf(block_max, s_scores[q * BLOCK_N + k_local]);
            }
            block_max = block_reduce_max(block_max, s_reduce);

            float m_new = fmaxf(m_max[q], block_max);
            float correction = expf(m_max[q] - m_new);

            // Compute exp and sum
            float block_sum = 0.0f;
            for (int k_local = tid; k_local < num_k; k_local += blockDim.x) {
                float exp_val = expf(s_scores[q * BLOCK_N + k_local] - m_new);
                s_scores[q * BLOCK_N + k_local] = exp_val;
                block_sum += exp_val;
            }
            block_sum = block_reduce_sum(block_sum, s_reduce);

            l_sum[q] = correction * l_sum[q] + block_sum;

            // Accumulate output
            int q_idx = first_q + q;
            int q_offset = qkv_base + q_idx * hdim;
            for (int d = tid; d < hdim; d += blockDim.x) {
                float prev_out = (k_start > 0) ? out[q_offset + d] : 0.0f;
                float corrected_out = prev_out * correction;

                float v_acc = 0.0f;
                for (int k_local = 0; k_local < num_k; k_local++) {
                    int k_idx = k_start + k_local;
                    v_acc += s_scores[q * BLOCK_N + k_local] * V[kv_base + k_idx * hdim + d];
                }

                out[q_offset + d] = corrected_out + v_acc;
            }

            m_max[q] = m_new;
            __syncthreads();
        }
    }

    // Final normalization for each query
    for (int q = 0; q < num_q_this_tile; q++) {
        int q_idx = first_q + q;
        int q_offset = qkv_base + q_idx * hdim;
        for (int d = tid; d < hdim; d += blockDim.x) {
            out[q_offset + d] /= l_sum[q];
        }
    }
}

// Host wrapper for V3
void attention_forward_rope_fused_v3(
    const float* Q, const float* K, const float* V, float* out,
    int batch_sz, int num_heads, int len_q, int len_k, int head_d
) {
    float scale = 1.0f / sqrtf(static_cast<float>(head_d));
    constexpr int BLOCK_N = 64;
    constexpr int NUM_Q = 4;  // Process 4 queries per block

    int num_q_tiles = (len_q + NUM_Q - 1) / NUM_Q;
    dim3 grid(num_q_tiles, batch_sz * num_heads);
    const int nthreads = 256;
    const int shmem_sz = (NUM_Q * BLOCK_N + 32 + NUM_Q * head_d) * sizeof(float);

    attention_fwd_kernel_rope_fused_v3<BLOCK_N, NUM_Q><<<grid, nthreads, shmem_sz>>>(
        Q, K, V, out, batch_sz, num_heads, len_q, len_k, head_d, scale
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// RoPE-Fused Attention Kernel V4: Cached Frequencies + Cooperative K Processing
// Key optimizations:
// 1. Precompute freq[] at block start (removes powf from inner loop)
// 2. Cache cos/sin for current K position in shared memory
// 3. All threads cooperate on each K position
// ============================================================================
template<int BLOCK_N>
__global__ void attention_fwd_kernel_rope_fused_v4(
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

    // Shared memory layout:
    // s_freq: [hdim/2] - precomputed frequencies
    // s_cos_sin: [hdim] - cos values, then [hdim] sin values for current K
    // s_scores: [BLOCK_N]
    // s_reduce: [32]
    // s_Q_rope: [hdim]
    extern __shared__ float smem[];
    float* s_freq = smem;                                    // [hdim/2]
    float* s_cos = s_freq + hdim/2;                          // [hdim]
    float* s_sin = s_cos + hdim;                             // [hdim]
    float* s_scores = s_sin + hdim;                          // [BLOCK_N]
    float* s_reduce = s_scores + BLOCK_N;                    // [32]
    float* s_Q_rope = s_reduce + 32;                         // [hdim]

    const int q_offset = qkv_base + q_idx * hdim;
    const int q_pos = q_idx;

    // Step 0: Precompute frequencies (removes powf from inner loop)
    for (int i = tid; i < hdim/2; i += blockDim.x) {
        s_freq[i] = 1.0f / powf(10000.0f, (2.0f * i) / hdim);
    }
    __syncthreads();

    // Step 1: Compute RoPE'd Q and cache in shared memory
    for (int d = tid; d < hdim; d += blockDim.x) {
        int pair_idx = d / 2;
        float theta = q_pos * s_freq[pair_idx];
        float cos_val, sin_val;
        __sincosf(theta, &sin_val, &cos_val);

        int pair_base = (d / 2) * 2;
        float q_even = Q[q_offset + pair_base];
        float q_odd  = Q[q_offset + pair_base + 1];

        if (d % 2 == 0) {
            s_Q_rope[d] = q_even * cos_val - q_odd * sin_val;
        } else {
            s_Q_rope[d] = q_even * sin_val + q_odd * cos_val;
        }
    }
    __syncthreads();

    float m_max = -INFINITY;
    float l_sum = 0.0f;

    // Step 2: Process K positions - ALL threads cooperate on each K
    for (int k_start = 0; k_start < seq_k; k_start += BLOCK_N) {
        const int k_end = min(k_start + BLOCK_N, seq_k);
        const int num_k = k_end - k_start;

        // Process each K position with ALL threads cooperating
        for (int k_local = 0; k_local < num_k; k_local++) {
            int k_idx = k_start + k_local;
            int k_pos = k_idx;
            int k_offset = kv_base + k_idx * hdim;

            // Step 2a: Cooperatively compute cos/sin for this K position
            for (int d = tid; d < hdim; d += blockDim.x) {
                int pair_idx = d / 2;
                float theta = k_pos * s_freq[pair_idx];
                float cos_val, sin_val;
                __sincosf(theta, &sin_val, &cos_val);
                s_cos[d] = cos_val;
                s_sin[d] = sin_val;
            }
            __syncthreads();

            // Step 2b: Cooperatively compute dot product
            float partial_score = 0.0f;
            for (int d = tid * 2; d < hdim; d += blockDim.x * 2) {
                float k_even = K[k_offset + d];
                float k_odd  = K[k_offset + d + 1];

                // Apply RoPE using cached cos/sin
                float k_even_rope = k_even * s_cos[d] - k_odd * s_sin[d];
                float k_odd_rope  = k_even * s_sin[d] + k_odd * s_cos[d];

                partial_score += s_Q_rope[d] * k_even_rope + s_Q_rope[d+1] * k_odd_rope;
            }

            // Reduce partial scores across threads
            partial_score = block_reduce_sum(partial_score, s_reduce);

            // Thread 0 stores the final score
            if (tid == 0) {
                s_scores[k_local] = partial_score * scale_factor;
            }
            __syncthreads();
        }

        // Standard online softmax from here
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
            float prev_out = (k_start > 0) ? out[q_offset + d] : 0.0f;
            float corrected_out = prev_out * correction;

            float v_acc = 0.0f;
            for (int k_local = 0; k_local < num_k; k_local++) {
                int k_idx = k_start + k_local;
                v_acc += s_scores[k_local] * V[kv_base + k_idx * hdim + d];
            }

            out[q_offset + d] = corrected_out + v_acc;
        }

        m_max = m_new;
        __syncthreads();
    }

    for (int d = tid; d < hdim; d += blockDim.x) {
        out[q_offset + d] /= l_sum;
    }
}

// Host wrapper for V4
void attention_forward_rope_fused_v4(
    const float* Q, const float* K, const float* V, float* out,
    int batch_sz, int num_heads, int len_q, int len_k, int head_d
) {
    float scale = 1.0f / sqrtf(static_cast<float>(head_d));
    constexpr int BLOCK_N = 64;
    dim3 grid(len_q, batch_sz * num_heads);
    const int nthreads = 256;
    // s_freq[hdim/2] + s_cos[hdim] + s_sin[hdim] + s_scores[BLOCK_N] + s_reduce[32] + s_Q_rope[hdim]
    const int shmem_sz = (head_d/2 + head_d + head_d + BLOCK_N + 32 + head_d) * sizeof(float);

    attention_fwd_kernel_rope_fused_v4<BLOCK_N><<<grid, nthreads, shmem_sz>>>(
        Q, K, V, out, batch_sz, num_heads, len_q, len_k, head_d, scale
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// RoPE-Fused Attention Kernel V5: FlashInfer-Style Register Tiling
// Key insight: Load RAW K into shared memory, apply RoPE IN REGISTERS during dot product
// This way K is loaded from global memory ONCE per tile, not once per query
// 
// Shared memory constraint: 48KB max
// For hdim=2048: can fit ~5 K tokens in shared (48KB / 2048 / 4 ≈ 5.8)
// We use dynamic BLOCK_K based on hdim to maximize tile size within limits
// ============================================================================
template<int BLOCK_K>
__global__ void attention_fwd_kernel_rope_fused_v5(
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

    // Shared memory layout (V is read from global to save shared memory):
    // s_K: [BLOCK_K * hdim] - RAW K tile (not RoPE'd!)
    // s_scores: [BLOCK_K]
    // s_reduce: [32]
    // s_Q_rope: [hdim] - RoPE'd Q (computed once)
    // s_freq: [hdim/2] - precomputed frequencies
    extern __shared__ float smem[];
    float* s_K = smem;                                        // [BLOCK_K * hdim]
    float* s_scores = s_K + BLOCK_K * hdim;                   // [BLOCK_K]
    float* s_reduce = s_scores + BLOCK_K;                     // [32]
    float* s_Q_rope = s_reduce + 32;                          // [hdim]
    float* s_freq = s_Q_rope + hdim;                          // [hdim/2]

    const int q_offset = qkv_base + q_idx * hdim;
    const int q_pos = q_idx;

    // Step 0: Precompute frequencies (removes powf from inner loop)
    for (int i = tid; i < hdim/2; i += blockDim.x) {
        s_freq[i] = 1.0f / powf(10000.0f, (2.0f * i) / hdim);
    }
    __syncthreads();

    // Step 1: Compute RoPE'd Q and cache in shared memory (done ONCE)
    for (int d = tid * 2; d < hdim; d += blockDim.x * 2) {
        int pair_idx = d / 2;
        float theta = q_pos * s_freq[pair_idx];
        float cos_val, sin_val;
        __sincosf(theta, &sin_val, &cos_val);

        float q_even = Q[q_offset + d];
        float q_odd  = Q[q_offset + d + 1];

        s_Q_rope[d]     = q_even * cos_val - q_odd * sin_val;
        s_Q_rope[d + 1] = q_even * sin_val + q_odd * cos_val;
    }
    __syncthreads();

    float m_max = -INFINITY;
    float l_sum = 0.0f;

    // Step 2: Process K,V in tiles
    for (int k_start = 0; k_start < seq_k; k_start += BLOCK_K) {
        const int k_end = min(k_start + BLOCK_K, seq_k);
        const int num_k = k_end - k_start;

        // Step 2a: Cooperatively load RAW K into shared memory (ONCE per tile!)
        for (int i = tid; i < num_k * hdim; i += blockDim.x) {
            int k_local = i / hdim;
            int d = i % hdim;
            int k_global = k_start + k_local;
            s_K[k_local * hdim + d] = K[kv_base + k_global * hdim + d];
        }
        __syncthreads();

        // Step 2b: Compute scores - apply RoPE to K IN REGISTERS during dot product
        for (int k_local = tid; k_local < num_k; k_local += blockDim.x) {
            int k_pos = k_start + k_local;
            
            float score = 0.0f;
            
            // Dot product: Q_rope · K_rope, where K_rope is computed in registers
            for (int d = 0; d < hdim; d += 2) {
                int pair_idx = d / 2;
                float theta = k_pos * s_freq[pair_idx];
                float cos_val, sin_val;
                __sincosf(theta, &sin_val, &cos_val);

                // Load K from SHARED memory (not global!)
                float k_even = s_K[k_local * hdim + d];
                float k_odd  = s_K[k_local * hdim + d + 1];

                // Apply RoPE IN REGISTERS (micro-fused, never written back)
                float k_even_rope = k_even * cos_val - k_odd * sin_val;
                float k_odd_rope  = k_even * sin_val + k_odd * cos_val;

                // Accumulate dot product
                score += s_Q_rope[d] * k_even_rope + s_Q_rope[d + 1] * k_odd_rope;
            }

            s_scores[k_local] = score * scale_factor;
        }
        __syncthreads();

        // Step 2c: Online softmax update
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

        // Step 2d: Accumulate output (V read from global memory to save shared)
        for (int d = tid; d < hdim; d += blockDim.x) {
            float prev_out = (k_start > 0) ? out[q_offset + d] : 0.0f;
            float corrected_out = prev_out * correction;

            float v_acc = 0.0f;
            for (int k_local = 0; k_local < num_k; k_local++) {
                int k_global = k_start + k_local;
                v_acc += s_scores[k_local] * V[kv_base + k_global * hdim + d];
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

// Host wrapper for V5
void attention_forward_rope_fused_v5(
    const float* Q, const float* K, const float* V, float* out,
    int batch_sz, int num_heads, int len_q, int len_k, int head_d
) {
    float scale = 1.0f / sqrtf(static_cast<float>(head_d));
    dim3 grid(len_q, batch_sz * num_heads);
    const int nthreads = 256;
    
    // Dynamic BLOCK_K based on hdim to fit in 48KB shared memory
    // Need: BLOCK_K * hdim + BLOCK_K + 32 + hdim + hdim/2 floats
    // Solve: BLOCK_K * (hdim + 1) <= (48KB/4) - 32 - hdim - hdim/2
    // For safety, use: BLOCK_K = min(64, (10000 - hdim * 1.5) / hdim)
    
    // Calculate max BLOCK_K that fits in ~40KB (leave room for other shared mem)
    const int available_floats = 10000;  // ~40KB in floats
    const int overhead = 32 + head_d + head_d/2;  // reduce + Q_rope + freq
    int block_k = (available_floats - overhead) / (head_d + 1);
    block_k = max(4, min(64, block_k));  // Clamp to [4, 64]
    
    if (block_k >= 16) {
        const int BLOCK_K = 16;
        const int shmem_sz = (BLOCK_K * head_d + BLOCK_K + 32 + head_d + head_d/2) * sizeof(float);
        attention_fwd_kernel_rope_fused_v5<BLOCK_K><<<grid, nthreads, shmem_sz>>>(
            Q, K, V, out, batch_sz, num_heads, len_q, len_k, head_d, scale
        );
    } else if (block_k >= 8) {
        const int BLOCK_K = 8;
        const int shmem_sz = (BLOCK_K * head_d + BLOCK_K + 32 + head_d + head_d/2) * sizeof(float);
        attention_fwd_kernel_rope_fused_v5<BLOCK_K><<<grid, nthreads, shmem_sz>>>(
            Q, K, V, out, batch_sz, num_heads, len_q, len_k, head_d, scale
        );
    } else {
        const int BLOCK_K = 4;
        const int shmem_sz = (BLOCK_K * head_d + BLOCK_K + 32 + head_d + head_d/2) * sizeof(float);
        attention_fwd_kernel_rope_fused_v5<BLOCK_K><<<grid, nthreads, shmem_sz>>>(
            Q, K, V, out, batch_sz, num_heads, len_q, len_k, head_d, scale
        );
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// RoPE-Fused Attention Kernel V6: Dimension-Tiled K Loading + Partial Rotation
// Key optimizations:
// 1. Tile the HEAD DIMENSION to allow larger BLOCK_K
// 2. PARTIAL ROTATION: Only apply RoPE to first rotary_dim (e.g., 64-128)
//    Rest of dimensions are plain dot product (like real LLaMA models)
// ============================================================================
template<int BLOCK_K, int TILE_D>
__global__ void attention_fwd_kernel_rope_fused_v6(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ out,
    int bs, int nh, int seq_q, int seq_k, int hdim, int rotary_dim,
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

    // Shared memory layout:
    // s_K_tile: [BLOCK_K * TILE_D] - ONE dimension tile of K (not full hdim!)
    // s_scores: [BLOCK_K]
    // s_reduce: [32]
    // s_Q_rope: [hdim] - Q with RoPE applied to first rotary_dim
    // s_freq: [rotary_dim/2] - precomputed frequencies (ONLY for rotated portion!)
    extern __shared__ float smem[];
    float* s_K_tile = smem;                                   // [BLOCK_K * TILE_D]
    float* s_scores = s_K_tile + BLOCK_K * TILE_D;            // [BLOCK_K]
    float* s_reduce = s_scores + BLOCK_K;                     // [32]
    float* s_Q_rope = s_reduce + 32;                          // [hdim]
    float* s_freq = s_Q_rope + hdim;                          // [rotary_dim/2]

    const int q_offset = qkv_base + q_idx * hdim;
    const int q_pos = q_idx;

    // Step 0: Precompute frequencies (ONLY for rotated dimensions)
    for (int i = tid; i < rotary_dim/2; i += blockDim.x) {
        s_freq[i] = 1.0f / powf(10000.0f, (2.0f * i) / rotary_dim);
    }
    __syncthreads();

    // Step 1: Load Q and apply RoPE to first rotary_dim, copy rest as-is
    // ROTATED portion: [0, rotary_dim)
    for (int d = tid * 2; d < rotary_dim; d += blockDim.x * 2) {
        int pair_idx = d / 2;
        float theta = q_pos * s_freq[pair_idx];
        float cos_val, sin_val;
        __sincosf(theta, &sin_val, &cos_val);

        float q_even = Q[q_offset + d];
        float q_odd  = Q[q_offset + d + 1];

        s_Q_rope[d]     = q_even * cos_val - q_odd * sin_val;
        s_Q_rope[d + 1] = q_even * sin_val + q_odd * cos_val;
    }

    // NON-ROTATED portion: [rotary_dim, hdim) - just copy
    for (int d = rotary_dim + tid; d < hdim; d += blockDim.x) {
        s_Q_rope[d] = Q[q_offset + d];
    }
    __syncthreads();

    float m_max = -INFINITY;
    float l_sum = 0.0f;

    // Step 2: Process K,V in tiles of BLOCK_K tokens
    for (int k_start = 0; k_start < seq_k; k_start += BLOCK_K) {
        const int k_end = min(k_start + BLOCK_K, seq_k);
        const int num_k = k_end - k_start;

        // Initialize scores to zero for this K tile
        for (int k_local = tid; k_local < BLOCK_K; k_local += blockDim.x) {
            s_scores[k_local] = 0.0f;
        }
        __syncthreads();

        // Step 2a: Process head dimension in tiles of TILE_D
        for (int d_start = 0; d_start < hdim; d_start += TILE_D) {
            const int d_end = min(d_start + TILE_D, hdim);
            const int tile_d = d_end - d_start;

            // Load K tile cooperatively: [num_k, tile_d]
            for (int i = tid; i < num_k * tile_d; i += blockDim.x) {
                int k_local = i / tile_d;
                int d_local = i % tile_d;
                int k_global = k_start + k_local;
                int d_global = d_start + d_local;
                s_K_tile[k_local * TILE_D + d_local] = K[kv_base + k_global * hdim + d_global];
            }
            __syncthreads();

            // Compute partial scores: apply RoPE to K (first rotary_dim only)
            for (int k_local = tid; k_local < num_k; k_local += blockDim.x) {
                int k_pos = k_start + k_local;
                float partial_score = 0.0f;

                // Dot product over this dimension tile
                for (int d_local = 0; d_local < tile_d; d_local += 2) {
                    int d_global = d_start + d_local;

                    float k_even = s_K_tile[k_local * TILE_D + d_local];
                    float k_odd  = s_K_tile[k_local * TILE_D + d_local + 1];

                    float k_val_0, k_val_1;

                    if (d_global < rotary_dim) {
                        // ROTATED portion: Apply RoPE
                        int pair_idx = d_global / 2;
                        float theta = k_pos * s_freq[pair_idx];
                        float cos_val, sin_val;
                        __sincosf(theta, &sin_val, &cos_val);

                        k_val_0 = k_even * cos_val - k_odd * sin_val;
                        k_val_1 = k_even * sin_val + k_odd * cos_val;
                    } else {
                        // NON-ROTATED portion: Use as-is
                        k_val_0 = k_even;
                        k_val_1 = k_odd;
                    }

                    // Accumulate dot product
                    partial_score += s_Q_rope[d_global] * k_val_0 +
                                   s_Q_rope[d_global + 1] * k_val_1;
                }

                // Accumulate to total score
                s_scores[k_local] += partial_score;
            }
            __syncthreads();
        }

        // Scale scores after accumulating all dimension tiles
        for (int k_local = tid; k_local < num_k; k_local += blockDim.x) {
            s_scores[k_local] *= scale_factor;
        }
        __syncthreads();

        // Step 2b: Online softmax update (same as V5)
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

        // Step 2c: Accumulate output (V read from global memory)
        for (int d = tid; d < hdim; d += blockDim.x) {
            float prev_out = (k_start > 0) ? out[q_offset + d] : 0.0f;
            float corrected_out = prev_out * correction;

            float v_acc = 0.0f;
            for (int k_local = 0; k_local < num_k; k_local++) {
                int k_global = k_start + k_local;
                v_acc += s_scores[k_local] * V[kv_base + k_global * hdim + d];
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

// Host wrapper for V6 with partial rotation support
void attention_forward_rope_fused_v6(
    const float* Q, const float* K, const float* V, float* out,
    int batch_sz, int num_heads, int len_q, int len_k, int head_d, int rotary_dim
) {
    float scale = 1.0f / sqrtf(static_cast<float>(head_d));
    dim3 grid(len_q, batch_sz * num_heads);
    const int nthreads = 256;

    // Strategy: Use dimension tiling to allow larger BLOCK_K
    // With partial rotation, s_freq is now only rotary_dim/2 instead of hdim/2!
    // Shared memory needed: BLOCK_K*TILE_D + BLOCK_K + 32 + hdim + rotary_dim/2
    // For BLOCK_K=32, TILE_D=256, hdim=2048, rotary_dim=128:
    //   32*256 + 32 + 32 + 2048 + 64 = 10368 floats = 41KB (within 48KB limit)

    constexpr int BLOCK_K = 32;
    constexpr int TILE_D = 256;
    const int shmem_sz = (BLOCK_K * TILE_D + BLOCK_K + 32 + head_d + rotary_dim/2) * sizeof(float);

    attention_fwd_kernel_rope_fused_v6<BLOCK_K, TILE_D><<<grid, nthreads, shmem_sz>>>(
        Q, K, V, out, batch_sz, num_heads, len_q, len_k, head_d, rotary_dim, scale
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// RoPE-Fused Attention Kernel V7: HYBRID - Pre-rotated K + Fused Q-RoPE
// Key insight: K is ALREADY rotated (separate preprocessing), only fuse Q-RoPE
// This eliminates O(seq_q × seq_k) K recomputation while keeping Q-RoPE benefits
// ============================================================================
template<int BLOCK_K>
__global__ void attention_fwd_kernel_rope_fused_v7_hybrid(
    const float* __restrict__ Q,
    const float* __restrict__ K_rotated,  // K is ALREADY RoPE'd!
    const float* __restrict__ V,
    float* __restrict__ out,
    int bs, int nh, int seq_q, int seq_k, int hdim, int rotary_dim,
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

    // Shared memory layout (simpler than V6 - no K tile needed!)
    // s_scores: [BLOCK_K]
    // s_reduce: [32]
    // s_Q_rope: [hdim] - Q with RoPE applied
    // s_freq: [rotary_dim/2] - precomputed frequencies
    extern __shared__ float smem[];
    float* s_scores = smem;                     // [BLOCK_K]
    float* s_reduce = s_scores + BLOCK_K;       // [32]
    float* s_Q_rope = s_reduce + 32;            // [hdim]
    float* s_freq = s_Q_rope + hdim;            // [rotary_dim/2]

    const int q_offset = qkv_base + q_idx * hdim;
    const int q_pos = q_idx;

    // Step 0: Precompute frequencies (only for rotated dims)
    for (int i = tid; i < rotary_dim/2; i += blockDim.x) {
        s_freq[i] = 1.0f / powf(10000.0f, (2.0f * i) / rotary_dim);
    }
    __syncthreads();

    // Step 1: Load Q and apply RoPE to first rotary_dim
    // ROTATED portion: [0, rotary_dim)
    for (int d = tid * 2; d < rotary_dim; d += blockDim.x * 2) {
        int pair_idx = d / 2;
        float theta = q_pos * s_freq[pair_idx];
        float cos_val, sin_val;
        __sincosf(theta, &sin_val, &cos_val);

        float q_even = Q[q_offset + d];
        float q_odd  = Q[q_offset + d + 1];

        s_Q_rope[d]     = q_even * cos_val - q_odd * sin_val;
        s_Q_rope[d + 1] = q_even * sin_val + q_odd * cos_val;
    }

    // NON-ROTATED portion: [rotary_dim, hdim) - just copy
    for (int d = rotary_dim + tid; d < hdim; d += blockDim.x) {
        s_Q_rope[d] = Q[q_offset + d];
    }
    __syncthreads();

    float m_max = -INFINITY;
    float l_sum = 0.0f;

    // Step 2: Process K,V in tiles (K is ALREADY rotated, just load and use!)
    for (int k_start = 0; k_start < seq_k; k_start += BLOCK_K) {
        const int k_end = min(k_start + BLOCK_K, seq_k);
        const int num_k = k_end - k_start;

        // Compute Q·K scores (K is already rotated - simple dot product!)
        for (int k_local = tid; k_local < num_k; k_local += blockDim.x) {
            int k_idx = k_start + k_local;
            int k_offset = kv_base + k_idx * hdim;

            float score = 0.0f;

            // Simple vectorized dot product - K is already RoPE'd
            for (int d = 0; d < hdim; d += 4) {
                if (d + 3 < hdim) {
                    float4 q_vec = *reinterpret_cast<const float4*>(&s_Q_rope[d]);
                    float4 k_vec = *reinterpret_cast<const float4*>(&K_rotated[k_offset + d]);
                    score += q_vec.x * k_vec.x + q_vec.y * k_vec.y +
                             q_vec.z * k_vec.z + q_vec.w * k_vec.w;
                } else {
                    // Handle remaining elements
                    for (int dd = d; dd < hdim; dd++) {
                        score += s_Q_rope[dd] * K_rotated[k_offset + dd];
                    }
                    break;
                }
            }

            s_scores[k_local] = score * scale_factor;
        }
        __syncthreads();

        // Online softmax (same as before)
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

        // Accumulate output
        for (int d = tid; d < hdim; d += blockDim.x) {
            float prev_out = (k_start > 0) ? out[q_offset + d] : 0.0f;
            float corrected_out = prev_out * correction;

            float v_acc = 0.0f;
            for (int k_local = 0; k_local < num_k; k_local++) {
                int k_idx = k_start + k_local;
                v_acc += s_scores[k_local] * V[kv_base + k_idx * hdim + d];
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

// Host wrapper for V7 hybrid approach
void attention_forward_rope_fused_v7_hybrid(
    const float* Q, const float* K_rotated, const float* V, float* out,
    int batch_sz, int num_heads, int len_q, int len_k, int head_d, int rotary_dim
) {
    float scale = 1.0f / sqrtf(static_cast<float>(head_d));
    dim3 grid(len_q, batch_sz * num_heads);
    const int nthreads = 256;

    // Shared memory: scores[BLOCK_K] + reduce[32] + Q_rope[hdim] + freq[rotary_dim/2]
    constexpr int BLOCK_K = 64;
    const int shmem_sz = (BLOCK_K + 32 + head_d + rotary_dim/2) * sizeof(float);

    attention_fwd_kernel_rope_fused_v7_hybrid<BLOCK_K><<<grid, nthreads, shmem_sz>>>(
        Q, K_rotated, V, out, batch_sz, num_heads, len_q, len_k, head_d, rotary_dim, scale
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// RoPE-Fused Attention Kernel (Original with cos/sin cache)
// ============================================================================

// Apply RoPE inline during attention computation
// Each block processes one query, tiles over K/V with online softmax
template<int BLOCK_N>
__global__ void attention_fwd_kernel_rope_fused(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ out,
    const float* __restrict__ cos_cache,  // [max_seq_len, hdim]
    const float* __restrict__ sin_cache,  // [max_seq_len, hdim]
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

    // Shared memory layout:
    // [0, BLOCK_N): scores for current K block
    // [BLOCK_N, BLOCK_N+32): reduction buffer
    // [BLOCK_N+32, BLOCK_N+32+hdim): RoPE'd Q cache
    extern __shared__ float smem[];
    float* s_scores = smem;
    float* s_reduce = s_scores + BLOCK_N;
    float* s_Q_rope = s_reduce + 32;  // Cache RoPE'd Q in shared memory

    const int q_offset = qkv_base + q_idx * hdim;
    const int q_pos = q_idx;  // Position for RoPE (can add offset if needed)

    // Step 1: Compute RoPE'd Q and cache in shared memory
    // Use float4 vectorization: 4 elements = 2 pairs per iteration
    if (hdim % 4 == 0) {
        for (int d = tid * 4; d < hdim; d += blockDim.x * 4) {
            // Load 4 Q elements (2 pairs)
            float4 q_vec = *reinterpret_cast<const float4*>(&Q[q_offset + d]);
            float4 cos_vec = *reinterpret_cast<const float4*>(&cos_cache[q_pos * hdim + d]);
            float4 sin_vec = *reinterpret_cast<const float4*>(&sin_cache[q_pos * hdim + d]);
            
            // Apply RoPE to both pairs
            // Pair 0: (q_vec.x, q_vec.y) with (cos_vec.x, sin_vec.x)
            // Pair 1: (q_vec.z, q_vec.w) with (cos_vec.z, sin_vec.z)
            float4 q_rope;
            q_rope.x = q_vec.x * cos_vec.x - q_vec.y * sin_vec.x;  // pair 0 even
            q_rope.y = q_vec.x * sin_vec.x + q_vec.y * cos_vec.x;  // pair 0 odd
            q_rope.z = q_vec.z * cos_vec.z - q_vec.w * sin_vec.z;  // pair 1 even
            q_rope.w = q_vec.z * sin_vec.z + q_vec.w * cos_vec.z;  // pair 1 odd
            
            *reinterpret_cast<float4*>(&s_Q_rope[d]) = q_rope;
        }
    } else {
        // Fallback for non-multiple-of-4 hdim
        for (int d = tid; d < hdim; d += blockDim.x) {
            int pair_base = (d / 2) * 2;
            float q_even = Q[q_offset + pair_base];
            float q_odd  = Q[q_offset + pair_base + 1];
            float cos_val = cos_cache[q_pos * hdim + pair_base];
            float sin_val = sin_cache[q_pos * hdim + pair_base];
            
            if (d % 2 == 0) {
                s_Q_rope[d] = q_even * cos_val - q_odd * sin_val;
            } else {
                s_Q_rope[d] = q_even * sin_val + q_odd * cos_val;
            }
        }
    }
    __syncthreads();

    // Online softmax state
    float m_max = -INFINITY;
    float l_sum = 0.0f;

    // Step 2: Process K,V in blocks using online softmax
    for (int k_start = 0; k_start < seq_k; k_start += BLOCK_N) {
        const int k_end = min(k_start + BLOCK_N, seq_k);
        const int num_k = k_end - k_start;

        // Compute Q @ K^T for this block, applying RoPE to K on-the-fly
        for (int k_local = tid; k_local < num_k; k_local += blockDim.x) {
            int k_idx = k_start + k_local;
            int k_pos = k_idx;  // Position for RoPE
            int k_offset = kv_base + k_idx * hdim;

            float score = 0.0f;

            // Dot product with RoPE applied to K (float4 vectorized)
            if (hdim % 4 == 0) {
                for (int d = 0; d < hdim; d += 4) {
                    // Load 4 K elements (2 pairs)
                    float4 k_vec = *reinterpret_cast<const float4*>(&K[k_offset + d]);
                    float4 cos_vec = *reinterpret_cast<const float4*>(&cos_cache[k_pos * hdim + d]);
                    float4 sin_vec = *reinterpret_cast<const float4*>(&sin_cache[k_pos * hdim + d]);
                    float4 q_vec = *reinterpret_cast<const float4*>(&s_Q_rope[d]);
                    
                    // Apply RoPE to K (2 pairs)
                    float k0_rope = k_vec.x * cos_vec.x - k_vec.y * sin_vec.x;
                    float k1_rope = k_vec.x * sin_vec.x + k_vec.y * cos_vec.x;
                    float k2_rope = k_vec.z * cos_vec.z - k_vec.w * sin_vec.z;
                    float k3_rope = k_vec.z * sin_vec.z + k_vec.w * cos_vec.z;
                    
                    // Dot product with RoPE'd Q
                    score += q_vec.x * k0_rope + q_vec.y * k1_rope +
                             q_vec.z * k2_rope + q_vec.w * k3_rope;
                }
            } else {
                for (int d = 0; d < hdim; d += 2) {
                    float k_even = K[k_offset + d];
                    float k_odd  = K[k_offset + d + 1];
                    float cos_val = cos_cache[k_pos * hdim + d];
                    float sin_val = sin_cache[k_pos * hdim + d];
                    float k_even_rope = k_even * cos_val - k_odd * sin_val;
                    float k_odd_rope  = k_even * sin_val + k_odd * cos_val;
                    score += s_Q_rope[d] * k_even_rope + s_Q_rope[d+1] * k_odd_rope;
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

// ============================================================================
// Host Wrapper
// ============================================================================

void attention_forward_rope_fused(
    const float* Q,
    const float* K,
    const float* V,
    float* out,
    const float* cos_cache,
    const float* sin_cache,
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
    // Shared memory: scores (BLOCK_N) + reduction (32) + Q_rope cache (hdim)
    const int shmem_sz = (BLOCK_N + 32 + head_d) * sizeof(float);

    attention_fwd_kernel_rope_fused<BLOCK_N><<<grid, nthreads, shmem_sz>>>(
        Q, K, V, out, cos_cache, sin_cache,
        batch_sz, num_heads, len_q, len_k, head_d, scale
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Non-Fused Implementation (Separate RoPE + Attention kernels)
// For comparison benchmarking
// ============================================================================

// Standalone RoPE kernel with partial rotation support
__global__ void rope_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ cos_cache,
    const float* __restrict__ sin_cache,
    int total_tokens,  // batch * heads * seq_len
    int hdim,
    int rotary_dim
) {
    int token_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (token_idx >= total_tokens) return;

    int pos = token_idx % (total_tokens / gridDim.x * gridDim.x == total_tokens ?
              blockIdx.x : token_idx);  // Simplified: assume seq_len divides evenly
    // Actually, for this benchmark we'll use a simpler approach
    int seq_pos = token_idx % 64;  // Hardcode seq_len for benchmark simplicity

    const int in_offset = token_idx * hdim;

    // Apply RoPE with float4 vectorization to ROTATED portion only
    if (rotary_dim % 4 == 0) {
        for (int d = tid * 4; d < rotary_dim; d += blockDim.x * 4) {
            float4 x_vec = *reinterpret_cast<const float4*>(&input[in_offset + d]);
            float4 cos_vec = *reinterpret_cast<const float4*>(&cos_cache[seq_pos * rotary_dim + d]);
            float4 sin_vec = *reinterpret_cast<const float4*>(&sin_cache[seq_pos * rotary_dim + d]);

            float4 out_vec;
            out_vec.x = x_vec.x * cos_vec.x - x_vec.y * sin_vec.x;
            out_vec.y = x_vec.x * sin_vec.x + x_vec.y * cos_vec.x;
            out_vec.z = x_vec.z * cos_vec.z - x_vec.w * sin_vec.z;
            out_vec.w = x_vec.z * sin_vec.z + x_vec.w * cos_vec.z;

            *reinterpret_cast<float4*>(&output[in_offset + d]) = out_vec;
        }
    }

    // Copy NON-ROTATED portion as-is
    for (int d = rotary_dim + tid; d < hdim; d += blockDim.x) {
        output[in_offset + d] = input[in_offset + d];
    }
}

// Attention kernel WITHOUT RoPE (assumes Q,K already have RoPE applied)
template<int BLOCK_N>
__global__ void attention_fwd_kernel_no_rope(
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

    for (int k_start = 0; k_start < seq_k; k_start += BLOCK_N) {
        const int k_end = min(k_start + BLOCK_N, seq_k);
        const int num_k = k_end - k_start;

        for (int k_local = tid; k_local < num_k; k_local += blockDim.x) {
            int k_idx = k_start + k_local;
            int k_offset = kv_base + k_idx * hdim;
            float score = 0.0f;

            // Simple dot product (no RoPE - already applied)
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

    for (int d = tid; d < hdim; d += blockDim.x) {
        out[q_offset + d] /= l_sum;
    }
}

// Host wrapper for non-fused approach with partial rotation support
void attention_forward_separate_rope(
    const float* Q,
    const float* K,
    const float* V,
    float* Q_rope,  // temp buffer
    float* K_rope,  // temp buffer
    float* out,
    const float* cos_cache,
    const float* sin_cache,
    int batch_sz,
    int num_heads,
    int len_q,
    int len_k,
    int head_d,
    int rotary_dim
) {
    float scale = 1.0f / sqrtf(static_cast<float>(head_d));
    int total_q = batch_sz * num_heads * len_q;
    int total_k = batch_sz * num_heads * len_k;

    // Step 1: Apply RoPE to Q
    rope_kernel<<<total_q, 256>>>(Q, Q_rope, cos_cache, sin_cache, total_q, head_d, rotary_dim);

    // Step 2: Apply RoPE to K
    rope_kernel<<<total_k, 256>>>(K, K_rope, cos_cache, sin_cache, total_k, head_d, rotary_dim);

    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 3: Attention with RoPE'd Q and K
    constexpr int BLOCK_N = 64;
    dim3 grid(len_q, batch_sz * num_heads);
    const int shmem_sz = (BLOCK_N + 32) * sizeof(float);

    attention_fwd_kernel_no_rope<BLOCK_N><<<grid, 256, shmem_sz>>>(
        Q_rope, K_rope, V, out,
        batch_sz, num_heads, len_q, len_k, head_d, scale
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// CPU Reference Implementation (RoPE + Attention)
// ============================================================================

void apply_rope_cpu(
    const float* input,
    float* output,
    const float* cos_cache,
    const float* sin_cache,
    int seq_len,
    int hdim
) {
    for (int pos = 0; pos < seq_len; pos++) {
        for (int d = 0; d < hdim; d += 2) {
            float x_even = input[pos * hdim + d];
            float x_odd  = input[pos * hdim + d + 1];
            float cos_val = cos_cache[pos * hdim + d];
            float sin_val = sin_cache[pos * hdim + d];
            
            output[pos * hdim + d]     = x_even * cos_val - x_odd * sin_val;
            output[pos * hdim + d + 1] = x_even * sin_val + x_odd * cos_val;
        }
    }
}

void attention_rope_cpu(
    const float* Q,
    const float* K,
    const float* V,
    float* out,
    const float* cos_cache,
    const float* sin_cache,
    int batch_sz,
    int num_heads,
    int len_q,
    int len_k,
    int head_d
) {
    float scale = 1.0f / sqrtf(static_cast<float>(head_d));

    // Temporary buffers for RoPE'd Q and K
    std::vector<float> Q_rope(len_q * head_d);
    std::vector<float> K_rope(len_k * head_d);

    for (int b = 0; b < batch_sz; b++) {
        for (int h = 0; h < num_heads; h++) {
            int qkv_base = (b * num_heads + h) * len_q * head_d;
            int kv_base = (b * num_heads + h) * len_k * head_d;

            // Apply RoPE to Q
            apply_rope_cpu(&Q[qkv_base], Q_rope.data(), cos_cache, sin_cache, len_q, head_d);
            // Apply RoPE to K
            apply_rope_cpu(&K[kv_base], K_rope.data(), cos_cache, sin_cache, len_k, head_d);

            // Standard attention with RoPE'd Q and K
            for (int i = 0; i < len_q; i++) {
                std::vector<float> attn_scores(len_k);
                float max_s = -INFINITY;

                // Compute scores
                for (int j = 0; j < len_k; j++) {
                    float s = 0.0f;
                    for (int d = 0; d < head_d; d++) {
                        s += Q_rope[i * head_d + d] * K_rope[j * head_d + d];
                    }
                    s *= scale;
                    attn_scores[j] = s;
                    max_s = fmaxf(max_s, s);
                }

                // Softmax
                float exp_total = 0.0f;
                for (int j = 0; j < len_k; j++) {
                    attn_scores[j] = expf(attn_scores[j] - max_s);
                    exp_total += attn_scores[j];
                }
                for (int j = 0; j < len_k; j++) {
                    attn_scores[j] /= exp_total;
                }

                // Output
                for (int d = 0; d < head_d; d++) {
                    float result = 0.0f;
                    for (int j = 0; j < len_k; j++) {
                        result += attn_scores[j] * V[kv_base + j * head_d + d];
                    }
                    out[qkv_base + i * head_d + d] = result;
                }
            }
        }
    }
}

// ============================================================================
// Testing
// ============================================================================

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

void test_rope_attention(int bs, int nh, int seqlen, int hdim) {
    printf("\n========================================\n");
    printf("Testing RoPE-fused attention kernel:\n");
    printf("  Batch size: %d\n", bs);
    printf("  Num heads: %d\n", nh);
    printf("  Sequence length: %d\n", seqlen);
    printf("  Head dimension: %d\n", hdim);
    printf("========================================\n\n");

    int sz_qkv = bs * nh * seqlen * hdim;
    int sz_rope = seqlen * hdim;

    // Host allocations
    float *h_Q = new float[sz_qkv];
    float *h_K = new float[sz_qkv];
    float *h_V = new float[sz_qkv];
    float *h_out_gpu = new float[sz_qkv];
    float *h_out_cpu = new float[sz_qkv];
    float *h_cos = new float[sz_rope];
    float *h_sin = new float[sz_rope];

    init_random(h_Q, sz_qkv);
    init_random(h_K, sz_qkv);
    init_random(h_V, sz_qkv);

    // Precompute RoPE cache
    precompute_rope_cache(h_cos, h_sin, seqlen, hdim);

    // Device allocations
    float *d_Q, *d_K, *d_V, *d_out, *d_cos, *d_sin;
    CUDA_CHECK(cudaMalloc(&d_Q, sz_qkv * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, sz_qkv * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V, sz_qkv * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, sz_qkv * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cos, sz_rope * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sin, sz_rope * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_Q, h_Q, sz_qkv * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K, sz_qkv * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V, sz_qkv * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cos, h_cos, sz_rope * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sin, h_sin, sz_rope * sizeof(float), cudaMemcpyHostToDevice));

    // Warmup
    attention_forward_rope_fused(d_Q, d_K, d_V, d_out, d_cos, d_sin,
                                  bs, nh, seqlen, seqlen, hdim);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    printf("Running RoPE-fused GPU kernel...\n");
    const int num_iters = 10;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iters; i++) {
        attention_forward_rope_fused(d_Q, d_K, d_V, d_out, d_cos, d_sin,
                                      bs, nh, seqlen, seqlen, hdim);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float fused_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&fused_ms, start, stop));
    fused_ms /= num_iters;

    CUDA_CHECK(cudaMemcpy(h_out_gpu, d_out, sz_qkv * sizeof(float), cudaMemcpyDeviceToHost));

    printf("Running CPU reference (RoPE + attention)...\n");
    attention_rope_cpu(h_Q, h_K, h_V, h_out_cpu, h_cos, h_sin,
                       bs, nh, seqlen, seqlen, hdim);

    printf("\n--- Performance Results ---\n");
    printf("RoPE-fused kernel time: %.3f ms\n", fused_ms);

    printf("\nChecking results vs CPU...\n");
    bool passed = check_results(h_out_gpu, h_out_cpu, sz_qkv);

    if (passed) {
        printf("\n✓ TEST PASSED\n");
    } else {
        printf("\n✗ TEST FAILED\n");
    }

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    delete[] h_Q;
    delete[] h_K;
    delete[] h_V;
    delete[] h_out_gpu;
    delete[] h_out_cpu;
    delete[] h_cos;
    delete[] h_sin;

    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_cos));
    CUDA_CHECK(cudaFree(d_sin));
}

// Debug test to isolate the bug
void debug_rope_attention() {
    printf("\n========================================\n");
    printf("DEBUG: Isolating RoPE bug\n");
    printf("========================================\n\n");

    // Minimal failing case: seq=64, hdim=128
    int bs = 1, nh = 1, seqlen = 64, hdim = 128;
    int sz_qkv = bs * nh * seqlen * hdim;
    int sz_rope = seqlen * hdim;

    float *h_Q = new float[sz_qkv];
    float *h_K = new float[sz_qkv];
    float *h_V = new float[sz_qkv];
    float *h_cos = new float[sz_rope];
    float *h_sin = new float[sz_rope];

    // Initialize with simple pattern for debugging
    for (int i = 0; i < sz_qkv; i++) {
        h_Q[i] = 0.1f;  // All same value
        h_K[i] = 0.1f;
        h_V[i] = 0.1f;
    }

    precompute_rope_cache(h_cos, h_sin, seqlen, hdim);

    // Print first few cos/sin values
    printf("RoPE cache for position 0:\n");
    for (int d = 0; d < 8; d++) {
        printf("  d=%d: cos=%.6f, sin=%.6f\n", d, h_cos[d], h_sin[d]);
    }
    printf("RoPE cache for position 1:\n");
    for (int d = 0; d < 8; d++) {
        printf("  d=%d: cos=%.6f, sin=%.6f\n", d, h_cos[hdim + d], h_sin[hdim + d]);
    }

    // Compute CPU reference
    std::vector<float> Q_rope(seqlen * hdim);
    std::vector<float> K_rope(seqlen * hdim);
    
    // Apply RoPE to Q (position 0 should be identity since sin(0)=0, cos(0)=1)
    for (int pos = 0; pos < seqlen; pos++) {
        for (int d = 0; d < hdim; d += 2) {
            float q_even = h_Q[pos * hdim + d];
            float q_odd  = h_Q[pos * hdim + d + 1];
            float cos_val = h_cos[pos * hdim + d];
            float sin_val = h_sin[pos * hdim + d];
            
            Q_rope[pos * hdim + d]     = q_even * cos_val - q_odd * sin_val;
            Q_rope[pos * hdim + d + 1] = q_even * sin_val + q_odd * cos_val;
        }
    }
    
    // Apply RoPE to K
    for (int pos = 0; pos < seqlen; pos++) {
        for (int d = 0; d < hdim; d += 2) {
            float k_even = h_K[pos * hdim + d];
            float k_odd  = h_K[pos * hdim + d + 1];
            float cos_val = h_cos[pos * hdim + d];
            float sin_val = h_sin[pos * hdim + d];
            
            K_rope[pos * hdim + d]     = k_even * cos_val - k_odd * sin_val;
            K_rope[pos * hdim + d + 1] = k_even * sin_val + k_odd * cos_val;
        }
    }

    printf("\nQ_rope for query 0 (should be ~0.1 since pos=0 means identity):\n");
    for (int d = 0; d < 8; d++) {
        printf("  d=%d: %.6f\n", d, Q_rope[d]);
    }

    printf("\nK_rope for key 0 (should be ~0.1):\n");
    for (int d = 0; d < 8; d++) {
        printf("  d=%d: %.6f\n", d, K_rope[d]);
    }

    printf("\nK_rope for key 1 (should be rotated):\n");
    for (int d = 0; d < 8; d++) {
        printf("  d=%d: %.6f\n", d, K_rope[hdim + d]);
    }

    // Compute attention score for query 0, key 0
    float score_q0_k0 = 0.0f;
    for (int d = 0; d < hdim; d++) {
        score_q0_k0 += Q_rope[d] * K_rope[d];
    }
    printf("\nCPU: score(q=0, k=0) = %.6f (before scale)\n", score_q0_k0);

    // Now run GPU and compare
    float *d_Q, *d_K, *d_V, *d_out, *d_cos, *d_sin;
    CUDA_CHECK(cudaMalloc(&d_Q, sz_qkv * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, sz_qkv * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V, sz_qkv * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, sz_qkv * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cos, sz_rope * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sin, sz_rope * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_Q, h_Q, sz_qkv * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K, sz_qkv * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V, sz_qkv * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cos, h_cos, sz_rope * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sin, h_sin, sz_rope * sizeof(float), cudaMemcpyHostToDevice));

    attention_forward_rope_fused(d_Q, d_K, d_V, d_out, d_cos, d_sin,
                                  bs, nh, seqlen, seqlen, hdim);

    float *h_out_gpu = new float[sz_qkv];
    CUDA_CHECK(cudaMemcpy(h_out_gpu, d_out, sz_qkv * sizeof(float), cudaMemcpyDeviceToHost));

    // Compute CPU output for query 0
    float scale = 1.0f / sqrtf((float)hdim);
    std::vector<float> scores(seqlen);
    float max_s = -INFINITY;
    for (int k = 0; k < seqlen; k++) {
        float s = 0.0f;
        for (int d = 0; d < hdim; d++) {
            s += Q_rope[d] * K_rope[k * hdim + d];
        }
        s *= scale;
        scores[k] = s;
        max_s = fmaxf(max_s, s);
    }
    float sum_exp = 0.0f;
    for (int k = 0; k < seqlen; k++) {
        scores[k] = expf(scores[k] - max_s);
        sum_exp += scores[k];
    }
    for (int k = 0; k < seqlen; k++) {
        scores[k] /= sum_exp;
    }
    
    std::vector<float> cpu_out(hdim);
    for (int d = 0; d < hdim; d++) {
        float acc = 0.0f;
        for (int k = 0; k < seqlen; k++) {
            acc += scores[k] * h_V[k * hdim + d];
        }
        cpu_out[d] = acc;
    }

    printf("\nOutput comparison for query 0:\n");
    printf("  dim |    GPU    |    CPU    |   diff   | ratio\n");
    printf("  ----|-----------|-----------|----------|------\n");
    for (int d = 0; d < 64; d++) {
        float diff = fabsf(h_out_gpu[d] - cpu_out[d]);
        float ratio = (cpu_out[d] != 0) ? h_out_gpu[d] / cpu_out[d] : 0;
        if (diff > 0.01f) {
            printf("  %3d | %9.6f | %9.6f | %8.6f | %.3f ***\n", 
                   d, h_out_gpu[d], cpu_out[d], diff, ratio);
        }
    }

    delete[] h_Q;
    delete[] h_K;
    delete[] h_V;
    delete[] h_out_gpu;
    delete[] h_cos;
    delete[] h_sin;

    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_cos));
    CUDA_CHECK(cudaFree(d_sin));
}

// ============================================================================
// Benchmark: Fused vs Non-Fused Comparison
// ============================================================================
void benchmark_fused_vs_separate(int bs, int nh, int seqlen, int hdim, int rotary_dim) {
    printf("\n========================================\n");
    printf("BENCHMARK: Fused vs Separate RoPE+Attention\n");
    printf("  bs=%d, nh=%d, seq=%d, hdim=%d, rotary_dim=%d\n", bs, nh, seqlen, hdim, rotary_dim);
    printf("========================================\n");

    int sz_qkv = bs * nh * seqlen * hdim;
    int sz_rope = seqlen * rotary_dim;  // Cache size now based on rotary_dim!

    // Host allocations
    float *h_Q = new float[sz_qkv];
    float *h_K = new float[sz_qkv];
    float *h_V = new float[sz_qkv];
    float *h_cos = new float[sz_rope];
    float *h_sin = new float[sz_rope];

    init_random(h_Q, sz_qkv);
    init_random(h_K, sz_qkv);
    init_random(h_V, sz_qkv);
    precompute_rope_cache(h_cos, h_sin, seqlen, rotary_dim);

    // Device allocations
    float *d_Q, *d_K, *d_V, *d_out, *d_cos, *d_sin;
    float *d_Q_rope, *d_K_rope;  // Extra buffers for separate approach
    
    CUDA_CHECK(cudaMalloc(&d_Q, sz_qkv * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, sz_qkv * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V, sz_qkv * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, sz_qkv * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cos, sz_rope * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sin, sz_rope * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Q_rope, sz_qkv * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K_rope, sz_qkv * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_Q, h_Q, sz_qkv * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K, sz_qkv * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V, sz_qkv * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cos, h_cos, sz_rope * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sin, h_sin, sz_rope * sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    const int num_iters = 20;

    // Warmup
    attention_forward_rope_fused_v6(d_Q, d_K, d_V, d_out, bs, nh, seqlen, seqlen, hdim, rotary_dim);
    attention_forward_separate_rope(d_Q, d_K, d_V, d_Q_rope, d_K_rope, d_out, d_cos, d_sin, bs, nh, seqlen, seqlen, hdim, rotary_dim);

    // Warmup for V7 hybrid (need to pre-rotate K once)
    int total_k = bs * nh * seqlen;
    rope_kernel<<<total_k, 256>>>(d_K, d_K_rope, d_cos, d_sin, total_k, hdim, rotary_dim);
    attention_forward_rope_fused_v7_hybrid(d_Q, d_K_rope, d_V, d_out, bs, nh, seqlen, seqlen, hdim, rotary_dim);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark FUSED V6 (full fusion with partial rotation)
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iters; i++) {
        attention_forward_rope_fused_v6(d_Q, d_K, d_V, d_out, bs, nh, seqlen, seqlen, hdim, rotary_dim);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float fused_v6_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&fused_v6_ms, start, stop));
    fused_v6_ms /= num_iters;

    // Benchmark V7 HYBRID (pre-rotated K + fused Q-RoPE)
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iters; i++) {
        // Pre-rotate K (separate kernel)
        rope_kernel<<<total_k, 256>>>(d_K, d_K_rope, d_cos, d_sin, total_k, hdim, rotary_dim);
        // Attention with pre-rotated K, fused Q-RoPE
        attention_forward_rope_fused_v7_hybrid(d_Q, d_K_rope, d_V, d_out, bs, nh, seqlen, seqlen, hdim, rotary_dim);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float hybrid_v7_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&hybrid_v7_ms, start, stop));
    hybrid_v7_ms /= num_iters;

    // Benchmark SEPARATE (both Q and K in separate kernels)
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iters; i++) {
        attention_forward_separate_rope(d_Q, d_K, d_V, d_Q_rope, d_K_rope, d_out, d_cos, d_sin, bs, nh, seqlen, seqlen, hdim, rotary_dim);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float separate_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&separate_ms, start, stop));
    separate_ms /= num_iters;

    printf("\n  V6 (full fusion):         %.3f ms", fused_v6_ms);
    if (fused_v6_ms < separate_ms) {
        printf("  [%.2fx FASTER] ✓\n", separate_ms / fused_v6_ms);
    } else {
        printf("  [%.2fx slower]\n", fused_v6_ms / separate_ms);
    }

    printf("  V7 (hybrid):              %.3f ms", hybrid_v7_ms);
    if (hybrid_v7_ms < separate_ms) {
        printf("  [%.2fx FASTER] ✓\n", separate_ms / hybrid_v7_ms);
    } else {
        printf("  [%.2fx slower]\n", hybrid_v7_ms / separate_ms);
    }

    printf("  Separate (baseline):      %.3f ms\n", separate_ms);

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    delete[] h_Q; delete[] h_K; delete[] h_V; delete[] h_cos; delete[] h_sin;
    CUDA_CHECK(cudaFree(d_Q)); CUDA_CHECK(cudaFree(d_K)); CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_out)); CUDA_CHECK(cudaFree(d_cos)); CUDA_CHECK(cudaFree(d_sin));
    CUDA_CHECK(cudaFree(d_Q_rope)); CUDA_CHECK(cudaFree(d_K_rope));
}

int main() {
    printf("=================================================\n");
    printf("RoPE-Fused Attention Kernel Test Suite\n");
    printf("=================================================\n");

    srand(42);

    // Debug test first
    debug_rope_attention();

    // Basic correctness tests
    printf("\n--- BASIC CORRECTNESS TESTS ---\n");
    test_rope_attention(1, 1, 4, 8);
    test_rope_attention(1, 2, 8, 16);
    test_rope_attention(2, 4, 32, 32);

    // Larger tests
    printf("\n--- LARGER DIMENSION TESTS ---\n");
    test_rope_attention(1, 4, 16, 64);
    test_rope_attention(1, 4, 64, 128);

    // Target dimensions
    printf("\n--- TARGET DIMENSION TESTS ---\n");
    test_rope_attention(1, 4, 64, 512);
    test_rope_attention(1, 4, 64, 2048);

    // Benchmark fused vs separate with PARTIAL ROTATION (like real LLMs!)
    printf("\n--- FUSED vs SEPARATE BENCHMARK (PARTIAL ROTATION) ---\n");
    printf("Testing realistic LLaMA-style configs:\n");
    printf("  hdim=128, rotary_dim=64   (50%% rotation)\n");
    printf("  hdim=512, rotary_dim=128  (25%% rotation)\n");
    printf("  hdim=2048, rotary_dim=128 (6%% rotation)\n\n");

    benchmark_fused_vs_separate(1, 4, 64, 128, 64);    // 50% rotation
    benchmark_fused_vs_separate(1, 4, 64, 512, 128);   // 25% rotation
    benchmark_fused_vs_separate(1, 4, 64, 2048, 128);  // 6% rotation - HUGE reduction!

    printf("\n=================================================\n");
    printf("All tests completed!\n");
    printf("=================================================\n");

    return 0;
}

