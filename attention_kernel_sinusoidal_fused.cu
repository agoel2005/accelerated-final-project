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
        if (lane == 0) shared[0] = val;
    }
    __syncthreads();

    return shared[0];
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
        if (lane == 0) shared[0] = val;
    }
    __syncthreads();

    return shared[0];
}

// Version 1: Fused kernel with sinusoidal embeddings
// Baseline approach: precomputed sin/cos tables
template<int BLOCK_N>
__global__ void attention_fwd_kernel_sinusoidal_fused(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ sin_table,  // [max_seq_len, hdim]
    const float* __restrict__ cos_table,  // [max_seq_len, hdim]
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
    const int sin_q_offset = q_idx * hdim;

    float m_max = -INFINITY;
    float l_sum = 0.0f;

    // Process K,V in blocks using online softmax
    for (int k_start = 0; k_start < seq_k; k_start += BLOCK_N) {
        const int k_end = min(k_start + BLOCK_N, seq_k);
        const int num_k = k_end - k_start;

        // Compute scores for this block with sinusoidal embeddings
        for (int k_local = tid; k_local < num_k; k_local += blockDim.x) {
            int k_idx = k_start + k_local;
            int k_offset = kv_base + k_idx * hdim;
            int sin_k_offset = k_idx * hdim;

            float score = 0.0f;

            // Dot product with sinusoidal embeddings applied
            for (int d = 0; d < hdim; d++) {
                // Apply sinusoidal embedding: Q' = Q + sin/cos, K' = K + sin/cos
                float q_embedded = Q[q_offset + d];
                float k_embedded = K[k_offset + d];

                // Add positional encoding (alternating sin/cos)
                if (d % 2 == 0) {
                    q_embedded += sin_table[sin_q_offset + d];
                    k_embedded += sin_table[sin_k_offset + d];
                } else {
                    q_embedded += cos_table[sin_q_offset + d];
                    k_embedded += cos_table[sin_k_offset + d];
                }

                score += q_embedded * k_embedded;
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

        // Update output with correction factor (V is not embedded)
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

// Version 2: Fused kernel with float4 vectorization
// Optimization: load sin/cos in float4 chunks to reduce memory transactions
template<int BLOCK_N>
__global__ void attention_fwd_kernel_sinusoidal_fused_v2(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ sin_table,
    const float* __restrict__ cos_table,
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
    const int sin_q_offset = q_idx * hdim;

    float m_max = -INFINITY;
    float l_sum = 0.0f;

    for (int k_start = 0; k_start < seq_k; k_start += BLOCK_N) {
        const int k_end = min(k_start + BLOCK_N, seq_k);
        const int num_k = k_end - k_start;

        for (int k_local = tid; k_local < num_k; k_local += blockDim.x) {
            int k_idx = k_start + k_local;
            int k_offset = kv_base + k_idx * hdim;
            int sin_k_offset = k_idx * hdim;

            float score = 0.0f;

            // float4 vectorized dot product with sinusoidal embeddings
            if (hdim % 4 == 0) {
                for (int d = 0; d < hdim; d += 4) {
                    float4 q_vec = *reinterpret_cast<const float4*>(&Q[q_offset + d]);
                    float4 k_vec = *reinterpret_cast<const float4*>(&K[k_offset + d]);
                    float4 sin_q = *reinterpret_cast<const float4*>(&sin_table[sin_q_offset + d]);
                    float4 sin_k = *reinterpret_cast<const float4*>(&sin_table[sin_k_offset + d]);
                    float4 cos_q = *reinterpret_cast<const float4*>(&cos_table[sin_q_offset + d]);
                    float4 cos_k = *reinterpret_cast<const float4*>(&cos_table[sin_k_offset + d]);

                    // Apply embeddings and accumulate
                    // d+0 (even): use sin, d+1 (odd): use cos, d+2 (even): use sin, d+3 (odd): use cos
                    float q_emb_0 = q_vec.x + sin_q.x;
                    float k_emb_0 = k_vec.x + sin_k.x;
                    float q_emb_1 = q_vec.y + cos_q.y;
                    float k_emb_1 = k_vec.y + cos_k.y;
                    float q_emb_2 = q_vec.z + sin_q.z;
                    float k_emb_2 = k_vec.z + sin_k.z;
                    float q_emb_3 = q_vec.w + cos_q.w;
                    float k_emb_3 = k_vec.w + cos_k.w;

                    score += q_emb_0 * k_emb_0 + q_emb_1 * k_emb_1 +
                             q_emb_2 * k_emb_2 + q_emb_3 * k_emb_3;
                }
            } else {
                for (int d = 0; d < hdim; d++) {
                    float q_embedded = Q[q_offset + d];
                    float k_embedded = K[k_offset + d];

                    if (d % 2 == 0) {
                        q_embedded += sin_table[sin_q_offset + d];
                        k_embedded += sin_table[sin_k_offset + d];
                    } else {
                        q_embedded += cos_table[sin_q_offset + d];
                        k_embedded += cos_table[sin_k_offset + d];
                    }

                    score += q_embedded * k_embedded;
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

// Version 3: Fused kernel with on-the-fly sin/cos computation
// Optimization: compute sin/cos instead of loading from table (trade memory for compute)
template<int BLOCK_N>
__global__ void attention_fwd_kernel_sinusoidal_fused_v3(
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
    const float base = 10000.0f;

    float m_max = -INFINITY;
    float l_sum = 0.0f;

    for (int k_start = 0; k_start < seq_k; k_start += BLOCK_N) {
        const int k_end = min(k_start + BLOCK_N, seq_k);
        const int num_k = k_end - k_start;

        for (int k_local = tid; k_local < num_k; k_local += blockDim.x) {
            int k_idx = k_start + k_local;
            int k_offset = kv_base + k_idx * hdim;

            float score = 0.0f;

            // Compute sin/cos on the fly
            for (int d = 0; d < hdim; d++) {
                float q_val = Q[q_offset + d];
                float k_val = K[k_offset + d];

                // Compute frequency and angle
                float freq = 1.0f / powf(base, (2.0f * (d / 2)) / (float)hdim);
                float angle_q = q_idx * freq;
                float angle_k = k_idx * freq;

                // Apply sin/cos based on dimension parity
                if (d % 2 == 0) {
                    q_val += sinf(angle_q);
                    k_val += sinf(angle_k);
                } else {
                    q_val += cosf(angle_q);
                    k_val += cosf(angle_k);
                }

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

// Version 4: V3 + cached frequencies (reduce expensive powf calls)
template<int BLOCK_N>
__global__ void attention_fwd_kernel_sinusoidal_fused_v4(
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
    float* s_freqs = s_reduce + 32;  // Cache frequencies in shared memory

    const int q_offset = qkv_base + q_idx * hdim;
    const float base = 10000.0f;

    // Precompute and cache frequencies (only once per block)
    for (int d = tid; d < hdim; d += blockDim.x) {
        s_freqs[d] = 1.0f / powf(base, (2.0f * (d / 2)) / (float)hdim);
    }
    __syncthreads();

    float m_max = -INFINITY;
    float l_sum = 0.0f;

    for (int k_start = 0; k_start < seq_k; k_start += BLOCK_N) {
        const int k_end = min(k_start + BLOCK_N, seq_k);
        const int num_k = k_end - k_start;

        for (int k_local = tid; k_local < num_k; k_local += blockDim.x) {
            int k_idx = k_start + k_local;
            int k_offset = kv_base + k_idx * hdim;

            float score = 0.0f;

            // Use cached frequencies
            for (int d = 0; d < hdim; d++) {
                float q_val = Q[q_offset + d];
                float k_val = K[k_offset + d];

                float freq = s_freqs[d];  // Load from shared memory
                float angle_q = q_idx * freq;
                float angle_k = k_idx * freq;

                if (d % 2 == 0) {
                    q_val += sinf(angle_q);
                    k_val += sinf(angle_k);
                } else {
                    q_val += cosf(angle_q);
                    k_val += cosf(angle_k);
                }

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

// Version 5: V4 + float4 vectorization + __sincosf() for simultaneous sin/cos
template<int BLOCK_N>
__global__ void attention_fwd_kernel_sinusoidal_fused_v5(
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
    float* s_freqs = s_reduce + 32;

    const int q_offset = qkv_base + q_idx * hdim;
    const float base = 10000.0f;

    // Cache frequencies
    for (int d = tid; d < hdim; d += blockDim.x) {
        s_freqs[d] = 1.0f / powf(base, (2.0f * (d / 2)) / (float)hdim);
    }
    __syncthreads();

    float m_max = -INFINITY;
    float l_sum = 0.0f;

    for (int k_start = 0; k_start < seq_k; k_start += BLOCK_N) {
        const int k_end = min(k_start + BLOCK_N, seq_k);
        const int num_k = k_end - k_start;

        for (int k_local = tid; k_local < num_k; k_local += blockDim.x) {
            int k_idx = k_start + k_local;
            int k_offset = kv_base + k_idx * hdim;

            float score = 0.0f;

            // Vectorized with __sincosf for better throughput
            if (hdim % 4 == 0) {
                for (int d = 0; d < hdim; d += 4) {
                    float4 q_vec = *reinterpret_cast<const float4*>(&Q[q_offset + d]);
                    float4 k_vec = *reinterpret_cast<const float4*>(&K[k_offset + d]);
                    float4 freq_vec = *reinterpret_cast<const float4*>(&s_freqs[d]);

                    // Compute all sin/cos using __sincosf
                    float sin_q0, cos_q0, sin_k0, cos_k0;
                    float sin_q1, cos_q1, sin_k1, cos_k1;
                    float sin_q2, cos_q2, sin_k2, cos_k2;
                    float sin_q3, cos_q3, sin_k3, cos_k3;

                    __sincosf(q_idx * freq_vec.x, &sin_q0, &cos_q0);
                    __sincosf(k_idx * freq_vec.x, &sin_k0, &cos_k0);
                    __sincosf(q_idx * freq_vec.y, &sin_q1, &cos_q1);
                    __sincosf(k_idx * freq_vec.y, &sin_k1, &cos_k1);
                    __sincosf(q_idx * freq_vec.z, &sin_q2, &cos_q2);
                    __sincosf(k_idx * freq_vec.z, &sin_k2, &cos_k2);
                    __sincosf(q_idx * freq_vec.w, &sin_q3, &cos_q3);
                    __sincosf(k_idx * freq_vec.w, &sin_k3, &cos_k3);

                    // d+0 (even): sin, d+1 (odd): cos, d+2 (even): sin, d+3 (odd): cos
                    float q_emb_0 = q_vec.x + sin_q0;
                    float k_emb_0 = k_vec.x + sin_k0;
                    float q_emb_1 = q_vec.y + cos_q1;
                    float k_emb_1 = k_vec.y + cos_k1;
                    float q_emb_2 = q_vec.z + sin_q2;
                    float k_emb_2 = k_vec.z + sin_k2;
                    float q_emb_3 = q_vec.w + cos_q3;
                    float k_emb_3 = k_vec.w + cos_k3;

                    score += q_emb_0 * k_emb_0 + q_emb_1 * k_emb_1 +
                             q_emb_2 * k_emb_2 + q_emb_3 * k_emb_3;
                }
            } else {
                for (int d = 0; d < hdim; d++) {
                    float q_val = Q[q_offset + d];
                    float k_val = K[k_offset + d];
                    float freq = s_freqs[d];

                    float sin_q, cos_q, sin_k, cos_k;
                    __sincosf(q_idx * freq, &sin_q, &cos_q);
                    __sincosf(k_idx * freq, &sin_k, &cos_k);

                    if (d % 2 == 0) {
                        q_val += sin_q;
                        k_val += sin_k;
                    } else {
                        q_val += cos_q;
                        k_val += cos_k;
                    }

                    score += q_val * k_val;
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

// Separate kernel approach: first apply sinusoidal embeddings, then attention
__global__ void apply_sinusoidal_embeddings(
    const float* __restrict__ input,
    const float* __restrict__ sin_table,
    const float* __restrict__ cos_table,
    float* __restrict__ output,
    int total_tokens,
    int hdim,
    int seqlen  // Need to know sequence length to extract position
) {
    int token_idx = blockIdx.x;

    if (token_idx >= total_tokens) return;

    // Extract position within sequence from global token index
    // For tensors [bs, nh, seqlen, hdim] flattened to [bs*nh*seqlen, hdim]
    int pos = token_idx % seqlen;

    // Each thread processes multiple dimensions (stride by blockDim.x)
    for (int d = threadIdx.x; d < hdim; d += blockDim.x) {
        int offset = token_idx * hdim + d;
        int sin_offset = pos * hdim + d;

        float val = input[offset];

        // Add positional encoding (alternating sin/cos)
        if (d % 2 == 0) {
            val += sin_table[sin_offset];
        } else {
            val += cos_table[sin_offset];
        }

        output[offset] = val;
    }
}

// Standard attention kernel (for separate approach)
template<int BLOCK_N>
__global__ void attention_fwd_kernel(
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
            for (int d = 0; d < hdim; d++) {
                score += Q[q_offset + d] * K[k_offset + d];
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

// CPU reference for sinusoidal embeddings
void generate_sinusoidal_embeddings_cpu(
    float* sin_table,
    float* cos_table,
    int max_seq_len,
    int hdim
) {
    const float base = 10000.0f;

    for (int pos = 0; pos < max_seq_len; pos++) {
        for (int i = 0; i < hdim; i++) {
            float freq = 1.0f / powf(base, (2.0f * (i / 2)) / (float)hdim);
            float angle = pos * freq;

            sin_table[pos * hdim + i] = sinf(angle);
            cos_table[pos * hdim + i] = cosf(angle);
        }
    }
}

// CPU attention with sinusoidal embeddings
void attention_sinusoidal_cpu(
    const float* Q,
    const float* K,
    const float* V,
    const float* sin_table,
    const float* cos_table,
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
                int sin_q_off = i * head_d;

                std::vector<float> attn_scores(len_k);
                float max_s = -INFINITY;

                for (int j = 0; j < len_k; j++) {
                    int koff = ((b * num_heads + h) * len_k + j) * head_d;
                    int sin_k_off = j * head_d;
                    float s = 0.0f;

                    for (int d = 0; d < head_d; d++) {
                        float q_val = Q[qoff + d];
                        float k_val = K[koff + d];

                        // Add sinusoidal embeddings
                        if (d % 2 == 0) {
                            q_val += sin_table[sin_q_off + d];
                            k_val += sin_table[sin_k_off + d];
                        } else {
                            q_val += cos_table[sin_q_off + d];
                            k_val += cos_table[sin_k_off + d];
                        }

                        s += q_val * k_val;
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

void test_sinusoidal_attention(int bs, int nh, int seqlen, int hdim) {
    printf("\n========================================\n");
    printf("Testing Sinusoidal Attention:\n");
    printf("  Batch size: %d, Num heads: %d\n", bs, nh);
    printf("  Sequence length: %d, Hidden dim: %d\n", seqlen, hdim);
    printf("========================================\n\n");

    int sz_qkv = bs * nh * seqlen * hdim;
    int sz_sin = seqlen * hdim;

    // Allocate host memory
    float *h_Q = new float[sz_qkv];
    float *h_K = new float[sz_qkv];
    float *h_V = new float[sz_qkv];
    float *h_sin_table = new float[sz_sin];
    float *h_cos_table = new float[sz_sin];
    float *h_out_fused = new float[sz_qkv];
    float *h_out_separate = new float[sz_qkv];
    float *h_out_cpu = new float[sz_qkv];

    init_random(h_Q, sz_qkv);
    init_random(h_K, sz_qkv);
    init_random(h_V, sz_qkv);

    // Generate sinusoidal embeddings
    generate_sinusoidal_embeddings_cpu(h_sin_table, h_cos_table, seqlen, hdim);

    // Allocate device memory
    float *d_Q, *d_K, *d_V, *d_sin, *d_cos, *d_out, *d_Q_emb, *d_K_emb;
    CUDA_CHECK(cudaMalloc(&d_Q, sz_qkv * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, sz_qkv * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V, sz_qkv * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sin, sz_sin * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cos, sz_sin * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, sz_qkv * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Q_emb, sz_qkv * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K_emb, sz_qkv * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_Q, h_Q, sz_qkv * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K, sz_qkv * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V, sz_qkv * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sin, h_sin_table, sz_sin * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cos, h_cos_table, sz_sin * sizeof(float), cudaMemcpyHostToDevice));

    float scale = 1.0f / sqrtf(static_cast<float>(hdim));
    constexpr int BLOCK_N = 64;
    const int nthreads = 256;
    const int shmem_sz = (BLOCK_N + 32) * sizeof(float);
    const int shmem_sz_v4 = (BLOCK_N + 32 + hdim) * sizeof(float);  // Extra space for frequencies

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    const int num_iters = 100;
    dim3 grid_fused(seqlen, bs * nh);

    // Test V1: Fused kernel (baseline with table lookups)
    printf("Testing FUSED V1 (table lookups)...\n");
    attention_fwd_kernel_sinusoidal_fused<BLOCK_N><<<grid_fused, nthreads, shmem_sz>>>(
        d_Q, d_K, d_V, d_sin, d_cos, d_out, bs, nh, seqlen, seqlen, hdim, scale
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iters; i++) {
        attention_fwd_kernel_sinusoidal_fused<BLOCK_N><<<grid_fused, nthreads, shmem_sz>>>(
            d_Q, d_K, d_V, d_sin, d_cos, d_out, bs, nh, seqlen, seqlen, hdim, scale
        );
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float fused_v1_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&fused_v1_ms, start, stop));
    fused_v1_ms /= num_iters;

    CUDA_CHECK(cudaMemcpy(h_out_fused, d_out, sz_qkv * sizeof(float), cudaMemcpyDeviceToHost));

    // Test V2: Fused kernel with float4 vectorization
    printf("Testing FUSED V2 (float4 vectorization)...\n");
    float *h_out_v2 = new float[sz_qkv];

    attention_fwd_kernel_sinusoidal_fused_v2<BLOCK_N><<<grid_fused, nthreads, shmem_sz>>>(
        d_Q, d_K, d_V, d_sin, d_cos, d_out, bs, nh, seqlen, seqlen, hdim, scale
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iters; i++) {
        attention_fwd_kernel_sinusoidal_fused_v2<BLOCK_N><<<grid_fused, nthreads, shmem_sz>>>(
            d_Q, d_K, d_V, d_sin, d_cos, d_out, bs, nh, seqlen, seqlen, hdim, scale
        );
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float fused_v2_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&fused_v2_ms, start, stop));
    fused_v2_ms /= num_iters;

    CUDA_CHECK(cudaMemcpy(h_out_v2, d_out, sz_qkv * sizeof(float), cudaMemcpyDeviceToHost));

    // Test V3: Fused kernel with on-the-fly sin/cos computation
    printf("Testing FUSED V3 (on-the-fly sin/cos)...\n");
    float *h_out_v3 = new float[sz_qkv];

    attention_fwd_kernel_sinusoidal_fused_v3<BLOCK_N><<<grid_fused, nthreads, shmem_sz>>>(
        d_Q, d_K, d_V, d_out, bs, nh, seqlen, seqlen, hdim, scale
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iters; i++) {
        attention_fwd_kernel_sinusoidal_fused_v3<BLOCK_N><<<grid_fused, nthreads, shmem_sz>>>(
            d_Q, d_K, d_V, d_out, bs, nh, seqlen, seqlen, hdim, scale
        );
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float fused_v3_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&fused_v3_ms, start, stop));
    fused_v3_ms /= num_iters;

    CUDA_CHECK(cudaMemcpy(h_out_v3, d_out, sz_qkv * sizeof(float), cudaMemcpyDeviceToHost));

    // Test V4: Fused kernel with cached frequencies
    printf("Testing FUSED V4 (cached frequencies)...\n");
    float *h_out_v4 = new float[sz_qkv];

    attention_fwd_kernel_sinusoidal_fused_v4<BLOCK_N><<<grid_fused, nthreads, shmem_sz_v4>>>(
        d_Q, d_K, d_V, d_out, bs, nh, seqlen, seqlen, hdim, scale
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iters; i++) {
        attention_fwd_kernel_sinusoidal_fused_v4<BLOCK_N><<<grid_fused, nthreads, shmem_sz_v4>>>(
            d_Q, d_K, d_V, d_out, bs, nh, seqlen, seqlen, hdim, scale
        );
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float fused_v4_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&fused_v4_ms, start, stop));
    fused_v4_ms /= num_iters;

    CUDA_CHECK(cudaMemcpy(h_out_v4, d_out, sz_qkv * sizeof(float), cudaMemcpyDeviceToHost));

    // Test V5: Fused kernel with float4 + __sincosf
    printf("Testing FUSED V5 (float4 + __sincosf + cached freqs)...\n");
    float *h_out_v5 = new float[sz_qkv];

    attention_fwd_kernel_sinusoidal_fused_v5<BLOCK_N><<<grid_fused, nthreads, shmem_sz_v4>>>(
        d_Q, d_K, d_V, d_out, bs, nh, seqlen, seqlen, hdim, scale
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iters; i++) {
        attention_fwd_kernel_sinusoidal_fused_v5<BLOCK_N><<<grid_fused, nthreads, shmem_sz_v4>>>(
            d_Q, d_K, d_V, d_out, bs, nh, seqlen, seqlen, hdim, scale
        );
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float fused_v5_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&fused_v5_ms, start, stop));
    fused_v5_ms /= num_iters;

    CUDA_CHECK(cudaMemcpy(h_out_v5, d_out, sz_qkv * sizeof(float), cudaMemcpyDeviceToHost));

    // Test separate kernel approach
    printf("Testing SEPARATE kernel approach...\n");

    // Apply embeddings (with fixed indexing and thread count)
    dim3 grid_emb(bs * nh * seqlen);
    const int emb_threads = 256;  // Use 256 threads, each processes hdim/256 dimensions
    apply_sinusoidal_embeddings<<<grid_emb, emb_threads>>>(d_Q, d_sin, d_cos, d_Q_emb, bs * nh * seqlen, hdim, seqlen);
    apply_sinusoidal_embeddings<<<grid_emb, emb_threads>>>(d_K, d_sin, d_cos, d_K_emb, bs * nh * seqlen, hdim, seqlen);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark separate
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iters; i++) {
        apply_sinusoidal_embeddings<<<grid_emb, emb_threads>>>(d_Q, d_sin, d_cos, d_Q_emb, bs * nh * seqlen, hdim, seqlen);
        apply_sinusoidal_embeddings<<<grid_emb, emb_threads>>>(d_K, d_sin, d_cos, d_K_emb, bs * nh * seqlen, hdim, seqlen);

        dim3 grid_attn(seqlen, bs * nh);
        attention_fwd_kernel<BLOCK_N><<<grid_attn, nthreads, shmem_sz>>>(
            d_Q_emb, d_K_emb, d_V, d_out, bs, nh, seqlen, seqlen, hdim, scale
        );
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float separate_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&separate_ms, start, stop));
    separate_ms /= num_iters;

    CUDA_CHECK(cudaMemcpy(h_out_separate, d_out, sz_qkv * sizeof(float), cudaMemcpyDeviceToHost));

    // CPU reference
    printf("Running CPU reference...\n");
    attention_sinusoidal_cpu(h_Q, h_K, h_V, h_sin_table, h_cos_table, h_out_cpu, bs, nh, seqlen, seqlen, hdim);

    // Results
    printf("\n========================================\n");
    printf("PERFORMANCE RESULTS\n");
    printf("========================================\n");
    printf("Fused V1 (table):      %.3f ms [baseline]\n", fused_v1_ms);
    printf("Fused V2 (float4):     %.3f ms  (%.2fx vs V1)\n", fused_v2_ms, fused_v1_ms / fused_v2_ms);
    printf("Fused V3 (on-fly):     %.3f ms  (%.2fx vs V1)\n", fused_v3_ms, fused_v1_ms / fused_v3_ms);
    printf("Fused V4 (cached freq):%.3f ms  (%.2fx vs V1)\n", fused_v4_ms, fused_v1_ms / fused_v4_ms);
    printf("Fused V5 (v4+sincosf): %.3f ms  (%.2fx vs V1)\n", fused_v5_ms, fused_v1_ms / fused_v5_ms);
    printf("Separate:              %.3f ms  (%.2fx vs V1)\n", separate_ms, fused_v1_ms / separate_ms);

    // Find best
    float best_time = fmin(fmin(fmin(fused_v3_ms, fused_v4_ms), fused_v5_ms), fused_v1_ms);
    printf("========================================\n");
    if (best_time == fused_v3_ms) printf("** BEST: V3 **\n");
    else if (best_time == fused_v4_ms) printf("** BEST: V4 **\n");
    else if (best_time == fused_v5_ms) printf("** BEST: V5 **\n");
    else printf("** BEST: V1 **\n");
    printf("========================================\n");

    printf("\nChecking fused V1 vs CPU...\n");
    bool passed_v1 = check_results(h_out_fused, h_out_cpu, sz_qkv);

    printf("\nChecking fused V2 vs CPU...\n");
    bool passed_v2 = check_results(h_out_v2, h_out_cpu, sz_qkv);

    printf("\nChecking fused V3 vs CPU...\n");
    bool passed_v3 = check_results(h_out_v3, h_out_cpu, sz_qkv);

    printf("\nChecking fused V4 vs CPU...\n");
    bool passed_v4 = check_results(h_out_v4, h_out_cpu, sz_qkv);

    printf("\nChecking fused V5 vs CPU...\n");
    bool passed_v5 = check_results(h_out_v5, h_out_cpu, sz_qkv);

    printf("\nChecking separate vs CPU...\n");
    bool passed_separate = check_results(h_out_separate, h_out_cpu, sz_qkv);

    if (passed_v1 && passed_v2 && passed_v3 && passed_v4 && passed_v5) {
        printf("\n✓ ALL FUSED KERNELS PASSED\n");
    } else {
        printf("\n✗ SOME TESTS FAILED\n");
    }

    delete[] h_out_v2;
    delete[] h_out_v3;
    delete[] h_out_v4;
    delete[] h_out_v5;

    // Cleanup
    delete[] h_Q;
    delete[] h_K;
    delete[] h_V;
    delete[] h_sin_table;
    delete[] h_cos_table;
    delete[] h_out_fused;
    delete[] h_out_separate;
    delete[] h_out_cpu;

    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_sin));
    CUDA_CHECK(cudaFree(d_cos));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_Q_emb));
    CUDA_CHECK(cudaFree(d_K_emb));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

int main() {
    printf("=================================================\n");
    printf("Sinusoidal Attention Fusion Benchmark\n");
    printf("=================================================\n");

    srand(42);

    // Test on 3 common LLM hidden dimensions
    printf("\n*** Testing hdim=512 ***\n");
    test_sinusoidal_attention(1, 4, 64, 512);

    printf("\n*** Testing hdim=2048 ***\n");
    test_sinusoidal_attention(1, 4, 64, 2048);

    printf("\n*** Testing hdim=4096 ***\n");
    test_sinusoidal_attention(1, 2, 32, 4096);

    printf("\n=================================================\n");
    printf("All tests completed!\n");
    printf("=================================================\n");

    return 0;
}
