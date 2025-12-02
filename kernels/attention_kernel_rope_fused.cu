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

void precompute_rope_cache(
    float* cos_cache,
    float* sin_cache,
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

            cos_cache[pos * rotary_dim + 2*i]     = cos_val;
            cos_cache[pos * rotary_dim + 2*i + 1] = cos_val;
            sin_cache[pos * rotary_dim + 2*i]     = sin_val;
            sin_cache[pos * rotary_dim + 2*i + 1] = sin_val;
        }
    }
}

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

__device__ __forceinline__ float rope_freq(int pair_idx, int hdim, float base = 10000.0f) {
    return 1.0f / powf(base, (2.0f * pair_idx) / hdim);
}

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

    const int num_q_this_tile = min(NUM_Q, seq_q - first_q);
    if (num_q_this_tile <= 0) return;

    const int tid = threadIdx.x;
    const int qkv_base = (b_idx * nh + h_idx) * seq_q * hdim;
    const int kv_base = (b_idx * nh + h_idx) * seq_k * hdim;

    extern __shared__ float smem[];
    float* s_scores = smem;
    float* s_reduce = s_scores + NUM_Q * BLOCK_N;
    float* s_Q_rope = s_reduce + 32;

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

    float m_max[NUM_Q], l_sum[NUM_Q];
    #pragma unroll
    for (int q = 0; q < NUM_Q; q++) {
        m_max[q] = -INFINITY;
        l_sum[q] = 0.0f;
    }

    for (int k_start = 0; k_start < seq_k; k_start += BLOCK_N) {
        const int k_end = min(k_start + BLOCK_N, seq_k);
        const int num_k = k_end - k_start;

        for (int k_local = tid; k_local < num_k; k_local += blockDim.x) {
            int k_idx = k_start + k_local;
            int k_pos = k_idx;
            int k_offset = kv_base + k_idx * hdim;

            float scores[NUM_Q];
            #pragma unroll
            for (int q = 0; q < NUM_Q; q++) scores[q] = 0.0f;

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

                #pragma unroll
                for (int q = 0; q < num_q_this_tile; q++) {
                    scores[q] += s_Q_rope[q * hdim + d] * k_even_rope +
                                 s_Q_rope[q * hdim + d + 1] * k_odd_rope;
                }
            }

            #pragma unroll
            for (int q = 0; q < num_q_this_tile; q++) {
                s_scores[q * BLOCK_N + k_local] = scores[q] * scale_factor;
            }
        }
        __syncthreads();

        for (int q = 0; q < num_q_this_tile; q++) {

            float block_max = -INFINITY;
            for (int k_local = tid; k_local < num_k; k_local += blockDim.x) {
                block_max = fmaxf(block_max, s_scores[q * BLOCK_N + k_local]);
            }
            block_max = block_reduce_max(block_max, s_reduce);

            float m_new = fmaxf(m_max[q], block_max);
            float correction = expf(m_max[q] - m_new);

            float block_sum = 0.0f;
            for (int k_local = tid; k_local < num_k; k_local += blockDim.x) {
                float exp_val = expf(s_scores[q * BLOCK_N + k_local] - m_new);
                s_scores[q * BLOCK_N + k_local] = exp_val;
                block_sum += exp_val;
            }
            block_sum = block_reduce_sum(block_sum, s_reduce);

            l_sum[q] = correction * l_sum[q] + block_sum;

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

    for (int q = 0; q < num_q_this_tile; q++) {
        int q_idx = first_q + q;
        int q_offset = qkv_base + q_idx * hdim;
        for (int d = tid; d < hdim; d += blockDim.x) {
            out[q_offset + d] /= l_sum[q];
        }
    }
}

void attention_forward_rope_fused_v3(
    const float* Q, const float* K, const float* V, float* out,
    int batch_sz, int num_heads, int len_q, int len_k, int head_d
) {
    float scale = 1.0f / sqrtf(static_cast<float>(head_d));
    constexpr int BLOCK_N = 64;
    constexpr int NUM_Q = 4;

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

    extern __shared__ float smem[];
    float* s_freq = smem;
    float* s_cos = s_freq + hdim/2;
    float* s_sin = s_cos + hdim;
    float* s_scores = s_sin + hdim;
    float* s_reduce = s_scores + BLOCK_N;
    float* s_Q_rope = s_reduce + 32;

    const int q_offset = qkv_base + q_idx * hdim;
    const int q_pos = q_idx;

    for (int i = tid; i < hdim/2; i += blockDim.x) {
        s_freq[i] = 1.0f / powf(10000.0f, (2.0f * i) / hdim);
    }
    __syncthreads();

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

    for (int k_start = 0; k_start < seq_k; k_start += BLOCK_N) {
        const int k_end = min(k_start + BLOCK_N, seq_k);
        const int num_k = k_end - k_start;

        for (int k_local = 0; k_local < num_k; k_local++) {
            int k_idx = k_start + k_local;
            int k_pos = k_idx;
            int k_offset = kv_base + k_idx * hdim;

            for (int d = tid; d < hdim; d += blockDim.x) {
                int pair_idx = d / 2;
                float theta = k_pos * s_freq[pair_idx];
                float cos_val, sin_val;
                __sincosf(theta, &sin_val, &cos_val);
                s_cos[d] = cos_val;
                s_sin[d] = sin_val;
            }
            __syncthreads();

            float partial_score = 0.0f;
            for (int d = tid * 2; d < hdim; d += blockDim.x * 2) {
                float k_even = K[k_offset + d];
                float k_odd  = K[k_offset + d + 1];

                float k_even_rope = k_even * s_cos[d] - k_odd * s_sin[d];
                float k_odd_rope  = k_even * s_sin[d] + k_odd * s_cos[d];

                partial_score += s_Q_rope[d] * k_even_rope + s_Q_rope[d+1] * k_odd_rope;
            }

            partial_score = block_reduce_sum(partial_score, s_reduce);

            if (tid == 0) {
                s_scores[k_local] = partial_score * scale_factor;
            }
            __syncthreads();
        }

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

void attention_forward_rope_fused_v4(
    const float* Q, const float* K, const float* V, float* out,
    int batch_sz, int num_heads, int len_q, int len_k, int head_d
) {
    float scale = 1.0f / sqrtf(static_cast<float>(head_d));
    constexpr int BLOCK_N = 64;
    dim3 grid(len_q, batch_sz * num_heads);
    const int nthreads = 256;

    const int shmem_sz = (head_d/2 + head_d + head_d + BLOCK_N + 32 + head_d) * sizeof(float);

    attention_fwd_kernel_rope_fused_v4<BLOCK_N><<<grid, nthreads, shmem_sz>>>(
        Q, K, V, out, batch_sz, num_heads, len_q, len_k, head_d, scale
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

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

    extern __shared__ float smem[];
    float* s_K = smem;
    float* s_scores = s_K + BLOCK_K * hdim;
    float* s_reduce = s_scores + BLOCK_K;
    float* s_Q_rope = s_reduce + 32;
    float* s_freq = s_Q_rope + hdim;

    const int q_offset = qkv_base + q_idx * hdim;
    const int q_pos = q_idx;

    for (int i = tid; i < hdim/2; i += blockDim.x) {
        s_freq[i] = 1.0f / powf(10000.0f, (2.0f * i) / hdim);
    }
    __syncthreads();

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

    for (int k_start = 0; k_start < seq_k; k_start += BLOCK_K) {
        const int k_end = min(k_start + BLOCK_K, seq_k);
        const int num_k = k_end - k_start;

        for (int i = tid; i < num_k * hdim; i += blockDim.x) {
            int k_local = i / hdim;
            int d = i % hdim;
            int k_global = k_start + k_local;
            s_K[k_local * hdim + d] = K[kv_base + k_global * hdim + d];
        }
        __syncthreads();

        for (int k_local = tid; k_local < num_k; k_local += blockDim.x) {
            int k_pos = k_start + k_local;

            float score = 0.0f;

            for (int d = 0; d < hdim; d += 2) {
                int pair_idx = d / 2;
                float theta = k_pos * s_freq[pair_idx];
                float cos_val, sin_val;
                __sincosf(theta, &sin_val, &cos_val);

                float k_even = s_K[k_local * hdim + d];
                float k_odd  = s_K[k_local * hdim + d + 1];

                float k_even_rope = k_even * cos_val - k_odd * sin_val;
                float k_odd_rope  = k_even * sin_val + k_odd * cos_val;

                score += s_Q_rope[d] * k_even_rope + s_Q_rope[d + 1] * k_odd_rope;
            }

            s_scores[k_local] = score * scale_factor;
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
                int k_global = k_start + k_local;
                v_acc += s_scores[k_local] * V[kv_base + k_global * hdim + d];
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

void attention_forward_rope_fused_v5(
    const float* Q, const float* K, const float* V, float* out,
    int batch_sz, int num_heads, int len_q, int len_k, int head_d
) {
    float scale = 1.0f / sqrtf(static_cast<float>(head_d));
    dim3 grid(len_q, batch_sz * num_heads);
    const int nthreads = 256;

    const int available_floats = 10000;
    const int overhead = 32 + head_d + head_d/2;
    int block_k = (available_floats - overhead) / (head_d + 1);
    block_k = max(4, min(64, block_k));

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

    extern __shared__ float smem[];
    float* s_K_tile = smem;
    float* s_scores = s_K_tile + BLOCK_K * TILE_D;
    float* s_reduce = s_scores + BLOCK_K;
    float* s_Q_rope = s_reduce + 32;
    float* s_freq = s_Q_rope + hdim;

    const int q_offset = qkv_base + q_idx * hdim;
    const int q_pos = q_idx;

    for (int i = tid; i < rotary_dim/2; i += blockDim.x) {
        s_freq[i] = 1.0f / powf(10000.0f, (2.0f * i) / rotary_dim);
    }
    __syncthreads();

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

    for (int d = rotary_dim + tid; d < hdim; d += blockDim.x) {
        s_Q_rope[d] = Q[q_offset + d];
    }
    __syncthreads();

    float m_max = -INFINITY;
    float l_sum = 0.0f;

    for (int k_start = 0; k_start < seq_k; k_start += BLOCK_K) {
        const int k_end = min(k_start + BLOCK_K, seq_k);
        const int num_k = k_end - k_start;

        for (int k_local = tid; k_local < BLOCK_K; k_local += blockDim.x) {
            s_scores[k_local] = 0.0f;
        }
        __syncthreads();

        for (int d_start = 0; d_start < hdim; d_start += TILE_D) {
            const int d_end = min(d_start + TILE_D, hdim);
            const int tile_d = d_end - d_start;

            for (int i = tid; i < num_k * tile_d; i += blockDim.x) {
                int k_local = i / tile_d;
                int d_local = i % tile_d;
                int k_global = k_start + k_local;
                int d_global = d_start + d_local;
                s_K_tile[k_local * TILE_D + d_local] = K[kv_base + k_global * hdim + d_global];
            }
            __syncthreads();

            for (int k_local = tid; k_local < num_k; k_local += blockDim.x) {
                int k_pos = k_start + k_local;
                float partial_score = 0.0f;

                for (int d_local = 0; d_local < tile_d; d_local += 2) {
                    int d_global = d_start + d_local;

                    float k_even = s_K_tile[k_local * TILE_D + d_local];
                    float k_odd  = s_K_tile[k_local * TILE_D + d_local + 1];

                    float k_val_0, k_val_1;

                    if (d_global < rotary_dim) {

                        int pair_idx = d_global / 2;
                        float theta = k_pos * s_freq[pair_idx];
                        float cos_val, sin_val;
                        __sincosf(theta, &sin_val, &cos_val);

                        k_val_0 = k_even * cos_val - k_odd * sin_val;
                        k_val_1 = k_even * sin_val + k_odd * cos_val;
                    } else {

                        k_val_0 = k_even;
                        k_val_1 = k_odd;
                    }

                    partial_score += s_Q_rope[d_global] * k_val_0 +
                                   s_Q_rope[d_global + 1] * k_val_1;
                }

                s_scores[k_local] += partial_score;
            }
            __syncthreads();
        }

        for (int k_local = tid; k_local < num_k; k_local += blockDim.x) {
            s_scores[k_local] *= scale_factor;
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
                int k_global = k_start + k_local;
                v_acc += s_scores[k_local] * V[kv_base + k_global * hdim + d];
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

void attention_forward_rope_fused_v6(
    const float* Q, const float* K, const float* V, float* out,
    int batch_sz, int num_heads, int len_q, int len_k, int head_d, int rotary_dim
) {
    float scale = 1.0f / sqrtf(static_cast<float>(head_d));
    dim3 grid(len_q, batch_sz * num_heads);
    const int nthreads = 256;

    constexpr int BLOCK_K = 32;
    constexpr int TILE_D = 256;
    const int shmem_sz = (BLOCK_K * TILE_D + BLOCK_K + 32 + head_d + rotary_dim/2) * sizeof(float);

    attention_fwd_kernel_rope_fused_v6<BLOCK_K, TILE_D><<<grid, nthreads, shmem_sz>>>(
        Q, K, V, out, batch_sz, num_heads, len_q, len_k, head_d, rotary_dim, scale
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template<int BLOCK_K>
__global__ void attention_fwd_kernel_rope_fused_v7_hybrid(
    const float* __restrict__ Q,
    const float* __restrict__ K_rotated,
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

    extern __shared__ float smem[];
    float* s_scores = smem;
    float* s_reduce = s_scores + BLOCK_K;
    float* s_Q_rope = s_reduce + 32;
    float* s_freq = s_Q_rope + hdim;

    const int q_offset = qkv_base + q_idx * hdim;
    const int q_pos = q_idx;

    // Precompute freq[i] = 1 / (10000^(2i/rotary_dim)) and cache in shmem
    for (int i = tid; i < rotary_dim/2; i += blockDim.x) {
        s_freq[i] = 1.0f / powf(10000.0f, (2.0f * i) / rotary_dim);
    }
    __syncthreads();

    // Apply RoPE to Q and store in shmem
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

    // Copy non-rotated dimensions to shmem
    for (int d = rotary_dim + tid; d < hdim; d += blockDim.x) {
        s_Q_rope[d] = Q[q_offset + d];
    }
    __syncthreads();

    // Flash attn softmax
    float m_max = -INFINITY;
    float l_sum = 0.0f;

    // Process K in tiles of BLOCK_K
    for (int k_start = 0; k_start < seq_k; k_start += BLOCK_K) {
        const int k_end = min(k_start + BLOCK_K, seq_k);
        const int num_k = k_end - k_start;

        // Attn scores: Q_rope x K_rotated
        for (int k_local = tid; k_local < num_k; k_local += blockDim.x) {
            int k_idx = k_start + k_local;
            int k_offset = kv_base + k_idx * hdim;

            float score = 0.0f;

            // Dot prod with float4
            for (int d = 0; d < hdim; d += 4) {
                if (d + 3 < hdim) {
                    float4 q_vec = *reinterpret_cast<const float4*>(&s_Q_rope[d]);
                    float4 k_vec = *reinterpret_cast<const float4*>(&K_rotated[k_offset + d]);
                    score += q_vec.x * k_vec.x + q_vec.y * k_vec.y +
                             q_vec.z * k_vec.z + q_vec.w * k_vec.w;
                } else {
                    for (int dd = d; dd < hdim; dd++) {
                        score += s_Q_rope[dd] * K_rotated[k_offset + dd];
                    }
                    break;
                }
            }

            s_scores[k_local] = score * scale_factor;
        }
        __syncthreads();

        // Find max score in this tile and subtract from all scores, stability trick
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

    // Normalize
    for (int d = tid; d < hdim; d += blockDim.x) {
        out[q_offset + d] /= l_sum;
    }
}

void attention_forward_rope_fused_v7_hybrid(
    const float* Q, const float* K_rotated, const float* V, float* out,
    int batch_sz, int num_heads, int len_q, int len_k, int head_d, int rotary_dim
) {
    float scale = 1.0f / sqrtf(static_cast<float>(head_d));
    dim3 grid(len_q, batch_sz * num_heads);
    const int nthreads = 256;

    constexpr int BLOCK_K = 64;
    const int shmem_sz = (BLOCK_K + 32 + head_d + rotary_dim/2) * sizeof(float);

    attention_fwd_kernel_rope_fused_v7_hybrid<BLOCK_K><<<grid, nthreads, shmem_sz>>>(
        Q, K_rotated, V, out, batch_sz, num_heads, len_q, len_k, head_d, rotary_dim, scale
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

template<int BLOCK_N>
__global__ void attention_fwd_kernel_rope_fused(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ out,
    const float* __restrict__ cos_cache,
    const float* __restrict__ sin_cache,
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

    if (hdim % 4 == 0) {
        for (int d = tid * 4; d < hdim; d += blockDim.x * 4) {

            float4 q_vec = *reinterpret_cast<const float4*>(&Q[q_offset + d]);
            float4 cos_vec = *reinterpret_cast<const float4*>(&cos_cache[q_pos * hdim + d]);
            float4 sin_vec = *reinterpret_cast<const float4*>(&sin_cache[q_pos * hdim + d]);

            float4 q_rope;
            q_rope.x = q_vec.x * cos_vec.x - q_vec.y * sin_vec.x;
            q_rope.y = q_vec.x * sin_vec.x + q_vec.y * cos_vec.x;
            q_rope.z = q_vec.z * cos_vec.z - q_vec.w * sin_vec.z;
            q_rope.w = q_vec.z * sin_vec.z + q_vec.w * cos_vec.z;

            *reinterpret_cast<float4*>(&s_Q_rope[d]) = q_rope;
        }
    } else {

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

            if (hdim % 4 == 0) {
                for (int d = 0; d < hdim; d += 4) {

                    float4 k_vec = *reinterpret_cast<const float4*>(&K[k_offset + d]);
                    float4 cos_vec = *reinterpret_cast<const float4*>(&cos_cache[k_pos * hdim + d]);
                    float4 sin_vec = *reinterpret_cast<const float4*>(&sin_cache[k_pos * hdim + d]);
                    float4 q_vec = *reinterpret_cast<const float4*>(&s_Q_rope[d]);

                    float k0_rope = k_vec.x * cos_vec.x - k_vec.y * sin_vec.x;
                    float k1_rope = k_vec.x * sin_vec.x + k_vec.y * cos_vec.x;
                    float k2_rope = k_vec.z * cos_vec.z - k_vec.w * sin_vec.z;
                    float k3_rope = k_vec.z * sin_vec.z + k_vec.w * cos_vec.z;

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

    const int shmem_sz = (BLOCK_N + 32 + head_d) * sizeof(float);

    attention_fwd_kernel_rope_fused<BLOCK_N><<<grid, nthreads, shmem_sz>>>(
        Q, K, V, out, cos_cache, sin_cache,
        batch_sz, num_heads, len_q, len_k, head_d, scale
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

__global__ void rope_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const float* __restrict__ cos_cache,
    const float* __restrict__ sin_cache,
    int total_tokens,
    int hdim,
    int rotary_dim
) {
    int token_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (token_idx >= total_tokens) return;

    int pos = token_idx % (total_tokens / gridDim.x * gridDim.x == total_tokens ?
              blockIdx.x : token_idx);

    int seq_pos = token_idx % 64;

    const int in_offset = token_idx * hdim;

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

    for (int d = rotary_dim + tid; d < hdim; d += blockDim.x) {
        output[in_offset + d] = input[in_offset + d];
    }
}

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

void attention_forward_separate_rope(
    const float* Q,
    const float* K,
    const float* V,
    float* Q_rope,
    float* K_rope,
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

    rope_kernel<<<total_q, 256>>>(Q, Q_rope, cos_cache, sin_cache, total_q, head_d, rotary_dim);

    rope_kernel<<<total_k, 256>>>(K, K_rope, cos_cache, sin_cache, total_k, head_d, rotary_dim);

    CUDA_CHECK(cudaDeviceSynchronize());

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

    std::vector<float> Q_rope(len_q * head_d);
    std::vector<float> K_rope(len_k * head_d);

    for (int b = 0; b < batch_sz; b++) {
        for (int h = 0; h < num_heads; h++) {
            int qkv_base = (b * num_heads + h) * len_q * head_d;
            int kv_base = (b * num_heads + h) * len_k * head_d;

            apply_rope_cpu(&Q[qkv_base], Q_rope.data(), cos_cache, sin_cache, len_q, head_d);

            apply_rope_cpu(&K[kv_base], K_rope.data(), cos_cache, sin_cache, len_k, head_d);

            for (int i = 0; i < len_q; i++) {
                std::vector<float> attn_scores(len_k);
                float max_s = -INFINITY;

                for (int j = 0; j < len_k; j++) {
                    float s = 0.0f;
                    for (int d = 0; d < head_d; d++) {
                        s += Q_rope[i * head_d + d] * K_rope[j * head_d + d];
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
                        result += attn_scores[j] * V[kv_base + j * head_d + d];
                    }
                    out[qkv_base + i * head_d + d] = result;
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

    precompute_rope_cache(h_cos, h_sin, seqlen, hdim);

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
    CUDA_CHECK(cudaDeviceSynchronize());

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

void debug_rope_attention() {
    printf("\n========================================\n");
    printf("DEBUG: Isolating RoPE bug\n");
    printf("========================================\n\n");

    int bs = 1, nh = 1, seqlen = 64, hdim = 128;
    int sz_qkv = bs * nh * seqlen * hdim;
    int sz_rope = seqlen * hdim;

    float *h_Q = new float[sz_qkv];
    float *h_K = new float[sz_qkv];
    float *h_V = new float[sz_qkv];
    float *h_cos = new float[sz_rope];
    float *h_sin = new float[sz_rope];

    for (int i = 0; i < sz_qkv; i++) {
        h_Q[i] = 0.1f;
        h_K[i] = 0.1f;
        h_V[i] = 0.1f;
    }

    precompute_rope_cache(h_cos, h_sin, seqlen, hdim);

    printf("RoPE cache for position 0:\n");
    for (int d = 0; d < 8; d++) {
        printf("  d=%d: cos=%.6f, sin=%.6f\n", d, h_cos[d], h_sin[d]);
    }
    printf("RoPE cache for position 1:\n");
    for (int d = 0; d < 8; d++) {
        printf("  d=%d: cos=%.6f, sin=%.6f\n", d, h_cos[hdim + d], h_sin[hdim + d]);
    }

    std::vector<float> Q_rope(seqlen * hdim);
    std::vector<float> K_rope(seqlen * hdim);

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

    float score_q0_k0 = 0.0f;
    for (int d = 0; d < hdim; d++) {
        score_q0_k0 += Q_rope[d] * K_rope[d];
    }
    printf("\nCPU: score(q=0, k=0) = %.6f (before scale)\n", score_q0_k0);

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

void benchmark_fused_vs_separate(int bs, int nh, int seqlen, int hdim, int rotary_dim) {
    printf("\n========================================\n");
    printf("BENCHMARK: Fused vs Separate RoPE+Attention\n");
    printf("  bs=%d, nh=%d, seq=%d, hdim=%d, rotary_dim=%d\n", bs, nh, seqlen, hdim, rotary_dim);
    printf("========================================\n");

    int sz_qkv = bs * nh * seqlen * hdim;
    int sz_rope = seqlen * rotary_dim;

    float *h_Q = new float[sz_qkv];
    float *h_K = new float[sz_qkv];
    float *h_V = new float[sz_qkv];
    float *h_cos = new float[sz_rope];
    float *h_sin = new float[sz_rope];

    init_random(h_Q, sz_qkv);
    init_random(h_K, sz_qkv);
    init_random(h_V, sz_qkv);
    precompute_rope_cache(h_cos, h_sin, seqlen, rotary_dim);

    float *d_Q, *d_K, *d_V, *d_out, *d_cos, *d_sin;
    float *d_Q_rope, *d_K_rope;

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

    attention_forward_rope_fused_v6(d_Q, d_K, d_V, d_out, bs, nh, seqlen, seqlen, hdim, rotary_dim);
    attention_forward_separate_rope(d_Q, d_K, d_V, d_Q_rope, d_K_rope, d_out, d_cos, d_sin, bs, nh, seqlen, seqlen, hdim, rotary_dim);

    int total_k = bs * nh * seqlen;
    rope_kernel<<<total_k, 256>>>(d_K, d_K_rope, d_cos, d_sin, total_k, hdim, rotary_dim);
    attention_forward_rope_fused_v7_hybrid(d_Q, d_K_rope, d_V, d_out, bs, nh, seqlen, seqlen, hdim, rotary_dim);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iters; i++) {
        attention_forward_rope_fused_v6(d_Q, d_K, d_V, d_out, bs, nh, seqlen, seqlen, hdim, rotary_dim);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float fused_v6_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&fused_v6_ms, start, stop));
    fused_v6_ms /= num_iters;

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iters; i++) {

        rope_kernel<<<total_k, 256>>>(d_K, d_K_rope, d_cos, d_sin, total_k, hdim, rotary_dim);

        attention_forward_rope_fused_v7_hybrid(d_Q, d_K_rope, d_V, d_out, bs, nh, seqlen, seqlen, hdim, rotary_dim);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float hybrid_v7_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&hybrid_v7_ms, start, stop));
    hybrid_v7_ms /= num_iters;

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

    debug_rope_attention();

    printf("\n--- BASIC CORRECTNESS TESTS ---\n");
    test_rope_attention(1, 1, 4, 8);
    test_rope_attention(1, 2, 8, 16);
    test_rope_attention(2, 4, 32, 32);

    printf("\n--- LARGER DIMENSION TESTS ---\n");
    test_rope_attention(1, 4, 16, 64);
    test_rope_attention(1, 4, 64, 128);

    printf("\n--- TARGET DIMENSION TESTS ---\n");
    test_rope_attention(1, 4, 64, 512);
    test_rope_attention(1, 4, 64, 2048);

    printf("\n--- FUSED vs SEPARATE BENCHMARK (PARTIAL ROTATION) ---\n");
    printf("Testing realistic LLaMA-style configs:\n");
    printf("  hdim=128, rotary_dim=64   (50%% rotation)\n");
    printf("  hdim=512, rotary_dim=128  (25%% rotation)\n");
    printf("  hdim=2048, rotary_dim=128 (6%% rotation)\n\n");

    benchmark_fused_vs_separate(1, 4, 64, 128, 64);
    benchmark_fused_vs_separate(1, 4, 64, 512, 128);
    benchmark_fused_vs_separate(1, 4, 64, 2048, 128);

    printf("\n=================================================\n");
    printf("All tests completed!\n");
    printf("=================================================\n");

    return 0;
}

