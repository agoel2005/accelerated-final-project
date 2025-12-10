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

// Precompute sin/cos tables
void precompute_sinusoidal_table(float* sin_table, float* cos_table, int seqlen, int hdim) {
    const float base = 10000.0f;

    for (int pos = 0; pos < seqlen; pos++) {
        for (int d = 0; d < hdim; d++) {
            float freq = 1.0f / powf(base, (2.0f * (d / 2)) / (float)hdim);
            float angle = pos * freq;
            sin_table[pos * hdim + d] = sinf(angle);
            cos_table[pos * hdim + d] = cosf(angle);
        }
    }
}

// Optimized kernel to add sinusoidal embeddings to K with float4
__global__ void add_sinusoidal_to_k_optimized(
    const float* __restrict__ K,
    const float* __restrict__ sin_table,
    const float* __restrict__ cos_table,
    float* __restrict__ K_emb,
    int total_tokens,
    int hdim,
    int seqlen
) {
    int token_idx = blockIdx.x;
    if (token_idx >= total_tokens) return;

    int pos = token_idx % seqlen;
    int offset_base = token_idx * hdim;
    int sin_offset_base = pos * hdim;

    // Float4 vectorization for 4x memory throughput
    if (hdim % 4 == 0) {
        for (int d = threadIdx.x * 4; d < hdim; d += blockDim.x * 4) {
            float4 k_vec = *reinterpret_cast<const float4*>(&K[offset_base + d]);
            float4 sin_vec = *reinterpret_cast<const float4*>(&sin_table[sin_offset_base + d]);
            float4 cos_vec = *reinterpret_cast<const float4*>(&cos_table[sin_offset_base + d]);

            // Apply sin to even indices, cos to odd indices
            k_vec.x += sin_vec.x;  // d+0 is even
            k_vec.y += cos_vec.y;  // d+1 is odd
            k_vec.z += sin_vec.z;  // d+2 is even
            k_vec.w += cos_vec.w;  // d+3 is odd

            *reinterpret_cast<float4*>(&K_emb[offset_base + d]) = k_vec;
        }
    } else {
        for (int d = threadIdx.x; d < hdim; d += blockDim.x) {
            float val = K[offset_base + d];
            if (d % 2 == 0) {
                val += sin_table[sin_offset_base + d];
            } else {
                val += cos_table[sin_offset_base + d];
            }
            K_emb[offset_base + d] = val;
        }
    }
}

// Hybrid V2: Compute Q embeddings on-the-fly (no shared memory for Q)
// K already has embeddings pre-added
template<int BLOCK_N>
__global__ void attention_fwd_kernel_sinusoidal_hybrid_v2(
    const float* __restrict__ Q,
    const float* __restrict__ K_emb,  // K with embeddings already added
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
    const int sin_offset = q_idx * hdim;

    // Online softmax state
    float m_max = -INFINITY;
    float l_sum = 0.0f;

    // Process K/V in blocks
    for (int k_start = 0; k_start < seq_k; k_start += BLOCK_N) {
        const int k_end = min(k_start + BLOCK_N, seq_k);
        const int num_k = k_end - k_start;

        // Compute Q@K^T scores
        // Q embeddings computed on-the-fly, K already has embeddings
        for (int k_local = tid; k_local < num_k; k_local += blockDim.x) {
            int k_idx = k_start + k_local;
            int k_offset = kv_base + k_idx * hdim;

            float score = 0.0f;

            // Vectorized dot product with on-the-fly Q embedding
            if (hdim % 4 == 0) {
                #pragma unroll 4
                for (int d = 0; d < hdim; d += 4) {
                    float4 q_vec = *reinterpret_cast<const float4*>(&Q[q_offset + d]);
                    float4 sin_vec = *reinterpret_cast<const float4*>(&sin_table[sin_offset + d]);
                    float4 cos_vec = *reinterpret_cast<const float4*>(&cos_table[sin_offset + d]);
                    float4 k_vec = *reinterpret_cast<const float4*>(&K_emb[k_offset + d]);

                    // Add embeddings to Q on-the-fly
                    float q_emb_0 = q_vec.x + sin_vec.x;  // d+0 is even
                    float q_emb_1 = q_vec.y + cos_vec.y;  // d+1 is odd
                    float q_emb_2 = q_vec.z + sin_vec.z;  // d+2 is even
                    float q_emb_3 = q_vec.w + cos_vec.w;  // d+3 is odd

                    score += q_emb_0 * k_vec.x + q_emb_1 * k_vec.y +
                             q_emb_2 * k_vec.z + q_emb_3 * k_vec.w;
                }
            } else {
                for (int d = 0; d < hdim; d++) {
                    float q_val = Q[q_offset + d];

                    // Add embedding on-the-fly
                    if (d % 2 == 0) {
                        q_val += sin_table[sin_offset + d];
                    } else {
                        q_val += cos_table[sin_offset + d];
                    }

                    score += q_val * K_emb[k_offset + d];
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

// Forward function for hybrid V2
void attention_forward_sinusoidal_hybrid_v2(
    const float* Q,
    const float* K,
    const float* V,
    const float* sin_table,
    const float* cos_table,
    float* K_emb,
    float* out,
    int batch_sz,
    int num_heads,
    int len_q,
    int len_k,
    int head_d
) {
    float scale = 1.0f / sqrtf(static_cast<float>(head_d));

    // Step 1: Add sinusoidal embeddings to K
    int total_k = batch_sz * num_heads * len_k;
    add_sinusoidal_to_k_optimized<<<total_k, 256>>>(K, sin_table, cos_table, K_emb, total_k, head_d, len_k);

    // Step 2: Run hybrid V2 attention (computes Q embeddings on-the-fly)
    constexpr int BLOCK_N = 64;
    dim3 grid(len_q, batch_sz * num_heads);
    const int nthreads = 256;
    const int shmem_sz = (BLOCK_N + 32) * sizeof(float);  // No Q embedding cache

    attention_fwd_kernel_sinusoidal_hybrid_v2<BLOCK_N><<<grid, nthreads, shmem_sz>>>(
        Q, K_emb, V, sin_table, cos_table, out,
        batch_sz, num_heads, len_q, len_k, head_d, scale
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Separate approach (baseline)
__global__ void add_sinusoidal_separate(
    const float* __restrict__ input,
    const float* __restrict__ sin_table,
    const float* __restrict__ cos_table,
    float* __restrict__ output,
    int total_tokens,
    int hdim,
    int seqlen
) {
    int token_idx = blockIdx.x;
    if (token_idx >= total_tokens) return;

    int pos = token_idx % seqlen;

    if (hdim % 4 == 0) {
        for (int d = threadIdx.x * 4; d < hdim; d += blockDim.x * 4) {
            int offset = token_idx * hdim + d;
            int sin_offset = pos * hdim + d;

            float4 in_vec = *reinterpret_cast<const float4*>(&input[offset]);
            float4 sin_vec = *reinterpret_cast<const float4*>(&sin_table[sin_offset]);
            float4 cos_vec = *reinterpret_cast<const float4*>(&cos_table[sin_offset]);

            in_vec.x += sin_vec.x;
            in_vec.y += cos_vec.y;
            in_vec.z += sin_vec.z;
            in_vec.w += cos_vec.w;

            *reinterpret_cast<float4*>(&output[offset]) = in_vec;
        }
    } else {
        for (int d = threadIdx.x; d < hdim; d += blockDim.x) {
            int offset = token_idx * hdim + d;
            int sin_offset = pos * hdim + d;

            float val = input[offset];
            if (d % 2 == 0) {
                val += sin_table[sin_offset];
            } else {
                val += cos_table[sin_offset];
            }
            output[offset] = val;
        }
    }
}

template<int BLOCK_N>
__global__ void attention_kernel(
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

void attention_forward_sinusoidal_separate(
    const float* Q,
    const float* K,
    const float* V,
    const float* sin_table,
    const float* cos_table,
    float* Q_emb,
    float* K_emb,
    float* out,
    int batch_sz,
    int num_heads,
    int len_q,
    int len_k,
    int head_d
) {
    float scale = 1.0f / sqrtf(static_cast<float>(head_d));

    int total_q = batch_sz * num_heads * len_q;
    int total_k = batch_sz * num_heads * len_k;

    add_sinusoidal_separate<<<total_q, 256>>>(Q, sin_table, cos_table, Q_emb, total_q, head_d, len_q);
    add_sinusoidal_separate<<<total_k, 256>>>(K, sin_table, cos_table, K_emb, total_k, head_d, len_k);

    constexpr int BLOCK_N = 64;
    dim3 grid(len_q, batch_sz * num_heads);
    const int nthreads = 256;
    const int shmem_sz = (BLOCK_N + 32) * sizeof(float);

    attention_kernel<BLOCK_N><<<grid, nthreads, shmem_sz>>>(
        Q_emb, K_emb, V, out,
        batch_sz, num_heads, len_q, len_k, head_d, scale
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// CPU reference
void attention_cpu_sinusoidal(
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

                std::vector<float> attn_scores(len_k);
                float max_s = -INFINITY;

                for (int j = 0; j < len_k; j++) {
                    int koff = ((b * num_heads + h) * len_k + j) * head_d;
                    float s = 0.0f;

                    for (int d = 0; d < head_d; d++) {
                        float q_val = Q[qoff + d];
                        float k_val = K[koff + d];

                        if (d % 2 == 0) {
                            q_val += sin_table[i * head_d + d];
                            k_val += sin_table[j * head_d + d];
                        } else {
                            q_val += cos_table[i * head_d + d];
                            k_val += cos_table[j * head_d + d];
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

void benchmark_hybrid_vs_separate(int bs, int nh, int seqlen, int hdim) {
    printf("\n========================================\n");
    printf("BENCHMARK: Hybrid V2 vs Separate\n");
    printf("  bs=%d, nh=%d, seq=%d, hdim=%d\n", bs, nh, seqlen, hdim);
    printf("========================================\n");

    int sz_qkv = bs * nh * seqlen * hdim;
    int sz_table = seqlen * hdim;

    float *h_Q = new float[sz_qkv];
    float *h_K = new float[sz_qkv];
    float *h_V = new float[sz_qkv];
    float *h_sin = new float[sz_table];
    float *h_cos = new float[sz_table];
    float *h_out_hybrid = new float[sz_qkv];
    float *h_out_separate = new float[sz_qkv];
    float *h_out_cpu = new float[sz_qkv];

    init_random(h_Q, sz_qkv);
    init_random(h_K, sz_qkv);
    init_random(h_V, sz_qkv);
    precompute_sinusoidal_table(h_sin, h_cos, seqlen, hdim);

    float *d_Q, *d_K, *d_V, *d_sin, *d_cos;
    float *d_Q_emb, *d_K_emb, *d_out;

    CUDA_CHECK(cudaMalloc(&d_Q, sz_qkv * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, sz_qkv * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V, sz_qkv * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sin, sz_table * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cos, sz_table * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Q_emb, sz_qkv * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K_emb, sz_qkv * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, sz_qkv * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_Q, h_Q, sz_qkv * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K, sz_qkv * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V, sz_qkv * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sin, h_sin, sz_table * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cos, h_cos, sz_table * sizeof(float), cudaMemcpyHostToDevice));

    // Warmup
    attention_forward_sinusoidal_hybrid_v2(d_Q, d_K, d_V, d_sin, d_cos, d_K_emb, d_out, bs, nh, seqlen, seqlen, hdim);
    attention_forward_sinusoidal_separate(d_Q, d_K, d_V, d_sin, d_cos, d_Q_emb, d_K_emb, d_out, bs, nh, seqlen, seqlen, hdim);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    const int num_iters = 100;

    // Benchmark hybrid V2
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iters; i++) {
        attention_forward_sinusoidal_hybrid_v2(d_Q, d_K, d_V, d_sin, d_cos, d_K_emb, d_out, bs, nh, seqlen, seqlen, hdim);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float hybrid_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&hybrid_ms, start, stop));
    hybrid_ms /= num_iters;

    CUDA_CHECK(cudaMemcpy(h_out_hybrid, d_out, sz_qkv * sizeof(float), cudaMemcpyDeviceToHost));

    // Benchmark separate
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iters; i++) {
        attention_forward_sinusoidal_separate(d_Q, d_K, d_V, d_sin, d_cos, d_Q_emb, d_K_emb, d_out, bs, nh, seqlen, seqlen, hdim);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float separate_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&separate_ms, start, stop));
    separate_ms /= num_iters;

    CUDA_CHECK(cudaMemcpy(h_out_separate, d_out, sz_qkv * sizeof(float), cudaMemcpyDeviceToHost));

    // CPU reference
    attention_cpu_sinusoidal(h_Q, h_K, h_V, h_sin, h_cos, h_out_cpu, bs, nh, seqlen, seqlen, hdim);

    printf("\n--- Performance Results ---\n");
    printf("  Hybrid V2 (on-the-fly Q): %.3f ms", hybrid_ms);
    if (hybrid_ms < separate_ms) {
        printf("  [%.2fx FASTER] ✓\n", separate_ms / hybrid_ms);
    } else {
        printf("  [%.2fx slower]\n", hybrid_ms / separate_ms);
    }

    printf("  Separate (baseline):      %.3f ms\n", separate_ms);

    float improvement = ((separate_ms - hybrid_ms) / separate_ms) * 100.0f;
    if (improvement > 0) {
        printf("  Speedup: +%.1f%%\n", improvement);
    } else {
        printf("  Slowdown: %.1f%%\n", improvement);
    }

    printf("\nChecking hybrid vs CPU...\n");
    bool hybrid_passed = check_results(h_out_hybrid, h_out_cpu, sz_qkv);

    printf("\nChecking separate vs CPU...\n");
    bool separate_passed = check_results(h_out_separate, h_out_cpu, sz_qkv);

    if (hybrid_passed && separate_passed) {
        printf("\n✓ ALL TESTS PASSED\n");
    } else {
        printf("\n✗ TESTS FAILED\n");
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    delete[] h_Q; delete[] h_K; delete[] h_V;
    delete[] h_sin; delete[] h_cos;
    delete[] h_out_hybrid; delete[] h_out_separate; delete[] h_out_cpu;

    CUDA_CHECK(cudaFree(d_Q)); CUDA_CHECK(cudaFree(d_K)); CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_sin)); CUDA_CHECK(cudaFree(d_cos));
    CUDA_CHECK(cudaFree(d_Q_emb)); CUDA_CHECK(cudaFree(d_K_emb)); CUDA_CHECK(cudaFree(d_out));
}

int main() {
    printf("=================================================\n");
    printf("Sinusoidal Hybrid V2 Attention Tests\n");
    printf("V2: Pre-compute K, compute Q on-the-fly\n");
    printf("=================================================\n");

    srand(42);

    benchmark_hybrid_vs_separate(1, 4, 64, 512);
    benchmark_hybrid_vs_separate(1, 4, 64, 2048);
    benchmark_hybrid_vs_separate(1, 2, 32, 4096);
    benchmark_hybrid_vs_separate(1, 1, 16, 8192);

    printf("\n=================================================\n");
    printf("All tests completed!\n");
    printf("=================================================\n");

    return 0;
}
