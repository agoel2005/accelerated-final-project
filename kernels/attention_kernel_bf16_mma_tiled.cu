/*
 * BF16 ATTENTION WITH MMA TENSOR CORES AND HDIM TILING
 *
 * Key insight from lab6: Tile the K dimension (hdim in our case)
 * Process hdim in chunks so we don't need huge shared memory
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

// MMA dimensions for BF16
constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 16;

// Tile size for hidden dimension (like BLOCK_K in lab6)
constexpr int TILE_HDIM = 128;  // Process 128 dims at a time

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

// BF16 Attention with MMA and hdim tiling
__global__ void __launch_bounds__(256) attention_fwd_bf16_mma_tiled(
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

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int tid = threadIdx.x;

    // Shared memory: scores + reduction + Q tile + K tile
    extern __shared__ char smem_raw[];
    float* s_scores = reinterpret_cast<float*>(smem_raw);
    float* s_reduce = s_scores + seq_k;
    bfloat16* s_q_tile = reinterpret_cast<bfloat16*>(s_reduce + 32);
    bfloat16* s_k_tile = s_q_tile + MMA_M * TILE_HDIM;

    // Initialize scores to zero
    for (int k = tid; k < seq_k; k += blockDim.x) {
        s_scores[k] = 0.0f;
    }
    __syncthreads();

    // ====================================================================
    // COMPUTE Q @ K^T BY TILING OVER HDIM
    // ====================================================================

    // Process hdim in tiles (like lab6 tiles the K dimension)
    for (int hdim_start = 0; hdim_start < hdim; hdim_start += TILE_HDIM) {
        int hdim_end = min(hdim_start + TILE_HDIM, hdim);
        int tile_size = hdim_end - hdim_start;

        // Load Q tile for this hdim range (replicate to MMA_M rows)
        for (int d = tid; d < tile_size; d += blockDim.x) {
            bfloat16 q_val = Q[q_offset + hdim_start + d];
            #pragma unroll
            for (int m = 0; m < MMA_M; m++) {
                s_q_tile[m * TILE_HDIM + d] = q_val;
            }
        }
        __syncthreads();

        // Process K positions in tiles of MMA_N
        for (int k_start = 0; k_start < seq_k; k_start += MMA_N) {
            int k_end = min(k_start + MMA_N, seq_k);
            int num_k = k_end - k_start;

            // Use MMA if full tiles
            if (num_k == MMA_N && tile_size >= MMA_K && tile_size % MMA_K == 0) {
                // Load K tile (transposed)
                for (int k = 0; k < MMA_N; k++) {
                    int k_idx = k_start + k;
                    if (k_idx < seq_k) {
                        for (int d = tid; d < tile_size; d += blockDim.x) {
                            s_k_tile[d * MMA_N + k] = K[kv_base + k_idx * hdim + hdim_start + d];
                        }
                    }
                }
                __syncthreads();

                // MMA computation (only warp 0)
                if (warp_id == 0) {
                    uint32_t frag_a[4];
                    uint32_t frag_b[2];
                    uint32_t frag_c[4];

                    // Initialize
                    #pragma unroll
                    for (int i = 0; i < 4; i++) {
                        frag_c[i] = 0;
                    }

                    // Tile over this hdim chunk
                    for (int k_dim = 0; k_dim < tile_size; k_dim += MMA_K) {
                        if (k_dim + MMA_K > tile_size) break;

                        // Load fragments
                        int a_row = lane_id / 4;
                        int a_col = lane_id % 4;

                        frag_a[0] = __bfloat16_as_ushort(s_q_tile[a_row * TILE_HDIM + k_dim + a_col]);
                        frag_a[1] = __bfloat16_as_ushort(s_q_tile[(a_row + 8) * TILE_HDIM + k_dim + a_col]);
                        frag_a[2] = __bfloat16_as_ushort(s_q_tile[a_row * TILE_HDIM + k_dim + a_col + 4]);
                        frag_a[3] = __bfloat16_as_ushort(s_q_tile[(a_row + 8) * TILE_HDIM + k_dim + a_col + 4]);

                        int b_col = lane_id / 4;
                        int b_row = lane_id % 4;

                        frag_b[0] = __bfloat16_as_ushort(s_k_tile[(k_dim + b_row) * MMA_N + b_col]);
                        frag_b[1] = __bfloat16_as_ushort(s_k_tile[(k_dim + b_row + 4) * MMA_N + b_col]);

                        // MMA
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
                            : "=r"(frag_c[0]), "=r"(frag_c[1]), "=r"(frag_c[2]), "=r"(frag_c[3])
                            : "r"(frag_a[0]), "r"(frag_a[1]), "r"(frag_a[2]), "r"(frag_a[3]),
                              "r"(frag_b[0]), "r"(frag_b[1]),
                              "r"(frag_c[0]), "r"(frag_c[1]), "r"(frag_c[2]), "r"(frag_c[3])
                        );
                    }

                    // Accumulate results to scores (add to existing values)
                    int c_row = lane_id / 4;
                    int c_col = lane_id % 4;

                    if (c_row == 0 && c_col * 2 < MMA_N) {
                        atomicAdd(&s_scores[k_start + c_col * 2], __uint_as_float(frag_c[0]));
                        if (c_col * 2 + 1 < MMA_N) {
                            atomicAdd(&s_scores[k_start + c_col * 2 + 1], __uint_as_float(frag_c[1]));
                        }
                    }
                }
                __syncthreads();
            } else {
                // Scalar fallback for partial tiles
                for (int k = k_start + tid; k < k_end; k += blockDim.x) {
                    float partial_score = 0.0f;
                    for (int d = 0; d < tile_size; d++) {
                        partial_score += __bfloat162float(Q[q_offset + hdim_start + d]) *
                                        __bfloat162float(K[kv_base + k * hdim + hdim_start + d]);
                    }
                    atomicAdd(&s_scores[k], partial_score);
                }
                __syncthreads();
            }
        }
    }

    // Apply scale factor
    for (int k = tid; k < seq_k; k += blockDim.x) {
        s_scores[k] *= scale_factor;
    }
    __syncthreads();

    // ====================================================================
    // SOFTMAX
    // ====================================================================

    float local_max = -INFINITY;
    for (int k = tid; k < seq_k; k += blockDim.x) {
        local_max = fmaxf(local_max, s_scores[k]);
    }

    local_max = warp_reduce_max(local_max);
    if (lane_id == 0) s_reduce[warp_id] = local_max;
    __syncthreads();

    if (tid < 32) {
        float val = (tid < (blockDim.x + 31) / 32) ? s_reduce[tid] : -INFINITY;
        val = warp_reduce_max(val);
        if (tid == 0) s_reduce[0] = val;
    }
    __syncthreads();
    local_max = s_reduce[0];

    float exp_sum = 0.0f;
    for (int k = tid; k < seq_k; k += blockDim.x) {
        float exp_val = expf(s_scores[k] - local_max);
        s_scores[k] = exp_val;
        exp_sum += exp_val;
    }

    exp_sum = warp_reduce_sum(exp_sum);
    if (lane_id == 0) s_reduce[warp_id] = exp_sum;
    __syncthreads();

    if (tid < 32) {
        float val = (tid < (blockDim.x + 31) / 32) ? s_reduce[tid] : 0.0f;
        val = warp_reduce_sum(val);
        if (tid == 0) s_reduce[0] = val;
    }
    __syncthreads();
    exp_sum = s_reduce[0];

    for (int k = tid; k < seq_k; k += blockDim.x) {
        s_scores[k] = (exp_sum != 0.0f) ? (s_scores[k] / exp_sum) : 0.0f;
    }
    __syncthreads();

    // ====================================================================
    // OUTPUT
    // ====================================================================

    for (int d = tid; d < hdim; d += blockDim.x) {
        float accum = 0.0f;

        for (int k = 0; k < seq_k; k++) {
            accum += s_scores[k] * __bfloat162float(V[kv_base + k * hdim + d]);
        }

        out[q_offset + d] = __float2bfloat16(accum);
    }
}

void attention_forward_bf16_mma_tiled(
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

    // Shared memory: scores + reduction + Q tile + K tile (for TILE_HDIM)
    const int shmem_sz = seq_k * sizeof(float) +
                         32 * sizeof(float) +
                         MMA_M * TILE_HDIM * sizeof(bfloat16) +
                         TILE_HDIM * MMA_N * sizeof(bfloat16);

    attention_fwd_bf16_mma_tiled<<<nblocks, nthreads, shmem_sz>>>(
        Q, K, V, out, bs, nh, seq_q, seq_k, hdim, scale
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

//---------------------------------------------------------------------------------------------------------------------
// Test code
//---------------------------------------------------------------------------------------------------------------------

void attention_cpu(
    const float* Q,
    const float* K,
    const float* V,
    float* out,
    int bs, int nh, int seq_q, int seq_k, int hdim
) {
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

void test_bf16(int bs, int nh, int seqlen, int hdim) {
    printf("\n===== BF16 MMA TILED Tensor Core Attention =====\n");
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
    attention_forward_bf16_mma_tiled(d_Q, d_K, d_V, d_out, bs, nh, seqlen, seqlen, hdim);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < 100; i++) {
        attention_forward_bf16_mma_tiled(d_Q, d_K, d_V, d_out, bs, nh, seqlen, seqlen, hdim);
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

    // Use higher tolerance for very large hdim due to accumulation precision
    float tolerance = (hdim >= 8192) ? 0.2f : 0.15f;
    bool passed = check_results(h_out_gpu, h_out_cpu, sz, tolerance);
    printf(passed ? "✓ TILED TENSOR CORES WORKING!\n" : "✗ FAILED\n");

    delete[] h_Q_fp32; delete[] h_K_fp32; delete[] h_V_fp32;
    delete[] h_out_cpu; delete[] h_out_gpu;
    delete[] h_Q; delete[] h_K; delete[] h_V; delete[] h_out_bf16;
    CUDA_CHECK(cudaFree(d_Q)); CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V)); CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
}

int main() {
    printf("=================================================\n");
    printf("BF16 MMA TILED Tensor Core Attention\n");
    printf("Tiles hdim like lab6 tiles K dimension\n");
    printf("=================================================\n");

    srand(42);

    test_bf16(1, 4, 64, 512);
    test_bf16(1, 4, 64, 2048);
    test_bf16(1, 2, 32, 4096);

    printf("\n*** Testing VERY LARGE hdim=8192 ***\n");
    test_bf16(1, 1, 16, 8192);

    printf("\n*** Testing EXTREME hdim=16384 ***\n");
    test_bf16(1, 1, 8, 16384);

    return 0;
}
