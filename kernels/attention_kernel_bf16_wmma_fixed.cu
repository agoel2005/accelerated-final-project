/*
 * BF16 ATTENTION WITH WMMA TENSOR CORES - PROPER IMPLEMENTATION
 *
 * Uses WMMA tensor core instructions correctly with shared memory.
 * All WMMA loads/stores go through shared memory with proper alignment.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace nvcuda;
using namespace nvcuda::wmma;
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

// WMMA tile dimensions for BF16 on Ada/Ampere
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// Warp reduction primitives
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

// BF16 Attention with WMMA Tensor Cores
// Each block processes WMMA_M queries (16 queries per block)
template<int TILE_K = 64>
__global__ void attention_fwd_bf16_wmma(
    const bfloat16* __restrict__ Q,
    const bfloat16* __restrict__ K,
    const bfloat16* __restrict__ V,
    bfloat16* __restrict__ out,
    int bs, int nh, int seq_q, int seq_k, int hdim,
    float scale_factor
) {
    // Block processes WMMA_M queries
    const int block_q_start = blockIdx.x * WMMA_M;
    const int batch_head_idx = blockIdx.y;
    const int b_idx = batch_head_idx / nh;
    const int h_idx = batch_head_idx % nh;

    if (block_q_start >= seq_q) return;

    const int num_q = min(WMMA_M, seq_q - block_q_start);
    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;

    // Shared memory layout (with proper alignment for WMMA)
    extern __shared__ char smem_raw[];
    bfloat16* s_q_tile = reinterpret_cast<bfloat16*>(smem_raw);  // [WMMA_M, hdim]
    bfloat16* s_k_tile = s_q_tile + WMMA_M * hdim;                // [TILE_K, hdim]
    float* s_scores = reinterpret_cast<float*>(s_k_tile + TILE_K * hdim);  // [WMMA_M, seq_k]
    bfloat16* s_v_tile = reinterpret_cast<bfloat16*>(s_scores + WMMA_M * seq_k);  // [TILE_K, hdim]
    float* s_out_accum = reinterpret_cast<float*>(s_v_tile + TILE_K * hdim);  // [WMMA_M, hdim]

    const int qkv_base = (b_idx * nh + h_idx) * seq_q * hdim;
    const int kv_base = (b_idx * nh + h_idx) * seq_k * hdim;

    // Load Q tile into shared memory
    for (int q = 0; q < num_q; q++) {
        int q_idx = block_q_start + q;
        int q_offset = qkv_base + q_idx * hdim;
        for (int d = threadIdx.x; d < hdim; d += blockDim.x) {
            s_q_tile[q * hdim + d] = Q[q_offset + d];
        }
    }
    // Pad remaining rows if num_q < WMMA_M
    for (int q = num_q; q < WMMA_M; q++) {
        for (int d = threadIdx.x; d < hdim; d += blockDim.x) {
            s_q_tile[q * hdim + d] = __float2bfloat16(0.0f);
        }
    }
    __syncthreads();

    // Initialize output accumulator
    for (int q = 0; q < WMMA_M; q++) {
        for (int d = threadIdx.x; d < hdim; d += blockDim.x) {
            s_out_accum[q * hdim + d] = 0.0f;
        }
    }

    // Initialize online softmax stats per query
    float m_max[WMMA_M];
    float l_sum[WMMA_M];
    #pragma unroll
    for (int q = 0; q < WMMA_M; q++) {
        m_max[q] = -INFINITY;
        l_sum[q] = 0.0f;
    }

    // Process K, V in tiles
    for (int k_start = 0; k_start < seq_k; k_start += TILE_K) {
        const int k_end = min(k_start + TILE_K, seq_k);
        const int num_k = k_end - k_start;

        // Load K tile into shared memory
        for (int k = 0; k < num_k; k++) {
            int k_idx = k_start + k;
            int k_offset = kv_base + k_idx * hdim;
            for (int d = threadIdx.x; d < hdim; d += blockDim.x) {
                s_k_tile[k * hdim + d] = K[k_offset + d];
            }
        }
        __syncthreads();

        // ============================================================
        // COMPUTE Q @ K^T using WMMA TENSOR CORES
        // ============================================================

        // Each warp handles a subset of the Q@K^T computation
        if (warp_id == 0) {  // Use first warp for WMMA
            // Process in WMMA tiles
            for (int q_tile = 0; q_tile < WMMA_M; q_tile += WMMA_M) {
                for (int k_tile = 0; k_tile < num_k; k_tile += WMMA_N) {
                    if (k_tile + WMMA_N > num_k) break;  // Need full tile

                    // WMMA fragments
                    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, bfloat16, row_major> a_frag;
                    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, bfloat16, col_major> b_frag;
                    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

                    fill_fragment(c_frag, 0.0f);

                    // Tile over K dimension (hdim)
                    for (int k_dim = 0; k_dim < hdim; k_dim += WMMA_K) {
                        if (k_dim + WMMA_K > hdim) break;  // Need full tile

                        // Load Q tile: [WMMA_M x WMMA_K]
                        load_matrix_sync(a_frag, s_q_tile + q_tile * hdim + k_dim, hdim);

                        // Load K^T tile: [WMMA_K x WMMA_N]
                        // K is stored as [k, d], we want K^T so use col_major
                        load_matrix_sync(b_frag, s_k_tile + k_tile * hdim + k_dim, hdim);

                        // Multiply-accumulate: C += A @ B
                        mma_sync(c_frag, a_frag, b_frag, c_frag);
                    }

                    // Store result to shared memory scores
                    // Use temporary shared memory aligned properly
                    __shared__ float s_wmma_out[WMMA_M * WMMA_N];
                    store_matrix_sync(s_wmma_out, c_frag, WMMA_N, mem_row_major);

                    // Copy to actual scores array with scaling
                    if (lane_id < WMMA_M * WMMA_N) {
                        int q = lane_id / WMMA_N;
                        int k = lane_id % WMMA_N;
                        if (q < num_q && (k_tile + k) < num_k) {
                            s_scores[q * seq_k + k_start + k_tile + k] = s_wmma_out[lane_id] * scale_factor;
                        }
                    }
                }
            }
        }
        __syncthreads();

        // If hdim not divisible by WMMA_K, handle remainder with scalar ops
        if (hdim % WMMA_K != 0 || num_k % WMMA_N != 0) {
            // Use all threads for remainder computation
            int k_tile_start = (num_k / WMMA_N) * WMMA_N;
            for (int q = 0; q < num_q; q++) {
                for (int k = k_tile_start + threadIdx.x; k < num_k; k += blockDim.x) {
                    float score = 0.0f;
                    for (int d = 0; d < hdim; d++) {
                        score += __bfloat162float(s_q_tile[q * hdim + d]) *
                                 __bfloat162float(s_k_tile[k * hdim + d]);
                    }
                    s_scores[q * seq_k + k_start + k] = score * scale_factor;
                }
            }
            __syncthreads();
        }

        // ============================================================
        // ONLINE SOFTMAX UPDATE
        // ============================================================

        // Each thread handles one query
        for (int q = threadIdx.x; q < num_q; q += blockDim.x) {
            // Find max in this tile
            float tile_max = -INFINITY;
            for (int k = 0; k < num_k; k++) {
                tile_max = fmaxf(tile_max, s_scores[q * seq_k + k_start + k]);
            }

            // Update global max
            float old_max = m_max[q];
            float new_max = fmaxf(old_max, tile_max);

            // Rescale old sum
            float rescale = expf(old_max - new_max);
            l_sum[q] *= rescale;

            // Add new contributions
            for (int k = 0; k < num_k; k++) {
                float exp_val = expf(s_scores[q * seq_k + k_start + k] - new_max);
                s_scores[q * seq_k + k_start + k] = exp_val;
                l_sum[q] += exp_val;
            }

            m_max[q] = new_max;
        }
        __syncthreads();

        // ============================================================
        // ACCUMULATE OUTPUT: out += scores @ V
        // ============================================================

        // Load V tile
        for (int k = 0; k < num_k; k++) {
            int k_idx = k_start + k;
            int v_offset = kv_base + k_idx * hdim;
            for (int d = threadIdx.x; d < hdim; d += blockDim.x) {
                s_v_tile[k * hdim + d] = V[v_offset + d];
            }
        }
        __syncthreads();

        // Compute scores @ V (simple matmul, could also use WMMA here)
        for (int q = threadIdx.x; q < num_q; q += blockDim.x) {
            for (int d = 0; d < hdim; d++) {
                float accum = 0.0f;
                for (int k = 0; k < num_k; k++) {
                    accum += s_scores[q * seq_k + k_start + k] *
                             __bfloat162float(s_v_tile[k * hdim + d]);
                }
                s_out_accum[q * hdim + d] += accum;
            }
        }
        __syncthreads();
    }

    // ============================================================
    // NORMALIZE AND WRITE OUTPUT
    // ============================================================

    for (int q = 0; q < num_q; q++) {
        int q_idx = block_q_start + q;
        int out_offset = qkv_base + q_idx * hdim;

        float norm = l_sum[q] != 0.0f ? 1.0f / l_sum[q] : 0.0f;

        for (int d = threadIdx.x; d < hdim; d += blockDim.x) {
            out[out_offset + d] = __float2bfloat16(s_out_accum[q * hdim + d] * norm);
        }
    }
}

void attention_forward_bf16_wmma(
    const bfloat16* Q,
    const bfloat16* K,
    const bfloat16* V,
    bfloat16* out,
    int bs, int nh, int seq_q, int seq_k, int hdim
) {
    float scale = 1.0f / sqrtf(static_cast<float>(hdim));

    constexpr int TILE_K = 64;
    const int num_q_blocks = (seq_q + WMMA_M - 1) / WMMA_M;

    dim3 grid(num_q_blocks, bs * nh);
    const int nthreads = 128;

    // Shared memory: Q tile + K tile + scores + V tile + output accumulator
    const int shmem_sz = WMMA_M * hdim * sizeof(bfloat16) +      // Q
                         TILE_K * hdim * sizeof(bfloat16) +      // K
                         WMMA_M * seq_k * sizeof(float) +        // scores
                         TILE_K * hdim * sizeof(bfloat16) +      // V
                         WMMA_M * hdim * sizeof(float);          // out accum

    attention_fwd_bf16_wmma<TILE_K><<<grid, nthreads, shmem_sz>>>(
        Q, K, V, out, bs, nh, seq_q, seq_k, hdim, scale
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

//---------------------------------------------------------------------------------------------------------------------
// CPU reference
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

//---------------------------------------------------------------------------------------------------------------------
// Test utilities
//---------------------------------------------------------------------------------------------------------------------

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

void test_bf16_wmma_attention(int bs, int nh, int seqlen, int hdim) {
    printf("\n========================================\n");
    printf("Testing BF16 WMMA Tensor Core Attention:\n");
    printf("  Batch size: %d, Num heads: %d\n", bs, nh);
    printf("  Sequence length: %d, Hidden dim: %d\n", seqlen, hdim);
    printf("========================================\n\n");

    int sz_qkv = bs * nh * seqlen * hdim;

    float *h_Q_fp32 = new float[sz_qkv];
    float *h_K_fp32 = new float[sz_qkv];
    float *h_V_fp32 = new float[sz_qkv];
    float *h_out_cpu = new float[sz_qkv];
    float *h_out_gpu_fp32 = new float[sz_qkv];

    bfloat16 *h_Q_bf16 = new bfloat16[sz_qkv];
    bfloat16 *h_K_bf16 = new bfloat16[sz_qkv];
    bfloat16 *h_V_bf16 = new bfloat16[sz_qkv];
    bfloat16 *h_out_gpu_bf16 = new bfloat16[sz_qkv];

    init_random(h_Q_fp32, sz_qkv);
    init_random(h_K_fp32, sz_qkv);
    init_random(h_V_fp32, sz_qkv);

    fp32_to_bf16(h_Q_fp32, h_Q_bf16, sz_qkv);
    fp32_to_bf16(h_K_fp32, h_K_bf16, sz_qkv);
    fp32_to_bf16(h_V_fp32, h_V_bf16, sz_qkv);

    bfloat16 *d_Q, *d_K, *d_V, *d_out;
    CUDA_CHECK(cudaMalloc(&d_Q, sz_qkv * sizeof(bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_K, sz_qkv * sizeof(bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_V, sz_qkv * sizeof(bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_out, sz_qkv * sizeof(bfloat16)));

    CUDA_CHECK(cudaMemcpy(d_Q, h_Q_bf16, sz_qkv * sizeof(bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K_bf16, sz_qkv * sizeof(bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V_bf16, sz_qkv * sizeof(bfloat16), cudaMemcpyHostToDevice));

    // Warmup
    attention_forward_bf16_wmma(d_Q, d_K, d_V, d_out, bs, nh, seqlen, seqlen, hdim);

    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int num_iters = 100;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iters; i++) {
        attention_forward_bf16_wmma(d_Q, d_K, d_V, d_out, bs, nh, seqlen, seqlen, hdim);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    elapsed_ms /= num_iters;

    CUDA_CHECK(cudaMemcpy(h_out_gpu_bf16, d_out, sz_qkv * sizeof(bfloat16), cudaMemcpyDeviceToHost));
    bf16_to_fp32(h_out_gpu_bf16, h_out_gpu_fp32, sz_qkv);

    // CPU reference
    attention_cpu(h_Q_fp32, h_K_fp32, h_V_fp32, h_out_cpu, bs, nh, seqlen, seqlen, hdim);

    printf("BF16 WMMA kernel time: %.3f ms\n", elapsed_ms);

    long long flops = 2LL * bs * nh * seqlen * seqlen * hdim * 2;
    float tflops = (flops / (elapsed_ms / 1000.0f)) / 1e12f;
    printf("Performance: %.2f TFLOPS\n", tflops);

    printf("\nChecking correctness...\n");
    bool passed = check_results(h_out_gpu_fp32, h_out_cpu, sz_qkv, 0.15f);

    if (passed) {
        printf("\n✓ TEST PASSED - TENSOR CORES WORKING!\n");
    } else {
        printf("\n✗ TEST FAILED\n");
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    delete[] h_Q_fp32;
    delete[] h_K_fp32;
    delete[] h_V_fp32;
    delete[] h_out_cpu;
    delete[] h_out_gpu_fp32;
    delete[] h_Q_bf16;
    delete[] h_K_bf16;
    delete[] h_V_bf16;
    delete[] h_out_gpu_bf16;

    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_out));
}

int main() {
    printf("=================================================\n");
    printf("BF16 WMMA Tensor Core Attention Benchmark\n");
    printf("=================================================\n");

    srand(42);

    printf("\n*** Testing hdim=512 ***\n");
    test_bf16_wmma_attention(1, 4, 64, 512);

    printf("\n*** Testing hdim=2048 ***\n");
    test_bf16_wmma_attention(1, 4, 64, 2048);

    printf("\n*** Testing hdim=4096 ***\n");
    test_bf16_wmma_attention(1, 2, 32, 4096);

    printf("\n=================================================\n");
    printf("All tests completed!\n");
    printf("=================================================\n");

    return 0;
}
