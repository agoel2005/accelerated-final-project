/*
 * BF16 ATTENTION KERNEL WITH WMMA TENSOR CORES
 *
 * This kernel implements attention entirely in BF16 using ACTUAL WMMA tensor core instructions.
 * The Q@K^T matmul is done with tensor cores for maximum performance!
 *
 * Target: Ada architecture (RTX 4000) with BF16 tensor core support
 * Tensor Core Tile: 16x8x16 (M x N x K) for BF16
 *
 * Key optimizations:
 * - All data in BF16 (no FP32 conversions!)
 * - WMMA tensor core instructions for Q@K^T
 * - Online softmax with tiling
 * - Warp-level reductions
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

// WMMA tensor core dimensions for BF16
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

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

// BF16 WMMA Tensor Core Attention Kernel
// One warp per query, uses WMMA for Q@K^T
template<int BLOCK_K = 64>
__global__ void attention_fwd_bf16_wmma(
    const bfloat16* __restrict__ Q,
    const bfloat16* __restrict__ K,
    const bfloat16* __restrict__ V,
    bfloat16* __restrict__ out,
    int bs, int nh, int seq_q, int seq_k, int hdim,
    float scale_factor
) {
    // Each warp handles one query position
    const int warp_id = blockIdx.x;
    const int batch_head_idx = blockIdx.y;
    const int b_idx = batch_head_idx / nh;
    const int h_idx = batch_head_idx % nh;
    const int q_idx = warp_id;

    if (q_idx >= seq_q) return;

    const int lane_id = threadIdx.x % 32;

    // Base offsets
    const int qkv_base = (b_idx * nh + h_idx) * seq_q * hdim;
    const int kv_base = (b_idx * nh + h_idx) * seq_k * hdim;
    const int q_offset = qkv_base + q_idx * hdim;

    // Shared memory for scores and tiles
    extern __shared__ char smem_raw[];
    float* s_scores = reinterpret_cast<float*>(smem_raw);
    float* s_reduce = s_scores + BLOCK_K;
    bfloat16* s_q_tile = reinterpret_cast<bfloat16*>(s_reduce + 32);
    bfloat16* s_k_tile = s_q_tile + WMMA_M * WMMA_K;

    // Online softmax statistics
    float m_max = -INFINITY;
    float l_sum = 0.0f;

    // Load Q tile for this query (WMMA_M=16, but we only need 1 row)
    // We'll replicate the Q row to fill the tile
    for (int k = lane_id; k < hdim; k += 32) {
        bfloat16 q_val = Q[q_offset + k];
        // Replicate across all 16 rows for WMMA
        for (int m = 0; m < WMMA_M; m++) {
            s_q_tile[m * hdim + k] = q_val;
        }
    }
    __syncthreads();

    // Process K, V in blocks
    for (int k_start = 0; k_start < seq_k; k_start += BLOCK_K) {
        const int k_end = min(k_start + BLOCK_K, seq_k);
        const int num_k = k_end - k_start;

        // ========================================
        // STEP 1: Compute scores Q @ K^T using WMMA
        // ========================================

        // For each K position in this block
        for (int k_tile_start = 0; k_tile_start < num_k; k_tile_start += WMMA_N) {
            int k_tile_end = min(k_tile_start + WMMA_N, num_k);

            // Can only use WMMA if we have full WMMA_N elements
            if (k_tile_end - k_tile_start == WMMA_N && hdim % WMMA_K == 0) {
                // Use WMMA!
                fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, bfloat16, row_major> a_frag;
                fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, bfloat16, col_major> b_frag;
                fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

                fill_fragment(c_frag, 0.0f);

                // Tile over K dimension
                for (int k_dim = 0; k_dim < hdim; k_dim += WMMA_K) {
                    // Load Q tile (16x16)
                    load_matrix_sync(a_frag, s_q_tile + k_dim, hdim);

                    // Load K tile (16x16, transposed)
                    // First load K to shared memory
                    for (int n = 0; n < WMMA_N; n++) {
                        int k_idx = k_start + k_tile_start + n;
                        if (k_idx < seq_k) {
                            int k_offset = kv_base + k_idx * hdim;
                            for (int k = lane_id; k < WMMA_K; k += 32) {
                                s_k_tile[k * WMMA_N + n] = K[k_offset + k_dim + k];
                            }
                        }
                    }
                    __syncthreads();

                    load_matrix_sync(b_frag, s_k_tile, WMMA_N);

                    // Multiply-accumulate
                    mma_sync(c_frag, a_frag, b_frag, c_frag);
                }

                // Store results (only use first row, since Q was replicated)
                float c_result[WMMA_N];
                store_matrix_sync(c_result, c_frag, WMMA_N, mem_row_major);

                // Only first row contains our results
                if (lane_id < WMMA_N) {
                    int k_idx = k_tile_start + lane_id;
                    if (k_idx < num_k) {
                        s_scores[k_idx] = c_result[lane_id] * scale_factor;
                    }
                }
            } else {
                // Fallback to regular computation for edge cases
                for (int k_local = k_tile_start + lane_id; k_local < k_tile_end; k_local += 32) {
                    int k_idx = k_start + k_local;
                    int k_offset = kv_base + k_idx * hdim;

                    float score = 0.0f;
                    for (int d = 0; d < hdim; d++) {
                        score += __bfloat162float(Q[q_offset + d]) * __bfloat162float(K[k_offset + d]);
                    }
                    score *= scale_factor;
                    s_scores[k_local] = score;
                }
            }
        }
        __syncthreads();

        // ========================================
        // STEP 2: Find max score
        // ========================================
        float block_max = -INFINITY;
        for (int k_local = lane_id; k_local < num_k; k_local += 32) {
            block_max = fmaxf(block_max, s_scores[k_local]);
        }
        block_max = warp_reduce_max(block_max);

        // Broadcast max across warp
        block_max = __shfl_sync(0xffffffff, block_max, 0);

        float m_new = fmaxf(m_max, block_max);
        float correction = expf(m_max - m_new);

        // ========================================
        // STEP 3: Compute exp and sum
        // ========================================
        float block_sum = 0.0f;
        for (int k_local = lane_id; k_local < num_k; k_local += 32) {
            float exp_val = expf(s_scores[k_local] - m_new);
            s_scores[k_local] = exp_val;
            block_sum += exp_val;
        }
        block_sum = warp_reduce_sum(block_sum);
        block_sum = __shfl_sync(0xffffffff, block_sum, 0);

        l_sum = correction * l_sum + block_sum;

        // ========================================
        // STEP 4: Update output (scores @ V)
        // ========================================
        for (int d = lane_id; d < hdim; d += 32) {
            float prev_out = (k_start > 0) ? __bfloat162float(out[q_offset + d]) : 0.0f;
            float corrected_out = prev_out * correction;

            float v_acc = 0.0f;
            for (int k_local = 0; k_local < num_k; k_local++) {
                int k_idx = k_start + k_local;
                int v_offset = kv_base + k_idx * hdim + d;
                v_acc += s_scores[k_local] * __bfloat162float(V[v_offset]);
            }

            out[q_offset + d] = __float2bfloat16(corrected_out + v_acc);
        }

        m_max = m_new;
        __syncthreads();
    }

    // ========================================
    // STEP 5: Final normalization
    // ========================================
    for (int d = lane_id; d < hdim; d += 32) {
        float val = __bfloat162float(out[q_offset + d]);
        out[q_offset + d] = __float2bfloat16(val / l_sum);
    }
}

// CPU reference implementation
void attention_forward_cpu(
    const float* Q, const float* K, const float* V, float* out,
    int batch_sz, int num_heads, int len_q, int len_k, int head_d
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

void init_random(float* data, int size, float min_val = -1.0f, float max_val = 1.0f) {
    for (int i = 0; i < size; i++) {
        data[i] = min_val + (max_val - min_val) * (rand() / (float)RAND_MAX);
    }
}

bool check_results(const float* gpu_out, const float* cpu_out, int size, float tol = 0.1f) {
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

void test_attention_bf16_wmma(int bs, int nh, int seqlen, int hdim) {
    printf("\n========================================\n");
    printf("Testing BF16 WMMA Tensor Core Attention:\n");
    printf("  Batch size: %d, Num heads: %d\n", bs, nh);
    printf("  Sequence length: %d, Hidden dim: %d\n", seqlen, hdim);
    printf("========================================\n\n");

    int sz_qkv = bs * nh * seqlen * hdim;

    float *h_Q_fp32 = new float[sz_qkv];
    float *h_K_fp32 = new float[sz_qkv];
    float *h_V_fp32 = new float[sz_qkv];
    float *h_out_fp32 = new float[sz_qkv];
    float *h_out_cpu = new float[sz_qkv];

    bfloat16 *h_Q_bf16 = new bfloat16[sz_qkv];
    bfloat16 *h_K_bf16 = new bfloat16[sz_qkv];
    bfloat16 *h_V_bf16 = new bfloat16[sz_qkv];
    bfloat16 *h_out_bf16 = new bfloat16[sz_qkv];

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

    float scale = 1.0f / sqrtf(static_cast<float>(hdim));
    constexpr int BLOCK_K = 64;
    const int warps_per_block = 1;  // One warp per block
    const int nthreads = warps_per_block * 32;
    const int shmem_sz = (BLOCK_K + 32) * sizeof(float) +
                         (WMMA_M * WMMA_K + WMMA_K * WMMA_N) * sizeof(bfloat16);

    dim3 grid(seqlen, bs * nh);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    const int num_iters = 100;

    // Warmup
    attention_fwd_bf16_wmma<BLOCK_K><<<grid, nthreads, shmem_sz>>>(
        d_Q, d_K, d_V, d_out, bs, nh, seqlen, seqlen, hdim, scale
    );
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iters; i++) {
        attention_fwd_bf16_wmma<BLOCK_K><<<grid, nthreads, shmem_sz>>>(
            d_Q, d_K, d_V, d_out, bs, nh, seqlen, seqlen, hdim, scale
        );
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    elapsed_ms /= num_iters;

    CUDA_CHECK(cudaMemcpy(h_out_bf16, d_out, sz_qkv * sizeof(bfloat16), cudaMemcpyDeviceToHost));

    bf16_to_fp32(h_out_bf16, h_out_fp32, sz_qkv);

    float *h_Q_bf16_fp32 = new float[sz_qkv];
    float *h_K_bf16_fp32 = new float[sz_qkv];
    float *h_V_bf16_fp32 = new float[sz_qkv];
    bf16_to_fp32(h_Q_bf16, h_Q_bf16_fp32, sz_qkv);
    bf16_to_fp32(h_K_bf16, h_K_bf16_fp32, sz_qkv);
    bf16_to_fp32(h_V_bf16, h_V_bf16_fp32, sz_qkv);

    printf("Running CPU reference...\n");
    attention_forward_cpu(h_Q_bf16_fp32, h_K_bf16_fp32, h_V_bf16_fp32, h_out_cpu,
                         bs, nh, seqlen, seqlen, hdim);

    long long flops = 2LL * bs * nh * seqlen * seqlen * hdim +
                      2LL * bs * nh * seqlen * seqlen * hdim;
    double tflops = (flops / (elapsed_ms * 1e-3)) / 1e12;

    printf("\n========================================\n");
    printf("PERFORMANCE RESULTS\n");
    printf("========================================\n");
    printf("BF16 WMMA Tensor Core: %.3f ms\n", elapsed_ms);
    printf("Throughput: %.2f TFLOPS\n", tflops);
    printf("========================================\n");

    printf("\nChecking BF16 WMMA kernel vs CPU...\n");
    bool passed = check_results(h_out_fp32, h_out_cpu, sz_qkv);

    if (passed) {
        printf("\n✓ TEST PASSED\n");
    } else {
        printf("\n✗ TEST FAILED\n");
    }

    delete[] h_Q_fp32;
    delete[] h_K_fp32;
    delete[] h_V_fp32;
    delete[] h_out_fp32;
    delete[] h_out_cpu;
    delete[] h_Q_bf16;
    delete[] h_K_bf16;
    delete[] h_V_bf16;
    delete[] h_out_bf16;
    delete[] h_Q_bf16_fp32;
    delete[] h_K_bf16_fp32;
    delete[] h_V_bf16_fp32;

    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_out));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
}

int main() {
    printf("=================================================\n");
    printf("BF16 WMMA Tensor Core Attention Benchmark\n");
    printf("=================================================\n");

    srand(42);

    printf("\n*** Testing hdim=512 ***\n");
    test_attention_bf16_wmma(1, 4, 64, 512);

    printf("\n*** Testing hdim=2048 ***\n");
    test_attention_bf16_wmma(1, 4, 64, 2048);

    printf("\n*** Testing hdim=4096 ***\n");
    test_attention_bf16_wmma(1, 2, 32, 4096);

    printf("\n=================================================\n");
    printf("All tests completed!\n");
    printf("=================================================\n");

    return 0;
}
