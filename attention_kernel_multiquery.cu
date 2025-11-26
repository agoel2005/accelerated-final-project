/*
 * MULTI-QUERY ATTENTION KERNEL - Optimized for Large Dimensions
 *
 * Key optimization: Process MULTIPLE queries per block to maximize K/V reuse
 *
 * Strategy:
 * - Each block processes BLOCK_Q queries (e.g., 4 or 16)
 * - Cache K/V tiles in shared memory
 * - All queries in block reuse the same K/V data
 * - Expected 10-20x speedup for large hdim!
 */

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

// Multi-query kernel: Process BLOCK_Q queries per block
template<int BLOCK_Q, int BLOCK_K>
__global__ void attention_multiquery_kernel(
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
    const int q_block_idx = blockIdx.x;
    const int q_start = q_block_idx * BLOCK_Q;

    if (q_start >= seq_q) return;

    const int num_q = min(BLOCK_Q, seq_q - q_start);
    const int tid = threadIdx.x;

    const int qkv_base = (b_idx * nh + h_idx) * seq_q * hdim;
    const int kv_base = (b_idx * nh + h_idx) * seq_k * hdim;

    // Shared memory layout
    extern __shared__ float smem[];
    float* s_K = smem;                           // BLOCK_K × hdim
    float* s_V = s_K + BLOCK_K * hdim;          // BLOCK_K × hdim
    float* s_scores = s_V + BLOCK_K * hdim;     // BLOCK_Q × BLOCK_K
    float* s_reduce = s_scores + BLOCK_Q * BLOCK_K;  // Reduction buffer

    // Each thread manages outputs for its assigned queries
    // For simplicity, process all queries (will optimize later)

    // Process K/V in tiles
    for (int k_start = 0; k_start < seq_k; k_start += BLOCK_K) {
        const int num_k = min(BLOCK_K, seq_k - k_start);

        // Step 1: Cooperatively load K tile into shared memory
        // Each thread loads elements to maximize coalescing
        for (int i = tid; i < num_k * hdim; i += blockDim.x) {
            int k_local = i / hdim;
            int d = i % hdim;
            int k_global = k_start + k_local;
            s_K[k_local * hdim + d] = K[kv_base + k_global * hdim + d];
        }

        // Step 2: Cooperatively load V tile into shared memory
        for (int i = tid; i < num_k * hdim; i += blockDim.x) {
            int k_local = i / hdim;
            int d = i % hdim;
            int k_global = k_start + k_local;
            s_V[k_local * hdim + d] = V[kv_base + k_global * hdim + d];
        }
        __syncthreads();

        // Step 3: Compute scores for all queries using shared K (REUSE!)
        for (int q_local = 0; q_local < num_q; q_local++) {
            int q_global = q_start + q_local;

            // Each thread computes scores for some keys
            for (int k_local = tid; k_local < num_k; k_local += blockDim.x) {
                float score = 0.0f;

                // Dot product Q[q_global] · K[k_local] with vectorization
                int q_off = qkv_base + q_global * hdim;

                if (hdim % 4 == 0) {
                    #pragma unroll 4
                    for (int d = 0; d < hdim; d += 4) {
                        float4 q_vec = *reinterpret_cast<const float4*>(&Q[q_off + d]);
                        float4 k_vec = *reinterpret_cast<const float4*>(&s_K[k_local * hdim + d]);
                        score += q_vec.x * k_vec.x + q_vec.y * k_vec.y +
                                 q_vec.z * k_vec.z + q_vec.w * k_vec.w;
                    }
                } else {
                    for (int d = 0; d < hdim; d++) {
                        score += Q[q_off + d] * s_K[k_local * hdim + d];
                    }
                }

                score *= scale_factor;
                s_scores[q_local * BLOCK_K + k_local] = score;
            }
            __syncthreads();

            // Step 4: Softmax for this query
            float local_max = -INFINITY;
            for (int k_local = tid; k_local < num_k; k_local += blockDim.x) {
                local_max = fmaxf(local_max, s_scores[q_local * BLOCK_K + k_local]);
            }
            local_max = block_reduce_max(local_max, s_reduce);
            if (tid == 0) s_reduce[0] = local_max;
            __syncthreads();
            local_max = s_reduce[0];

            float exp_sum = 0.0f;
            for (int k_local = tid; k_local < num_k; k_local += blockDim.x) {
                float exp_val = expf(s_scores[q_local * BLOCK_K + k_local] - local_max);
                s_scores[q_local * BLOCK_K + k_local] = exp_val;
                exp_sum += exp_val;
            }
            exp_sum = block_reduce_sum(exp_sum, s_reduce);
            if (tid == 0) s_reduce[0] = exp_sum;
            __syncthreads();
            exp_sum = s_reduce[0];

            for (int k_local = tid; k_local < num_k; k_local += blockDim.x) {
                s_scores[q_local * BLOCK_K + k_local] = safe_div(s_scores[q_local * BLOCK_K + k_local], exp_sum);
            }
            __syncthreads();

            // Step 5: Compute output using shared V (REUSE!)
            for (int d = tid; d < hdim; d += blockDim.x) {
                float accum = 0.0f;
                for (int k_local = 0; k_local < num_k; k_local++) {
                    accum += s_scores[q_local * BLOCK_K + k_local] * s_V[k_local * hdim + d];
                }

                // Accumulate across K tiles
                if (k_start == 0) {
                    out[qkv_base + q_global * hdim + d] = accum;
                } else {
                    out[qkv_base + q_global * hdim + d] += accum;
                }
            }
            __syncthreads();
        }
    }
}

void attention_forward_multiquery(
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

    // Adaptive configuration based on hdim to stay within shared memory limits
    // Target: 2 * BLOCK_K * hdim * 4 + BLOCK_Q * BLOCK_K * 4 < 48000 bytes
    int BLOCK_Q, BLOCK_K;

    if (head_d <= 512) {
        BLOCK_Q = 4;
        BLOCK_K = 16;  // 2*16*512*4 + 4*16*4 = 65,792 bytes... still too much!
    } else if (head_d <= 2048) {
        BLOCK_Q = 4;
        BLOCK_K = 4;   // 2*4*2048*4 + 4*4*4 = 65,600 bytes... still too much!
    } else {
        BLOCK_Q = 4;
        BLOCK_K = 4;
    }

    // Actually, let's calculate properly
    // 2 * BLOCK_K * head_d * 4 < 45000 (leave some room)
    int max_block_k = 45000 / (2 * head_d * 4);
    if (max_block_k < 4) max_block_k = 4;
    if (max_block_k > 32) max_block_k = 32;

    BLOCK_K = max_block_k;
    BLOCK_Q = 4;

    const int num_q_blocks = (len_q + BLOCK_Q - 1) / BLOCK_Q;
    dim3 grid(num_q_blocks, batch_sz * num_heads);
    const int nthreads = 256;

    // Shared memory: K tile + V tile + scores + reduction buffer
    const int shmem_sz = (2 * BLOCK_K * head_d +  // K + V tiles
                          BLOCK_Q * BLOCK_K +       // Scores
                          32) * sizeof(float);      // Reduction

    printf("Multi-query kernel config:\n");
    printf("  hdim=%d\n", head_d);
    printf("  BLOCK_Q=%d, BLOCK_K=%d\n", BLOCK_Q, BLOCK_K);
    printf("  Grid: %d blocks (%d q_blocks × %d heads)\n",
           num_q_blocks * batch_sz * num_heads, num_q_blocks, batch_sz * num_heads);
    printf("  Shared memory: %d bytes (%.1f KB)\n", shmem_sz, shmem_sz / 1024.0f);

    if (shmem_sz > 48000) {
        printf("  WARNING: Shared memory exceeds 48KB limit!\n");
        return;
    }

    // Use template specialization for common cases
    if (BLOCK_Q == 4 && BLOCK_K == 16) {
        attention_multiquery_kernel<4, 16><<<grid, nthreads, shmem_sz>>>(
            Q, K, V, out, batch_sz, num_heads, len_q, len_k, head_d, scale
        );
    } else if (BLOCK_Q == 4 && BLOCK_K == 8) {
        attention_multiquery_kernel<4, 8><<<grid, nthreads, shmem_sz>>>(
            Q, K, V, out, batch_sz, num_heads, len_q, len_k, head_d, scale
        );
    } else if (BLOCK_Q == 4 && BLOCK_K == 4) {
        attention_multiquery_kernel<4, 4><<<grid, nthreads, shmem_sz>>>(
            Q, K, V, out, batch_sz, num_heads, len_q, len_k, head_d, scale
        );
    } else {
        printf("  ERROR: No kernel template for BLOCK_Q=%d, BLOCK_K=%d\n", BLOCK_Q, BLOCK_K);
        return;
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// CPU reference (same as before)
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

// Testing code
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
    printf("Testing Multi-Query Attention:\n");
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
    attention_forward_multiquery(d_Q, d_K, d_V, d_out, bs, nh, seqlen, seqlen, hdim);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    printf("Running multi-query kernel...\n");
    const int num_iters = 20;
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iters; i++) {
        attention_forward_multiquery(d_Q, d_K, d_V, d_out, bs, nh, seqlen, seqlen, hdim);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float kernel_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));
    kernel_ms /= num_iters;

    CUDA_CHECK(cudaMemcpy(h_out_gpu, d_out, sz_q * sizeof(float), cudaMemcpyDeviceToHost));

    printf("Running CPU reference...\n");
    attention_cpu(h_Q, h_K, h_V, h_out_cpu, bs, nh, seqlen, seqlen, hdim);

    printf("\n--- Performance Results ---\n");
    printf("Multi-query kernel time: %.3f ms\n", kernel_ms);

    long long flops = 2LL * bs * nh * seqlen * seqlen * hdim;
    flops += 2LL * bs * nh * seqlen * seqlen * hdim;
    float tflops = (flops / (kernel_ms / 1000.0f)) / 1e12f;
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
    printf("Multi-Query Attention Kernel Optimization\n");
    printf("Target: Process multiple queries per block\n");
    printf("=================================================\n");

    srand(42);

    printf("\n*** Testing hdim=512 ***\n");
    test_attention(1, 4, 64, 512);

    printf("\n*** Testing hdim=2048 ***\n");
    test_attention(1, 4, 64, 2048);

    printf("\n*** Testing hdim=4096 ***\n");
    test_attention(1, 2, 32, 4096);

    printf("\n*** Testing hdim=8192 ***\n");
    test_attention(1, 1, 16, 8192);

    printf("\n=================================================\n");
    printf("All tests completed!\n");
    printf("=================================================\n");

    return 0;
}
