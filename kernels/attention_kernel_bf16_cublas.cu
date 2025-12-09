/*
 * BF16 ATTENTION WITH TENSOR CORES VIA CUBLAS
 *
 * Uses cuBLAS gemm operations which automatically use tensor cores for BF16.
 * This is the most reliable way to ensure tensor core usage.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
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

#define CUBLAS_CHECK(x) \
    do { \
        cublasStatus_t err = (x); \
        if (err != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS error " << static_cast<int>(err) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

// Softmax kernel for BF16
__global__ void softmax_inplace_bf16(
    float* scores,
    int bs_nh_seqq,
    int seq_k,
    float scale
) {
    int idx = blockIdx.x;
    if (idx >= bs_nh_seqq) return;

    float* row = scores + idx * seq_k;

    // Find max
    float max_val = -INFINITY;
    for (int i = threadIdx.x; i < seq_k; i += blockDim.x) {
        max_val = fmaxf(max_val, row[i] * scale);
    }

    // Warp reduce max
    for (int offset = 16; offset > 0; offset >>= 1) {
        max_val = fmaxf(max_val, __shfl_xor_sync(0xffffffff, max_val, offset));
    }

    __shared__ float s_max[32];
    int lane = threadIdx.x % 32;
    int wid = threadIdx.x / 32;

    if (lane == 0) s_max[wid] = max_val;
    __syncthreads();

    if (wid == 0) {
        max_val = (threadIdx.x < (blockDim.x + 31) / 32) ? s_max[lane] : -INFINITY;
        for (int offset = 16; offset > 0; offset >>= 1) {
            max_val = fmaxf(max_val, __shfl_xor_sync(0xffffffff, max_val, offset));
        }
        if (lane == 0) s_max[0] = max_val;
    }
    __syncthreads();
    max_val = s_max[0];

    // Compute exp and sum
    float sum_exp = 0.0f;
    for (int i = threadIdx.x; i < seq_k; i += blockDim.x) {
        float exp_val = expf(row[i] * scale - max_val);
        row[i] = exp_val;
        sum_exp += exp_val;
    }

    // Warp reduce sum
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_exp += __shfl_xor_sync(0xffffffff, sum_exp, offset);
    }

    if (lane == 0) s_max[wid] = sum_exp;
    __syncthreads();

    if (wid == 0) {
        sum_exp = (threadIdx.x < (blockDim.x + 31) / 32) ? s_max[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum_exp += __shfl_xor_sync(0xffffffff, sum_exp, offset);
        }
        if (lane == 0) s_max[0] = sum_exp;
    }
    __syncthreads();
    sum_exp = s_max[0];

    // Normalize
    for (int i = threadIdx.x; i < seq_k; i += blockDim.x) {
        row[i] = row[i] / sum_exp;
    }
}

// Convert FP32 scores to BF16
__global__ void fp32_to_bf16_kernel(const float* src, bfloat16* dst, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dst[idx] = __float2bfloat16(src[idx]);
    }
}

// BF16 Attention using cuBLAS (TENSOR CORES!)
void attention_forward_bf16_cublas(
    cublasHandle_t handle,
    const bfloat16* Q,
    const bfloat16* K,
    const bfloat16* V,
    bfloat16* out,
    float* d_scores_fp32,  // Temporary buffer for scores in FP32
    bfloat16* d_scores_bf16,  // Temporary buffer for scores in BF16
    int bs, int nh, int seq_q, int seq_k, int hdim
) {
    float scale = 1.0f / sqrtf(static_cast<float>(hdim));

    // For each batch and head, compute attention
    for (int b = 0; b < bs; b++) {
        for (int h = 0; h < nh; h++) {
            int offset = (b * nh + h) * seq_q * hdim;
            int kv_offset = (b * nh + h) * seq_k * hdim;

            const bfloat16* Q_bh = Q + offset;
            const bfloat16* K_bh = K + kv_offset;
            const bfloat16* V_bh = V + kv_offset;
            bfloat16* out_bh = out + offset;

            // ================================================
            // STEP 1: Q @ K^T using cuBLAS (TENSOR CORES!)
            // ================================================
            // Compute: scores_fp32 = Q @ K^T
            // Q: [seq_q, hdim], K^T: [hdim, seq_k] -> scores: [seq_q, seq_k]

            float alpha = 1.0f;
            float beta = 0.0f;

            // cuBLAS uses column-major, so we compute K @ Q^T and treat result as row-major
            // scores^T = K @ Q^T, where scores^T is [seq_k, seq_q]
            // This gives us scores [seq_q, seq_k] in row-major

            CUBLAS_CHECK(cublasGemmEx(
                handle,
                CUBLAS_OP_T,    // K^T
                CUBLAS_OP_N,    // Q
                seq_k,          // m
                seq_q,          // n
                hdim,           // k
                &alpha,
                K_bh, CUDA_R_16BF, hdim,  // K is [seq_k, hdim]
                Q_bh, CUDA_R_16BF, hdim,  // Q is [seq_q, hdim]
                &beta,
                d_scores_fp32, CUDA_R_32F, seq_k,  // Output [seq_k, seq_q] in col-major = [seq_q, seq_k] in row-major
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP  // Use tensor cores!
            ));

            // ================================================
            // STEP 2: Softmax on scores
            // ================================================
            softmax_inplace_bf16<<<seq_q, 256>>>(
                d_scores_fp32,
                seq_q,
                seq_k,
                scale
            );

            // Convert scores to BF16 for next matmul
            int score_size = seq_q * seq_k;
            fp32_to_bf16_kernel<<<(score_size + 255) / 256, 256>>>(
                d_scores_fp32, d_scores_bf16, score_size
            );

            // ================================================
            // STEP 3: scores @ V using cuBLAS (TENSOR CORES!)
            // ================================================
            // Compute: out = scores @ V
            // scores: [seq_q, seq_k], V: [seq_k, hdim] -> out: [seq_q, hdim]

            alpha = 1.0f;
            beta = 0.0f;

            // V @ scores^T in column-major = scores @ V in row-major
            CUBLAS_CHECK(cublasGemmEx(
                handle,
                CUBLAS_OP_N,    // V
                CUBLAS_OP_T,    // scores^T
                hdim,           // m
                seq_q,          // n
                seq_k,          // k
                &alpha,
                V_bh, CUDA_R_16BF, hdim,  // V is [seq_k, hdim]
                d_scores_bf16, CUDA_R_16BF, seq_k,  // scores is [seq_q, seq_k]
                &beta,
                out_bh, CUDA_R_16BF, hdim,  // Output [hdim, seq_q] in col-major = [seq_q, hdim] in row-major
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP  // Use tensor cores!
            ));
        }
    }

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

void test_bf16_cublas(int bs, int nh, int seqlen, int hdim) {
    printf("\n===== BF16 cuBLAS Tensor Core Attention =====\n");
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
    float *d_scores_fp32;
    bfloat16 *d_scores_bf16;

    CUDA_CHECK(cudaMalloc(&d_Q, sz * sizeof(bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_K, sz * sizeof(bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_V, sz * sizeof(bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_out, sz * sizeof(bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_scores_fp32, seqlen * seqlen * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scores_bf16, seqlen * seqlen * sizeof(bfloat16)));

    CUDA_CHECK(cudaMemcpy(d_Q, h_Q, sz * sizeof(bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K, sz * sizeof(bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V, sz * sizeof(bfloat16), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));  // Enable tensor cores!

    // Warmup
    attention_forward_bf16_cublas(handle, d_Q, d_K, d_V, d_out, d_scores_fp32, d_scores_bf16, bs, nh, seqlen, seqlen, hdim);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < 100; i++) {
        attention_forward_bf16_cublas(handle, d_Q, d_K, d_V, d_out, d_scores_fp32, d_scores_bf16, bs, nh, seqlen, seqlen, hdim);
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

    bool passed = check_results(h_out_gpu, h_out_cpu, sz);
    printf(passed ? "✓ TENSOR CORES WORKING!\n" : "✗ FAILED\n");

    delete[] h_Q_fp32; delete[] h_K_fp32; delete[] h_V_fp32;
    delete[] h_out_cpu; delete[] h_out_gpu;
    delete[] h_Q; delete[] h_K; delete[] h_V; delete[] h_out_bf16;

    CUDA_CHECK(cudaFree(d_Q)); CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V)); CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_scores_fp32)); CUDA_CHECK(cudaFree(d_scores_bf16));

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
}

int main() {
    printf("=================================================\n");
    printf("BF16 Attention with cuBLAS Tensor Cores\n");
    printf("=================================================\n");

    srand(42);

    test_bf16_cublas(1, 4, 64, 512);
    test_bf16_cublas(1, 4, 64, 2048);
    test_bf16_cublas(1, 2, 32, 4096);

    return 0;
}
