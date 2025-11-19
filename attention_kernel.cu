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

__global__ void attention_fwd_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ out,
    int bs,
    int nh,
    int seq_q,
    int seq_k,
    int hdim,
    float scale_factor
) {
    int idx = blockIdx.x;
    int total_q = bs * nh * seq_q;

    if (idx >= total_q) return;

    extern __shared__ float smem[];
    float* scores = smem;

    int b_idx = idx / (nh * seq_q);
    int h_idx = (idx / seq_q) % nh;

    int q_off = idx * hdim;
    int kv_off = (b_idx * nh + h_idx) * seq_k * hdim;

    int tid = threadIdx.x;

    float local_max = -INFINITY;

    for (int kpos = tid; kpos < seq_k; kpos += blockDim.x) {
        float score = 0.0f;
        int k_off = kv_off + kpos * hdim;

        for (int d = 0; d < hdim; d++) {
            score += Q[q_off + d] * K[k_off + d];
        }

        score *= scale_factor;
        scores[kpos] = score;
        local_max = fmaxf(local_max, score);
    }

    __shared__ float max_val;
    if (tid == 0) {
        max_val = -INFINITY;
    }
    __syncthreads();

    atomicMax((int*)&max_val, __float_as_int(local_max));
    __syncthreads();
    local_max = max_val;

    float exp_sum = 0.0f;
    for (int kpos = tid; kpos < seq_k; kpos += blockDim.x) {
        float exp_val = expf(scores[kpos] - local_max);
        scores[kpos] = exp_val;
        exp_sum += exp_val;
    }

    __shared__ float total_exp;
    if (tid == 0) {
        total_exp = 0.0f;
    }
    __syncthreads();

    atomicAdd(&total_exp, exp_sum);
    __syncthreads();
    exp_sum = total_exp;

    for (int kpos = tid; kpos < seq_k; kpos += blockDim.x) {
        scores[kpos] = safe_div(scores[kpos], exp_sum);
    }
    __syncthreads();

    for (int d = tid; d < hdim; d += blockDim.x) {
        float accum = 0.0f;

        for (int kpos = 0; kpos < seq_k; kpos++) {
            int v_off = kv_off + kpos * hdim + d;
            accum += scores[kpos] * V[v_off];
        }

        out[q_off + d] = accum;
    }
}

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
    const int shmem_sz = len_k * sizeof(float);

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

bool check_results(const float* gpu_out, const float* cpu_out, int size, float tol = 1e-3f) {
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
    attention_forward(d_Q, d_K, d_V, d_out, bs, nh, seqlen, seqlen, hdim);

    CUDA_CHECK(cudaMemcpy(h_out_gpu, d_out, sz_q * sizeof(float), cudaMemcpyDeviceToHost));

    printf("Running CPU reference...\n");
    attention_cpu(h_Q, h_K, h_V, h_out_cpu, bs, nh, seqlen, seqlen, hdim);

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

    test_attention(1, 1, 4, 8);

    test_attention(1, 2, 8, 16);

    test_attention(2, 4, 32, 32);

    test_attention(2, 8, 64, 64);

    test_cross_attention(1, 2, 16, 32, 32);

    test_attention(1, 4, 16, 64);

    printf("\n=================================================\n");
    printf("All tests completed!\n");
    printf("=================================================\n");

    return 0;
}
