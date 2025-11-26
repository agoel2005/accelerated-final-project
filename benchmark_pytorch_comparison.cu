/*
 * PyTorch-compatible benchmark for our attention kernel
 * This can be compiled and run on the remote server to compare with PyTorch
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <iostream>

#define CUDA_CHECK(x) \
    do { \
        cudaError_t err = (x); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)

// Import the working kernel from attention_kernel.cu
extern "C" void attention_forward_optimized(
    const float* Q, const float* K, const float* V, float* out,
    int batch_sz, int num_heads, int len_q, int len_k, int head_d);

void benchmark_kernel(int bs, int nh, int seq_len, int hdim, const char* desc) {
    printf("\n========================================\n");
    printf("%s\n", desc);
    printf("  bs=%d, nh=%d, seq=%d, hdim=%d\n", bs, nh, seq_len, hdim);
    printf("========================================\n");

    int sz_qkv = bs * nh * seq_len * hdim;

    // Allocate memory
    float *d_Q, *d_K, *d_V, *d_out;
    CUDA_CHECK(cudaMalloc(&d_Q, sz_qkv * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, sz_qkv * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V, sz_qkv * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, sz_qkv * sizeof(float)));

    // Initialize with random data
    CUDA_CHECK(cudaMemset(d_Q, 0, sz_qkv * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_K, 0, sz_qkv * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_V, 0, sz_qkv * sizeof(float)));

    // Warmup
    for (int i = 0; i < 10; i++) {
        attention_forward_optimized(d_Q, d_K, d_V, d_out, bs, nh, seq_len, seq_len, hdim);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    const int num_iters = 100;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < num_iters; i++) {
        attention_forward_optimized(d_Q, d_K, d_V, d_out, bs, nh, seq_len, seq_len, hdim);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    elapsed_ms /= num_iters;

    // Calculate TFLOPS
    long long flops = 2LL * bs * nh * seq_len * seq_len * hdim * 2;
    double tflops = (flops / (elapsed_ms * 1e-3)) / 1e12;

    printf("Time: %.3f ms\n", elapsed_ms);
    printf("Performance: %.2f TFLOPS\n", tflops);
    printf("Memory transferred (estimate): %.2f MB\n",
           (sz_qkv * 4 * 3.0) / (1024.0 * 1024.0));

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_out));
}

int main() {
    printf("=================================================\n");
    printf("Attention Kernel Performance Benchmark\n");
    printf("Compare these results with PyTorch!\n");
    printf("=================================================\n");

    // Get GPU info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("\nGPU: %s\n", prop.name);
    printf("Memory Bandwidth: %.0f GB/s\n",
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    printf("Peak FP32: %.2f TFLOPS (theoretical)\n\n",
           2.0 * prop.clockRate * prop.multiProcessorCount *
           prop.maxThreadsPerMultiProcessor / 1.0e9);

    // Benchmark configurations matching PyTorch tests
    benchmark_kernel(1, 4, 64, 512, "hdim=512 (warmup)");
    benchmark_kernel(1, 4, 64, 2048, "hdim=2048");
    benchmark_kernel(1, 2, 32, 4096, "hdim=4096");
    benchmark_kernel(1, 1, 16, 8192, "hdim=8192");

    printf("\n=================================================\n");
    printf("TO COMPARE WITH PYTORCH:\n");
    printf("Run this Python code:\n");
    printf("=================================================\n");
    printf("import torch\n");
    printf("import torch.nn.functional as F\n");
    printf("import time\n\n");
    printf("# hdim=2048 example\n");
    printf("q = torch.randn(1, 4, 64, 2048, device='cuda')\n");
    printf("k = torch.randn(1, 4, 64, 2048, device='cuda')\n");
    printf("v = torch.randn(1, 4, 64, 2048, device='cuda')\n\n");
    printf("# Warmup\n");
    printf("for _ in range(10):\n");
    printf("    out = F.scaled_dot_product_attention(q, k, v)\n");
    printf("torch.cuda.synchronize()\n\n");
    printf("# Benchmark\n");
    printf("start = torch.cuda.Event(enable_timing=True)\n");
    printf("end = torch.cuda.Event(enable_timing=True)\n");
    printf("start.record()\n");
    printf("for _ in range(100):\n");
    printf("    out = F.scaled_dot_product_attention(q, k, v)\n");
    printf("end.record()\n");
    printf("torch.cuda.synchronize()\n");
    printf("print(f'PyTorch time: {start.elapsed_time(end)/100:.3f} ms')\n");
    printf("=================================================\n");

    return 0;
}
