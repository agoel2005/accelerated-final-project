#!/usr/bin/env python3
"""
Benchmark our custom CUDA attention kernel against PyTorch's optimized implementation.
This will help us understand if there's room for improvement.
"""

import torch
import torch.nn.functional as F
import time
import subprocess
import os

def benchmark_pytorch_attention(batch_sz, num_heads, seq_len, head_dim, num_iters=100, warmup=10):
    """Benchmark PyTorch's built-in scaled_dot_product_attention."""

    # Create random tensors on GPU
    Q = torch.randn(batch_sz, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float32)
    K = torch.randn(batch_sz, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float32)
    V = torch.randn(batch_sz, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float32)

    # Warmup
    for _ in range(warmup):
        out = F.scaled_dot_product_attention(Q, K, V)
    torch.cuda.synchronize()

    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(num_iters):
        out = F.scaled_dot_product_attention(Q, K, V)
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end) / num_iters

    # Calculate FLOPs
    # Attention: Q @ K^T (bs * nh * len_q * len_k * head_d * 2)
    #          + softmax@V (bs * nh * len_q * len_k * head_d * 2)
    flops = 2 * batch_sz * num_heads * seq_len * seq_len * head_dim * 2
    tflops = (flops / (elapsed_ms * 1e-3)) / 1e12

    return elapsed_ms, tflops, out

def benchmark_naive_pytorch(batch_sz, num_heads, seq_len, head_dim, num_iters=100, warmup=10):
    """Benchmark naive PyTorch attention (non-fused) for comparison."""

    Q = torch.randn(batch_sz, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float32)
    K = torch.randn(batch_sz, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float32)
    V = torch.randn(batch_sz, num_heads, seq_len, head_dim, device='cuda', dtype=torch.float32)

    scale = 1.0 / (head_dim ** 0.5)

    # Warmup
    for _ in range(warmup):
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
    torch.cuda.synchronize()

    # Benchmark
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(num_iters):
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end) / num_iters

    flops = 2 * batch_sz * num_heads * seq_len * seq_len * head_dim * 2
    tflops = (flops / (elapsed_ms * 1e-3)) / 1e12

    return elapsed_ms, tflops, out

def main():
    print("=" * 80)
    print("PyTorch Attention Benchmark")
    print("Comparing our CUDA kernel against PyTorch's optimized implementation")
    print("=" * 80)

    # Test configurations matching our CUDA kernel tests
    configs = [
        # (batch_sz, num_heads, seq_len, head_dim, description)
        (1, 4, 64, 512, "hdim=512 (warmup for large dims)"),
        (1, 4, 64, 2048, "hdim=2048"),
        (1, 2, 32, 4096, "hdim=4096"),
        (1, 1, 16, 8192, "hdim=8192"),
    ]

    print("\n" + "=" * 80)
    print("LARGE HIDDEN DIMENSION BENCHMARKS")
    print("=" * 80)

    for batch_sz, num_heads, seq_len, head_dim, desc in configs:
        print(f"\n--- {desc} ---")
        print(f"Configuration: bs={batch_sz}, nh={num_heads}, seq={seq_len}, hdim={head_dim}")

        # Calculate total FLOPs
        flops = 2 * batch_sz * num_heads * seq_len * seq_len * head_dim * 2
        print(f"Total FLOPs: {flops / 1e6:.2f} MFLOPs")

        # Benchmark PyTorch optimized
        try:
            time_opt, tflops_opt, _ = benchmark_pytorch_attention(
                batch_sz, num_heads, seq_len, head_dim
            )
            print(f"\nPyTorch Optimized (F.scaled_dot_product_attention):")
            print(f"  Time: {time_opt:.3f} ms")
            print(f"  Performance: {tflops_opt:.2f} TFLOPS")
        except Exception as e:
            print(f"\nPyTorch Optimized failed: {e}")
            time_opt, tflops_opt = None, None

        # Benchmark naive PyTorch
        try:
            time_naive, tflops_naive, _ = benchmark_naive_pytorch(
                batch_sz, num_heads, seq_len, head_dim
            )
            print(f"\nPyTorch Naive (matmul + softmax):")
            print(f"  Time: {time_naive:.3f} ms")
            print(f"  Performance: {tflops_naive:.2f} TFLOPS")

            if time_opt:
                speedup = time_naive / time_opt
                print(f"  Speedup vs Optimized: {speedup:.2f}x")
        except Exception as e:
            print(f"\nPyTorch Naive failed: {e}")

        print()

    # Compare with our CUDA kernel results (from summary)
    print("\n" + "=" * 80)
    print("COMPARISON WITH OUR CUDA KERNEL")
    print("=" * 80)

    print("\nOur Best Results (attention_kernel_large_dims.cu):")
    our_results = [
        ("hdim=512", 0.76),
        ("hdim=2048", 0.86),
        ("hdim=4096", 0.27),
        ("hdim=8192", 0.05),
    ]

    for desc, tflops in our_results:
        print(f"  {desc}: {tflops:.2f} TFLOPS")

    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print("""
Key Questions to Answer:
1. How does PyTorch's optimized attention compare to ours?
   - If much faster: They likely use FlashAttention or similar optimizations
   - If similar: Our kernel is near-optimal for this workload!

2. Is naive PyTorch faster than our kernel?
   - If yes: We have bugs or inefficiencies
   - If no: We're doing reasonably well

3. What's the speedup of PyTorch optimized vs naive?
   - This shows the theoretical maximum gain from advanced optimizations
   - If small (<2x): Memory bandwidth is the bottleneck (as we suspected)
   - If large (>5x): Compute or better tiling helps significantly

Next Steps Based on Results:
- If PyTorch optimized >> our kernel: Study FlashAttention implementation
- If PyTorch similar to ours: Focus on increasing batch size for better GPU utilization
- Profile with nsys/ncu to understand actual memory bandwidth utilization
""")

if __name__ == "__main__":
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        exit(1)

    device_name = torch.cuda.get_device_name(0)
    print(f"\nGPU: {device_name}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}\n")

    main()
