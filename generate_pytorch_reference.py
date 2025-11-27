#!/usr/bin/env python3
"""
Generate PyTorch reference outputs for CUDA attention kernel validation.
This creates test data files that can be loaded by the CUDA kernel for comparison.
"""

import torch
import torch.nn.functional as F
import numpy as np
import struct


def attention_pytorch(Q, K, V):
    """PyTorch reference implementation of scaled dot-product attention."""
    scale = 1.0 / (Q.size(-1) ** 0.5)
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)
    return output


def generate_test_case(batch_sz, num_heads, seq_len, head_d, seed=42, device='cuda'):
    """Generate a single test case with PyTorch reference output."""
    torch.manual_seed(seed)

    Q = torch.randn(batch_sz, num_heads, seq_len, head_d, device=device, dtype=torch.float32)
    K = torch.randn(batch_sz, num_heads, seq_len, head_d, device=device, dtype=torch.float32)
    V = torch.randn(batch_sz, num_heads, seq_len, head_d, device=device, dtype=torch.float32)

    # Compute reference using PyTorch
    output_pytorch = F.scaled_dot_product_attention(Q, K, V)

    # Also compute with manual implementation to verify
    output_manual = attention_pytorch(Q, K, V)

    # Verify they match
    max_diff = (output_pytorch - output_manual).abs().max().item()

    return {
        'Q': Q.cpu().numpy(),
        'K': K.cpu().numpy(),
        'V': V.cpu().numpy(),
        'output': output_pytorch.cpu().numpy(),
        'max_diff': max_diff,
        'config': (batch_sz, num_heads, seq_len, head_d)
    }


def save_binary(filename, data, output_dir="pytorch_reference_data"):
    """Save numpy array as binary file (C-contiguous)."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    data = np.ascontiguousarray(data, dtype=np.float32)
    data.tofile(filepath)
    return data.size


def print_stats(name, data):
    """Print statistics about the data."""
    print(f"  {name}:")
    print(f"    Shape: {data.shape}")
    print(f"    Range: [{data.min():.6f}, {data.max():.6f}]")
    print(f"    Mean: {data.mean():.6f}, Std: {data.std():.6f}")


def main():
    print("=" * 80)
    print("PyTorch Reference Data Generator")
    print("=" * 80)

    if not torch.cuda.is_available():
        print("\nNote: CUDA not available, using CPU")
        print("This is fine for generating reference data - the outputs will be identical!")
        print("(Only timing results would differ, but we're not benchmarking here)\n")
        device = 'cpu'
        cuda_version = "N/A"
    else:
        device = 'cuda'
        cuda_version = torch.version.cuda
        print(f"\nGPU: {torch.cuda.get_device_name(0)}")

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {cuda_version}")

    # Test configurations
    configs = [
        (1, 1, 4, 8, "tiny", 42),
        (1, 2, 8, 16, "small", 43),
        (1, 4, 64, 512, "hdim512", 44),
        (1, 4, 64, 2048, "hdim2048", 45),
        (1, 2, 32, 4096, "hdim4096", 46),
        (1, 1, 16, 8192, "hdim8192", 47),
    ]

    print("\n" + "=" * 80)
    print("Generating Test Cases")
    print("=" * 80)
    print("Output directory: pytorch_reference_data/\n")

    for batch_sz, num_heads, seq_len, head_d, name, seed in configs:
        print(f"\n{name}: bs={batch_sz}, nh={num_heads}, seq={seq_len}, hdim={head_d}")

        test_data = generate_test_case(batch_sz, num_heads, seq_len, head_d, seed, device)

        # Save binary files to pytorch_reference_data/
        prefix = f"pytorch_ref_{name}"
        q_size = save_binary(f"{prefix}_Q.bin", test_data['Q'])
        k_size = save_binary(f"{prefix}_K.bin", test_data['K'])
        v_size = save_binary(f"{prefix}_V.bin", test_data['V'])
        out_size = save_binary(f"{prefix}_output.bin", test_data['output'])

        print(f"  PyTorch vs manual max diff: {test_data['max_diff']:.2e}")
        print(f"  Saved {q_size} floats per tensor")

        print_stats("Q", test_data['Q'])
        print_stats("Output", test_data['output'])

    # Generate a comprehensive test report
    print("\n" + "=" * 80)
    print("Benchmark PyTorch Performance")
    print("=" * 80)

    bench_configs = [
        (1, 4, 64, 512),
        (1, 4, 64, 2048),
        (1, 2, 32, 4096),
        (1, 1, 16, 8192),
    ]

    print(f"\n{'Config':<25} {'Time (ms)':<12} {'TFLOPS':<10}")
    print("-" * 50)

    for batch_sz, num_heads, seq_len, head_d in bench_configs:
        Q = torch.randn(batch_sz, num_heads, seq_len, head_d, device=device, dtype=torch.float32)
        K = torch.randn(batch_sz, num_heads, seq_len, head_d, device=device, dtype=torch.float32)
        V = torch.randn(batch_sz, num_heads, seq_len, head_d, device=device, dtype=torch.float32)

        # Warmup
        for _ in range(10):
            _ = F.scaled_dot_product_attention(Q, K, V)
        if device == 'cuda':
            torch.cuda.synchronize()

        # Benchmark
        if device == 'cuda':
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(100):
                _ = F.scaled_dot_product_attention(Q, K, V)
            end.record()
            torch.cuda.synchronize()
            time_ms = start.elapsed_time(end) / 100
        else:
            import time
            start = time.time()
            for _ in range(100):
                _ = F.scaled_dot_product_attention(Q, K, V)
            time_ms = (time.time() - start) * 1000 / 100

        # Calculate TFLOPS
        flops = 2 * batch_sz * num_heads * seq_len * seq_len * head_d * 2
        tflops = (flops / (time_ms * 1e-3)) / 1e12

        config_str = f"hdim={head_d}, seq={seq_len}"
        print(f"{config_str:<25} {time_ms:>8.3f}    {tflops:>8.2f}")

    print("\n" + "=" * 80)
    print("How to Use These Files")
    print("=" * 80)
    print("""
In your CUDA kernel, load the binary files:

    // Example for hdim2048 test case
    FILE* f = fopen("pytorch_reference_data/pytorch_ref_hdim2048_Q.bin", "rb");
    fread(h_Q, sizeof(float), 1*4*64*2048, f);
    fclose(f);

    // Then compare your output with pytorch_reference_data/pytorch_ref_hdim2048_output.bin

Files are saved in C-contiguous order (row-major):
    Q[batch][head][seq][hdim]

Expected results (from FINAL_SUMMARY.md):
    hdim=512:  Optimized should be ~0.062 ms (1.91x speedup)
    hdim=2048: Optimized should be ~0.214 ms (2.02x speedup)
    hdim=4096: Optimized should be ~0.126 ms (1.48x speedup)
    hdim=8192: Optimized should be ~0.194 ms (1.03x speedup)
""")

    print("\n" + "=" * 80)
    print("All reference data generated successfully!")
    print("=" * 80)j


if __name__ == "__main__":
    main()
