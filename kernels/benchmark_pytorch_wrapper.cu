/*
 * Wrapper to run PyTorch benchmark via telerun
 * This is a CUDA file that executes the Python benchmark script
 */

#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <iostream>

const char* PYTHON_SCRIPT = R"PYTHON(
import torch
import torch.nn.functional as F
import sys

print("=" * 80)
print("PyTorch Attention Benchmark on Telerun GPU")
print("=" * 80)
print()

# GPU info
if not torch.cuda.is_available():
    print("ERROR: CUDA not available!")
    sys.exit(1)

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"PyTorch Version: {torch.__version__}")
total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"GPU Memory: {total_mem:.1f} GB")
print()

# Test configurations
configs = [
    (1, 4, 64, 512, "hdim=512"),
    (1, 4, 64, 2048, "hdim=2048"),
    (1, 2, 32, 4096, "hdim=4096"),
    (1, 1, 16, 8192, "hdim=8192"),
]

results = []

for batch_sz, num_heads, seq_len, head_d, name in configs:
    print(f"{'=' * 80}")
    print(f"{name}: bs={batch_sz}, nh={num_heads}, seq={seq_len}, hdim={head_d}")
    print(f"{'=' * 80}")

    Q = torch.randn(batch_sz, num_heads, seq_len, head_d, device='cuda', dtype=torch.float32)
    K = torch.randn(batch_sz, num_heads, seq_len, head_d, device='cuda', dtype=torch.float32)
    V = torch.randn(batch_sz, num_heads, seq_len, head_d, device='cuda', dtype=torch.float32)

    flops = 2 * batch_sz * num_heads * seq_len * seq_len * head_d * 2

    # PyTorch Optimized
    for _ in range(10):
        _ = F.scaled_dot_product_attention(Q, K, V)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(100):
        _ = F.scaled_dot_product_attention(Q, K, V)
    end.record()
    torch.cuda.synchronize()

    time_opt = start.elapsed_time(end) / 100
    tflops_opt = (flops / (time_opt * 1e-3)) / 1e12

    print(f"\nPyTorch Optimized:")
    print(f"  Time: {time_opt:.3f} ms")
    print(f"  Performance: {tflops_opt:.2f} TFLOPS")

    # PyTorch Naive
    scale = 1.0 / (head_d ** 0.5)
    for _ in range(10):
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        attn = F.softmax(scores, dim=-1)
        _ = torch.matmul(attn, V)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(100):
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        attn = F.softmax(scores, dim=-1)
        _ = torch.matmul(attn, V)
    end.record()
    torch.cuda.synchronize()

    time_naive = start.elapsed_time(end) / 100
    tflops_naive = (flops / (time_naive * 1e-3)) / 1e12

    print(f"\nPyTorch Naive:")
    print(f"  Time: {time_naive:.3f} ms")
    print(f"  Performance: {tflops_naive:.2f} TFLOPS")
    print(f"  Speedup: {time_naive / time_opt:.2f}x")
    print()

    results.append((name, time_opt, time_naive, tflops_opt))

# Summary table
print("\n" + "=" * 80)
print("SUMMARY: PyTorch on This GPU")
print("=" * 80)
print(f"\n{'Config':<15} {'Optimized':<15} {'Naive':<15} {'Speedup':<10} {'TFLOPS':<10}")
print("-" * 70)
for name, t_opt, t_naive, tflops in results:
    print(f"{name:<15} {t_opt:>10.3f} ms  {t_naive:>10.3f} ms  {t_naive/t_opt:>6.2f}x    {tflops:>6.2f}")

print("\n" + "=" * 80)
print("EXPECTED CUDA KERNEL RESULTS")
print("=" * 80)
print(f"\n{'Config':<15} {'Expected':<15} {'Speedup vs Baseline':<20}")
print("-" * 55)
print(f"{'hdim=512':<15} {'0.062 ms':<15} {'1.91x':<20}")
print(f"{'hdim=2048':<15} {'0.214 ms':<15} {'2.02x':<20}")
print(f"{'hdim=4096':<15} {'0.126 ms':<15} {'1.48x':<20}")
print(f"{'hdim=8192':<15} {'0.194 ms':<15} {'1.03x':<20}")

print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)
print()
print("Compare PyTorch Optimized times above with CUDA kernel times.")
print()
print("To run the CUDA kernel:")
print("  python ../telerun/telerun.py submit attention_kernel.cu")
print()
print("Look for 'Optimized kernel time' in the output and compare.")
print("=" * 80)
)PYTHON";

int main() {
    std::cout << "================================================================================\n";
    std::cout << "PyTorch Benchmark Wrapper\n";
    std::cout << "================================================================================\n\n";

    std::cout << "Writing embedded Python script...\n";

    // Write Python script to file
    std::ofstream script("pytorch_bench.py");
    if (!script.is_open()) {
        std::cerr << "ERROR: Could not create Python script file\n";
        return 1;
    }
    script << PYTHON_SCRIPT;
    script.close();

    std::cout << "Executing PyTorch benchmark...\n\n";

    // Try python3 first
    int result = system("python3 pytorch_bench.py");

    if (result != 0) {
        std::cout << "\nTrying 'python' instead of 'python3'...\n";
        result = system("python pytorch_bench.py");
    }

    if (result != 0) {
        std::cerr << "\nERROR: PyTorch benchmark failed\n";
        std::cerr << "PyTorch may not be installed on the remote system\n";
        return 1;
    }

    std::cout << "\n================================================================================\n";
    std::cout << "PyTorch Benchmark Complete\n";
    std::cout << "================================================================================\n";

    return 0;
}
