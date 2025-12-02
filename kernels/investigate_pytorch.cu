/*
 * Investigate PyTorch behavior with different batch sizes and backends
 */

#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <iostream>

const char* PYTHON_SCRIPT = R"PYTHON(
import torch
import torch.nn.functional as F

print("=" * 80)
print("Investigating PyTorch Attention Backend Selection")
print("=" * 80)
print()

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")
print()

# Check which backend PyTorch is using
print("Testing different backends for hdim=8192:")
print()

batch_sz, num_heads, seq_len, head_d = 1, 1, 16, 8192

Q = torch.randn(batch_sz, num_heads, seq_len, head_d, device='cuda', dtype=torch.float32)
K = torch.randn(batch_sz, num_heads, seq_len, head_d, device='cuda', dtype=torch.float32)
V = torch.randn(batch_sz, num_heads, seq_len, head_d, device='cuda', dtype=torch.float32)

# Test with backend selection
backends = [
    ("auto", None),
    ("flash", torch.nn.attention.SDPBackend.FLASH_ATTENTION),
    ("mem_efficient", torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION),
    ("math", torch.nn.attention.SDPBackend.MATH),
]

print(f"Config: bs={batch_sz}, nh={num_heads}, seq={seq_len}, hdim={head_d}\n")

for name, backend in backends:
    try:
        if backend is None:
            # Auto selection
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
            time_ms = start.elapsed_time(end) / 100
            print(f"{name:20s}: {time_ms:.3f} ms")
        else:
            # Force specific backend
            with torch.nn.attention.sdpa_kernel([backend]):
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
                time_ms = start.elapsed_time(end) / 100
                print(f"{name:20s}: {time_ms:.3f} ms")
    except Exception as e:
        print(f"{name:20s}: FAILED - {e}")

# Compare with manual naive
print()
print("Manual naive implementation:")
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
time_ms = start.elapsed_time(end) / 100
print(f"{'naive':20s}: {time_ms:.3f} ms")

print("\n" + "=" * 80)
print("Analysis:")
print("=" * 80)
print("""
If FlashAttention is much slower than naive:
- FlashAttention has overhead for very small batches
- Your configuration (bs=1, seq=16) is too small for FlashAttention
- Your custom kernel is optimized for this exact case

If MATH backend is fastest:
- PyTorch is using cuBLAS/cuDNN for matmuls
- Still slower than your kernel due to multiple kernel launches

Your kernel advantage:
- Single fused kernel (no intermediate memory)
- Optimized for small batch, large hdim
- Online softmax reduces memory traffic
""")

print("\n" + "=" * 80)
print("Test larger batch size for comparison:")
print("=" * 80)

# Test with larger batch
batch_sz = 8
seq_len = 128
print(f"\nConfig: bs={batch_sz}, nh=4, seq={seq_len}, hdim=2048\n")

Q = torch.randn(batch_sz, 4, seq_len, 2048, device='cuda', dtype=torch.float32)
K = torch.randn(batch_sz, 4, seq_len, 2048, device='cuda', dtype=torch.float32)
V = torch.randn(batch_sz, 4, seq_len, 2048, device='cuda', dtype=torch.float32)

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
print(f"PyTorch optimized: {time_opt:.3f} ms")

# Naive
scale = 1.0 / (2048 ** 0.5)
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
print(f"Naive: {time_naive:.3f} ms")
print(f"Speedup: {time_naive / time_opt:.2f}x")

print("\nWith larger batches, PyTorch optimizations should work better.")
)PYTHON";

int main() {
    std::ofstream script("investigate.py");
    script << PYTHON_SCRIPT;
    script.close();

    std::cout << "Running investigation...\n\n";
    int result = system("python3 investigate.py");
    if (result != 0) {
        result = system("python investigate.py");
    }
    return result;
}
