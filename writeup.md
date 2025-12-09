# Fusing Positional Embeddings into Attention Kernels

Advay Goel and Dylan Yu

## Introduction

For memory-bound GPU operations, we commonly use a technique called *kernel fusion* to eliminate redundant memory traffic. When you launch separate kernels, intermediate results must be written to global memory and read back microseconds later. FlashAttention demonstrated this for attention mechanisms: by fusing the QK^T matmul, softmax, and score-value matmul into a single kernel with online computation, it achieved speedups over naive implementations.

We started this project with a simple hypothesis: if fusing attention's internal operations works so well, what about with positional embeddings? Transformers first started with sinusoidal embeddings, and modern transformers have adopted rotational positional embeddings (RoPE). In nearly every modern implementation, these run as separate preprocessing kernels before the attention computation. That's two extra kernel launches and two full passes over the Q and K matrices. So we thought: maybe we could do better?

As it turns out, people are smart and if something as simple as "let's add RoPE!" worked, we would have seen it already. But interestingly, fusion with sinusoidal embeddings did work, which begets two questions:

1. Why did sinusoidal embeddings work and RoPE not?
2. Is there a way that we could get RoPE to fuse, even if not fully?

This writeup covers our optimization of fused PE-attention kernels for large hidden dimensions (512-16384), showing what worked, what failed, and why.

## Understanding the Baseline

We first analyzed attention's arithmetic intensity (the ratio of FLOPs to bytes transferred) to determine whether the operation is memory-bound or compute-bound.

For a single query with hidden dimension 8192 and sequence length 16:

```
Q@K^T phase:
- FLOPs: 2 x 16 x 8192 = 262,144
- Memory: Q (32 KB) + K (512 KB) = 544 KB
- Arithmetic Intensity: 0.47 FLOPs/byte

softmax@V phase:
- FLOPs: 2 x 16 x 8192 = 262,144
- Memory: V (512 KB) + scores (64 B) = 512 KB
- Arithmetic Intensity: 0.50 FLOPs/byte
```

On the RTX 4000 Ada (our target GPU with 26.7 TFLOPS FP32 and 360 GB/s memory bandwidth), you need at least 74 FLOPs/byte to be compute-bound. At 0.5 FLOPs/byte, we're **severely memory-bound** -- about 150x below the threshold. This means tensor cores, fancy math tricks, and compute optimizations won't help -- we need to reduce memory traffic.

Our baseline FP32 implementation already included online softmax with K/V tiling (processing keys and values in blocks of 64), which we inherited from the FlashAttention approach. This achieved respectable performance:

| Hidden Dim | Time | TFLOPS | Status |
|-----------|------|--------|--------|
| 512 | 0.062 ms | 2.16 | Good |
| 2048 | 0.214 ms | 2.51 | Good |
| 4096 | 0.126 ms | 4.26 | Best |
| 8192 | 0.194 ms | 0.69 | Struggling |

The performance cliff at hdim=8192 was our first clue. At this scale, the working set (1 MB for K + V) exceeds the L2 cache, causing constant DRAM traffic. We were achieving only 87 GB/s out of the RTX 4000 Ada's 360 GB/s peak bandwidth -- just 24% utilization.

## Sinusoidal Embeddings

Sinusoidal positional embeddings, introduced in the original Transformer paper, have a simple form:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/hdim))
PE(pos, 2i+1) = cos(pos / 10000^(2i/hdim))
```

You compute these values and add them to Q and K before attention. The standard approach uses three kernels:
1. Apply embeddings to Q to get Q'
2. Apply embeddings to K to get K'
3. Run attention on Q', K'

This means writing Q_embedded and K_embedded to global memory, then reading them back. For hdim=2048 and seq=64, that's 1 MB of redundant traffic.

### V1: Naive Fusion

Our first attempt seemed obvious -- precompute sin/cos lookup tables and fuse the embedding application into the attention kernel:

```cuda
// For each Q-K pair in attention:
float q_val = Q[d];
float k_val = K[d];
float sin_val = sin_table[pos_q][d];
float cos_val = cos_table[pos_q][d];

q_val += (d % 2 == 0) ? sin_val : cos_val;
k_val += (d % 2 == 0) ? sin_val : cos_val;

score += q_val * k_val;
```

**Results:** Way too slow.

| hdim | Fused (V1) | Speedup |
|------|-----------|---------|
| 512 | 0.242 ms | **0.42x**  |
| 2048 | 0.950 ms | **0.43x** |
| 4096 | 0.378 ms | **0.58x** |

The problem was obvious once we profiled it: we were loading sin/cos values from global memory for every single Q-K pair. For a 64x64 attention matrix with hdim=2048, that's 64 x 64 x 2048 = 8.4 million loads from global memory, all for values we could compute on the fly.

### V2-V3: Trading Memory for Compute

Let's replace table lookups with on-the-fly computation:

```cuda
float freq = 1.0f / powf(10000.0f, (2.0f * (d/2)) / hdim);
float angle = pos * freq;
float sin_val = sinf(angle);
float cos_val = cosf(angle);
```

| hdim | V1 (table) | V3 (on-the-fly) | Speedup |
|------|-----------|-----------------|---------|
| 512 | 0.242 ms | **0.114 ms** | **2.13x faster** |
| 2048 | 0.950 ms | **0.407 ms** | **2.33x faster** |
| 4096 | 0.378 ms | **0.219 ms** | **1.72x faster** |

Sweet! By trading expensive memory loads for cheap SFU operations, we beat even the baseline separate kernel approach.

### V4-V5: Cached Frequencies + Vectorization

Two more optimizations:

**Cached frequencies (V4):** The `freq` calculation uses `powf`, which is expensive. But frequencies only depend on the dimension index, not position. We precomputed them once in shared memory:

```cuda
__shared__ float s_freq[MAX_HDIM];
if (threadIdx.x < hdim) {
    s_freq[threadIdx.x] = 1.0f / powf(10000.0f, (2.0f * (threadIdx.x/2)) / hdim);
}
__syncthreads();
```

**Float4 vectorization + __sincosf (V5):** We loaded Q, K, and frequencies as `float4` vectors (4 floats at once) and used `__sincosf()` to compute sin and cos simultaneously in a single SFU operation.

Final results:

| hdim | Baseline Separate | V5 Fused | Speedup |
|------|------------------|----------|---------|
| 512 | 0.118 ms | **0.058 ms** | **2.0x faster** |
| 2048 | 0.421 ms | **0.220 ms** | **1.9x faster** |
| 4096 | 0.208 ms | **0.149 ms** | **1.4x faster** |

This achieved a 1.4-2.0x speedup over the separate kernel baseline.

## RoPE

RoPE looked similar enough to sinusoidal embeddings that we expected similar results. The embedding formula rotates pairs of dimensions:

```
for each pair (x, y) at dimension d:
  freq = 1 / (10000^(2d/hdim))
  angle = position * freq
  x' = x * cos(angle) - y * sin(angle)
  y' = x * sin(angle) + y * cos(angle)
```

Like sinusoidal embeddings, the standard implementation uses separate kernels for Q and K, then runs attention. We tried applying the same optimizations: on-the-fly computation with cached frequencies, float4 vectorization, and __sincosf.

### V1-V2: Full Fusion?

Besides the optimizations above, we cached the rotated Q in shared memory but recomputed K's rotation for each Q-K pair:

```cuda
// Cache Q_rope in shared memory (computed once)
for (int d = 0; d < hdim; d += 2) {
    float freq = s_freq[d/2];
    float angle = pos_q * freq;
    float cos_val, sin_val;
    __sincosf(angle, &sin_val, &cos_val);
    s_Q_rope[d]   = Q[d] * cos_val - Q[d+1] * sin_val;
    s_Q_rope[d+1] = Q[d] * sin_val + Q[d+1] * cos_val;
}

// Recompute K_rope for each K vector (every Q needs it)
for (int k_idx = 0; k_idx < seq_k; k_idx++) {
    // Compute K_rope again...
    score += s_Q_rope[d] * k_rope[d];  // Can't cache K_rope!
}
```

**Results:** Not good.

| hdim | Fused | Separate | Performance |
|------|-------|----------|------------|
| 128 | 0.053 ms | 0.032 ms | **1.66x slower** |
| 512 | 0.183 ms | 0.069 ms | **2.65x slower** |
| 2048 | 0.753 ms | 0.230 ms | **3.27x slower** |

The problem was algorithmic. In the separate kernel approach:
- Apply RoPE to all Q vectors: 64 vectors x 1 time = **64 RoPE operations**
- Apply RoPE to all K vectors: 64 vectors x 1 time = **64 RoPE operations**
- Total: **128 RoPE operations**

In the fused approach:
- Apply RoPE to Q: 64 vectors x 1 time = 64 operations
- Apply RoPE to K: 64 vectors x 64 queries = **4,096 operations**
- Total: **4,160 RoPE operations**

We were doing **32x more work** to save a single kernel launch and intermediate buffer. Even though both approaches are memory-bound, adding 30x more computation *and* memory traffic doesn't help.

### V3-V6: More Optimizations

We tried several approaches to salvage this:

**V3 (Multi-query per block):** Process 4 queries per block to amortize K rotation across them. Still 8x redundant computation compared to separate kernels.

**V4 (Cached frequencies):** Precompute frequencies in shared memory to remove `powf` from inner loop. Only helped for hdim=2048, still 2.5x slower than separate.

**V5 (K in shared memory):** Load full K tiles into shared memory and rotate there. This was really bad for large hdim:

| hdim | V5 (K in shmem) | Separate | Performance |
|------|----------------|----------|------------|
| 2048 | 1.940 ms | 0.230 ms | **8.5x slower** |

The problem turned out to be shared memory limits. With only 48 KB available and needing space for [BLOCK_K x 2048 x 2 bytes], we could only fit 4 K vectors per tile, requiring 16 sequential tile loads with synchronization overhead.

**V6 (Dimension tiling):** Tile both the sequence dimension *and* hdim dimension. Better than V5 (3.3x faster), but still 2.5x slower than separate kernels.

### V7: Partial Fusion?

The winning strategy: fuse only Q rotation, keep K rotation separate.

```
1. Pre-rotate K using separate kernel: O(seq_k) operations
2. Fuse Q rotation into attention: O(seq_q) operations
3. Total: O(seq_q + seq_k) operations (same as fully separate!)
```

This "hybrid" approach eliminates the redundant K computation while keeping Q fusion benefits:

| hdim | Hybrid (V7) | Fully Separate | Speedup |
|------|------------|----------------|---------|
| 128 | **0.020 ms** | 0.032 ms | **1.66x faster** |
| 512 | **0.047 ms** | 0.069 ms | **1.45x faster** |
| 2048 | **0.162 ms** | 0.230 ms | **1.40x faster** |

**Why V7 works:**

Q rotation can be fused because each query is processed independently -- we rotate Q once when loading it, then use it for all K comparisons. This is O(seq_q) work.

K rotation cannot be fused because each K vector needs to be compared against all queries. Fusing means rotating K once per query (O(seq_q x seq_k) work), which is 64x more expensive than rotating it once in a separate kernel (O(seq_k) work).

By fusing only Q's rotation, we:
- Save one kernel launch (no separate Q RoPE kernel)
- Eliminate one global memory buffer (Q_rope)
- Improve cache locality (Q loaded once, rotated inline)
- Avoid O(seq^2) redundant computation

## So Why Did Sinusoidal Work?

The answer lies in how the operations interact with the data.

Sinusoidal embeddings add to each element independently:

```cuda
Q'[i] = Q[i] + sin(pos / 10000^(2i/hdim))  // No dependencies between elements
```

Whereas RoPE rotates pairs of elements together:

```cuda
Q'[i] = Q[i] * cos(...) - Q[i+1] * sin(...)
Q'[i+1] = Q[i] * sin(...) + Q[i+1] * cos(...)
```

This coupling means:
1. **Can't vectorize RoPE as easily:** float4 loads don't align with rotation pairs
2. **Can't cache partial results:** The rotation is all-or-nothing
3. **More total arithmetic:** 4 multiplies + 2 adds per pair vs 1 add per element
4. **SFU less effective:** Need both sin and cos applied to both elements

For sinusoidal embeddings:
- Computing on-the-fly: ~4 FLOPs per element (1 add + SFU)
- Loading from table: 2 memory transactions

For RoPE:
- Computing on-the-fly: ~6 FLOPs per element (4 muls + 2 adds + SFU)
- Loading rotated values: Can't cache them without O(seq^2) redundancy

The crucial insight: **fusion only helps when you can compute values on-demand without redundancy**. Sinusoidal embeddings depend only on position and dimension index. RoPE depends on the *values* being rotated, so recomputing it means recalculating for every Q-K pair.

For RoPE, the hybrid approach works because Q appears in the outer loop (one rotation per query) while K appears in the inner loop (would require seq_q rotations). Fusing the outer loop variable succeeds; fusing the inner loop variable fails.

## Conclusion

We set out to fuse positional embeddings into attention kernels and achieved 1.4-2.0x speedups for sinusoidal embeddings and 1.4-1.6x for RoPE (via partial fusion).

For sinusoidal embeddings, full fusion succeeded because sin/cos values depend only on position and dimension index -- computing them on-the-fly during the QK dot product costs about the same as loading from memory, and eliminates intermediate buffers.

For RoPE, full fusion failed because it introduced O(seq_q x seq_k) redundant computation -- rotating K once per query instead of once total. The hybrid solution (fuse Q rotation, separate K rotation) achieved speedups by fusing the outer loop variable while keeping the inner loop variable separate.

The broader lesson: **you should only fuse embeddings that don't depend on the input values**. Sinusoidal embeddings are pure functions of position and dimension, but RoPE rotates the actual Q/K values, so fusing it into the attention loop means recomputing rotations for every Q-K pair instead of once per token.

This explains why production frameworks keep positional embeddings as separate preprocessing: for prefill workloads where seq_q and seq_k are approximately equal, any value-dependent embedding like RoPE would introduce O(N^2) redundant rotations. Even sinusoidal embeddings might not be worth fusing if the framework already has optimized separate kernels. Our results suggest sinusoidal fusion could be worthwhile (1.4-2.0x speedup), while RoPE needs the hybrid approach to avoid catastrophic slowdowns.

Future work could explore:
1. **FP8/INT8 embeddings**: Lower precision positional encodings could reduce memory traffic further on newer GPUs
2. **Learned positional embeddings**: ALiBi and other learned schemes might have different fusion characteristics
3. **Decode workloads**: Hybrid RoPE fusion should work better during inference (seq_q=1-4, seq_k=1000s) since Q rotation redundancy is minimal

---

## Appendix: Benchmarking and Validation

### Benchmarking Methodology

Throughout this project, we validated correctness against PyTorch's `F.scaled_dot_product_attention`. Here's our methodology:

**Test Data Generation:**
```python
# generate_pytorch_reference.py
def generate_reference(batch_size, num_heads, seq_len, hidden_dim):
    Q = torch.randn(batch_size, num_heads, seq_len, hidden_dim)
    K = torch.randn(batch_size, num_heads, seq_len, hidden_dim)
    V = torch.randn(batch_size, num_heads, seq_len, hidden_dim)

    output = F.scaled_dot_product_attention(Q, K, V)

    # Save to binary for CUDA kernel testing
    save_reference(Q, K, V, output, f"pytorch_ref_{hidden_dim}.bin")
```

**CUDA Validation:**
```cuda
// Load PyTorch reference data
load_reference_data("pytorch_ref_2048.bin", h_Q, h_K, h_V, h_expected);

// Run our kernel
attention_forward_optimized(d_Q, d_K, d_V, d_output, ...);

// Compare results
float max_diff = compare_outputs(h_output, h_expected, total_elements);
bool passed = (max_diff < 5e-5);  // Tolerance for FP32
```

We used **telerun** to execute tests on remote GPU servers with NVIDIA RTX 4000 Ada Generation GPUs (6144 CUDA cores, 26.7 TFLOPS FP32, 360 GB/s bandwidth). All benchmarks reported in this writeup were run on this hardware.

**Correctness Results:** All implementations passing with max error < 5e-5 for FP32.
