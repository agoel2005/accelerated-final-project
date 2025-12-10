# Fusing Positional Embeddings into Attention Kernels

Advay Goel and Dylan Yu

## Introduction

For memory-bound GPU operations, we commonly use a technique called *kernel fusion* to eliminate redundant memory traffic. When you launch separate kernels, intermediate results must be written to global memory and read back microseconds later. FlashAttention demonstrated this for attention mechanisms: by fusing the QK^T matmul, softmax, and score-value matmul into a single kernel with online computation, it achieved speedups over naive implementations.

We started this project with a simple hypothesis: if fusing attention's internal operations works so well, what about with positional embeddings? Transformers first started with sinusoidal embeddings, and modern transformers have adopted rotational positional embeddings (RoPE). In nearly every modern implementation, these run as separate preprocessing kernels before the attention computation. This requires two extra kernel launches and two full passes over the Q and K matrices. We thought that by fusing these into a singular kernel, we could maybe do better. 

This writeup covers our optimization of fused PE-attention kernels for large hidden dimensions (512-8192), showing what worked, what failed, and why.

## Related Work

There are several attention optimization attempts due to attention's important role in Transformers. Attention optimization is mainly dominated by the FlashAttention family but there are several other algorithms as well. 

Despite this, positional embeddings remain largely unfused in existing systems. All major frameworks, including PyTorch, TensorFlow, and JAX, implement positional embeddings as separate preprocessing kernels that run before attention. To our knowledge, no existing publicly-available implementation fuses sinusoidal positional embeddings into attention kernels, despite the potential for eliminating redundant memory traffic. However, for RoPE, there are a few existing optimizations. 

Most recent work explore optimizing RoPe by combining the rotation operation with the QK matmul, rather than the full attention pipeline. For example, vLLM and TensorRT-LLM kernels apply RoPE inline during matrix multiplication, but still treat Q and K rotation as separate from attention itself. Moreover, the FlashInfer library seems to provide a fused kernel for RoPE and attention, though it does not appear to be documented well and is difficult to benchmark against.


## Baseline Attention Kernel

We first analyzed attention's arithmetic intensity to determine whether the operation is memory-bound or compute-bound.

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

On the RTX 4000 Ada, you need at least 74 FLOPs/byte to be compute-bound. At 0.5 FLOPs/byte, we're severely memory-bound. This means we need to reduce memory traffic.

To start, we wrote an optimized attention kernel with similar optimizations to FlashAttention in order to create baselines and build on top of. We included online softmax with K/V tiling and float4 vectorization. One important choice of note was we decided to keep the data type as FP32. Some (failed) implementations using Tensor Cores used BF16. However, we were unable to get tensor core performance to match our FP32 optimized attention (likely due to the memory-bound nature of the problem). While FP16 also seemed desirable, it turned out to be slower due to all of the conversions back to FP32 in order to compute functions like exp() for softmax. Ultimately, we achieved the following performance:

| Hidden Dim | Time | TFLOPS | Status |
|-----------|------|--------|--------|
| 512 | 0.042 ms | 2.16 | Good |
| 2048 | 0.184 ms | 2.51 | Good |
| 4096 | 0.126 ms | 4.26 | Best |
| 8192 | 0.194 ms | 0.69 | Struggling |

We also wrote a script to compare this performance to PyTorch's implementations of attention. PyTorch required time was as follows:

| Hidden Dim | Optmized Attention | Naive Attention |
|-----------|------|--------|--------|
| 512 | 0.032 ms | 0.077 ms | 
| 2048 | 0.114 ms | 0.219 ms |
| 4096 | 0.190 ms | 0.412 ms | 
| 8192 | 0.362 ms | 0.081 ms |

We aren't as fast as Pytorch's implementation, but our attention kernel performs reasonably well to the point where results with fused attention become more meaningful. From the data, we ultimately realized the TFLOPS performance cliff at hdim=8192. At this scale, the working set (1 MB for K + V) exceeds the L2 cache, causing constant DRAM traffic. We were achieving only 87 GB/s out of the RTX 4000 Ada's 360 GB/s peak bandwidth, just 24% utilization.

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

**Results:** 

| hdim | Fused (V1) | Speedup |
|------|-----------|---------|
| 512 | 0.242 ms | **0.42x**  |
| 2048 | 0.950 ms | **0.43x** |
| 4096 | 0.378 ms | **0.58x** |

This was obviously slow since we were loading sin/cos values from global memory for every single Q-K pair. For a 64x64 attention matrix with hdim=2048, that's 64 x 64 x 2048 = 8.4 million loads from global memory.

### V2-V3: On The Fly Computation

To fix this, we considered calculating the sine/cosine values on the fly:

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

This led to us beating the table approach.

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

RoPE looked similar to sinusoidal embeddings so we expected similar results. The embedding formula rotates pairs of dimensions:

```
for each pair (x, y) at dimension d:
  freq = 1 / (10000^(2d/hdim))
  angle = position * freq
  x' = x * cos(angle) - y * sin(angle)
  y' = x * sin(angle) + y * cos(angle)
```

Like sinusoidal embeddings, the standard implementation uses separate kernels for Q and K, then runs attention. We tried applying the same optimizations: on-the-fly computation with cached frequencies, float4 vectorization, and __sincosf.

### V1-V2: Full Fusion

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

**Results:** 

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
- Apply RoPE to K: 64 vectors x 64 queries = 4,096 operations
- Total: **4,160 RoPE operations**

We were doing **32x more work** to save a single kernel launch and intermediate buffer. Even though both approaches are memory-bound, this made the process significantly slower.

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

### V7: Partial Fusion

Our final idea was to fuse only the Q rotation but keep K rotation separate. 

```
1. Pre-rotate K using separate kernel: O(seq_k) operations
2. Fuse Q rotation into attention: O(seq_q) operations
3. Total: O(seq_q + seq_k) operations (same as fully separate!)
```

The reason why this approach works so well is because it eliminates redoing the computations on K while maintaining the benefits of fusing Q's PE. The results were positive, matching our hypothesis:

| hdim | Hybrid (V7) | Fully Separate | Speedup |
|------|------------|----------------|---------|
| 128 | **0.020 ms** | 0.032 ms | **1.66x faster** |
| 512 | **0.047 ms** | 0.069 ms | **1.45x faster** |
| 2048 | **0.162 ms** | 0.230 ms | **1.40x faster** |

**Why V7 works:**

Q rotation can be fused because each query is processed independently -- we rotate Q once when loading it, then use it for all K comparisons. This is O(seq_q) work.

K rotation cannot be fused because each K vector needs to be compared against all queries. Fusing means rotating K once per query (O(seq_q x seq_k) work), which is seq_q times more expensive than rotating it once in a separate kernel (O(seq_k) work).

By fusing only Q's rotation, we:
- Save one kernel launch (no separate Q RoPE kernel)
- Eliminate one global memory buffer (Q_rope)
- Improve cache locality (Q loaded once, rotated inline)
- Avoid redundant computation

## Why Did Sinusoidal Work?

We thought that since RoPE required partial fusion, why didn't sinusoidal: 

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
2. **Can't cache partial results** 
3. **More total arithmetic:** 4 multiplies + 2 adds per pair vs 1 add per element
4. **SFU less effective:** Need both sin and cos applied to both elements

We also tried using partial-fusion for sinusoidal but the results were actually slower, leading us to stick with the full fusion approach mentioned earlier.

## Discussion and Limitations

### Performance Results Summary

Our experiments demonstrated that kernel fusion for positional embeddings is highly dependent on the embedding type and computational structure:

**Sinusoidal Embeddings:**

We achieved 1.4-2.0x speedup over separate kernels by elimianting ~1MB of redundant memory traffic. We accomplished it via a full fusion.


**RoPE Embeddings:**

We achieved a 1.4-1.6x speedup over separate kernels via a partial fusion where we only fused the Q matrix's positional encoding. For attempts with full fusion, the runtime increased drastically, with results being 2.6-8.5x slower.


### Limitations

**1. Hardware Constraints**

Our kernels were optimized for and tested exclusively on the NVIDIA RTX 4000 Ada Generation GPU. RTX 4000 Ada is no longer a SOTA GPU, so our performance characteristics would differ if we ran it on newer GPUs like H100s. It would also allow us to use much larger hdim sizes and be even quicker than we currently are. 

**2. Workload Assumptions**

Our benchmarks focused on workloads where seq_q ≈ seq_k, batch sizes were 1-4, sequence lengths were 16-64, and hdim size was 512-8192. 

In the real world, decoders are often autoregressive, where seq_k >> seq_q. Additionally, the batch size and sequence length would be a lot larger, and different precision types are used other than just FP32. 

**3. Attention Limitations**

Our base attention kernel was a little slower than PyTorch's built-in optimized Attention function, which is likely built upon FlashAttention 2. If we used this more optimized kernel, it is possible that results could also be different. 

**4. Positional Embedding Coverage**

We only tested two embedding types, sinusoidal and RopE. Two big classes of embeddings that we didn't test were learned embeddings (which seem quite difficult to build out-of-the-box fusions for) and hierarchical embeddings. 


### Directions for Future Work


**1.Tensor Core and Mixed Precision**

We were unable to optimize our attention kernels via tensor cores. If properly used, they would likely speed up our kernel significantly and bring it closer to SOTA on PyTorch. Doing so might elicit new behavior in the fused kernel as well as providing new opportunities to add fusions/optimizations. 

In doing so, we would also shift to BF16, which enables lower precision. However, a potential challenge would be functions like exponentiation, sine, and cosine all require FP32, so there would be a lot of conversions. 

**2. Fusing More Positional Embeddings**

As mentioned above, two classes of PEs that we didn't fuse were learned embeddings and hierarchical. Creating a system for fusing hierarchical embeddings seems like an interesting next step. 

**3. Further Thought into Algorithms**

These were our best attempts at fusing sinusoidal and RoPE embeddings. It is likely that better algorithms exist for fusion that we did not yet think of. These might yield even better performance than what we achieved.


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

---


