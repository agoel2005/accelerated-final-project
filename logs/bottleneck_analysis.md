# Attention Kernel Bottleneck Analysis

## Current Performance (attention_kernel_large_dims.cu)

| hdim  | TFLOPS | Time (ms) | Observations |
|-------|--------|-----------|--------------|
| 512   | 0.76   | 0.044     | ✓ Good performance |
| 2048  | 0.86   | 0.156     | ✓ Best performance |
| 4096  | 0.27   | 0.124     | ⚠️ 3x drop! |
| 8192  | 0.05   | 0.183     | ❌ 17x drop! |

**Key Observation**: Performance collapses for large hidden dimensions!

---

## Root Cause Analysis

### 1. **Arithmetic Intensity (AI) - Memory Bound!**

**Formula**: AI = FLOPs / Bytes Loaded

For attention with 1 query, seq_k=16, hdim=8192:

**Q@K^T Phase:**
- FLOPs: 2 × 16 × 8192 = 262,144
- Memory: Q (8192×4) + K (16×8192×4) = 557,056 bytes
- **AI = 0.47 FLOPs/byte** ❌ Very low!

**softmax@V Phase:**
- FLOPs: 2 × 16 × 8192 = 262,144
- Memory: V (16×8192×4) + scores (16×4) = 524,352 bytes
- **AI = 0.50 FLOPs/byte** ❌ Very low!

**Compare to GPU Peak:**
- Need AI > 100 FLOPs/byte to be compute-bound
- We have AI < 1 FLOPs/byte
- **Conclusion: Severely memory-bound!**

---

### 2. **Low GPU Utilization**

**Current Launch Config:**
```cpp
Grid size: batch_sz × num_heads × len_q
For our tests: 1 × 4 × 64 = 256 blocks (or even just 16 blocks for hdim=8192)
Block size: 256 threads
```

**Problem:**
- Modern GPUs have 80-140 SMs (Streaming Multiprocessors)
- With only 256 blocks, many SMs sit idle
- For hdim=8192 test: Only 16 blocks → terrible utilization!

---

### 3. **No Data Reuse Across Queries**

**Current Design**: One query per block
- Each block loads its own copy of K (seq_k × hdim floats)
- Each block loads its own copy of V (seq_k × hdim floats)
- **K and V are loaded Q times with ZERO sharing!**

**Example** (64 queries, hdim=8192):
- K loaded: 64 times × 16 × 8192 × 4 bytes = 33 MB
- V loaded: 64 times × 16 × 8192 × 4 bytes = 33 MB
- **Total redundant memory: 66 MB!**

---

### 4. **Cache Thrashing for Large hdim**

**L1 Cache**: ~128 KB per SM
**L2 Cache**: ~40 MB total (shared across GPU)

**Working Set Analysis:**

| hdim | K size | V size | Total | Fits in L1? | Fits in L2? |
|------|--------|--------|-------|-------------|-------------|
| 512  | 32 KB  | 32 KB  | 64 KB | ❌ (barely) | ✓ Yes |
| 2048 | 128 KB | 128 KB | 256 KB | ❌ No | ✓ Yes |
| 4096 | 256 KB | 256 KB | 512 KB | ❌ No | ⚠️ Tight |
| 8192 | 512 KB | 512 KB | 1 MB | ❌ No | ❌ No |

**For hdim=8192:**
- Working set (1 MB) >> L2 cache per query
- Constant DRAM access
- **Memory bandwidth saturated!**

---

### 5. **Shared Memory Underutilization**

**Current Usage:**
```cpp
shmem_sz = (len_k + 32) * sizeof(float)
For seq_k=64: 384 bytes (only 0.8% of available 48KB!)
```

**Wasted Opportunity:**
- Have 48 KB shared memory available
- Only using 384 bytes for score storage
- Could cache K/V tiles in shared memory for reuse!

---

## Memory Bandwidth Analysis

**Measured Performance:**
- hdim=8192: 0.05 TFLOPS, 0.183 ms
- FLOPs: 2 × 1 × 1 × 16 × 16 × 8192 × 2 = 8.4M FLOPs
- Memory transferred (estimate):
  - Q: 16 × 8192 × 4 = 524 KB
  - K: 16 × 16 × 8192 × 4 = 8.4 MB (loaded 16 times)
  - V: 16 × 16 × 8192 × 4 = 8.4 MB (loaded 16 times)
  - Total: ~17 MB
- **Memory BW achieved**: 17 MB / 0.183 ms = 93 GB/s

**GPU Theoretical Peak** (A100/H100):
- A100: ~2000 GB/s
- **We're achieving only 4.7% of peak bandwidth!**

---

## Why Performance Drops for Large hdim

1. **Small hdim (512, 2048):**
   - Working set fits in L2 cache
   - Good cache hit rate
   - 0.76-0.86 TFLOPS

2. **Large hdim (4096, 8192):**
   - Working set >> cache
   - Every access goes to DRAM
   - **Cache miss rate → 100%**
   - 0.05-0.27 TFLOPS (10x+ slower!)

---

## The Core Problem: Sequential Query Processing

```
Current: Process queries ONE AT A TIME
Query 1: Load K → Load V → Compute → Discard K,V
Query 2: Load K → Load V → Compute → Discard K,V
Query 3: Load K → Load V → Compute → Discard K,V
...

Problem: Load K and V repeatedly with no reuse!
```

---

## How FlashAttention Solves This

FlashAttention tiles **both Q and K dimensions**:

```
1. Load Q tile (e.g., 16 queries) into shared memory
2. For each K/V tile:
   - Load K tile into shared memory
   - Compute scores for all 16 queries (REUSE K!)
   - Load V tile
   - Update all 16 outputs (REUSE V!)
3. Use online softmax to avoid storing full attention matrix
```

**Key Benefits:**
- K/V loaded once and reused across 16 queries
- Working set fits in shared memory
- 16x less DRAM traffic
- 16x better GPU utilization

---

## Concrete Improvement Opportunities

### 1. **Process Multiple Queries Per Block** (HIGHEST IMPACT)
- Change from 1 query/block to 16 queries/block
- Cache K/V tiles in shared memory
- **Expected speedup: 10-20x for large hdim**

### 2. **Increase Block Count** (GPU Utilization)
- More blocks = better SM utilization
- Especially important for small sequence lengths

### 3. **Better Cache Utilization**
- Reorder memory accesses for better locality
- Tile Q, K, V to fit in shared memory

### 4. **Reduce Memory Footprint**
- Use online softmax (no full score matrix)
- Stream K/V in tiles

---

## Next Steps (Prioritized)

1. **Implement Multi-Query Processing** ⭐⭐⭐
   - Biggest potential gain
   - Process 4-16 queries per block
   - Cache K/V in shared memory

2. **Profile Current Kernel** ⭐⭐
   - Use `nsys` or `ncu` to see actual memory BW
   - Measure cache hit rates
   - Identify exact bottleneck

3. **Optimize Memory Patterns** ⭐
   - Better coalescing
   - Reduce bank conflicts

4. **Compare with PyTorch** ⭐⭐⭐
   - Benchmark against PyTorch F.scaled_dot_product_attention
   - See how much room for improvement exists
