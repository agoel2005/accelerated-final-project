# RoPE Fusion Implementation Log

## Goal
Fuse RoPE into attention kernel to eliminate redundant memory traffic for Q/K.

## Starting Point
- Base kernel: attention_fwd_kernel_large_hdim in attention_kernel.cu
- Performance: 1.48-2.02x speedup over baseline (hdim=512-4096)
- Memory-bound kernel → reducing memory traffic should help

---

## V1: Basic Implementation

1. Created attention_kernel_rope_fused.cu
   - Precompute cos/sin on CPU
   - Cache RoPE'd Q in shared memory
   - Apply RoPE to K on-the-fly

2. Bug: hdim>=128 fail with GPU values ~2x CPU
   - Pattern: GPU=-0.233, CPU=-0.126 (ratio 1.85x)
   - Errors start at index 33 (dim 32 = warp boundary)
   - 4055/32768 elements wrong (12%)

3. Root cause: block_reduce_max/sum don't broadcast to all threads
   - Only warp 0 gets correct value
   - Other warps use stale values → wrong softmax normalization

4. Fix: broadcast final result via shared memory
   - Added `shared[0] = val` after reduction
   - Return `shared[0]` instead of local `val`
   - All tests pass: hdim=8,16,32,64,128,512,2048

5. Performance (no vectorization):
   - hdim=128: 0.058ms
   - hdim=512: 0.204ms
   - hdim=2048: 0.901ms

6. Added float4 vectorization
   - Rationale: memory-bound kernel, float4 loads 16B vs 4B
   - Applied to Q RoPE load and K RoPE + dot product
   - Results: 9-19% speedup (0.053ms, 0.178ms, 0.757ms)

7. Benchmark vs separate kernels - UNEXPECTED:
   - Fused SLOWER than separate by 1.66-3.27x
   - hdim=128: 0.053ms vs 0.032ms (1.66x slower)
   - hdim=512: 0.183ms vs 0.069ms (2.65x slower)
   - hdim=2048: 0.753ms vs 0.230ms (3.27x slower)

8. Analysis: why fusion fails
   - Fused loads cos/sin from global mem for each Q-K pair (64 × hdim loads per query)
   - Separate: RoPE kernel loads cos/sin once per token, attention is simple dot products
   - Saved memory from not storing Q_rope/K_rope outweighed by repeated cos/sin loads

---

## V2: On-the-fly cos/sin

9. Compute cos/sin on-the-fly instead of loading from cache
   - Use __sincosf() (fast SFU)
   - Trade memory bandwidth for compute

10. Results - significant improvement:
    - hdim=128: 0.035ms (WINS over separate 0.042ms - 1.2x speedup!)
    - hdim=512: 0.112ms (still slower than separate 0.070ms)
    - hdim=2048: 0.423ms (still slower than separate 0.230ms)

11. Conclusion: works for small hdim only
    - hdim<=128: use fused V2
    - hdim>=512: use separate (inner loop overhead accumulates)

---

## V3: Multi-query per block

12. Process 4 queries per block to amortize K RoPE
    - Apply RoPE to K once, reuse for all 4 queries
    - Shared mem: 4 × hdim for Q_rope (4 × 2048 = 32KB, fits)

13. Results: 22-27% faster than V2, still slower than separate
    - hdim=128: 0.035ms (tie with separate 0.034ms)
    - hdim=512: 0.091ms (slower than separate 0.077ms)
    - hdim=2048: 0.313ms (slower than separate 0.234ms)

14. Analysis: still O(seq_q/4 × seq_k) K RoPE recomputation
    - V3: 16 × 64 = 1024 K RoPE ops
    - Separate: 64 K RoPE ops
    - Gap: 16x more K RoPE
    - Would need 64 queries/block but shared mem limit is 48KB (need 512KB)

---

## V4: Cached frequencies

15. Precompute freq[i] = 1/pow(10000, 2i/hdim) at block start
    - Removes powf from inner loop
    - All threads cooperate on each K position

16. Results: only helps for hdim=2048
    - hdim=128: 0.058ms (1.7x slower than V3)
    - hdim=512: 0.097ms (slightly slower)
    - hdim=2048: 0.273ms (12.5% faster than V3, still slower than separate)

17. Analysis: shared mem overhead + sync barriers outweigh powf savings for small hdim

---

## V5: K in shared memory (FAILED)

18. Load full K tile [BLOCK_K × hdim] into shared memory
    - Apply RoPE in registers during dot product

19. Results: CATASTROPHIC for large hdim
    - hdim=128: 0.046ms (1.4x slower)
    - hdim=512: 0.261ms (3.8x slower)
    - hdim=2048: 1.940ms (8.5x slower!)

20. Root cause: shared memory overflow
    - Available: 40KB = 10000 floats
    - Overhead (Q_rope, freq, reduce): 3104 floats
    - BLOCK_K = (10000-3104)/(2048+1) = 3.36 → 4
    - Only 4 K vectors per tile → 16 tiles total (massive sync overhead)

---

## V6: Dimension tiling

21. Tile HEAD DIMENSION instead of just K sequence
    - Load K in chunks [BLOCK_K × TILE_D] where TILE_D=256
    - For hdim=2048: 8 dimension tiles
    - BLOCK_K=32 instead of 4 → only 2 K tiles instead of 16

22. Results: 3.3x faster than V5, still slower than separate
    - hdim=128: 0.050ms (1.61x slower)
    - hdim=512: 0.161ms (2.33x slower)
    - hdim=2048: 0.583ms (2.52x slower, but 3.3x faster than V5)

23. Fundamental limitation: O(seq_q × seq_k) K RoPE recomputation
    - Fused: 64 × 64 = 4096 K RoPE ops
    - Separate: 64 K RoPE ops
    - Gap: 64x more operations
    - Cannot cache all K_rope (need 512KB, have 48KB)

---

## Partial Rotation (rotary_dim)

24. Expert feedback: real LLMs only rotate subset of dimensions
    - LLaMA configs:
      * hdim=128, rotary_dim=64 (50%)
      * hdim=256, rotary_dim=128 (50%)
      * hdim=2048, rotary_dim=128 (6%)

25. Updated V6 to only rotate first rotary_dim dimensions
    - Rest copied as-is (no sin/cos/powf)
    - Shared mem savings: s_freq now [rotary_dim/2] not [hdim/2]

26. Results: NO significant improvement
    - hdim=128, rotary_dim=64: 0.055ms (1.73x slower)
    - hdim=512, rotary_dim=128: 0.159ms (2.32x slower)
    - hdim=2048, rotary_dim=128: 0.567ms (2.50x slower)

27. Why: O(seq_q × seq_k) recomputation still dominates
    - For hdim=2048, rotary_dim=128:
      * Separate: 64Q + 64K = 128 tokens × 128 dims = 16k ops
      * Fused V6: 64 × 64 × 128 = 524k ops
      * Still 32x more work

28. Insight: partial rotation necessary for decode workloads (1-4 queries)
    - Our workload: 64×64 symmetric prefill
    - FlashInfer's workload: 1-4 queries × 8k-32k KV cache

---

## V7: HYBRID (Pre-rotated K + Fused Q-RoPE)

29. Middle ground approach:
    - Pre-rotate K using separate kernel (O(seq_k))
    - Fuse Q-RoPE into attention (O(seq_q))
    - Total: O(seq_q + seq_k) = same as separate

30. Implementation:
    - Kernel accepts already-rotated K
    - Apply RoPE to Q on-the-fly during load
    - Simple float4 dot product for Q·K (K already rotated)
    - BLOCK_K=64 (no dimension tiling needed)

31. Results: SUCCESS - beats separate across all configs
    - hdim=128, rotary_dim=64: 0.020ms (1.66x faster than separate)
    - hdim=512, rotary_dim=128: 0.047ms (1.45x faster)
    - hdim=2048, rotary_dim=128: 0.162ms (1.40x faster)

32. Why V7 wins:
    - No O(seq_q × seq_k) K RoPE recomputation
    - Saves one kernel launch (no separate Q RoPE)
    - Better cache locality (Q RoPE + dot product together)
    - No Q_rope buffer needed

33. Performance breakdown (hdim=2048, rotary_dim=128):
    - V6 (full fusion): 0.574ms
      * K RoPE: 64 × 64 × 128 = 524k ops (kills performance)
    - Separate: 0.226ms
      * Q RoPE + K RoPE + Attention: 3 kernel launches
    - V7 (hybrid): 0.162ms
      * K RoPE (separate) + Q RoPE+Attention (fused): 2 kernels

---

## Final Conclusions

34. Full fusion (V2-V6) fails due to O(seq_q × seq_k) K recomputation
    - 64x multiplier cannot be overcome with shared mem (need 512KB, have 48KB)

35. Hybrid approach (V7) wins by avoiding redundant computation
    - Pre-rotate K once (separate kernel)
    - Fuse only Q-RoPE into attention
    - 1.4-1.66x speedup over separate

36. Key insight: only fuse what benefits from fusion
    - Q-RoPE fuses well with attention (better cache locality, saved launches)
    - K-RoPE belongs in separate preprocessing
    - Matches FlashInfer's production approach for decode
