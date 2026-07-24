# Kernel #2 — Fused Masked Attention (tensor cores)

A hand-written scaled-dot-product-attention kernel for the niche PyTorch's
backends serve worst: **attention with a custom block mask** — the pattern
SmolVLA's prefix/expert layers require (bidirectional prefix + causal action
suffix), which disqualifies flash and forces the memory-efficient backend.

Result at the model's masked expert sites (bf16, head_dim 64, batch 1,
Jetson Thor, locked clocks; 300-iter averages):

| site (B,H,M,N) | F.sdpa (mem-efficient, masked) | ours | speedup |
|---|---|---|---|
| (1,15,50,241) ×80/inference | 40.7 µs | **31.3 µs** | **1.30×** |
| (1,15,50,291) ×80/inference | 40.6 µs | **37.2 µs** | **1.09×** |
| (1,15,241,241) ×16/inference | 47.2 µs | 142.1 µs | 0.33× — see routing |

The prefix site (241×241) is not our kernel's to win: its mask is all-True
by construction at fixed shapes, so the correct treatment is dropping the
mask and letting **flash** run it (14.5 µs — 3.3× faster than masked SDPA).
Full routing: vision → flash, prefix → flash (mask dropped), expert sites →
`vlak::fused_attention`, anything off-envelope → F.sdpa.

## Design

Flash-2 structure, one kernel launch per site:

```
grid: (B·H) x ceil(M/BMT)   BMT = 32 (M <= 96) or 64;  256 threads (8 warps)
per block:
  Q tile resident in smem; K/V tiles stream through DOUBLE-BUFFERED smem
  (cp.async: tile t+1 copies while tile t computes)
  per KV tile:
    S = Q K^T      WMMA bf16 16x16x16 fragments, fp32 accumulate,
                   (row-frag x col-frag) jobs distributed over all 8 warps
    mask + online softmax   in-smem scalar phase, V2_THREADS/BMT threads
                   per row: masked max -> merged (m,l) via the associative
                   update (softmax v4's mlMerge) -> P in bf16 -> O rescale
    O += P V       WMMA, all warps
  epilogue: O / l -> output      (scores NEVER touch HBM)
```

The mask enters in one of three modes:
- `prefix_len=P` — **the mask as arithmetic**: visible(col; row) = col < P
  or col ≤ (N−M)+row. No mask tensor exists at all — no memory traffic, no
  byte loads; eligibility is computed from indices. Additionally each KV
  tile is classified once per block: fully-visible tiles run a branchless
  fast path, fully-invisible tiles are skipped outright (no mma, no
  softmax). This is the FlexAttention idea (mask-as-code) hand-built for
  one pattern.
- `attn_mask` — generic (B,M,N) bool for arbitrary patterns.
- neither — unmasked fast path.

## The iteration ladder (all measured at (1,15,50,291), masked)

| version | change | µs | vs bar |
|---|---|---|---|
| v1 | scalar warp-per-row scaffold (correctness first) | ~107 | 0.38× |
| v2.0 | WMMA tiles, smem-mediated softmax | 125 | 0.32× |
| v2.1 | scalar phase 1 → 2 threads/row | 80 | 0.51× |
| v2.2 | cp.async double-buffered K/V + 256 threads | 55.5 | 0.73× |
| v2.3 | BMT=32 template (fills 20 SMs at M=50) + wmma jobs on all 8 warps | 39.2 | 1.03× |
| v2.4 | analytic mask + tile classification + fast/skip paths | **37.2** | **1.09×** |

(v2.0 being *slower* than v1 is the price of scaffolding: correct
structure first, speed after — same method as softmax v0→v4.)

## Lessons worth keeping

1. **Measure the real bar.** Unmasked F.sdpa (flash) runs 14–18 µs at these
   shapes; *masked* F.sdpa runs 37–55 µs. The niche only exists because the
   mask disqualifies flash — sizing it against the wrong bar would have
   either scared us off or declared a false win.
2. **At 1 block/SM, overlap is everything.** The single biggest step
   (v2.2) came from cp.async double buffering — with ~100 KB of smem per
   block there is no second block to hide latency behind.
3. **Small-M shapes starve the grid.** 15 heads × 1 tile = 15 blocks on
   20 SMs; templating BMT=32 doubled the grid and was worth ~1.3× alone.
4. **A mask that is data is traffic; a mask that is code is free.** The
   analytic form removed ~200 KB of bool reads per call and all mask
   branches from full tiles.
5. **Fast paths breed tail bugs.** The fully-visible fast path skipped the
   `col < N` guard, letting zero-padded tail columns into the softmax at
   any N not divisible by the tile width. Caught by the exact-equivalence
   test between the analytic and tensor-mask paths — bit-identical outputs
   are a stronger oracle than tolerance comparisons.
6. **Not every site is yours to win.** The prefix site's all-True mask
   means flash — once un-blocked — beats everything by 3×+. Routing beats
   heroics; the kernel keeps only the sites where it is the measured
   winner.

## Envelope

Validated: bf16, head_dim 64, batch·heads parallelism, the SmolVLA mask
family, M ≤ ~100 (BMT=32 path). Outside it: large M loses (the per-M-tile
K/V re-stream is a known scaling flaw), other dtypes fall to the scalar v1
path, other head_dims are rejected. Integration must guard the envelope and
route to F.sdpa beyond it.

## Reproduce

```bash
python3 -m pytest tests/test_attention.py -q      # 35 cases, incl. exact
                                                  # analytic==tensor equivalence
python3 - <<'EOF'
import torch, torch.nn.functional as F
from kernels.attention import fused_attention
from tests.test_attention import smolvla_mask
q = torch.randn(1,15,50,64, device='cuda', dtype=torch.bfloat16)
k = torch.randn(1,15,291,64, device='cuda', dtype=torch.bfloat16)
v = torch.randn(1,15,291,64, device='cuda', dtype=torch.bfloat16)
print(fused_attention(q, k, v, prefix_len=241).shape)
EOF
```
