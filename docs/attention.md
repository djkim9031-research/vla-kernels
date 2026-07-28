# Kernel #2 — Fused Masked Attention (tensor cores)

A hand-written scaled-dot-product-attention kernel for the niche PyTorch's
backends serve worst: **attention with a custom mask** — which every
prefix/expert site in SmolVLA carries (a padded-language dead band in all of
them, plus a causal action staircase in the joint layers), disqualifying
flash and forcing the memory-efficient backend.

Results at the model's REAL masks (extracted from a live inference —
`results/real_masks.pt` — not reconstructed; bf16, head_dim 64, Jetson Thor,
7 repetitions × 200-iter averages):

| site (B,H,M,N) | metric | F.sdpa eager call (mem-efficient) | ours | verdict |
|---|---|---|---|---|
| (1,15,50,241) ×80/inf | median | 60.4 µs | **33.3 µs** | **1.8× faster** |
| | best-case | 48.8 µs | 33.2 µs | 1.5× faster |
| (1,15,50,291) ×80/inf | median | 40.6 µs | 39.2 µs | parity (1.04×) |
| | worst-case | **110.0 µs** | **40.9 µs** | **2.7× tighter tail** |
| (1,15,241,241) ×16/inf | median | ~69 µs | 150 µs | theirs — routed to F.sdpa |

Scope of the comparison column: these are **eager-mode SDPA calls**, which
pay GPU-side mask preparation on every invocation. Inside a compiled graph
that preparation is hoisted and fused away, and the picture inverts — see
the integration section below. The table above is the right comparison for
eager callers; it is not the cost of the mem-efficient *kernel*.

The stability column matters as much as the speed column for a robot control
loop: across every session and mask, our kernel holds a ~2 µs spread while
the mem-efficient backend's masked path swings 40–110 µs.

Verified mask census (one live inference, all sites): every expert/prefix
mask contains a never-visible column band (the padded language slots — cols
196–239 for this task string), so **flash is legitimately ineligible at all
of them**; the joint sites additionally carry the causal action staircase
(verified equal to tril). Routing: vision → flash (all-ones patch mask
dropped — the one genuinely ceremonial mask, fixed earlier); both expert
sites → `vlak::fused_attention`; prefix → F.sdpa.

**Corrections history (2026-07-24):** an earlier revision claimed 1.30× at
(50,241) under a synthetic staircase mask (prefix=191) that does not occur
in the model, and asserted the prefix/cross masks were all-True "by
construction" — forgetting that the language block is padded. Both claims
are withdrawn and replaced by the real-mask measurements above. Lesson
recorded below.

**Corrections history (2026-07-27):** the table's 1.8× was previously
presented without qualifying the baseline. Integration profiling showed the
eager SDPA call spends most of its time on per-call mask preparation, not
in the attention kernel itself; under torch.compile that overhead vanishes
and the mem-efficient kernel runs 21.6 µs — faster than ours. The claim now
carries its scope. Full account in the integration section.

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

## Integration into the compiled model (compiled-vk, 2026-07-27)

The `compiled-vk` variant routes the 160 expert attention calls per
inference to `vlak::fused_attention` inside the compiled + CUDA-graphed
model (prefix stays on F.sdpa, vision on flash). Getting it to run
correctly surfaced two real bugs, and getting numbers out of it corrected
the op-level story.

**Bug 1 — stream capture escape.** The kernels launched with bare
`<<<...>>>`, which targets the legacy default stream. Eager mode masks this
(the legacy stream synchronizes with everything); CUDA graph capture runs
on a side stream, so the launches escaped capture and replays served stale
output buffers — surfacing as NaN actions three layers downstream. A
minimal repro (capture one op call, replay with new input values, compare
against eager) showed replays returning the *previous* answer. Every launch
now targets `c10::cuda::getCurrentCUDAStream()`.

**Bug 2 — fake-impl strides.** The registered fake returned
`torch.empty_like(q)`, which inherits the caller's view strides; the kernel
returns contiguous output. Inductor plans buffer layouts from the fake and
asserts at runtime on the mismatch. The fake now promises contiguous
strides. (Cache note: correcting a fake does not invalidate compiled
artifacts — stale Inductor caches must be cleared by hand.)

**Result.** Accuracy retention 0.999951 — gate PASS. Latency: 63.5 ms p50
vs the same-day compiled-ro control at 57.8 ms. Per-kernel profiling inside
the graph:

| expert-site attention | in-graph cost |
|---|---|
| `fmha_cutlassF` (mem-efficient, under compiled-ro) | 21.6 µs/call |
| `fused_attention_wmma_bf16` (ours, under compiled-vk) | 36.4 µs/call |

Our kernel measures consistently in both settings (33.3 µs isolated →
36.4 µs in-graph; L2 contention accounts for the difference). The baseline
does not: the 60.4 µs eager SDPA figure was dominated by per-call mask
preparation kernels that compilation hoists away. Against the mem-efficient
kernel's true 21.6 µs, ours loses 1.7× at these shapes.

`compiled-ro` therefore remains the shipping variant; `compiled-vk` stays
in the tree as a working, gate-passing integration whose value is the
finding itself. Beating tuned cutlass at M=50 would need a further kernel
generation (v2.5) with uncertain payoff against known cheaper wins
elsewhere in the budget.

## v3: attack the region, not the kernel (2026-07-27)

v3 (`vlak::fused_attention_gqa`, `compiled-vk3` variant) rebuilds the op
around what the library kernel cannot know about this model:

- **GQA-native**: query head h reads kv head h/G from the unexpanded
  5-head cache; the expand copies vanish and in-kernel K/V reads drop 3x
  (sibling heads share lines through L2).
- **Layout-native**: q/k/v consumed at their natural (B, seq, heads, 64)
  strides; output written (B, M, H, 64) contiguous so the trailing
  reshape is a view. Every transpose/contiguous staging kernel vanishes.
- **Analytic two-boundary mask**: visible(r, c) = c < P ? (c < ds ||
  c >= de) : staircase — solved per site by a one-time eager probe that
  requires EXACT equality with the captured mask, falling back to sdpa
  otherwise. No mask tensor, no mask traffic.
- **Register (m, l) + shuffle reductions**: a row's TPR stat-sharing
  lanes live in one warp, so the smem stats and two of six per-tile
  block barriers go away (adversarial 5-lens review, compute-sanitizer
  x3, and a 300-trial randomized stress all clean).
- **Register-resident columns (v3.2)**: the softmax phase holds its
  scores in registers between the max and exp passes instead of a smem
  writeback/re-read (the softmax-v4 lesson) — worth ~3 us/call.

Locked-clock scoreboard at the expert sites, in-graph and same-session:

| | compiled-ro | compiled-vk3 |
|---|---|---|
| attention kernel | fmha_cutlassF 19.9 us/call | ours 21.6 us/call |
| expand/cast/mask-prep entourage | 2.0 us/call | none |
| **region** | **21.9 us/call** | **21.6 us/call** |

**Verdict: region parity.** The mem-efficient kernel keeps a 1.7 us
kernel-vs-kernel edge; the zero-entourage design cancels it. End-to-end
p50 differences between the two variants (both ~54-56 ms, retention
0.99994 gate PASS) sit inside run-to-run noise. Isolated timings match
in-graph timings for our op throughout (21.2 vs 21.6 us) — the eager-call
inflation documented above never applies to it.

Measured dead ends, kept for the record: **fragment skipping** (masking
two fully-dead 16-col WMMA fragments per cross tile) LOST ~2.4 us — in a
barrier-synchronized block, skipping work for some warps saves nothing
unless the slowest participant gets faster, and the branch overhead is
paid everywhere. **Warp-specialized softmax hiding** was ruled out by
phase attribution before implementation: loads/barriers ~8.7 us, mma
~7.7 us, softmax ~8.0 us (now ~4.8) — both phases are issue-bound on the
same warps, so a 4/4 split doubles each phase and the overlap cannot pay.
**enable_gqa=True** drops to the fp32 math backend when a mask is present
(~280 us/call): the GQA read savings are unreachable via stock sdpa.

## v4: the register pipeline (2026-07-27) — the bar falls

v3.2 reached region parity but still paid WMMA's structural tax: opaque
fragments force the score tile through shared memory (store S, scalar
softmax, store P, reload) with two extra block barriers per K/V tile. v4
(`vlak::fused_attention_gqa_v4`, `compiled-vk4`) rebuilds the inner loop
on **CUTLASS/CuTe** (vendored as a submodule at `third_party/cutlass`),
using the SM80 mma atom (`SM80_16x8x16_F32BF16BF16F32_TN`) whose
thread<->element mapping is architecturally defined — a compile probe
pinned the sm_110 menu first (mma.sync yes; tcgen05 no; wgmma is
Hopper-only):

- softmax runs ON the accumulator registers: row max/sum via two
  xor-shuffles among the 4 lanes that share a row (an SM80-atom fact)
- exp(S) becomes P's A-fragment by a pure CuTe layout relabeling of the
  same registers (`logical_divide` of the accumulator layout — the C and
  A fragments share (row, column-pair) ownership)
- fragment loads go through CuTe LDSM copy atoms (`SM75_U32x4_LDSM_N`
  for Q/K, the transposing `SM75_U16x8_LDSM_T` for V); mask coordinates
  come from CuTe identity tensors — no hand-derived lane arithmetic and
  no inline PTX anywhere in the kernel
- S and P never touch shared memory; barriers drop to 2 per tile
  (K/V visibility + buffer protection)
- 64-thread blocks at ~41 KB smem -> 2 blocks/SM (v3's 100 KB blocks
  pinned occupancy at 1), giving cross-block latency hiding
- the TiledMMA carries an N-permutation tile of 16 so B fragments match
  the LDSM atoms' 8-value granularity (the flash-attention shaping)
- inherits v3's contracts unchanged: GQA-native, arbitrary strides,
  (B, M, H, 64) contiguous output, probed analytic mask

Locked-clock results (7 sessions, medians):

| | v3.2 | v4 | fmha kernel (bar) |
|---|---|---|---|
| cross (50x241) | 21.2 us | **16.5 us** | ~19.9 |
| self (50x291) | 26.2 us | **18.6 us** | ~19.9 (avg) |

First kernel-vs-kernel win, on top of the zero-entourage region. End to
end (paired alternating A/B, same session): compiled-ro 53.1/53.5 ms vs
**compiled-vk4 49.5/49.7 ms p50** — a strict improvement in both rounds,
so by the shipping rule compiled-vk4 becomes the default. Retention
0.999956 gate PASS; LoadGen SingleStream p90 51.0 ms and Offline 20.1
samples/s, both VALID. Cumulative: eager 198.4 -> ~49.6 ms = **4.0x at
~20 Hz**. Correctness: full case grid + strided + v3-equivalence +
compile-composability + 30-trial stress in tests/test_attention_v4.py;
racecheck/synccheck/memcheck clean.

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
6. **Not every site is yours to win.** At the prefix's 241×241 shape the
   mem-efficient backend beats us ~2×; it keeps that site. Routing beats
   heroics; the kernel keeps only the sites where it is the measured winner.
7. **Benchmark the model's inputs, not your reconstruction of them.** Two
   published claims died to this: a synthetic staircase mask at a shape
   where the model uses a different pattern, and an "all-True by
   construction" argument that forgot the language padding. The fix that
   made claims durable: extract the masks from a live inference and assert
   their structure (column dead-bands, tril staircase) before benchmarking
   against them.
8. **Report tails, not just medians.** The masked mem-efficient path is
   bimodal (40–110 µs across sessions); ours is flat. Median-only reporting
   would have hidden the most deployment-relevant difference.
9. **Launch on the current stream, always.** Bare `<<<...>>>` submits to the
   legacy default stream and works in eager mode for years — then silently
   escapes CUDA graph capture. The failure mode is stale outputs, not an
   error. `getCurrentCUDAStream()` on every launch site is the cost of
   composing with the modern stack.
10. **An eager-mode baseline is not an in-graph baseline.** The same F.sdpa
    call costs 60 µs eager and 22 µs compiled — the difference is per-call
    mask preparation that compilation hoists. An op-level win measured
    against eager calls says nothing about winning inside torch.compile;
    the only benchmark that settles integration questions is the integrated
    model, profiled per kernel.

## Analytic-mask caveat

The `prefix_len` analytic path models the idealized no-padding pattern and
does NOT represent the deployed masks (which carry the pad dead band); the
real sites are served through the tensor-mask path. Extending the analytic
form with a second boundary (real-language end + state column) is queued —
the pattern remains arithmetic.

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
