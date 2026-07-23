# Kernel #1 — Fused Row Softmax

Softmax over the last dim of a `[rows, cols]` tensor — the reduction pattern at
the core of transformer attention. Four hand-written CUDA versions plus two
library baselines, benchmarked warm and profiled cold with Nsight Compute on
Jetson Thor (sm_110, CUDA 13.0, clocks locked via `bench/rig.sh`).

## Variants

| variant | strategy | key idea |
|---------|----------|----------|
| v0 | one **thread** per row | naive baseline, uncoalesced serial loads |
| v1 | one **warp** per row | `__shfl_xor` register reductions, no barriers |
| v2 | one **block** per row | shared-mem block reduce, safe softmax (3 passes) |
| v3 | one **block** per row | **online softmax**: fused max+sum, 2 passes |
| **v4** | one **block** per row | **fused (m,l) tree + raking combine + register-resident rows** — the synthesis of the profiling findings below |
| cudnn | `cudnnSoftmaxForward` | vendor baseline (register-resident design) |
| cub | v2 structure on `cub::BlockReduce` | library raking reduction vs our hand-rolled one |

v4 folds in one lesson per profiled competitor: a single fused (m,l)
online reduction tree instead of separate max and sum trees; CUB's raking
cross-warp combine (2 barriers total vs v2/v3's 4–6); and cuDNN's
register-resident trick at a modest budget (RPT=8 floats/thread, so rows up
to 2048 columns are read from global memory exactly once and written from
registers) without cuDNN's occupancy collapse. Wider rows fall back to a
streaming path with the same reduction. All dtypes: fp32, fp16, **bf16**
(fp32 accumulation throughout).

`softmax(x, "auto")` routes by measured crossover: cols > 2048 → v4;
else rows ≥ 32 → v1; else v4. cuBLAS has no softmax op; it becomes the
baseline at the GEMM kernel instead.

The op is registered with the dispatcher (`TORCH_LIBRARY(vlak)` +
`register_fake`), so `softmax()` traces as a single node under
`torch.compile(fullgraph=True)` / `torch.export` — the same custom-op
pattern the fused attention kernel uses.

## Warm benchmark (locked clocks, 100-iter loop)

Selected rows; full sweep in `results/softmax.csv` (5 shapes × fp32/fp16/bf16
× 7 variants, 105 rows). Times in ms.

| shape · dtype | torch | v1 | **v4** | cudnn | cub |
|---|---|---|---|---|---|
| 2048×512 fp32 | 0.0154 | **0.0124** | 0.0347 | 0.0127 | 0.0472 |
| 512×2048 fp32 | 0.0205 | 0.0144 | **0.0124** | 0.0291 | 0.0164 |
| 512×2048 fp16 | 0.0309 | **0.0124** | 0.0128 | 0.0226 | 0.0164 |
| 512×2048 bf16 | 0.0437 | 0.0125 | **0.0124** | 0.0226 | 0.0173 |
| 256×4096 fp32 | 0.0371 | 0.0165 | **0.0136** | 0.0206 | 0.0144 |
| 256×4096 fp16 | 0.0370 | 0.0144 | **0.0136** | 0.0184 | 0.0143 |
| 256×4096 bf16 | 0.0374 | 0.0145 | **0.0132** | 0.0188 | 0.0144 |
| 64×16384 fp32 | 0.0267 | 0.0309 | **0.0162** | 0.0254 | 0.0164 |
| 64×16384 fp16 | 0.0388 | 0.0246 | **0.0132** | 0.0230 | 0.0148 |
| 64×16384 bf16 | 0.0274 | 0.0285 | **0.0138** | 0.0226 | 0.0150 |

The winner is shape-dependent, and v4 now owns everything from 2048 columns
up:

- **Wide rows (cols ≥ ~2048, any row count):** v4 leads — up to 3.5× over
  torch (bf16 512×2048), ~2.9× at fp16 64×16384, and ahead of CUB at the
  shape where CUB previously beat our v2/v3. bf16 mirrors fp16 throughout.
- **Many rows, narrow cols:** v1 keeps the crown (matches cuDNN at
  2048×512); at 512 columns a block per row leaves parallelism on the table
  that v1's warp-per-row gets for free.
- **Tiny rows (1024×128):** cuDNN edges everyone; launch overhead dominates.

A lesson worth keeping from v4's first iteration: the register-resident path
initially reused the *online* update across its 8 register elements — an
8-step serial chain of exponentials per thread (ncu: SM 40.6%, 26.8 µs cold).
Switching to the classic two-phase form on the registers (independent max
tree, then independent exp sum) cut SM pressure to 31.7% and the time to
25.7 µs. **Online softmax is required for streaming data and for cross-thread
merges — on register-resident data it is pure overhead.**

Cold-cache (ncu, 512×2048 fp32): v4 25.7 µs — L2 hit 0.2% (each byte read
once, the cuDNN pattern) at 90.6% occupancy (vs cuDNN's 31.8%). CUB's
three-pass kernel remains slightly ahead cold at this one shape (22.7 µs,
L1-resident re-reads); warm, v4 leads it by ~25%.

An earlier run of this table (June) was taken without locked clocks and
overstated the speedups (e.g. 4.45× vs torch at 512×2048 fp16; the
locked-clock number is 2.49×). All current numbers are under
`nvpmodel -m 0` + `jetson_clocks`.

## Cold-cache profiles (Nsight Compute, 512×2048 fp32)

One shape, all six kernels, so the comparison is on identical work. Raw
export: `results/softmax_all6_ncu.csv`, parsed by `bench/roofline.py`.

| | v0 | v1 | v2 | v3 | cudnn | cub |
|---|---|---|---|---|---|---|
| duration (µs) | 719.4 | 29.3 | 44.2 | 39.0 | 42.8 | **22.7** |
| SM % | 0.9 | 16.3 | 27.4 | 28.7 | 28.8 | 43.0 |
| L1 / L2 hit % | 72/88 | 31/27 | 49/2 | 30/5 | **0/0.7** | 47/5 |
| occupancy % | 8.3 | 46.1 | 89.8 | 87.5 | **31.8** | 88.6 |
| registers/thread | 40 | 40 | 21 | 26 | **88** | 40 |
| executed instructions | 470k | 523k | 1373k | 1340k | 1358k | **1128k** |
| warp cycles/instr | 39.1 | 30.9 | 33.3 | 33.1 | **12.0** | 22.1 |

What the counters say about each design:

- **v0** — with one thread per row, 512 threads fill only 4 blocks: 16 of
  Thor's 20 SMs are idle, and the L1 pipe saturates (91.6% throughput) on
  serial 4-byte dependent loads. A latency-serialized kernel on a fifth of
  the chip.
- **v1** — the leanest instruction stream of the parallel variants (523k vs
  ~1.3M for block-per-row): warp shuffles need no shared memory, no barriers,
  no broadcast. 46% occupancy is enough to keep the memory system busy, so
  the extra occupancy of v2/v3 buys nothing here.
- **v2 vs v3** — near-identical profiles; the online single pass saves ~5 µs.
  v2's higher L1 hit rate (49%) is its extra read pass hitting cache, which
  is why the saved pass matters little while a row (8 KB) still fits in L1.
  v3's advantage grows once rows outgrow L1 — the regime the flash-attention
  kernel lives in.
- **cudnn** — the kernel name (`softmax_fw_kernel_resident`) and counters
  reveal the design: 88 registers/thread, L1/L2 hit ≈ 0. It loads each row
  into registers once and never re-reads memory — minimal traffic — but the
  register cost caps occupancy at 31.8%, and at that occupancy it cannot hide
  memory latency. Register residency vs occupancy is a real trade, and on
  Thor at this shape it loses.
- **cub** — identical memory behavior to v2, but 18% fewer instructions and
  34% fewer stall cycles: `cub::BlockReduce`'s raking strategy does the
  cross-warp combine with fewer synchronization points than our two-stage
  shuffle + broadcast (~4–6 `__syncthreads()` per round-trip). Highest SM%
  and MEM% of all six — it keeps more of the machine busy per cycle.

Cold vs warm flips the ranking: cold (each row touched once — the regime real
inference lives in) CUB leads; warm loops favor v1's minimal instruction
count once L2 holds the data.

## Method notes (Tegra)

- Thor's ncu exposes no DRAM-throughput counter; the SOL "Memory Throughput"
  column is L2-pipe utilization. Bandwidth statements here derive from
  bytes-moved / duration, which mixes DRAM and cache traffic — where a figure
  exceeds the ~273 GB/s LPDDR5X spec, caches were absorbing part of the
  stream.
- ncu flushes caches between replay passes, so its durations are cold-cache
  and differ from the warm benchmark loop by design.
- `ncu` needs `sudo` on Tegra; pass `HOME=$HOME` so root's python finds the
  user-installed torch.

## Takeaways (v4 shipped)

1. Reduction implementation efficiency matters as much as memory-pass count
   at cache-resident row sizes — v4's raking combine + fused (m,l) tree
   closed the gap CUB exposed, and the register-resident write pass closed
   cuDNN's traffic advantage without its occupancy cost.
2. No single decomposition wins all shapes: v1 (warp-per-row) and v4
   (block-per-row) split the table along the measured `auto` crossover.
3. The v4 building blocks are the fused-attention kernel's per-tile
   machinery: the `(m, l)` merge is associative, so the same state handles
   flash-style tiling and split-KV combines. Next stop: Phase D.

## Reproduce

```bash
bash bench/rig.sh                                    # lock clocks (sudo)
python3 -m pytest tests/test_softmax.py -q           # correctness, 76 cases
python3 bench/bench_kernel.py --op softmax --dtypes fp32,fp16 --out results/softmax.csv
sudo HOME=$HOME /usr/local/cuda/bin/ncu --set full --launch-skip 1 --csv \
  python3 -c "import torch; from kernels.softmax import softmax, softmax_cudnn, softmax_cub; \
x = torch.randn(512, 2048, device='cuda'); [softmax(x, v) for v in ('v0','v1','v2','v3')]; \
softmax_cudnn(x); softmax_cub(x); torch.cuda.synchronize()" > results/softmax_all6_ncu.csv
python3 bench/roofline.py results/softmax_all6_ncu.csv
```
