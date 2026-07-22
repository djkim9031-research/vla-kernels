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
| cudnn | `cudnnSoftmaxForward` | vendor baseline (register-resident design) |
| cub | v2 structure on `cub::BlockReduce` | library raking reduction vs our hand-rolled one |

`softmax(x, "auto")` routes by shape: rows ≥ 32 → v1, few wide rows → v3.
cuBLAS has no softmax op; it becomes the baseline at the GEMM kernel instead.

## Warm benchmark (locked clocks, 100-iter loop)

Selected rows; full sweep in `results/softmax.csv` (5 shapes × fp32/fp16 × 6
variants). Times in ms.

| shape · dtype | torch | v1 | v3 | cudnn | cub |
|---|---|---|---|---|---|
| 2048×512 fp32 | 0.0145 | **0.0124** | 0.0328 | 0.0123 | 0.0473 |
| 512×2048 fp32 | 0.0204 | **0.0144** | 0.0205 | 0.0290 | 0.0164 |
| 512×2048 fp16 | 0.0308 | **0.0124** | 0.0210 | 0.0226 | 0.0164 |
| 256×4096 fp16 | 0.0376 | 0.0143 | 0.0206 | 0.0196 | **0.0142** |
| 64×16384 fp32 | 0.0267 | 0.0308 | 0.0333 | 0.0226 | **0.0164** |
| 64×16384 fp16 | 0.0280 | 0.0247 | 0.0297 | 0.0248 | **0.0144** |

The winner is shape-dependent:

- **Many rows (attention regime):** v1 matches cuDNN at 2048×512 and is
  1.8–2.0× faster at 512×2048; up to 2.6× over `torch.softmax` (fp16).
- **Few wide rows:** the CUB-reduction variant leads (1.6–1.9× over torch at
  64×16384); row-level parallelism runs out and block-per-row with an
  efficient reduction wins.
- **Tiny rows (1024×128):** cuDNN edges everyone; launch overhead dominates.

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

## Takeaways → v4

1. Reduction implementation efficiency matters as much as memory-pass count
   at cache-resident row sizes. Adopt the raking pattern (or
   `cub::BlockReduce` directly) in the block-per-row path.
2. Register residency is the minimal-traffic ideal but must be budgeted to
   keep ≥ 32 warps/SM — steal it in moderation for the attention kernel.
3. Planned v4: raking reduce + online max/sum + moderate register blocking;
   expected to lead the cold-cache table across shapes. The online `(m, l)`
   merge in v3 is associative, so the same state generalizes to multi-block
   rows and split-KV attention.

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
