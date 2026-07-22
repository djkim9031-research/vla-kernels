# Kernel #1 — Fused Row Softmax

Softmax over the last dim of a `[rows, cols]` tensor — the canonical reduction
kernel, and a real hot op in transformer attention (and in VLA action/logit
heads). Four versions are kept so the optimization path is reproducible.

## Versions

| ver | strategy | key idea | passes over row |
|-----|----------|----------|-----------------|
| v0 | one **thread** per row | naive baseline, uncoalesced global reads | 3 |
| v1 | one **warp** per row | `__shfl_xor` warp reductions, coalesced stride | 3 (in regs) |
| v2 | one **block** per row | shared-mem block reduce, safe softmax | 3 |
| v3 | one **block** per row | **online softmax** (fused max+sum) | **2** |

`softmax(x, "auto")` picks v1 for many-row shapes (attention) and v3 for
few-row/wide shapes.

## Results (Jetson Thor, sm_110, CUDA 13.0, locked clocks)

Speedup vs `torch.softmax` (higher is better); `err` = max abs vs fp32 reference.

**fp16, representative shapes:**

| shape | v0 | v1 | v2 | v3 | torch (ms) |
|-------|----|----|----|----|-----------|
| 2048×512  | 0.03× | **1.33×** | 0.29× | 0.29× | 0.0165 |
| 512×2048  | 0.03× | **4.45×** | 1.16× | 2.65× | 0.0548 |
| 256×4096  | 0.02× | **4.24×** | 1.33× | 1.38× | 0.0610 |

**fp32, wide rows (online-softmax payoff):**

| shape | v2 (safe) | v3 (online) |
|-------|-----------|-------------|
| 512×2048 | 138.8 GB/s | **404.6 GB/s** |

(Full sweep: `results/softmax.csv`.)

## Findings

1. **Coalescing dominates.** v0's one-thread-per-row layout is ~30–700× slower
   than warp/block layouts — neighbouring threads must touch neighbouring
   addresses. This is the single biggest lever.
2. **Warp-per-row (v1) wins the attention regime.** With many rows there's
   enough warp-level parallelism to fill the GPU, and v1 avoids `__syncthreads`
   entirely — beating cuDNN/`torch.softmax` by up to **4.5×** in fp16 here.
3. **Online softmax (v3) pays off when rows are few and wide.** Folding the max
   and sum into one streaming pass cuts one full read of the row (3→2 passes);
   v3 nearly **3×** v2 at 512×2048 fp32. This is the same trick flash-attention
   uses — the bridge to kernel #3.
4. **fp32 accumulation keeps parity.** Even fp16 inputs reduce in fp32, so max
   abs error stays ≤ 3e-3 vs the reference.

## Reproduce

```bash
python3 -m pytest tests/test_softmax.py -q                       # correctness
python3 bench/bench_kernel.py --op softmax --dtypes fp32,fp16 --out results/softmax.csv
sudo ncu --set full --csv -o results/softmax_ncu \
    python3 bench/bench_kernel.py --op softmax --once            # roofline capture
python3 bench/roofline.py results/softmax_ncu.csv
```

## Next

- Vectorized loads (`float4`/`__half2`) in v2/v3 for the wide-row path.
- Fuse into attention (kernels/attention/) — replace SDPA, reuse the online-sum.
