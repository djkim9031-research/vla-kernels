# SmolVLA on Thor — variant comparison (locked clocks, single day)

All numbers measured 2026-07-23 on Jetson Thor at MAXN + `jetson_clocks`, in the
project Docker image, same checkpoint (`lerobot/smolvla_base`, bf16), same
synthetic observation set. Chunk inference = `predict_action_chunk` (50 actions).
An earlier partial run at unlocked clocks was discarded: DVFS shifts the CPU/GPU
balance, so cross-day ratios are not clock-invariant.

## The table

| variant | what it is | p50 / p99 (ms) | chunk Hz | LoadGen SS p90 | Offline /s | parity cosine | 99% gate |
|---|---|---|---|---|---|---|---|
| original | eager PyTorch | 198.4 / 236.6 | 5.0 | 210.1 ms | 4.92 | 1.0 (ref) | — |
| tuned | eager + fused-softmax patch | 204.6 / 215.8 | 4.9 | — | — | 0.999997 | PASS |
| compiled | torch.compile (Inductor) | 94.3 / 108.4 | 10.6 | 85.6 ms | — | 0.999960 | PASS |
| compiled-ro | + CUDA Graphs (reduce-overhead) | 73.8 / 82.0 | 13.5 | 76.4 ms | 13.75 | 0.999967 | PASS |
| compiled-ro + SDPA fix | + flash-pinned vision attention | 66.1 / 73.3 | 15.1 | — | — | 0.999953 | PASS |
| + eager→SDPA routing | expert/prefix attention via F.sdpa | **54.3 / 56.4** | **18.4** | — | — | 0.999952 | PASS |

**SDPA fix (2026-07-24, Phase D step 2):** profiling caught Inductor lowering
the 36 vision SDPA calls to the memory-efficient backend (11.8 ms) where eager
dispatches flash (3.1 ms). Root cause: SmolVLM manufactures an all-ones patch
mask; eager's all-ones shortcut drops it (flash eligible) but tracing must
materialize it (flash disqualified). Since the mask is all-ones by
construction for fixed cameras, `vla/variants.py` nulls it at the source
(`create_bidirectional_mask` → None) and pins flash-first backend priority.
−7.9 ms for zero kernel code; flash_fwd verified in the profile.

All LoadGen results VALID (SingleStream 60 s window; Offline 330 queries for
original, 850 for compiled-ro at its higher rate).

Compile cost: first call ~200 s cold (Dynamo + Inductor + Triton + autotune);
warm restarts via `TORCHINDUCTOR_CACHE_DIR` skip most of it. Dynamo reports
**7 graph breaks / 11 subgraphs** — the flow-matching denoising loop and queue
logic fragment the capture, so this is not a single end-to-end graph.

## Reading the deltas

- **original → compiled (2.1×):** Inductor's fusion doing exactly what the
  shape reconnaissance predicted it could — the eager fp32 attention chains
  (176 softmax sites) and the surrounding elementwise/norm ops collapse into
  generated Triton kernels; matmuls stay cuBLAS. The memory-round-trip tax and
  most per-op dispatch disappear.
- **compiled → compiled-ro (another 1.28×):** pure launch-tax removal. CUDA
  Graphs replay the 11 subgraphs' launch sequences as single submissions —
  ~20 ms of CPU-side launch/dispatch overhead per chunk, gone.
- **tuned (−3%):** re-confirmed at matched clocks — swapping one op inside the
  eager attention chain adds wrapper overhead without removing round trips.
  Op-level wins need fusion granularity to survive end-to-end.
- **Accuracy:** every variant passes the ≥0.99 retention proxy; compiled
  variants drift slightly more (0.99996 vs 0.999997) from op reordering and
  fused-math differences — far inside the gate.

## What this sets up (Phase C/D bar)

The bar for the hand-written fused-attention kernel is no longer eager's
198 ms — it is **compiled-ro's 73.8 ms**. Two consequences:

1. Phase D's kernel should be registered as a **custom op that composes with
   torch.compile** (replacing the eager attention inside the compiled graph),
   not an either/or alternative — beat Inductor's generated kernels *inside*
   the region, keep its fusion everywhere else, keep CUDA Graphs on top.
2. Remaining known headroom, in order: the vision tower's 36 bf16 SDPA calls
   (untouched by our kernels until bf16 lands), the 7 graph breaks (each a
   CPU sync + eager region), and precision (bf16 end-to-end / INT8 via
   TensorRT — Phase E).

**Custom-kernel integration (compiled-vk, 2026-07-27, Phase D close-out):**
routes the 160 expert attention calls to `vlak::fused_attention` inside the
compiled graph. Correct after the stream-capture and fake-stride fixes
(retention 0.999951, gate PASS) but slower: 63.5 ms p50 vs the same-day
compiled-ro control at 57.8 ms. In-graph profiling attributes it: the
mem-efficient kernel costs 21.6 µs/call under compilation (its 60 µs eager
figure was per-call mask preparation, which compiling hoists), ours
36.4 µs. compiled-ro stays the shipping variant; the eager-vs-in-graph
baseline gap is written up in [docs/attention.md](../docs/attention.md).
Day-to-day note: the 57.8 ms control vs the 54.3 ms below is thermal/clock
drift across sessions — variants are only compared within a session.

**v3 GQA-native kernel (compiled-vk3, 2026-07-27):** rebuilt op — GQA-native
(no expand), layout-native (no transposes), analytic probed mask (no mask
tensor), register (m,l) + register-resident softmax columns. Locked-clock
in-graph region at the expert sites: ours 21.6 us/call with zero entourage
vs sdpa's 19.9 + 2.0 prep = 21.9 — **region parity**; e2e p50 differences
(both ~54-56 ms, gate PASS 0.99994) are inside run-to-run noise. Full
design, scoreboard, and the measured dead ends (fragment skipping, warp
specialization, enable_gqa) in [docs/attention.md](../docs/attention.md).
Session hygiene reminder these runs re-taught: verify locked clocks
(devfreq cur_freq == max) AND a quiet GPU before benching — an unlocked
day masqueraded as variant regressions twice.

**fp32 timestep embedding (compiled-vk7, 2026-07-30): the new default,
~36.0 ms.** Found by ncu + kernel-time attribution on compiled-vk5: 9% of
the inference (8-10 kernels x ~340 us) was the flow loop's sinusoidal
timestep embedding running in FLOAT64 — lerobot's get_safe_dtype only
downgrades float64 on mps/xpu/cpu, so on CUDA every denoising step paid
fp64 linspace/pow/sin/cos on Thor's 1/32-rate fp64 pipe, then cast the
result to bf16 one line later. The _sinusoid_fp32 context computes the
embedding in fp32 — bit-identical after the bf16 cast (23 mantissa bits
vs bf16's 8). Gate PASS at the bf16 noise floor: cosine 0.999959,
joint_worst_rel 0.0635 (hardened per-joint gate, bar 0.12). Paired
same-session A/B: **36.03/35.98 vs compiled-vk5 40.53/39.25 ms p50**
(strict, both rounds, −3.3 ms vs the warm round); p99 36.4; ~27.8 Hz;
LoadGen SS p90 37.66 ms VALID. The fp64 kernels are absent from the vk7
profile (device time 38.5 → 35.3 ms/call). For scale: this puts the torch
pipeline AHEAD of a TensorRT 10.13 bf16 engine of the stock model
(median 36.45 ms), which fails the same per-joint gate at 0.40. Third
instance of the campaign's oldest lesson: the cheapest milliseconds come
from dtype defaults, not new kernels.

**Graph-break elimination (compiled-vk5, 2026-07-28): the new default,
and the deployment milestone.** torch._dynamo.explain traced ALL 11
breaks to one line: SmolVLM's vision embeddings assign position ids via
boolean-mask scatter (`position_ids[mask] = pos_ids[mask]` -> aten.nonzero
-> dynamic shape), once per camera. For fixed full-resolution cameras
that mask is all-ones — the third ceremonial-mask incident — so the
scatter is the identity; _vision_posids_static assigns directly (the
fractional-coordinate computation kept verbatim). Result: **ONE graph,
zero breaks, fullgraph=True enforced** so regressions fail loudly.
Paired A/B: 39.2/39.2 vs compiled-vk4c 43.3/43.2 ms p50 — strict both
rounds (−4.1 ms: the CPU gaps plus cross-boundary fusion the split
graphs forbade). p99 39.5 — single-graph replay is near-deterministic.
Retention 0.999959 gate PASS; LoadGen SS p90 39.7 ms + Offline 25.4/s
VALID. The whole-graph property also unblocks torch.export/AOTI/TRT for
Phase E. Cumulative: eager 198.4 -> ~39.2 ms = **5.1x, ~25 Hz**.

**Padding compaction (compiled-vk4c, 2026-07-28):** The
44 padded-language columns (census-proven inert: masked as keys
everywhere, positions cumsum-skip them) are dropped at embed_prefix via
a probe-captured static index — prefix 241 -> 197, shrinking the encode
GEMMs, the K/V cache, and the expert kernel's key lengths (cross 197,
self 247; our v4 now averages 13.9 us/call in-graph). Paired A/B:
42.7/43.1 vs compiled-vk4x 43.7/43.6 ms p50 — strict both rounds
(-0.7 ms; the old -3-5 ms estimate predated vk4x, which had already
removed the prefix's fp32 costs). Retention 0.999954-0.999959 gate PASS.
Static-per-task: a changed instruction re-probes at variant setup.
Best observed p50: **42.7 ms**. Cumulative: eager 198.4 -> ~43 ms =
**4.6x, ~23 Hz**.

**Stray-fp32 fix (compiled-vk4x, 2026-07-28):** A dtype
census found the checkpoint's freshly initialized modules — the 8
cross-attention projection blocks (odd expert layers) and both action-time
MLPs — loaded as fp32 while everything pretrained went bf16. Their ~456
GEMMs/inference ran as fp32 sgemm on CUDA cores (~8 ms) and forced a
per-call cast of the cross K/V cache at every expert attention site. The
fix is a cast context (vla/variants.py: _expert_bf16) — no kernels.
Paired same-session A/B: **46.0/45.4 vs compiled-vk4 55.0/54.1 ms p50**
(strict, both rounds, −8.8 ms avg); retention 0.999951 gate PASS; LoadGen
SS p90 50.9 ms + Offline 21.3/s VALID. Best observed p50: **45.0 ms**.
Cumulative: eager 198.4 -> ~45 ms = **4.4x, ~22 Hz**. Same disease as the
eager-attention upcast of Phase D, found the same way: follow the
fp32-on-CUDA-cores kernels in the profile.

**v4 register-pipeline kernel (compiled-vk4, 2026-07-27):**
CUTLASS/CuTe rebuild of the v3 op on the SM80 mma atom — S and P stay in
registers (softmax on the accumulators via lane shuffles; exp(S) -> P by
a CuTe layout relabeling; LDSM copy atoms for fragments; identity-tensor
mask coordinates; no inline PTX), 2 barriers/tile, 64-thread blocks at
2 blocks/SM. Op-level: cross 16.5 us / self 18.6 — beats the fmha kernel
itself (19.9) with zero entourage. Paired same-session A/B: 49.5/49.7 vs
compiled-ro 53.1/53.5 ms p50 (strict win, both rounds), gate PASS
0.999956, LoadGen SS p90 51.0 ms + Offline 20.1/s VALID. Cumulative:
eager 198.4 -> ~49.6 ms = 4.0x, ~20 Hz.

**Eager→SDPA routing (2026-07-24, Phase D step 3a):** lerobot's expert/prefix
attention is hand-written (fp32-upcast QK^T → where(mask) → softmax → PV, 176
sites). Op-level measurement showed F.sdpa 3–6× faster at those shapes; the
routing context (`vla/variants.py`) swaps the class's eager_attention_forward
for an sdpa call preserving mask semantics and GQA expansion. −11.8 ms — more
than the softmax/round-trip estimate, because the eager path's explicit fp32
upcast was also running the attention GEMMs on CUDA cores; sdpa runs them on
bf16 tensor cores with internal fp32 accumulation. Cumulative: eager 198.4 →
54.3 ms (3.65×) at accuracy retention 0.99995.