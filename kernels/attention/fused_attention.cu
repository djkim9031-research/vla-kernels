// Fused scaled-dot-product attention (non-causal), first version.
//
// Replaces the eager 3-op chain  QK^T -> softmax -> PV  with one kernel:
// the score matrix never touches global memory. Structure: one WARP per
// query row (the softmax-v1 decomposition, grown a GEMM on each side), with
// the online (m, l) state and output accumulator held in registers — the
// flash-attention recurrence built on the same associative merge as
// softmax v4.
//
// Targets the measured SmolVLA sites: head_dim 64, q_len 50/241,
// kv_len 241/291, 15 heads. This version favors clarity over peak: K/V are
// re-read per query row (L2 absorbs the reuse across warps of the same
// head); shared-memory K/V tiling is the planned next iteration.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <cfloat>
#include "../common/cuda_utils.cuh"

namespace {

constexpr int HEAD_DIM = 64;          // dims per lane = HEAD_DIM / 32 = 2

// mask layout: (B, M, N) bool, broadcast over heads; nullptr = no mask
template <typename scalar_t>
__global__ void fused_attention_warp_row(
    const scalar_t* __restrict__ q,   // (BH, M, D) contiguous
    const scalar_t* __restrict__ k,   // (BH, N, D)
    const scalar_t* __restrict__ v,   // (BH, N, D)
    scalar_t* __restrict__ o,         // (BH, M, D)
    const bool* __restrict__ mask, int H,
    int BH, int M, int N, float scale) {
  int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  int total_rows = BH * M;
  if (warp_id >= total_rows) return;

  int bh = warp_id / M;
  int m_row = warp_id % M;

  const scalar_t* qr = q + ((size_t)bh * M + m_row) * HEAD_DIM;
  const scalar_t* kb = k + (size_t)bh * N * HEAD_DIM;
  const scalar_t* vb = v + (size_t)bh * N * HEAD_DIM;
  const bool* mrow = mask ? mask + ((size_t)(bh / H) * M + m_row) * N : nullptr;

  // this lane's two dims of the query row, output accumulator, (m,l) state
  float q0 = (float)qr[lane];
  float q1 = (float)qr[lane + WARP_SIZE];
  float o0 = 0.f, o1 = 0.f;
  float m_run = -FLT_MAX, l_run = 0.f;

  for (int j = 0; j < N; ++j) {
    if (mrow && !mrow[j]) continue;   // masked key: contributes nothing
    const scalar_t* kj = kb + (size_t)j * HEAD_DIM;
    // dot(q, k_j): each lane covers its two dims, warp-sum the partials
    float part = q0 * (float)kj[lane] + q1 * (float)kj[lane + WARP_SIZE];
#pragma unroll
    for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
      part += __shfl_xor_sync(FULL_MASK, part, off);
    float s = part * scale;                    // score, identical in all lanes

    // online (m,l) update + rescale of the running output (flash recurrence)
    float m_new = fmaxf(m_run, s);
    float alpha = expf(m_run - m_new);         // rescales previous state
    float p = expf(s - m_new);                 // this key's weight (unnormalized)
    l_run = l_run * alpha + p;
    const scalar_t* vj = vb + (size_t)j * HEAD_DIM;
    o0 = o0 * alpha + p * (float)vj[lane];
    o1 = o1 * alpha + p * (float)vj[lane + WARP_SIZE];
    m_run = m_new;
  }

  float inv = 1.f / l_run;
  scalar_t* orow = o + ((size_t)bh * M + m_row) * HEAD_DIM;
  orow[lane] = (scalar_t)(o0 * inv);
  orow[lane + WARP_SIZE] = (scalar_t)(o1 * inv);
}

// ---- v2: tensor-core tiles (bf16, WMMA), smem-mediated online softmax ------
// Flash-2 structure at BLOCK_M x BLOCK_N = 64x64: Q tile resident in smem,
// K/V tiles stream through smem, S = QK^T and O += PV on WMMA bf16 fragments
// (fp32 accumulate), the online (m,l) + mask logic runs scalar on the S tile
// in smem between the two matmuls. The score tile never touches HBM; the
// bool mask is read once per tile and fused into the stats pass.
constexpr int BM = 64, BN = 64, LD = 80;    // LD: smem leading dim (16-elem aligned)
constexpr int V2_THREADS = 256;             // 8 warps: 4 run wmma, all 8 load + softmax

// 16-byte async copy into smem (cp.async on sm80+): the enabler for
// double-buffered K/V — the next tile streams in while this one computes
__device__ __forceinline__ void cpa16(void* dst, const void* src) {
  __pipeline_memcpy_async(dst, src, 16);
}

using nvcuda::wmma::fragment;
using nvcuda::wmma::matrix_a;
using nvcuda::wmma::matrix_b;
using nvcuda::wmma::accumulator;
using nvcuda::wmma::row_major;
using nvcuda::wmma::col_major;
using nvcuda::wmma::load_matrix_sync;
using nvcuda::wmma::store_matrix_sync;
using nvcuda::wmma::fill_fragment;
using nvcuda::wmma::mma_sync;
using nvcuda::wmma::mem_row_major;

template <int BMT>
__global__ void fused_attention_wmma_bf16(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v, __nv_bfloat16* __restrict__ o,
    const bool* __restrict__ mask, int prefix_len, int H,
    int BH, int M, int N, float scale) {
  extern __shared__ unsigned char smem_raw[];
  constexpr int TPR = V2_THREADS / BMT;      // scalar threads per query row
  __nv_bfloat16* Qs = (__nv_bfloat16*)smem_raw;                    // BMT x LD
  __nv_bfloat16* Kt[2] = {Qs + BMT * LD, Qs + BMT * LD + BN * LD}; // double-buffered
  __nv_bfloat16* Vt[2] = {Kt[1] + BN * LD, Kt[1] + 2 * BN * LD};
  __nv_bfloat16* Pb = Vt[1] + BN * LD;                             // BMT x LD
  float* S  = (float*)(Pb + BMT * LD);                             // BMT x LD
  float* Oa = S + BMT * LD;                                        // BMT x LD
  float* mrun = Oa + BMT * LD;                                     // BMT
  float* lrun = mrun + BMT;                                        // BMT
  float* pmax = lrun + BMT;                                        // BMT x TPR
  float* psum = pmax + TPR * BMT;                                  // BMT x TPR

  int tiles_m = (M + BMT - 1) / BMT;
  int bh = blockIdx.x / tiles_m;
  int m0 = (blockIdx.x % tiles_m) * BMT;                            // first q row
  int tid = threadIdx.x, warp = tid / 32;

  const __nv_bfloat16* qb = q + (size_t)bh * M * HEAD_DIM;
  const __nv_bfloat16* kb = k + (size_t)bh * N * HEAD_DIM;
  const __nv_bfloat16* vb = v + (size_t)bh * N * HEAD_DIM;
  const bool* mb = mask ? mask + (size_t)(bh / H) * M * N : nullptr;

  // load Q tile (zero-pad rows beyond M); init stats and O accumulator
  for (int i = tid; i < BMT * HEAD_DIM; i += V2_THREADS) {
    int r = i / HEAD_DIM, c = i % HEAD_DIM;
    Qs[r * LD + c] = (m0 + r < M) ? qb[(size_t)(m0 + r) * HEAD_DIM + c]
                                  : __float2bfloat16(0.f);
  }
  for (int i = tid; i < BMT * HEAD_DIM; i += V2_THREADS) Oa[(i / HEAD_DIM) * LD + i % HEAD_DIM] = 0.f;
  if (tid < BMT) { mrun[tid] = -FLT_MAX; lrun[tid] = 0.f; }

  // async prefetch of one K/V tile into buffer `buf` (8 x 16B chunks per row;
  // rows past N are zero-filled synchronously)
  auto prefetch = [&](int n0, int buf) {
    for (int ch = tid; ch < BN * 8; ch += V2_THREADS) {
      int r = ch / 8, seg = ch % 8;
      if (n0 + r < N) {
        cpa16(Kt[buf] + r * LD + seg * 8, kb + (size_t)(n0 + r) * HEAD_DIM + seg * 8);
        cpa16(Vt[buf] + r * LD + seg * 8, vb + (size_t)(n0 + r) * HEAD_DIM + seg * 8);
      } else {
        float4 z = {0.f, 0.f, 0.f, 0.f};
        *(float4*)(Kt[buf] + r * LD + seg * 8) = z;
        *(float4*)(Vt[buf] + r * LD + seg * 8) = z;
      }
    }
    __pipeline_commit();
  };

  prefetch(0, 0);
  int n_tiles = (N + BN - 1) / BN;
  // analytic-mask staircase base: visible(col; row) = col < prefix_len
  //                                || col <= startNM + row   (suffix causal)
  int startNM = N - M;
  int rows_lo = m0, rows_hi = min(m0 + BMT, M) - 1;

  for (int t = 0; t < n_tiles; ++t) {
    int n0 = t * BN, cur = t & 1;
    if (t + 1 < n_tiles) {
      prefetch(n0 + BN, cur ^ 1);       // next tile streams while this computes
      __pipeline_wait_prior(1);         // current tile's copies done
    } else {
      __pipeline_wait_prior(0);
    }

    // classify this KV tile against the (analytic) mask: 2 = fully visible
    // (branchless fast path), 1 = partial (per-element check), 0 = invisible
    // (skip all compute; uniform across the block so barriers stay balanced)
    int tstate;
    bool tail = n0 + BN > N;            // tile contains zero-padded cols >= N:
                                        // NEVER eligible for the no-check path
    if (prefix_len >= 0) {
      int last = n0 + BN - 1;
      bool full = !tail && ((last < prefix_len) || (last <= startNM + rows_lo));
      bool empty = (n0 >= prefix_len) && (n0 > startNM + rows_hi);
      tstate = full ? 2 : (empty ? 0 : 1);
    } else {
      tstate = (mb || tail) ? 1 : 2;    // per-element checks unless clean+unmasked
    }
    if (tstate == 0) { __syncthreads(); continue; }
    __syncthreads();

    // S(BMT x 64) = Q @ K^T: (row-frag x col-frag) jobs spread over all 8 warps
    for (int job = warp; job < (BMT / 16) * 4; job += V2_THREADS / 32) {
      int rf = job / 4, nf = job % 4;
      fragment<accumulator, 16, 16, 16, float> acc;
      fill_fragment(acc, 0.f);
      for (int kf = 0; kf < 4; ++kf) {
        fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> af;
        fragment<matrix_b, 16, 16, 16, __nv_bfloat16, col_major> bf;
        load_matrix_sync(af, Qs + rf * 16 * LD + kf * 16, LD);
        load_matrix_sync(bf, Kt[cur] + nf * 16 * LD + kf * 16, LD);  // K^T via col_major
        mma_sync(acc, af, bf, acc);
      }
      store_matrix_sync(S + rf * 16 * LD + nf * 16, acc, LD, mem_row_major);
    }
    __syncthreads();

    // scalar phase over all 256 threads: TPR threads per query row — mask+max,
    // then exp/P/O-rescale, then one thread commits the merged (m,l).
    {
      int r = tid / TPR, quart = tid % TPR, c0 = quart * (BN / TPR);
      int grow = m0 + r;
      const bool* mr = (mb && grow < M) ? mb + (size_t)grow * N + n0 : nullptr;
      bool row_ok = grow < M;

      float tmax = -FLT_MAX;
      if (tstate == 2) {                 // fully visible: no per-element checks
        for (int c = c0; c < c0 + BN / TPR; ++c) {
          float val = row_ok ? S[r * LD + c] * scale : -FLT_MAX;
          S[r * LD + c] = val;
          tmax = fmaxf(tmax, val);
        }
      } else if (prefix_len >= 0) {      // partial, analytic: mask is arithmetic
        for (int c = c0; c < c0 + BN / TPR; ++c) {
          int gc = n0 + c;
          bool valid = row_ok && gc < N &&
                       (gc < prefix_len || gc <= startNM + grow);
          float val = valid ? S[r * LD + c] * scale : -FLT_MAX;
          S[r * LD + c] = val;
          tmax = fmaxf(tmax, val);
        }
      } else {                           // partial, tensor mask
        for (int c = c0; c < c0 + BN / TPR; ++c) {
          bool valid = row_ok && (n0 + c < N) && (!mr || mr[c]);
          float val = valid ? S[r * LD + c] * scale : -FLT_MAX;
          S[r * LD + c] = val;
          tmax = fmaxf(tmax, val);
        }
      }
      pmax[r * TPR + quart] = tmax;
      __syncthreads();

      float m_new = mrun[r];
      for (int i = 0; i < TPR; ++i) m_new = fmaxf(m_new, pmax[r * TPR + i]);
      float alpha = 1.f, lsum = 0.f;
      if (row_ok && m_new > -FLT_MAX) {
        alpha = expf(mrun[r] - m_new);
        for (int c = c0; c < c0 + BN / TPR; ++c) {
          float p = expf(S[r * LD + c] - m_new);     // exp(-inf)=0 for masked
          Pb[r * LD + c] = __float2bfloat16(p);
          lsum += p;
        }
        for (int d = quart * (HEAD_DIM / TPR); d < (quart + 1) * (HEAD_DIM / TPR); ++d)
          Oa[r * LD + d] *= alpha;
      } else {
        for (int c = c0; c < c0 + BN / TPR; ++c) Pb[r * LD + c] = __float2bfloat16(0.f);
      }
      psum[r * TPR + quart] = lsum;
      __syncthreads();

      if (quart == 0 && row_ok && m_new > -FLT_MAX) {
        float ls = 0.f;
        for (int i = 0; i < TPR; ++i) ls += psum[r * TPR + i];
        lrun[r] = lrun[r] * alpha + ls;
        mrun[r] = m_new;
      }
    }
    __syncthreads();

    // O(BMT x 64) += P @ V: jobs spread over all warps
    for (int job = warp; job < (BMT / 16) * 4; job += V2_THREADS / 32) {
      int rf = job / 4, nf = job % 4;
      fragment<accumulator, 16, 16, 16, float> acc;
      load_matrix_sync(acc, Oa + rf * 16 * LD + nf * 16, LD, mem_row_major);
      for (int kf = 0; kf < 4; ++kf) {
        fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> af;
        fragment<matrix_b, 16, 16, 16, __nv_bfloat16, row_major> bf;
        load_matrix_sync(af, Pb + rf * 16 * LD + kf * 16, LD);
        load_matrix_sync(bf, Vt[cur] + kf * 16 * LD + nf * 16, LD);
        mma_sync(acc, af, bf, acc);
      }
      store_matrix_sync(Oa + rf * 16 * LD + nf * 16, acc, LD, mem_row_major);
    }
    __syncthreads();
  }

  // write out: O / l
  if (tid < BMT) {
    int r = tid, grow = m0 + r;
    if (grow < M) {
      float inv = 1.f / lrun[r];
      __nv_bfloat16* orow = o + ((size_t)bh * M + grow) * HEAD_DIM;
      for (int d = 0; d < HEAD_DIM; ++d)
        orow[d] = __float2bfloat16(Oa[r * LD + d] * inv);
    }
  }
}

constexpr size_t v2_smem(int bmt) {
  return (size_t)(2 * bmt + 4 * BN) * LD * sizeof(__nv_bfloat16) +
         ((size_t)2 * bmt * LD + 2 * (V2_THREADS / 32) * bmt + 2 * bmt) * sizeof(float);
}

}  // namespace

torch::Tensor fused_attention(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                              double scale,
                              std::optional<torch::Tensor> attn_mask,
                              int64_t prefix_len) {
  TORCH_CHECK(q.is_cuda() && q.dim() == 4, "expected 4D CUDA (B,H,M,D)");
  TORCH_CHECK(q.size(3) == HEAD_DIM, "this kernel is specialized to head_dim 64");
  TORCH_CHECK(k.size(3) == HEAD_DIM && v.size(3) == HEAD_DIM, "k/v head_dim 64");
  TORCH_CHECK(k.size(2) == v.size(2), "k/v length mismatch");
  q = q.contiguous(); k = k.contiguous(); v = v.contiguous();

  int B = q.size(0), H = q.size(1), M = q.size(2);
  int N = k.size(2);
  int BH = B * H;
  auto o = torch::empty_like(q);

  const bool* mp = nullptr;
  torch::Tensor mask;
  if (attn_mask.has_value()) {
    mask = attn_mask->contiguous();
    TORCH_CHECK(mask.scalar_type() == torch::kBool, "attn_mask must be bool");
    TORCH_CHECK(mask.dim() == 3 && mask.size(0) == B && mask.size(1) == M &&
                mask.size(2) == N, "attn_mask must be (B, M, N)");
    mp = mask.data_ptr<bool>();
  }

  float sc = scale > 0 ? (float)scale : 1.f / sqrtf((float)HEAD_DIM);

  if (q.scalar_type() == torch::kBFloat16) {   // v2: tensor-core path
    // small-M sites get BMT=32: doubles the grid (fills SMs) + halves Q/S/O smem
    bool small = M <= 96;
    static bool smem_ok = [] {
      bool a = cudaFuncSetAttribute(fused_attention_wmma_bf16<32>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, v2_smem(32)) == cudaSuccess;
      bool b = cudaFuncSetAttribute(fused_attention_wmma_bf16<64>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, v2_smem(64)) == cudaSuccess;
      return a && b;
    }();
    TORCH_CHECK(smem_ok, "failed to reserve dynamic smem for v2");
    if (small) {
      int tiles_m = (M + 31) / 32;
      fused_attention_wmma_bf16<32><<<BH * tiles_m, V2_THREADS, v2_smem(32),
          c10::cuda::getCurrentCUDAStream()>>>(
          (const __nv_bfloat16*)q.data_ptr(), (const __nv_bfloat16*)k.data_ptr(),
          (const __nv_bfloat16*)v.data_ptr(), (__nv_bfloat16*)o.data_ptr(),
          mp, (int)prefix_len, H, BH, M, N, sc);
    } else {
      int tiles_m = (M + 63) / 64;
      fused_attention_wmma_bf16<64><<<BH * tiles_m, V2_THREADS, v2_smem(64),
          c10::cuda::getCurrentCUDAStream()>>>(
          (const __nv_bfloat16*)q.data_ptr(), (const __nv_bfloat16*)k.data_ptr(),
          (const __nv_bfloat16*)v.data_ptr(), (__nv_bfloat16*)o.data_ptr(),
          mp, (int)prefix_len, H, BH, M, N, sc);
    }
  } else {                                     // v1: scalar warp-per-row path
    TORCH_CHECK(prefix_len < 0,
                "analytic prefix mask requires the bf16 tensor-core path");
    int threads = 128;
    int rows_per_block = threads / WARP_SIZE;
    int blocks = (BH * M + rows_per_block - 1) / rows_per_block;
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        q.scalar_type(), "fused_attention", [&] {
          fused_attention_warp_row<scalar_t><<<blocks, threads, 0,
              c10::cuda::getCurrentCUDAStream()>>>(
              q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(),
              v.data_ptr<scalar_t>(), o.data_ptr<scalar_t>(),
              mp, H, BH, M, N, sc);
        });
  }
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "fused_attention launch failed: ",
              cudaGetErrorString(err));
  return o;
}

TORCH_LIBRARY_FRAGMENT(vlak, m) {
  m.def("fused_attention(Tensor q, Tensor k, Tensor v, float scale=-1., "
        "Tensor? attn_mask=None, int prefix_len=-1) -> Tensor");
}
TORCH_LIBRARY_IMPL(vlak, CUDA, m) {
  m.impl("fused_attention", &fused_attention);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_attention", &fused_attention,
        "fused SDPA (head_dim 64; bool mask or analytic prefix+staircase mask)",
        pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"),
        pybind11::arg("scale") = -1.0, pybind11::arg("attn_mask") = std::nullopt,
        pybind11::arg("prefix_len") = -1);
}
