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
#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_bf16.h>
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
constexpr int V2_THREADS = 128;             // 4 warps x 16 query rows

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

__global__ void fused_attention_wmma_bf16(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v, __nv_bfloat16* __restrict__ o,
    const bool* __restrict__ mask, int H, int BH, int M, int N, float scale) {
  extern __shared__ unsigned char smem_raw[];
  __nv_bfloat16* Qs = (__nv_bfloat16*)smem_raw;                    // BM x LD
  __nv_bfloat16* Kt = Qs + BM * LD;                                // BN x LD
  __nv_bfloat16* Vt = Kt + BN * LD;                                // BN x LD
  __nv_bfloat16* Pb = Vt + BN * LD;                                // BM x LD
  float* S  = (float*)(Pb + BM * LD);                              // BM x LD
  float* Oa = S + BM * LD;                                         // BM x LD
  float* mrun = Oa + BM * LD;                                      // BM
  float* lrun = mrun + BM;                                         // BM
  float* pmax = lrun + BM;                                         // BM x 2
  float* psum = pmax + 2 * BM;                                     // BM x 2

  int tiles_m = (M + BM - 1) / BM;
  int bh = blockIdx.x / tiles_m;
  int m0 = (blockIdx.x % tiles_m) * BM;                            // first q row
  int tid = threadIdx.x, warp = tid / 32;

  const __nv_bfloat16* qb = q + (size_t)bh * M * HEAD_DIM;
  const __nv_bfloat16* kb = k + (size_t)bh * N * HEAD_DIM;
  const __nv_bfloat16* vb = v + (size_t)bh * N * HEAD_DIM;
  const bool* mb = mask ? mask + (size_t)(bh / H) * M * N : nullptr;

  // load Q tile (zero-pad rows beyond M); init stats and O accumulator
  for (int i = tid; i < BM * HEAD_DIM; i += V2_THREADS) {
    int r = i / HEAD_DIM, c = i % HEAD_DIM;
    Qs[r * LD + c] = (m0 + r < M) ? qb[(size_t)(m0 + r) * HEAD_DIM + c]
                                  : __float2bfloat16(0.f);
  }
  for (int i = tid; i < BM * HEAD_DIM; i += V2_THREADS) Oa[(i / HEAD_DIM) * LD + i % HEAD_DIM] = 0.f;
  if (tid < BM) { mrun[tid] = -FLT_MAX; lrun[tid] = 0.f; }
  __syncthreads();

  for (int n0 = 0; n0 < N; n0 += BN) {
    // stage K/V tiles (zero-pad cols beyond N)
    for (int i = tid; i < BN * HEAD_DIM; i += V2_THREADS) {
      int r = i / HEAD_DIM, c = i % HEAD_DIM;
      bool ok = n0 + r < N;
      Kt[r * LD + c] = ok ? kb[(size_t)(n0 + r) * HEAD_DIM + c] : __float2bfloat16(0.f);
      Vt[r * LD + c] = ok ? vb[(size_t)(n0 + r) * HEAD_DIM + c] : __float2bfloat16(0.f);
    }
    __syncthreads();

    // S(64x64) = Q @ K^T on tensor cores; each warp owns 16 rows
    for (int nf = 0; nf < 4; ++nf) {
      fragment<accumulator, 16, 16, 16, float> acc;
      fill_fragment(acc, 0.f);
      for (int kf = 0; kf < 4; ++kf) {
        fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> af;
        fragment<matrix_b, 16, 16, 16, __nv_bfloat16, col_major> bf;
        load_matrix_sync(af, Qs + warp * 16 * LD + kf * 16, LD);
        load_matrix_sync(bf, Kt + nf * 16 * LD + kf * 16, LD);   // K^T via col_major
        mma_sync(acc, af, bf, acc);
      }
      store_matrix_sync(S + warp * 16 * LD + nf * 16, acc, LD, mem_row_major);
    }
    __syncthreads();

    // scalar phase, parallel over all 128 threads: 2 threads per query row,
    // each owning a 32-column half — mask+max, then exp/P/O-rescale, then one
    // thread commits the merged (m,l). Three cheap barriers replace the old
    // single-thread 64+64-element serial crawl.
    {
      int r = tid / 2, half = tid % 2, c0 = half * (BN / 2);
      int grow = m0 + r;
      const bool* mr = (mb && grow < M) ? mb + (size_t)grow * N + n0 : nullptr;
      bool row_ok = grow < M;

      float tmax = -FLT_MAX;
      for (int c = c0; c < c0 + BN / 2; ++c) {
        bool valid = row_ok && (n0 + c < N) && (!mr || mr[c]);
        float val = valid ? S[r * LD + c] * scale : -FLT_MAX;
        S[r * LD + c] = val;
        tmax = fmaxf(tmax, val);
      }
      pmax[r * 2 + half] = tmax;
      __syncthreads();

      float m_new = fmaxf(mrun[r], fmaxf(pmax[r * 2], pmax[r * 2 + 1]));
      float alpha = 1.f, lsum = 0.f;
      if (row_ok && m_new > -FLT_MAX) {
        alpha = expf(mrun[r] - m_new);
        for (int c = c0; c < c0 + BN / 2; ++c) {
          float p = expf(S[r * LD + c] - m_new);     // exp(-inf)=0 for masked
          Pb[r * LD + c] = __float2bfloat16(p);
          lsum += p;
        }
        for (int d = half * (HEAD_DIM / 2); d < (half + 1) * (HEAD_DIM / 2); ++d)
          Oa[r * LD + d] *= alpha;
      } else {
        for (int c = c0; c < c0 + BN / 2; ++c) Pb[r * LD + c] = __float2bfloat16(0.f);
      }
      psum[r * 2 + half] = lsum;
      __syncthreads();

      if (half == 0 && row_ok && m_new > -FLT_MAX) {
        lrun[r] = lrun[r] * alpha + psum[r * 2] + psum[r * 2 + 1];
        mrun[r] = m_new;
      }
    }
    __syncthreads();

    // O(64x64) += P @ V on tensor cores
    for (int nf = 0; nf < 4; ++nf) {
      fragment<accumulator, 16, 16, 16, float> acc;
      load_matrix_sync(acc, Oa + warp * 16 * LD + nf * 16, LD, mem_row_major);
      for (int kf = 0; kf < 4; ++kf) {
        fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> af;
        fragment<matrix_b, 16, 16, 16, __nv_bfloat16, row_major> bf;
        load_matrix_sync(af, Pb + warp * 16 * LD + kf * 16, LD);
        load_matrix_sync(bf, Vt + kf * 16 * LD + nf * 16, LD);
        mma_sync(acc, af, bf, acc);
      }
      store_matrix_sync(Oa + warp * 16 * LD + nf * 16, acc, LD, mem_row_major);
    }
    __syncthreads();
  }

  // write out: O / l
  if (tid < BM) {
    int r = tid, grow = m0 + r;
    if (grow < M) {
      float inv = 1.f / lrun[r];
      __nv_bfloat16* orow = o + ((size_t)bh * M + grow) * HEAD_DIM;
      for (int d = 0; d < HEAD_DIM; ++d)
        orow[d] = __float2bfloat16(Oa[r * LD + d] * inv);
    }
  }
}

constexpr size_t V2_SMEM =
    (4 * BM * LD) * sizeof(__nv_bfloat16) + (2 * BM * LD + 6 * BM) * sizeof(float);

}  // namespace

torch::Tensor fused_attention(torch::Tensor q, torch::Tensor k, torch::Tensor v,
                              double scale,
                              std::optional<torch::Tensor> attn_mask) {
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
    static bool smem_ok = [] {
      return cudaFuncSetAttribute(fused_attention_wmma_bf16,
                                  cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  V2_SMEM) == cudaSuccess;
    }();
    TORCH_CHECK(smem_ok, "failed to reserve ", V2_SMEM, "B dynamic smem");
    int tiles_m = (M + BM - 1) / BM;
    fused_attention_wmma_bf16<<<BH * tiles_m, V2_THREADS, V2_SMEM>>>(
        (const __nv_bfloat16*)q.data_ptr(), (const __nv_bfloat16*)k.data_ptr(),
        (const __nv_bfloat16*)v.data_ptr(), (__nv_bfloat16*)o.data_ptr(),
        mp, H, BH, M, N, sc);
  } else {                                     // v1: scalar warp-per-row path
    int threads = 128;
    int rows_per_block = threads / WARP_SIZE;
    int blocks = (BH * M + rows_per_block - 1) / rows_per_block;
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        q.scalar_type(), "fused_attention", [&] {
          fused_attention_warp_row<scalar_t><<<blocks, threads>>>(
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
        "Tensor? attn_mask=None) -> Tensor");
}
TORCH_LIBRARY_IMPL(vlak, CUDA, m) {
  m.impl("fused_attention", &fused_attention);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_attention", &fused_attention, "fused SDPA (head_dim 64, opt. bool mask)",
        pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"),
        pybind11::arg("scale") = -1.0, pybind11::arg("attn_mask") = std::nullopt);
}
