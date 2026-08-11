// v4s: v4 with swizzled dense smem instead of padded rows.
//
// v4 spaces its shared-memory rows at 72 elements (144 B = 16 B x 9, odd)
// so the 8 rows an ldmatrix touches start on 8 distinct bank arcs; the 8
// pad elements per row are dead bytes. v4s stores rows dense (64 elements
// = one full 128 B bank row) and gets the same conflict-freedom from a
// Swizzle<3,3,3> layout: the low three row bits XOR into the 16 B-chunk
// bits, so a column of chunks spreads across all 32 banks. Writers
// (cp.async staging) and readers (ldmatrix partitioning) address through
// the same composed layout object, so the scramble cancels.
//
// Everything else — fragments, masking, online softmax, the exp(S)->P
// relabeling, barriers, epilogue — is v4 unchanged; outputs should be
// bit-identical (same atoms, same accumulation order). Shared memory
// drops 41.6 KB -> 36.9 KB per block.
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <cfloat>

#include <cute/tensor.hpp>

namespace vlak_v4s {

using namespace cute;

constexpr int HD = 64;             // head_dim
constexpr int V4_THREADS = 64;     // 2 warps
constexpr int BM4 = 32;            // query rows per block (16 per warp)
constexpr int BN4 = 64;            // keys per K/V tile

__device__ __forceinline__ void cpa16v4(void* dst, const void* src) {
  __pipeline_memcpy_async(dst, src, 16);
}

// exp(S) -> P relabeling, identical to v4
template <typename Layout>
__device__ __forceinline__ auto acc_layout_as_Aregs(Layout acc) {
  using X = Underscore;
  auto l = logical_divide(acc, Shape<X, X, _2>{});
  return make_layout(make_layout(get<0>(l), get<2, 0>(l)), get<1>(l),
                     get<2, 1>(l));
}

__global__ void __launch_bounds__(V4_THREADS, 2)
fused_attention_v4s_gqa(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v, __nv_bfloat16* __restrict__ o,
    long qB, long qM, long qH, long kB, long kN, long kH,
    long vB, long vN, long vH,
    int H, int G, int tiles_m, int M, int N, float scale,
    int P, int ds, int de) {
  extern __shared__ unsigned char smem_raw[];
  bfloat16_t* Qs = (bfloat16_t*)smem_raw;                        // BM4 x HD
  bfloat16_t* Kt[2] = {Qs + BM4 * HD, Qs + BM4 * HD + BN4 * HD};
  bfloat16_t* Vt[2] = {Kt[1] + BN4 * HD, Kt[1] + 2 * BN4 * HD};

  int bh = blockIdx.x / tiles_m;
  int m0 = (blockIdx.x % tiles_m) * BM4;
  int b = bh / H, h = bh % H, kvh = h / G;
  int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
  int wm0 = m0 + warp * 16;                       // this warp's first row

  const __nv_bfloat16* qp = q + (size_t)b * qB + (size_t)h * qH;
  const __nv_bfloat16* kp = k + (size_t)b * kB + (size_t)kvh * kH;
  const __nv_bfloat16* vp = v + (size_t)b * vB + (size_t)kvh * vH;

  // ---- the swizzled layouts: rows dense (64 elem = 128 B), the three row
  // bits XORed into the chunk bits. ONE formula each, shared by the
  // cp.async writers below and the ldmatrix partitioning later. The XOR
  // touches only bits >= the 16 B chunk, so chunks stay contiguous and
  // 16 B-aligned for cp.async.
  auto swzQ = composition(Swizzle<3, 3, 3>{},
      Layout<Shape<Int<BM4>, Int<HD>>, Stride<Int<HD>, _1>>{});   // (row, d)
  auto swzKV = composition(Swizzle<3, 3, 3>{},
      Layout<Shape<Int<BN4>, Int<HD>>, Stride<Int<HD>, _1>>{});   // (key, d)
  auto swzVt = composition(Swizzle<3, 3, 3>{},
      Layout<Shape<Int<HD>, Int<BN4>>, Stride<_1, Int<HD>>>{});   // (d, key) view

  // ---- gmem -> smem staging (cp.async), zero-padding beyond M / N -------
  for (int ch = tid; ch < BM4 * 8; ch += V4_THREADS) {
    int r = ch / 8, seg = ch % 8;
    if (m0 + r < M)
      cpa16v4(Qs + swzQ(r, seg * 8), qp + (size_t)(m0 + r) * qM + seg * 8);
    else {
      float4 z = {0.f, 0.f, 0.f, 0.f};
      *(float4*)(Qs + swzQ(r, seg * 8)) = z;
    }
  }
  __pipeline_commit();

  auto prefetch = [&](int n0, int buf) {
    for (int ch = tid; ch < BN4 * 8; ch += V4_THREADS) {
      int r = ch / 8, seg = ch % 8;
      if (n0 + r < N) {
        cpa16v4(Kt[buf] + swzKV(r, seg * 8), kp + (size_t)(n0 + r) * kN + seg * 8);
        cpa16v4(Vt[buf] + swzKV(r, seg * 8), vp + (size_t)(n0 + r) * vN + seg * 8);
      } else {
        float4 z = {0.f, 0.f, 0.f, 0.f};
        *(float4*)(Kt[buf] + swzKV(r, seg * 8)) = z;
        *(float4*)(Vt[buf] + swzKV(r, seg * 8)) = z;
      }
    }
    __pipeline_commit();
  };
  prefetch(0, 0);

  // ---- CuTe machinery (identical atoms to v4) ---------------------------
  TiledMMA tiled_mma = make_tiled_mma(SM80_16x8x16_F32BF16BF16F32_TN{},
                                      Layout<Shape<_1, _1, _1>>{},
                                      Tile<_16, _16, _16>{});
  auto thr_mma = tiled_mma.get_thread_slice(lane);

  // warp's Q slice: local_tile of the WHOLE swizzled tensor (a raw pointer
  // offset would break the XOR pattern; local_tile carries it along)
  Tensor sQ = make_tensor(make_smem_ptr(Qs), swzQ);
  Tensor sQw = local_tile(sQ, Shape<_16, Int<HD>>{}, make_coord(warp, 0));

  auto smem_copy_QK = make_tiled_copy_A(
      Copy_Atom<SM75_U32x4_LDSM_N, bfloat16_t>{}, tiled_mma);
  auto thr_copy_Q = smem_copy_QK.get_thread_slice(lane);
  Tensor tSrQ = thr_mma.partition_fragment_A(sQw);
  {
    __pipeline_wait_prior(0);
    __syncthreads();
    Tensor tSsQ = thr_copy_Q.partition_S(sQw);
    Tensor tSrQv = thr_copy_Q.retile_D(tSrQ);
    copy(smem_copy_QK, tSsQ, tSrQv);
  }

  Tensor tOrO = partition_fragment_C(tiled_mma, Shape<_16, Int<HD>>{});
  clear(tOrO);
  float m_run[2] = {-FLT_MAX, -FLT_MAX}, l_run[2] = {0.f, 0.f};

  Tensor cS = make_identity_tensor(Shape<_16, Int<BN4>>{});
  Tensor tScS = thr_mma.partition_C(cS);
  Tensor cO = make_identity_tensor(Shape<_16, Int<HD>>{});
  Tensor tOcO = thr_mma.partition_C(cO);

  auto smem_copy_V = make_tiled_copy_B(
      Copy_Atom<SM75_U16x8_LDSM_T, bfloat16_t>{}, tiled_mma);
  auto thr_copy_V = smem_copy_V.get_thread_slice(lane);
  auto thr_copy_K = make_tiled_copy_B(
      Copy_Atom<SM75_U32x4_LDSM_N, bfloat16_t>{}, tiled_mma)
      .get_thread_slice(lane);

  int n_tiles = (N + BN4 - 1) / BN4;
  int startNM = N - M;
  int rows_lo = m0, rows_hi = min(m0 + BM4, M) - 1;
  bool nodead = de <= ds;

  for (int t = 0; t < n_tiles; ++t) {
    int n0 = t * BN4, cur = t & 1;
    if (t + 1 < n_tiles) { prefetch(n0 + BN4, cur ^ 1); __pipeline_wait_prior(1); }
    else                 { __pipeline_wait_prior(0); }

    int last = n0 + BN4 - 1;
    bool tail = n0 + BN4 > N;
    bool full = !tail && ((last < P && (nodead || last < ds || n0 >= de)) ||
                          (n0 >= P && last <= startNM + rows_lo));
    bool empty = (!nodead && n0 >= ds && last < de && last < P) ||
                 (n0 >= P && n0 > startNM + rows_hi);
    if (empty) { __syncthreads(); continue; }
    __syncthreads();

    // ---- S = Q K^T on registers -----------------------------------------
    Tensor sK = make_tensor(make_smem_ptr(Kt[cur]), swzKV);
    Tensor tSrK = thr_mma.partition_fragment_B(sK);
    {
      Tensor tSsK = thr_copy_K.partition_S(sK);
      Tensor tSrKv = thr_copy_K.retile_D(tSrK);
      copy(smem_copy_QK, tSsK, tSrKv);
    }
    Tensor tSrS = partition_fragment_C(tiled_mma, Shape<_16, Int<BN4>>{});
    clear(tSrS);
    gemm(tiled_mma, tSrS, tSrQ, tSrK, tSrS);

    // ---- mask + online softmax, entirely in registers --------------------
    float tmax[2] = {-FLT_MAX, -FLT_MAX};
    CUTE_UNROLL
    for (int i = 0; i < size(tSrS); ++i) {
      int lr = get<0>(tScS(i));
      int gr = wm0 + lr;
      int gc = n0 + get<1>(tScS(i));
      bool vis;
      if (full) vis = (gr < M);
      else vis = (gr < M) && (gc < N) &&
                 (gc < P ? (gc < ds || gc >= de) : (gc <= startNM + gr));
      float val = vis ? tSrS(i) * scale : -FLT_MAX;
      tSrS(i) = val;
      tmax[lr >= 8] = fmaxf(tmax[lr >= 8], val);
    }
    CUTE_UNROLL
    for (int off = 2; off > 0; off >>= 1) {
      tmax[0] = fmaxf(tmax[0], __shfl_xor_sync(0xffffffff, tmax[0], off));
      tmax[1] = fmaxf(tmax[1], __shfl_xor_sync(0xffffffff, tmax[1], off));
    }
    float m_new[2], alpha[2], lsum[2] = {0.f, 0.f};
    CUTE_UNROLL
    for (int rr = 0; rr < 2; ++rr) {
      m_new[rr] = fmaxf(m_run[rr], tmax[rr]);
      alpha[rr] = (m_new[rr] > -FLT_MAX) ? expf(m_run[rr] - m_new[rr]) : 1.f;
    }
    Tensor tSrP = make_tensor<bfloat16_t>(tSrS.layout());
    CUTE_UNROLL
    for (int i = 0; i < size(tSrS); ++i) {
      int rr = get<0>(tScS(i)) >= 8;
      float p = (m_new[rr] > -FLT_MAX) ? expf(tSrS(i) - m_new[rr]) : 0.f;
      lsum[rr] += p;
      tSrP(i) = bfloat16_t(p);
    }
    CUTE_UNROLL
    for (int off = 2; off > 0; off >>= 1) {
      lsum[0] += __shfl_xor_sync(0xffffffff, lsum[0], off);
      lsum[1] += __shfl_xor_sync(0xffffffff, lsum[1], off);
    }
    CUTE_UNROLL
    for (int rr = 0; rr < 2; ++rr) {
      l_run[rr] = l_run[rr] * alpha[rr] + lsum[rr];
      m_run[rr] = m_new[rr];
    }

    // ---- O = O*alpha + P V ----------------------------------------------
    CUTE_UNROLL
    for (int i = 0; i < size(tOrO); ++i)
      tOrO(i) *= alpha[get<0>(tOcO(i)) >= 8];

    Tensor sVt = make_tensor(make_smem_ptr(Vt[cur]), swzVt);
    Tensor tOrVt = thr_mma.partition_fragment_B(sVt);
    {
      Tensor tOsVt = thr_copy_V.partition_S(sVt);
      Tensor tOrVv = thr_copy_V.retile_D(tOrVt);
      copy(smem_copy_V, tOsVt, tOrVv);
    }
    Tensor tOrP = make_tensor(tSrP.data(), acc_layout_as_Aregs(tSrP.layout()));
    gemm(tiled_mma, tOrO, tOrP, tOrVt, tOrO);

    __syncthreads();
  }

  // ---- write out (B, M, H, 64) contiguous: O / l ------------------------
  CUTE_UNROLL
  for (int i = 0; i < size(tOrO); ++i) {
    int lr = get<0>(tOcO(i));
    int gr = wm0 + lr;
    int rr = lr >= 8;
    if (gr < M && l_run[rr] > 0.f) {
      __nv_bfloat16* orow = o + (((size_t)b * M + gr) * H + h) * HD;
      orow[get<1>(tOcO(i))] = __float2bfloat16(tOrO(i) / l_run[rr]);
    }
  }
}

constexpr size_t v4s_smem() {
  return (size_t)(BM4 + 4 * BN4) * HD * sizeof(__nv_bfloat16);
}

}  // namespace vlak_v4s

torch::Tensor fused_attention_gqa_v4s(torch::Tensor q, torch::Tensor k,
                                      torch::Tensor v, double scale,
                                      int64_t prefix_len, int64_t dead_start,
                                      int64_t dead_end) {
  TORCH_CHECK(q.is_cuda() && q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
              "expected 4D CUDA tensors (B, seq, heads, 64)");
  TORCH_CHECK(q.scalar_type() == torch::kBFloat16 &&
              k.scalar_type() == torch::kBFloat16 &&
              v.scalar_type() == torch::kBFloat16, "v4s is bf16-only");
  constexpr int HD = 64;
  TORCH_CHECK(q.size(3) == HD && k.size(3) == HD && v.size(3) == HD,
              "head_dim must be 64");
  TORCH_CHECK(q.stride(3) == 1 && k.stride(3) == 1 && v.stride(3) == 1,
              "head_dim must be the contiguous dim");
  int B = q.size(0), M = q.size(1), H = q.size(2);
  int N = k.size(1), Hkv = k.size(2);
  TORCH_CHECK(k.size(0) == B && v.size(0) == B && v.size(1) == N &&
              v.size(2) == Hkv && k.sizes() == v.sizes(), "k/v shape mismatch");
  TORCH_CHECK(H % Hkv == 0, "query heads must be a multiple of kv heads");
  for (auto& t : {q, k, v})
    TORCH_CHECK((t.stride(0) % 8 == 0) && (t.stride(1) % 8 == 0) &&
                    (t.stride(2) % 8 == 0) &&
                    (reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0),
                "strides/base must keep 16B alignment for cp.async");

  int P = prefix_len >= 0 ? (int)prefix_len : N;
  TORCH_CHECK(P <= N, "prefix_len exceeds kv length");
  TORCH_CHECK(0 <= dead_start && dead_start <= dead_end && dead_end <= P,
              "dead band must satisfy 0 <= ds <= de <= prefix_len");

  auto o = torch::empty({B, M, H, HD}, q.options());
  float sc = scale > 0 ? (float)scale : 1.f / sqrtf((float)HD);
  int G = H / Hkv;
  int tiles_m = (M + vlak_v4s::BM4 - 1) / vlak_v4s::BM4;

  static bool smem_ok4s = [] {
    return cudaFuncSetAttribute(vlak_v4s::fused_attention_v4s_gqa,
             cudaFuncAttributeMaxDynamicSharedMemorySize,
             vlak_v4s::v4s_smem()) == cudaSuccess;
  }();
  TORCH_CHECK(smem_ok4s, "failed to reserve dynamic smem for v4s");

  vlak_v4s::fused_attention_v4s_gqa<<<B * H * tiles_m, vlak_v4s::V4_THREADS,
      vlak_v4s::v4s_smem(), c10::cuda::getCurrentCUDAStream()>>>(
      (const __nv_bfloat16*)q.data_ptr(), (const __nv_bfloat16*)k.data_ptr(),
      (const __nv_bfloat16*)v.data_ptr(), (__nv_bfloat16*)o.data_ptr(),
      q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1),
      k.stride(2), v.stride(0), v.stride(1), v.stride(2),
      H, G, tiles_m, M, N, sc, P, (int)dead_start, (int)dead_end);
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "fused_attention_gqa_v4s launch failed: ",
              cudaGetErrorString(err));
  return o;
}

TORCH_LIBRARY_FRAGMENT(vlak, m) {
  m.def("fused_attention_gqa_v4s(Tensor q, Tensor k, Tensor v, float scale=-1., "
        "int prefix_len=-1, int dead_start=0, int dead_end=0) -> Tensor");
}
TORCH_LIBRARY_IMPL(vlak, CUDA, m) {
  m.impl("fused_attention_gqa_v4s", &fused_attention_gqa_v4s);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_attention_gqa_v4s", &fused_attention_gqa_v4s,
        "v4 with swizzled dense smem (no padded rows), analytic mask",
        pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"),
        pybind11::arg("scale") = -1.0, pybind11::arg("prefix_len") = -1,
        pybind11::arg("dead_start") = 0, pybind11::arg("dead_end") = 0);
}
