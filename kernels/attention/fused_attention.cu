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
#include <cfloat>
#include "../common/cuda_utils.cuh"

namespace {

constexpr int HEAD_DIM = 64;          // dims per lane = HEAD_DIM / 32 = 2

template <typename scalar_t>
__global__ void fused_attention_warp_row(
    const scalar_t* __restrict__ q,   // (BH, M, D) contiguous
    const scalar_t* __restrict__ k,   // (BH, N, D)
    const scalar_t* __restrict__ v,   // (BH, N, D)
    scalar_t* __restrict__ o,         // (BH, M, D)
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

  // this lane's two dims of the query row, output accumulator, (m,l) state
  float q0 = (float)qr[lane];
  float q1 = (float)qr[lane + WARP_SIZE];
  float o0 = 0.f, o1 = 0.f;
  float m_run = -FLT_MAX, l_run = 0.f;

  for (int j = 0; j < N; ++j) {
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

}  // namespace

torch::Tensor fused_attention(torch::Tensor q, torch::Tensor k,
                              torch::Tensor v, double scale) {
  TORCH_CHECK(q.is_cuda() && q.dim() == 4, "expected 4D CUDA (B,H,M,D)");
  TORCH_CHECK(q.size(3) == HEAD_DIM, "this kernel is specialized to head_dim 64");
  TORCH_CHECK(k.size(3) == HEAD_DIM && v.size(3) == HEAD_DIM, "k/v head_dim 64");
  TORCH_CHECK(k.size(2) == v.size(2), "k/v length mismatch");
  q = q.contiguous(); k = k.contiguous(); v = v.contiguous();

  int B = q.size(0), H = q.size(1), M = q.size(2);
  int N = k.size(2);
  int BH = B * H;
  auto o = torch::empty_like(q);

  float sc = scale > 0 ? (float)scale : 1.f / sqrtf((float)HEAD_DIM);
  int threads = 128;                             // 4 warps -> 4 query rows/block
  int rows_per_block = threads / WARP_SIZE;
  int blocks = (BH * M + rows_per_block - 1) / rows_per_block;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      q.scalar_type(), "fused_attention", [&] {
        fused_attention_warp_row<scalar_t><<<blocks, threads>>>(
            q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(),
            v.data_ptr<scalar_t>(), o.data_ptr<scalar_t>(), BH, M, N, sc);
      });
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "fused_attention launch failed: ",
              cudaGetErrorString(err));
  return o;
}

TORCH_LIBRARY_FRAGMENT(vlak, m) {
  m.def("fused_attention(Tensor q, Tensor k, Tensor v, float scale=-1.) -> Tensor");
}
TORCH_LIBRARY_IMPL(vlak, CUDA, m) {
  m.impl("fused_attention", &fused_attention);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fused_attention", &fused_attention, "fused SDPA (non-causal, head_dim 64)",
        pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"),
        pybind11::arg("scale") = -1.0);
}
