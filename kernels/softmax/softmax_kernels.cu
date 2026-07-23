// Fused row-softmax kernels (softmax over the last dim of a [rows, cols] tensor).
// Progressive versions, all kept so the optimization story is reproducible:
//   v0  naive   : one thread per row, 3 global passes, no shared mem
//   v1  warp    : one warp per row, shuffle reductions
//   v2  block   : one block per row, shared-mem block reduce, safe softmax (3 reads)
//   v3  online  : one block per row, single-pass max+sum (online softmax, 2 reads)
//
// Math is always accumulated in fp32 (even for fp16 input) for numerical parity
// with torch.softmax.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cfloat>
#include "../common/cuda_utils.cuh"

namespace {

// ---- v0: naive, one thread per row -----------------------------------------
template <typename scalar_t>
__global__ void softmax_v0(const scalar_t* __restrict__ x,
                           scalar_t* __restrict__ y, int rows, int cols) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  const scalar_t* xr = x + (size_t)row * cols;
  scalar_t* yr = y + (size_t)row * cols;

  float m = -FLT_MAX;
  for (int c = 0; c < cols; ++c) m = fmaxf(m, (float)xr[c]);
  float s = 0.f;
  for (int c = 0; c < cols; ++c) s += expf((float)xr[c] - m);
  float inv = 1.f / s;
  for (int c = 0; c < cols; ++c) yr[c] = (scalar_t)((expf((float)xr[c] - m)) * inv);
}

// ---- v1: one warp per row --------------------------------------------------
template <typename scalar_t>
__global__ void softmax_v1(const scalar_t* __restrict__ x,
                           scalar_t* __restrict__ y, int rows, int cols) {
  int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
  int lane = threadIdx.x % WARP_SIZE;
  if (warp_id >= rows) return;
  const scalar_t* xr = x + (size_t)warp_id * cols;
  scalar_t* yr = y + (size_t)warp_id * cols;

  float m = -FLT_MAX;
  for (int c = lane; c < cols; c += WARP_SIZE) m = fmaxf(m, (float)xr[c]);
  m = warpReduceMax(m);

  float s = 0.f;
  for (int c = lane; c < cols; c += WARP_SIZE) s += expf((float)xr[c] - m);
  s = warpReduceSum(s);
  float inv = 1.f / s;

  for (int c = lane; c < cols; c += WARP_SIZE)
    yr[c] = (scalar_t)(expf((float)xr[c] - m) * inv);
}

// ---- v2: one block per row, safe softmax (3 read passes) -------------------
template <typename scalar_t>
__global__ void softmax_v2(const scalar_t* __restrict__ x,
                           scalar_t* __restrict__ y, int rows, int cols) {
  int row = blockIdx.x;
  if (row >= rows) return;
  const scalar_t* xr = x + (size_t)row * cols;
  scalar_t* yr = y + (size_t)row * cols;

  extern __shared__ float smem[];          // n_warps slots
  __shared__ float bcast;                  // broadcast slot

  float m = -FLT_MAX;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) m = fmaxf(m, (float)xr[c]);
  m = blockReduceMax(m, smem);
  if (threadIdx.x == 0) bcast = m;
  __syncthreads();
  m = bcast;
  __syncthreads();

  float s = 0.f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) s += expf((float)xr[c] - m);
  s = blockReduceSum(s, smem);
  if (threadIdx.x == 0) bcast = s;
  __syncthreads();
  float inv = 1.f / bcast;

  for (int c = threadIdx.x; c < cols; c += blockDim.x)
    yr[c] = (scalar_t)(expf((float)xr[c] - m) * inv);
}

// ---- (m,l) fused online-softmax reduction helpers (used by v4) -------------
// Merge two online-softmax states: the associative operator behind
// flash-attention's split-KV combine.
__device__ __forceinline__ void mlMerge(float& m, float& l, float mo, float lo) {
  float m_new = fmaxf(m, mo);
  l = l * expf(m - m_new) + lo * expf(mo - m_new);
  m = m_new;
}

// ---- v3: one block per row, online softmax (2 read passes) -----------------
// Each thread streams its strided elements once, keeping a running (max, sum).
// Threads' partial stats are combined, then a single write pass normalizes.
template <typename scalar_t>
__global__ void softmax_v3(const scalar_t* __restrict__ x,
                           scalar_t* __restrict__ y, int rows, int cols) {
  int row = blockIdx.x;
  if (row >= rows) return;
  const scalar_t* xr = x + (size_t)row * cols;
  scalar_t* yr = y + (size_t)row * cols;

  extern __shared__ float smem[];
  __shared__ float m_bcast, l_bcast;

  // pass 1: streaming local (max, sum-of-exp relative to local max)
  float m_local = -FLT_MAX;
  float l_local = 0.f;
  for (int c = threadIdx.x; c < cols; c += blockDim.x) {
    float v = (float)xr[c];
    float m_new = fmaxf(m_local, v);
    l_local = l_local * expf(m_local - m_new) + expf(v - m_new);
    m_local = m_new;
  }

  // combine local maxima across the block
  float m_block = blockReduceMax(m_local, smem);
  if (threadIdx.x == 0) m_bcast = m_block;
  __syncthreads();
  m_block = m_bcast;
  __syncthreads();

  // rescale each thread's partial sum to the block max, then reduce
  l_local *= expf(m_local - m_block);
  float l_block = blockReduceSum(l_local, smem);
  if (threadIdx.x == 0) l_bcast = l_block;
  __syncthreads();
  float inv = 1.f / l_bcast;

  // pass 2: write normalized output
  for (int c = threadIdx.x; c < cols; c += blockDim.x)
    yr[c] = (scalar_t)(expf((float)xr[c] - m_block) * inv);
}

// ---- v4: block per row — fused (m,l) tree + raking combine + registers -----
// Three upgrades over v2/v3, each from the profiling evidence:
//   * ONE fused (m,l) reduction tree instead of separate max and sum trees
//   * raking cross-warp combine: warp partials -> smem -> warp 0 folds them;
//     2 barriers total (v2/v3 pay ~4-6 via the generic blockReduce+broadcast)
//   * register-resident rows (cols <= BLOCK*RPT): elements loaded once into
//     registers, write pass never re-reads global memory — one read total,
//     at a register budget (~RPT floats/thread) that keeps occupancy high
template <typename scalar_t, int BLOCK, int RPT>
__global__ void softmax_v4_reg(const scalar_t* __restrict__ x,
                               scalar_t* __restrict__ y, int rows, int cols) {
  int row = blockIdx.x;
  if (row >= rows) return;
  const scalar_t* xr = x + (size_t)row * cols;
  scalar_t* yr = y + (size_t)row * cols;

  // Elements are register-resident, so the classic two-phase form applies:
  // independent max tree, then independent exp sum — no serial dependency
  // chain of exponentials (the online update is only needed when values
  // cannot be held, i.e. in the streaming fallback below).
  float v[RPT];
  float m = -FLT_MAX, l = 0.f;
#pragma unroll
  for (int r = 0; r < RPT; ++r) {
    int c = threadIdx.x + r * BLOCK;
    v[r] = c < cols ? (float)xr[c] : -FLT_MAX;
    m = fmaxf(m, v[r]);
  }
#pragma unroll
  for (int r = 0; r < RPT; ++r) {
    int c = threadIdx.x + r * BLOCK;
    if (c < cols) l += expf(v[r] - m);
  }
  // fused (m,l) intra-warp tree
#pragma unroll
  for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
    mlMerge(m, l, __shfl_xor_sync(FULL_MASK, m, off),
            __shfl_xor_sync(FULL_MASK, l, off));

  // raking cross-warp combine
  constexpr int NW = BLOCK / WARP_SIZE;
  __shared__ float sm[NW], sl[NW];
  int wid = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
  if (lane == 0) { sm[wid] = m; sl[wid] = l; }
  __syncthreads();
  if (wid == 0) {
    m = lane < NW ? sm[lane] : -FLT_MAX;
    l = lane < NW ? sl[lane] : 0.f;
#pragma unroll
    for (int off = NW / 2; off > 0; off >>= 1)
      mlMerge(m, l, __shfl_xor_sync(FULL_MASK, m, off),
              __shfl_xor_sync(FULL_MASK, l, off));
    if (lane == 0) { sm[0] = m; sl[0] = l; }
  }
  __syncthreads();
  m = sm[0];
  float inv = 1.f / sl[0];

  // write pass straight from registers — no second global read
#pragma unroll
  for (int r = 0; r < RPT; ++r) {
    int c = threadIdx.x + r * BLOCK;
    if (c < cols) yr[c] = (scalar_t)(expf(v[r] - m) * inv);
  }
}

// streaming fallback for rows wider than the register budget: same fused
// tree + raking combine, but elements re-read for the write pass (v3-style)
template <typename scalar_t, int BLOCK>
__global__ void softmax_v4_stream(const scalar_t* __restrict__ x,
                                  scalar_t* __restrict__ y, int rows, int cols) {
  int row = blockIdx.x;
  if (row >= rows) return;
  const scalar_t* xr = x + (size_t)row * cols;
  scalar_t* yr = y + (size_t)row * cols;

  float m = -FLT_MAX, l = 0.f;
  for (int c = threadIdx.x; c < cols; c += BLOCK) {
    float val = (float)xr[c];
    float m_new = fmaxf(m, val);
    l = l * expf(m - m_new) + expf(val - m_new);
    m = m_new;
  }
#pragma unroll
  for (int off = WARP_SIZE / 2; off > 0; off >>= 1)
    mlMerge(m, l, __shfl_xor_sync(FULL_MASK, m, off),
            __shfl_xor_sync(FULL_MASK, l, off));

  constexpr int NW = BLOCK / WARP_SIZE;
  __shared__ float sm[NW], sl[NW];
  int wid = threadIdx.x / WARP_SIZE, lane = threadIdx.x % WARP_SIZE;
  if (lane == 0) { sm[wid] = m; sl[wid] = l; }
  __syncthreads();
  if (wid == 0) {
    m = lane < NW ? sm[lane] : -FLT_MAX;
    l = lane < NW ? sl[lane] : 0.f;
#pragma unroll
    for (int off = NW / 2; off > 0; off >>= 1)
      mlMerge(m, l, __shfl_xor_sync(FULL_MASK, m, off),
              __shfl_xor_sync(FULL_MASK, l, off));
    if (lane == 0) { sm[0] = m; sl[0] = l; }
  }
  __syncthreads();
  m = sm[0];
  float inv = 1.f / sl[0];
  for (int c = threadIdx.x; c < cols; c += BLOCK)
    yr[c] = (scalar_t)(expf((float)xr[c] - m) * inv);
}

inline int n_warps_for(int threads) { return (threads + WARP_SIZE - 1) / WARP_SIZE; }

}  // namespace

// Dispatch: softmax over the last dim of a contiguous 2D tensor.
torch::Tensor softmax_lastdim(torch::Tensor x, int64_t version) {
  TORCH_CHECK(x.is_cuda(), "input must be CUDA");
  TORCH_CHECK(x.dim() == 2, "expected a 2D [rows, cols] tensor");
  x = x.contiguous();
  auto y = torch::empty_like(x);
  int rows = x.size(0);
  int cols = x.size(1);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      x.scalar_type(), "softmax_lastdim", [&] {
        const scalar_t* xp = x.data_ptr<scalar_t>();
        scalar_t* yp = y.data_ptr<scalar_t>();
        if (version == 0) {
          int threads = 128;
          int blocks = (rows + threads - 1) / threads;
          softmax_v0<scalar_t><<<blocks, threads>>>(xp, yp, rows, cols);
        } else if (version == 1) {
          int threads = 128;                       // 4 warps -> 4 rows / block
          int rows_per_block = threads / WARP_SIZE;
          int blocks = (rows + rows_per_block - 1) / rows_per_block;
          softmax_v1<scalar_t><<<blocks, threads>>>(xp, yp, rows, cols);
        } else if (version == 2) {
          int threads = 256;
          size_t shmem = n_warps_for(threads) * sizeof(float);
          softmax_v2<scalar_t><<<rows, threads, shmem>>>(xp, yp, rows, cols);
        } else if (version == 3) {
          int threads = 256;
          size_t shmem = n_warps_for(threads) * sizeof(float);
          softmax_v3<scalar_t><<<rows, threads, shmem>>>(xp, yp, rows, cols);
        } else {  // version 4 — fused (m,l) + raking; registers when row fits
          constexpr int BLOCK = 256, RPT = 8;
          if (cols <= BLOCK * RPT)
            softmax_v4_reg<scalar_t, BLOCK, RPT><<<rows, BLOCK>>>(xp, yp, rows, cols);
          else
            softmax_v4_stream<scalar_t, BLOCK><<<rows, BLOCK>>>(xp, yp, rows, cols);
        }
      });
  cudaError_t launch_err = cudaGetLastError();
  TORCH_CHECK(launch_err == cudaSuccess,
              "softmax kernel launch failed: ", cudaGetErrorString(launch_err));
  return y;
}

// Real operator registration (the custom-op pattern Phase D reuses for the
// fused attention): dispatcher-visible, so torch.compile / torch.export / the
// C++ AOTI runtime can all resolve vlak::softmax by name. The meta ("fake")
// implementation lives in Python via torch.library.register_fake.
TORCH_LIBRARY_FRAGMENT(vlak, m) {
  m.def("softmax(Tensor x, int version=4) -> Tensor");
}
TORCH_LIBRARY_IMPL(vlak, CUDA, m) {
  m.impl("softmax", &softmax_lastdim);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("softmax_lastdim", &softmax_lastdim,
        "row softmax (last dim), version-selectable",
        pybind11::arg("x"), pybind11::arg("version") = 4);
}
