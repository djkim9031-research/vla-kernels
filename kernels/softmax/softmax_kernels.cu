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

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
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
        } else {  // version 3 (online) — default best
          int threads = 256;
          size_t shmem = n_warps_for(threads) * sizeof(float);
          softmax_v3<scalar_t><<<rows, threads, shmem>>>(xp, yp, rows, cols);
        }
      });
  cudaError_t launch_err = cudaGetLastError();
  TORCH_CHECK(launch_err == cudaSuccess,
              "softmax kernel launch failed: ", cudaGetErrorString(launch_err));
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("softmax_lastdim", &softmax_lastdim,
        "row softmax (last dim), version-selectable",
        pybind11::arg("x"), pybind11::arg("version") = 3);
}
