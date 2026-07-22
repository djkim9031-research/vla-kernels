// Shared CUDA helpers: warp/block reductions, vectorized load types, error checks.
// Hand-written for the vla-kernels project. No third-party kernel code.
#pragma once

#include <cuda_runtime.h>
#include <cfloat>

#define FULL_MASK 0xffffffffu
#define WARP_SIZE 32

// ---- Warp-level reductions (butterfly via shuffle) -------------------------
template <typename T>
__device__ __forceinline__ T warpReduceMax(T val) {
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    T other = __shfl_xor_sync(FULL_MASK, val, offset);
    val = other > val ? other : val;
  }
  return val;
}

template <typename T>
__device__ __forceinline__ T warpReduceSum(T val) {
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    val += __shfl_xor_sync(FULL_MASK, val, offset);
  }
  return val;
}

// ---- Block-level reductions (warp reduce -> shared -> warp reduce) ----------
// `shared` must hold at least (blockDim.x / WARP_SIZE) elements.
template <typename T>
__device__ __forceinline__ T blockReduceMax(T val, T* shared) {
  int lane = threadIdx.x % WARP_SIZE;
  int wid = threadIdx.x / WARP_SIZE;
  val = warpReduceMax(val);
  if (lane == 0) shared[wid] = val;
  __syncthreads();
  int n_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
  val = (threadIdx.x < n_warps) ? shared[lane] : -FLT_MAX;
  if (wid == 0) val = warpReduceMax(val);
  return val;  // valid in thread 0; callers broadcast via shared if needed
}

template <typename T>
__device__ __forceinline__ T blockReduceSum(T val, T* shared) {
  int lane = threadIdx.x % WARP_SIZE;
  int wid = threadIdx.x / WARP_SIZE;
  val = warpReduceSum(val);
  if (lane == 0) shared[wid] = val;
  __syncthreads();
  int n_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
  val = (threadIdx.x < n_warps) ? shared[lane] : T(0);
  if (wid == 0) val = warpReduceSum(val);
  return val;
}

// ---- Host-side error check -------------------------------------------------
#define CUDA_CHECK(expr)                                                      \
  do {                                                                        \
    cudaError_t _err = (expr);                                               \
    if (_err != cudaSuccess) {                                                \
      throw std::runtime_error(std::string("CUDA error: ") +                  \
                               cudaGetErrorString(_err) + " at " + __FILE__ + \
                               ":" + std::to_string(__LINE__));               \
    }                                                                         \
  } while (0)
