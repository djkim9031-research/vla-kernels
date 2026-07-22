// Library baselines for the softmax comparison:
//   - cudnn : cudnnSoftmaxForward (ACCURATE, per-row via MODE_INSTANCE) — the
//             vendor softmax. cuBLAS has no softmax; it becomes the baseline
//             at the GEMM kernel instead.
//   - cub   : our v2 block-per-row structure, but with cub::BlockReduce doing
//             the reductions — library primitives vs the hand-rolled ones in
//             cuda_utils.cuh.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cub/block/block_reduce.cuh>
#include <cfloat>

#define CUDNN_CHECK(expr)                                                  \
  do {                                                                     \
    cudnnStatus_t _s = (expr);                                            \
    TORCH_CHECK(_s == CUDNN_STATUS_SUCCESS, "cuDNN error: ",              \
                cudnnGetErrorString(_s));                                  \
  } while (0)

// ---- cuDNN softmax ---------------------------------------------------------
torch::Tensor softmax_cudnn(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2, "expected 2D CUDA tensor");
  x = x.contiguous();
  auto y = torch::empty_like(x);
  int rows = x.size(0), cols = x.size(1);

  static cudnnHandle_t handle = nullptr;
  if (!handle) CUDNN_CHECK(cudnnCreate(&handle));
  CUDNN_CHECK(cudnnSetStream(handle, at::cuda::getCurrentCUDAStream()));

  // cache the descriptor for the last shape/dtype (bench reuses one shape)
  static cudnnTensorDescriptor_t desc = nullptr;
  static int c_rows = -1, c_cols = -1;
  static cudnnDataType_t c_dt = CUDNN_DATA_FLOAT;
  cudnnDataType_t dt = x.scalar_type() == torch::kFloat16 ? CUDNN_DATA_HALF
                                                          : CUDNN_DATA_FLOAT;
  if (!desc) CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc));
  if (rows != c_rows || cols != c_cols || dt != c_dt) {
    // N=rows, C=cols, H=W=1; MODE_INSTANCE reduces over C*H*W per N => per-row
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, dt,
                                           rows, cols, 1, 1));
    c_rows = rows; c_cols = cols; c_dt = dt;
  }

  const float alpha = 1.f, beta = 0.f;
  CUDNN_CHECK(cudnnSoftmaxForward(handle, CUDNN_SOFTMAX_ACCURATE,
                                  CUDNN_SOFTMAX_MODE_INSTANCE, &alpha, desc,
                                  x.data_ptr(), &beta, desc, y.data_ptr()));
  return y;
}

// ---- CUB-primitive softmax (block per row, safe 3-pass) --------------------
namespace {

struct MaxOp {  // cub::Max was removed in CCCL 3.x
  __device__ __forceinline__ float operator()(float a, float b) const {
    return fmaxf(a, b);
  }
};

template <typename scalar_t, int BLOCK>
__global__ void softmax_cub_kernel(const scalar_t* __restrict__ x,
                                   scalar_t* __restrict__ y, int rows,
                                   int cols) {
  using BlockReduce = cub::BlockReduce<float, BLOCK>;
  __shared__ typename BlockReduce::TempStorage tmp;
  __shared__ float bcast;

  int row = blockIdx.x;
  if (row >= rows) return;
  const scalar_t* xr = x + (size_t)row * cols;
  scalar_t* yr = y + (size_t)row * cols;

  float m = -FLT_MAX;
  for (int c = threadIdx.x; c < cols; c += BLOCK) m = fmaxf(m, (float)xr[c]);
  m = BlockReduce(tmp).Reduce(m, MaxOp{});      // valid in thread 0
  if (threadIdx.x == 0) bcast = m;
  __syncthreads();
  m = bcast;
  __syncthreads();                               // tmp reuse barrier

  float s = 0.f;
  for (int c = threadIdx.x; c < cols; c += BLOCK) s += expf((float)xr[c] - m);
  s = BlockReduce(tmp).Sum(s);
  if (threadIdx.x == 0) bcast = s;
  __syncthreads();
  float inv = 1.f / bcast;

  for (int c = threadIdx.x; c < cols; c += BLOCK)
    yr[c] = (scalar_t)(expf((float)xr[c] - m) * inv);
}

}  // namespace

torch::Tensor softmax_cub(torch::Tensor x) {
  TORCH_CHECK(x.is_cuda() && x.dim() == 2, "expected 2D CUDA tensor");
  x = x.contiguous();
  auto y = torch::empty_like(x);
  int rows = x.size(0), cols = x.size(1);
  constexpr int BLOCK = 256;

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "softmax_cub", [&] {
    softmax_cub_kernel<scalar_t, BLOCK><<<rows, BLOCK>>>(
        x.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), rows, cols);
  });
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "softmax_cub launch failed: ",
              cudaGetErrorString(err));
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("softmax_cudnn", &softmax_cudnn, "cuDNN softmax (per-row, accurate)");
  m.def("softmax_cub", &softmax_cub, "CUB BlockReduce softmax (block per row)");
}
