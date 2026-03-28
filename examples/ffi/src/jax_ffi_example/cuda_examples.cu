/* Copyright 2024 The JAX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

//----------------------------------------------------------------------------//
//                            Forward pass                                    //
//----------------------------------------------------------------------------//

// c = a * (b+1)
// This strawman operation works well for demo purposes because:
// 1. it's simple enough to be quickly understood,
// 2. it's complex enough to require intermediate outputs in grad computation,
//    like many operations in practice do, and
// 3. it does not have a built-in implementation in JAX.
__global__ void FooFwdKernel(const float *a, const float *b, float *c,
                             float *b_plus_1,  // intermediate output b+1
                             size_t n) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t grid_stride = blockDim.x * gridDim.x;
  for (size_t i = tid; i < n; i += grid_stride) {
    b_plus_1[i] = b[i] + 1.0f;
    c[i] = a[i] * b_plus_1[i];
  }
}

// Host function wrapper that launches the kernel with hardcoded grid/block
// size. Note, it uses types from XLA FFI. The return type must be ffi::Error.
// Buffer type provides buffer dimensions, so the "n" argument here is not
// strictly necessary, but it allows us to demonstrate the use of attributes
// (.Attr in the FFI handler definition above).
ffi::Error FooFwdHost(cudaStream_t stream, ffi::Buffer<ffi::F32> a,
                      ffi::Buffer<ffi::F32> b, ffi::ResultBuffer<ffi::F32> c,
                      ffi::ResultBuffer<ffi::F32> b_plus_1, size_t n) {
  const int block_dim = 128;
  const int grid_dim = 1;
  // Note how we access regular Buffer data vs Result Buffer data:
  FooFwdKernel<<<grid_dim, block_dim, /*shared_mem=*/0, stream>>>(
      a.typed_data(), b.typed_data(), c->typed_data(), b_plus_1->typed_data(),
      n);
  // Check for launch time errors. Note that this function may also
  // return error codes from previous, asynchronous launches. This
  // means that an error status returned here could have been caused
  // by a different kernel previously launched by XLA.
  cudaError_t last_error = cudaGetLastError();
  if (last_error != cudaSuccess) {
    return ffi::Error::Internal(
        std::string("CUDA error: ") + cudaGetErrorString(last_error));
  }
  return ffi::Error::Success();
}

// Creates symbol FooFwd with C linkage that can be loaded using Python ctypes
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    FooFwd, FooFwdHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()  // stream
        .Arg<ffi::Buffer<ffi::F32>>()              // a
        .Arg<ffi::Buffer<ffi::F32>>()              // b
        .Ret<ffi::Buffer<ffi::F32>>()              // c
        .Ret<ffi::Buffer<ffi::F32>>()              // b_plus_1
        .Attr<int64_t>("n"),
    {xla::ffi::Traits::kCmdBufferCompatible});  // cudaGraph enabled

//----------------------------------------------------------------------------//
//                            Backward pass                                   //
//----------------------------------------------------------------------------//

// compute da = dc * (b+1), and
//         db = dc * a
__global__ void FooBwdKernel(const float *c_grad,    // incoming gradient wrt c
                             const float *a,         // original input a
                             const float *b_plus_1,  // intermediate output b+1
                             float *a_grad,          // outgoing gradient wrt a
                             float *b_grad,          // outgoing gradient wrt b
                             size_t n) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t grid_stride = blockDim.x * gridDim.x;
  for (size_t i = tid; i < n; i += grid_stride) {
    // In practice on GPUs b_plus_1 can be recomputed for practically free
    // instead of storing it out and reusing, so the reuse here is a bit
    // contrived. We do it to demonstrate residual/intermediate output passing
    // between the forward and the backward pass which becomes useful when
    // recomputation is more expensive than reuse.
    a_grad[i] = c_grad[i] * b_plus_1[i];
    b_grad[i] = c_grad[i] * a[i];
  }
}

ffi::Error FooBwdHost(cudaStream_t stream,
                      ffi::Buffer<ffi::F32> c_grad,
                      ffi::Buffer<ffi::F32> a,
                      ffi::ResultBuffer<ffi::F32> b_plus_1,
                      ffi::ResultBuffer<ffi::F32> a_grad,
                      ffi::ResultBuffer<ffi::F32> b_grad,
                      size_t n) {
  const int block_dim = 128;
  const int grid_dim = 1;
  FooBwdKernel<<<grid_dim, block_dim, /*shared_mem=*/0, stream>>>(
      c_grad.typed_data(), a.typed_data(), b_plus_1->typed_data(),
      a_grad->typed_data(), b_grad->typed_data(), n);
  cudaError_t last_error = cudaGetLastError();
  if (last_error != cudaSuccess) {
    return ffi::Error::Internal(
        std::string("CUDA error: ") + cudaGetErrorString(last_error));
  }
  return ffi::Error::Success();
}

// Creates symbol FooBwd with C linkage that can be loaded using Python ctypes
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    FooBwd, FooBwdHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()  // stream
        .Arg<ffi::Buffer<ffi::F32>>()    // c_grad
        .Arg<ffi::Buffer<ffi::F32>>()    // a
        .Arg<ffi::Buffer<ffi::F32>>()    // b_plus_1
        .Ret<ffi::Buffer<ffi::F32>>()    // a_grad
        .Ret<ffi::Buffer<ffi::F32>>()    // b_grad
        .Attr<int64_t>("n"),
    {xla::ffi::Traits::kCmdBufferCompatible});  // cudaGraph enabled


//                            GELU activation                                //
//----------------------------------------------------------------------------//

// GELU kernel: y = 0.5 * x * (1 + erf(x / sqrt(2)))
__global__ void GeluKernel(const float *x, float *y, size_t n) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t grid_stride = blockDim.x * gridDim.x;
  for (size_t i = tid; i < n; i += grid_stride) {
    float value = x[i];
    y[i] = 0.5f * value * (1.0f + erf(value / sqrtf(2.0f)));
  }
}

// Host function wrapper for GELU
ffi::Error GeluHost(cudaStream_t stream, ffi::Buffer<ffi::F32> self, ffi::ResultBuffer<ffi::F32> out) {
  size_t n = self.element_count();
  const int block_dim = 128;
  const int grid_dim = (n + block_dim - 1) / block_dim;
  
  GeluKernel<<<grid_dim, block_dim, /*shared_mem=*/0, stream>>>(
      self.typed_data(), out->typed_data(), n);
  
  cudaError_t last_error = cudaGetLastError();
  if (last_error != cudaSuccess) {
    return ffi::Error::Internal(
        std::string("CUDA error: ") + cudaGetErrorString(last_error));
  }
  return ffi::Error::Success();
}

// Creates symbol AscendGelu with C linkage that can be loaded using Python ctypes
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    MyGelu, GeluHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()  // stream
        .Arg<ffi::Buffer<ffi::F32>>()              // self
        .Ret<ffi::Buffer<ffi::F32>>(),             // out
    {xla::ffi::Traits::kCmdBufferCompatible});  // cudaGraph enabled

//----------------------------------------------------------------------------//

//----------------------------------------------------------------------------//
//                         Matrix Multiplication                             //
//----------------------------------------------------------------------------//

// Simple matrix multiplication kernel: C = A * B
// A: [m, k], B: [k, n], C: [m, n]
__global__ void MatMulKernel(const float *A, const float *B, float *C, 
                             int64_t m, int64_t n, int64_t k) {
  size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (row < m && col < n) {
    float sum = 0.0f;
    for (size_t i = 0; i < k; ++i) {
      sum += A[row * k + i] * B[i * n + col];
    }
    C[row * n + col] = sum;
  }
}

// Host function wrapper for matrix multiplication
ffi::Error MatMulHost(cudaStream_t stream, 
                      ffi::Buffer<ffi::F32> A,
                      ffi::Buffer<ffi::F32> B,
                      ffi::ResultBuffer<ffi::F32> C) {
  // Extract dimensions from buffers
  // A has shape [m, k], B has shape [k, n]
  const auto& a_dims = A.dimensions();
  const auto& b_dims = B.dimensions();
  
  int64_t m = a_dims[0];  // rows of A
  int64_t k_from_a = a_dims[1];  // columns of A
  int64_t k_from_b = b_dims[0];  // rows of B
  int64_t n = b_dims[1];  // columns of B
  
  // Validate that inner dimensions match
  if (k_from_a != k_from_b) {
    return ffi::Error::Internal(
        "Matrix dimension mismatch: A's columns (" + std::to_string(k_from_a) +
        ") != B's rows (" + std::to_string(k_from_b) + ")");
  }
  
  // Define block and grid dimensions
  dim3 block_dim(16, 16);  // 16x16 threads per block
  dim3 grid_dim((n + block_dim.x - 1) / block_dim.x,
                (m + block_dim.y - 1) / block_dim.y);
  
  // Launch kernel
  MatMulKernel<<<grid_dim, block_dim, /*shared_mem=*/0, stream>>>(
      A.typed_data(), B.typed_data(), C->typed_data(), m, n, k_from_a);
  
  // Check for errors
  cudaError_t last_error = cudaGetLastError();
  if (last_error != cudaSuccess) {
    return ffi::Error::Internal(
        std::string("CUDA error: ") + cudaGetErrorString(last_error));
  }
  return ffi::Error::Success();
}

// Creates symbol MatMul with C linkage that can be loaded using Python ctypes
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    MatMul, MatMulHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()  // stream
        .Arg<ffi::Buffer<ffi::F32>>()              // A: [m, k]
        .Arg<ffi::Buffer<ffi::F32>>()              // B: [k, n]
        .Ret<ffi::Buffer<ffi::F32>>(),             // C: [m, n]
    {xla::ffi::Traits::kCmdBufferCompatible});  // cudaGraph enabled

//----------------------------------------------------------------------------//
