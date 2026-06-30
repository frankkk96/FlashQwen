#pragma once
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdlib>

#include "log.h"

namespace fq {

using bf16 = __nv_bfloat16;

#define CUDA_CHECK(call)                                    \
  do {                                                      \
    cudaError_t _e = (call);                                \
    if (_e != cudaSuccess) {                                \
      LOG_ERROR("CUDA error %s:%d: %s", __FILE__, __LINE__, \
                cudaGetErrorString(_e));                    \
      std::exit(1);                                         \
    }                                                       \
  } while (0)

#define CUBLAS_CHECK(call)                                         \
  do {                                                             \
    cublasStatus_t _s = (call);                                    \
    if (_s != CUBLAS_STATUS_SUCCESS) {                             \
      LOG_ERROR("cuBLAS error %s:%d: %d", __FILE__, __LINE__, _s); \
      std::exit(1);                                                \
    }                                                              \
  } while (0)

// Owning RAII handle for a cudaMalloc'd device buffer. Move-only; frees on
// destruction (so every buffer has exactly one owner — no central free list).
// The type marks the memory as device-resident and there is no operator*, so it
// can't be dereferenced on the host; pass D() to kernels / cuda APIs.
template <class T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(size_t n) {
    void* p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, n * sizeof(T)));
    p_ = static_cast<T*>(p);
  }
  ~DeviceBuffer() { Free(); }
  DeviceBuffer(DeviceBuffer&& o) noexcept : p_(o.p_) { o.p_ = nullptr; }
  DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
    if (this != &o) {
      Free();
      p_ = o.p_;
      o.p_ = nullptr;
    }
    return *this;
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  T* D() const { return p_; }

  void Upload(const T* src, size_t n, size_t dst_off = 0) {
    CUDA_CHECK(cudaMemcpy(p_ + dst_off, src, n * sizeof(T),
                          cudaMemcpyHostToDevice));
  }

  void Upload(const T* src, size_t n, cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(p_, src, n * sizeof(T), cudaMemcpyHostToDevice,
                               stream));
  }

 private:
  void Free() {
    if (p_) cudaFree(p_);
    p_ = nullptr;
  }
  T* p_ = nullptr;
};

}
