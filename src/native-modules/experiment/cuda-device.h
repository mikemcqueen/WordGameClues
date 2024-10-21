#pragma once

#include <cuda_runtime.h>
#include "cuda-types.h"

class CudaDevice {
  CudaDevice() {
    cudaError_t err;
    err = cudaDeviceGetAttribute(&num_sm_, cudaDevAttrMultiProcessorCount, 0);
    cm::assert_cuda_success(err, "cudaDeviceGetAttribute");
    err = cudaDeviceGetAttribute(&max_threads_per_sm_,
        cudaDevAttrMaxThreadsPerMultiProcessor, 0);
    cm::assert_cuda_success(err, "cudaDeviceGetAttribute");
  }

public:
  static CudaDevice& get() {
    static CudaDevice instance;
    return instance;
  }

  int num_sm() { return num_sm_; }
  int max_threads_per_sm() { return max_threads_per_sm_; }

private:
  int num_sm_;
  int max_threads_per_sm_;
};
