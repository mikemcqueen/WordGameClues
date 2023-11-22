#ifndef INCLUDE_CUDA_TYPES_H // TODO: CUDA_COMMON
#define INCLUDE_CUDA_TYPES_H

#include <cassert>
#include <iostream>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>
#include <cuda_runtime.h>

namespace cm {

using result_t = uint8_t;
using index_t = uint32_t;
using combo_index_t = uint64_t;

using IndexList = std::vector<index_t>;
using ComboIndexList = std::vector<combo_index_t>;

using ComboIndexSpan = std::span<const combo_index_t>;
using ComboIndexSpanPair = std::pair<ComboIndexSpan, ComboIndexSpan>;

inline void assert_cuda_success(cudaError err, std::string_view sv) {
  if (err != cudaSuccess) {
    std::cerr << sv << ", error " << cudaGetErrorString(err) << std::endl;
    assert(0);
  }
}

inline auto cuda_alloc_results(size_t num_results,
  cudaStream_t stream = cudaStreamPerThread) {
  // alloc results
  auto results_bytes = num_results * sizeof(result_t);
  result_t* device_results;
  cudaError_t err = cudaMallocAsync((void**)&device_results, results_bytes, stream);
  assert_cuda_success(err, "alloc results");
  return device_results;
}

inline void cuda_zero_results(result_t* results, size_t num_results,
  cudaStream_t stream = cudaStreamPerThread) {
  // memset results to zero
  auto results_bytes = num_results * sizeof(result_t);
  cudaError_t err = cudaMemsetAsync(results, 0, results_bytes, stream);
  assert_cuda_success(err, "zero results");
}

inline void cuda_free(void* ptr) {
  cudaError_t err = cudaFree(ptr);
  assert_cuda_success(err, "cuda_free");
}

}  // namespace cm

#endif //  INCLUDE_CUDA_TYPES_H
