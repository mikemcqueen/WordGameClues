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

inline void cuda_free(void* ptr) {
  cudaError_t err = cudaFree(ptr);
  assert_cuda_success(err, "cuda_free");
}

}  // namespace cm

#endif //  INCLUDE_CUDA_TYPES_H
