#ifndef INCLUDE_MERGE_FILTER_COMMON_H
#define INCLUDE_MERGE_FILTER_COMMON_H

#pragma once
#include <vector>
#include <cuda_runtime.h>
#include "cuda-types.h"

namespace cm {

template <typename T> 
auto make_start_indices(const std::vector<T>& vecs) {
  IndexList start_indices{};
  index_t index{};
  for (const auto& v : vecs) {
    start_indices.push_back(index);
    // TODO: assert(index + v.size() < std::numeric_limits<index_t>::max());
    index += index_t(v.size());
  }
  return start_indices;
}

// this is probably a duplicate of the one in filter-support.cpp.
// it's used by filter-types.h
[[nodiscard]] inline auto cuda_alloc_copy_start_indices(
    const IndexList& start_indices, cudaStream_t stream,
    std::string_view malloc_tag = "const index/size data") {  // "start_indices"
  cudaError_t err{};
  // alloc indices
  auto indices_bytes = start_indices.size() * sizeof(index_t);
  index_t* device_indices{};
  cuda_malloc_async((void**)&device_indices, indices_bytes, stream, malloc_tag);
  // copy indices
  err = cudaMemcpyAsync(device_indices, start_indices.data(), indices_bytes,
      cudaMemcpyHostToDevice, stream);
  assert_cuda_success(err, "copy start indices");
  return device_indices;
}
  
}  // namespace cm

#endif // INCLUDE_MERGE_FILTER_COMMON_H
