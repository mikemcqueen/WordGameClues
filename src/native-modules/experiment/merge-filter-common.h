#ifndef INCLUDE_MERGE_FILTER_COMMON_H
#define INCLUDE_MERGE_FILTER_COMMON_H

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
    index += v.size();
  }
  return start_indices;
}

inline auto alloc_copy_start_indices(
  const IndexList& start_indices, size_t* num_bytes = nullptr) {
  cudaStream_t stream = cudaStreamPerThread;
  cudaError_t err = cudaSuccess;
  // alloc indices
  auto indices_bytes = start_indices.size() * sizeof(index_t);
  index_t* device_indices{};
  err = cudaMallocAsync((void**)&device_indices, indices_bytes, stream);
  assert((err == cudaSuccess) && "alloc start indices");
  // copy indices
  err = cudaMemcpyAsync(device_indices, start_indices.data(),
    indices_bytes, cudaMemcpyHostToDevice, stream);
  assert((err == cudaSuccess) && "copy start indices");
  if (num_bytes) {
    *num_bytes = indices_bytes;
  }
  return device_indices;
}

}  // namespace cm

#endif // INCLUDE_MERGE_FILTER_COMMON_H
