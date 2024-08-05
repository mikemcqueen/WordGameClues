#ifndef INCLUDE_CUDA_TYPES_H // TODO: CUDA_COMMON
#define INCLUDE_CUDA_TYPES_H

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>
#include <cuda_runtime.h>

namespace cm {

// aliases

using result_t = uint8_t;
using index_t = uint32_t;
using fat_index_t = uint64_t;

template <typename T> using IndexListBase = std::vector<T>;

using IndexList = IndexListBase<index_t>;
using FatIndexList = IndexListBase<fat_index_t>;

template <typename T>
requires std::is_same_v<T, index_t> || std::is_same_v<T, fat_index_t>
using IndexSpan = std::span<const T>;

template <typename T>
using IndexSpanPairBase = std::pair<IndexSpan<T>, IndexSpan<T>>;

using IndexSpanPair = IndexSpanPairBase<index_t>;

//using FatIndexSpan = IndexSpan<fat_index_t>;
using FatIndexSpanPair = IndexSpanPairBase<fat_index_t>;

namespace device {  // on-device data structures

template <typename T>  //
struct VariationIndexData {
  T* device_data;              // one chunk of allocated data; other pointers
                               // below point inside this chunk.
  T* indices;
  index_t num_indices;         // size of indices array
  index_t num_variations;      // size of the following two arrays
  index_t* num_indices_per_variation;
  index_t* variation_offsets;  // offsets into indices

  constexpr IndexSpan<T> get_index_span(int variation) const {
    return {&indices[variation_offsets[variation]],
      num_indices_per_variation[variation]};
  }
};

using VariationIndices = VariationIndexData<index_t>;
using FatVariationIndices = VariationIndexData<fat_index_t>;

struct SourceCompatibilityData;

}  // namespace device

// functions

void cuda_malloc_async(
    void** ptr, size_t bytes, cudaStream_t stream, std::string_view tag);
void cuda_free(void* ptr);
void cuda_memory_dump(std::string_view header = "cuda_memory_dump");
size_t cuda_get_free_mem();

inline void assert_cuda_success(cudaError err, std::string_view sv) {
  if (err != cudaSuccess) {
    std::cerr << sv << ", error " << cudaGetErrorString(err) << std::endl;
    assert(0);
  }
}

template <typename T = result_t>
[[nodiscard]] inline auto cuda_alloc_results(size_t num_results,
    cudaStream_t stream = cudaStreamPerThread,
    std::string_view tag = "results") {
  // alloc results
  auto results_bytes = num_results * sizeof(T);
  T* device_results;
  // cudaError_t err = cudaMallocAsync((void**)&device_results, results_bytes, stream);
  // assert_cuda_success(err, "alloc results");
  cuda_malloc_async((void**)&device_results, results_bytes, stream, tag);
  return device_results;
}

template <typename T = result_t>
inline void cuda_zero_results(
  T* results, size_t num_results, cudaStream_t stream = cudaStreamPerThread) {
  // memset results to zero
  auto results_bytes = num_results * sizeof(T);
  cudaError_t err = cudaMemsetAsync(results, 0, results_bytes, stream);
  assert_cuda_success(err, "zero results");
}

#if 0
// every_combo_index is actually a better name due to predicate fn
inline auto for_each_combo_index(combo_index_t combo_idx,
    const std::vector<IndexList>& idx_lists, const auto& pred) {
  for (index_t list_idx{}; list_idx < idx_lists.size(); ++list_idx) {
    const auto& idx_list = idx_lists.at(list_idx);
    auto src_idx = idx_list.at(combo_idx % idx_list.size());
    if (!pred(list_idx, src_idx)) return false;
    combo_idx /= idx_list.size();
  }
  return true;
}
#endif

class CudaEvent {
public:
  CudaEvent(cudaStream_t stream = cudaStreamPerThread, bool record_now = true)
      : stream_(stream) {
    auto err = cudaEventCreate(&event_);
    assert_cuda_success(err, "cudaEventCreate");
    if (record_now) {
      record();
    }
  }

  ~CudaEvent() {
    auto err = cudaEventDestroy(event_);
    assert_cuda_success(err, "cudaEventDestroy");
  }

  auto event() const {
    return event_;
  }

  void record() const {
    auto err = cudaEventRecord(event_, stream_);
    assert_cuda_success(err, "cudaEventRecord");
  }

  void synchronize() const {
    auto err = cudaEventSynchronize(event_);
    assert_cuda_success(err, "cudaEventSynchronize");
  }

  long synchronize(const CudaEvent& start_event) const {
    synchronize();
    return elapsed(start_event);
  }

  long elapsed(const CudaEvent& start_event) const {
    float elapsed_ms;
    auto err = cudaEventElapsedTime(&elapsed_ms, start_event.event(), event_);
    assert_cuda_success(err, "cudaEventElapsedTime");
    return std::lround(elapsed_ms);
  }

private:
  cudaEvent_t event_;
  cudaStream_t stream_;
};

}  // namespace cm

#endif //  INCLUDE_CUDA_TYPES_H
