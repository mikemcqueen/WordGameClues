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
#include "variations.h"

namespace cm {

// aliases

using result_t = uint8_t;
using index_t = uint32_t;
using fat_index_t = uint64_t;
using atomic64_t = unsigned long long int;

template <typename T> using IndexListBase = std::vector<T>;

using IndexList = IndexListBase<index_t>;
using FatIndexList = IndexListBase<fat_index_t>;

using IndexSpan = std::span<const index_t>;
using IndexSpanPair = std::pair<IndexSpan, IndexSpan>;

struct VariationIndexOffset {
  variation_index_t variation_index;
  variation_index_t padding_;
  index_t offset;
};

struct UniqueVariations {
  Variations variations{};
  // sum of num_indices of all prior UniqueVariations in an array
  index_t start_idx;
  // first index into xxx_data.compat_indices
  index_t first_compat_idx;
  index_t num_indices;
};

namespace device {  // on-device data structures

struct VariationIndices {
  constexpr IndexSpan get_index_span(index_t variation) const {
    // TODO: ifdef ASSERTS
    assert(variation < num_variations);
    return {&indices[variation_offsets[variation]],
        num_indices_per_variation[variation]};
  }

  index_t* device_data;        // one chunk of allocated data; other pointers
                               // below point inside this chunk.
  index_t* indices;
  index_t* num_indices_per_variation;
  index_t* variation_offsets;  // offsets into indices
  index_t num_indices;         // size of indices array (unused i think)
  index_t num_variations;      // size of num_indices & variation_offsets arrays
};

// this was all some attempt at supporting OrVariationIndices, an idea which
// has died on the vine and which I should have left relegated to an orphan
// branch. Lots of unnecessary template noise leftover here.
/*
template <typename T> struct VariationIndices;
template <> struct VariationIndices<index_t> : VariationIndicesBase<index_t> {
constexpr IndexSpan<index_t> get_index_span(int variation) const {
  assert(0);
  return {&indices[variation_index_offsets[variation].offset],
      num_indices_per_variation[variation]};
}

VariationIndexOffset* variation_index_offsets;
};

struct VariationIndices : VariationIndicesBase<index_t> {
  constexpr IndexSpan get_index_span(int variation) const {
    return {&indices[variation_offsets[variation]],
      num_indices_per_variation[variation]};
  }

};
using OrVariationIndices = VariationIndices<index_t>;
*/

using XorVariationIndices = VariationIndices;

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
