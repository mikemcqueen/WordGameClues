#ifndef INCLUDE_CUDA_TYPES_H // TODO: CUDA_COMMON
#define INCLUDE_CUDA_TYPES_H

#pragma once

#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <span>
#include <string_view>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>
#include <cuda_runtime.h>
#include "variations.h"

namespace cm {

constexpr unsigned kMaxOrArgs = 20;
constexpr auto kMaxSums = 32;
constexpr auto kMaxStreams = 4;  // 2 swarms x 2 streams each

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

// types

struct CompatSourceIndex {
  constexpr CompatSourceIndex(index_t data) : data_(data) {}
  CompatSourceIndex(int count, index_t idx) : data_(make_data(count, idx)) {}

  static index_t make_data(int count, index_t idx) {
    return index_t(count) << 27 | (idx & 0x07ffffff);
  }

  constexpr auto count() const { return data_ >> 27; }
  constexpr auto index() const { return data_ & 0x07ffffff; }

  auto data() const { return data_; }

private:
  index_t data_;
};

struct CompatSourceIndices {
  constexpr CompatSourceIndices() = default; 
  CompatSourceIndices(CompatSourceIndex first, CompatSourceIndex second)
      : data_(make_data(first, second)) {}

  static fat_index_t make_data(CompatSourceIndex first,
      CompatSourceIndex second) {
    auto first_data = first.count() < second.count()
        ? first.data()
        : (first.index() < second.index() ? first.data() : second.data());
    auto second_data = first_data == first.data() ? second.data() : first.data();
    return fat_index_t(first_data) << 32 | second_data;
  }

  constexpr CompatSourceIndex first() const {  //
    return index_t(data_ >> 32);
  }

  constexpr CompatSourceIndex second() const {
    return index_t(data_ & 0xffffffff);
  }

  auto data() const { return data_; }

private:
  fat_index_t data_{};
};

using CompatSourceIndicesList = std::vector<CompatSourceIndices>;
using CompatSourceIndicesListCRef =
    std::reference_wrapper<const CompatSourceIndicesList>;
using CompatSourceIndicesSet = std::unordered_set<CompatSourceIndices>;

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

using XorVariationIndices = VariationIndices;

struct SourceCompatibilityData;

}  // namespace device

// functions

void cuda_malloc_async(
    void** ptr, size_t bytes, cudaStream_t stream, std::string_view tag);
void cuda_free(void* ptr);
void cuda_free_async(void* ptr, cudaStream_t stream);
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
  const auto results_bytes = num_results * sizeof(T);
  T* device_results;
  cuda_malloc_async((void**)&device_results, results_bytes, stream, tag);
  return device_results;
}

template <typename T = result_t>
inline void cuda_zero_results(
  T* results, size_t num_results, cudaStream_t stream) {
  // memset results to zero
  const auto num_bytes = num_results * sizeof(T);
  cudaError_t err = cudaMemsetAsync(results, 0, num_bytes, stream);
  assert_cuda_success(err, "zero results");
}

class CudaEvent {
public:
  CudaEvent() {
    auto err = cudaEventCreate(&event_);
    assert_cuda_success(err, "cudaEventCreate");
  }

  CudaEvent(cudaStream_t stream_to_record) : CudaEvent() {
    record(stream_to_record);
  }

  // disable copy/assign
  CudaEvent(const CudaEvent&) = delete;
  CudaEvent& operator=(const CudaEvent&) = delete;
  // enable move
  CudaEvent(CudaEvent&&) = default;

  ~CudaEvent() {
    auto err = cudaEventDestroy(event_);
    assert_cuda_success(err, "cudaEventDestroy");
  }

  auto event() const {
    return event_;
  }

  void record(cudaStream_t stream) const {
    auto err = cudaEventRecord(event_, stream);
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
};

}  // namespace cm

#endif //  INCLUDE_CUDA_TYPES_H
