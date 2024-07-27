#ifndef INCLUDE_CUDA_TYPES_H // TODO: CUDA_COMMON
#define INCLUDE_CUDA_TYPES_H

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <span>
#include <string_view>
#include <utility>
#include <vector>
#include <cuda_runtime.h>

namespace cm {

// aliases

using result_t = uint8_t;
using compat_src_result_t = result_t;
using index_t = uint32_t;
using combo_index_t = uint64_t;

using IndexList = std::vector<index_t>;
using ComboIndexList = std::vector<combo_index_t>;

using ComboIndexSpan = std::span<const combo_index_t>;
using ComboIndexSpanPair = std::pair<ComboIndexSpan, ComboIndexSpan>;

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

  /*
char* cuda_strcpy(char* dest, const char* src);
char* cuda_strcat(char* dest, const char* src);
int cuda_itoa(int value, char *sp, int radix = 10);
  */

class CudaEvent {
public:
  CudaEvent(const cudaStream_t stream = cudaStreamPerThread, bool record_now = true) : stream_(stream) {
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
