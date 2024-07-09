#ifndef INCLUDE_CUDA_TYPES_H // TODO: CUDA_COMMON
#define INCLUDE_CUDA_TYPES_H

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <span>
#include <utility>
#include <vector>
#include <cuda_runtime.h>

namespace cm {

using result_t = uint8_t;
using compat_src_result_t = result_t;
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

template <typename T = result_t>
inline auto cuda_alloc_results(
  size_t num_results, cudaStream_t stream = cudaStreamPerThread) {
  // alloc results
  auto results_bytes = num_results * sizeof(T);
  T* device_results;
  cudaError_t err = cudaMallocAsync((void**)&device_results, results_bytes, stream);
  assert_cuda_success(err, "alloc results");
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

inline void cuda_free(void* ptr) {
  cudaError_t err = cudaFree(ptr);
  assert_cuda_success(err, "cuda_free");
}

inline auto cuda_get_free_mem() {
  size_t free;
  size_t total;
  cudaError_t err = cudaMemGetInfo(&free, &total);
  assert_cuda_success(err, "cudaMemGetInfo");
  return free;
}

inline constexpr char* cuda_strcpy(char* dest, const char* src) {
  int i = 0;
  do {
    dest[i] = src[i];
  } while (src[i++] != 0);
  return dest;
}

inline constexpr char* cuda_strcat(char* dest, const char* src) {
  int i = 0;
  while (dest[i] != 0)
    i++;
  cuda_strcpy(dest + i, src);
  return dest;
}

// Yet, another good itoa implementation
// returns: the length of the number string
// https://stackoverflow.com/questions/3440726/what-is-the-proper-way-of-implementing-a-good-itoa-function
inline constexpr int cuda_itoa(int value, char *sp, int radix = 10) {
  char tmp[32];  // be careful with the length of the buffer
  char* tp = tmp;
  int i;
  unsigned v;

  int sign = (radix == 10 && value < 0);
  if (sign)
    v = -value;
  else
    v = (unsigned)value;

  while (v || tp == tmp) {
    i = v % radix;
    v /= radix;
    if (i < 10)
      *tp++ = i + '0';
    else
      *tp++ = i + 'a' - 10;
  }

  int len = tp - tmp;

  if (sign) {
    *sp++ = '-';
    len++;
  }

  while (tp > tmp)
    *sp++ = *--tp;

  *sp = '\0';
  return len;
}

class CudaEvent {
public:
  CudaEvent(const cudaStream_t stream = cudaStreamPerThread, bool record_now = true) : stream_(stream) {
    auto err = cudaEventCreate(&event_);
    assert_cuda_success(err, "cudaEventCreate");
    if (record_now) {
      record();
    }
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
