#pragma once

#include "source-index.h"
#include "cuda-types.h"

namespace cm {
  
class IndexStates;

struct StreamBase {
  StreamBase() = delete;
  StreamBase(int idx, cudaStream_t stream)
      : stream_idx(idx), cuda_stream(stream), xor_kernel_start(stream, false),
        xor_kernel_stop(stream, false), or_kernel_start(stream, false),
        or_kernel_stop(stream, false) {}

  int stream_idx{-1};
  cudaStream_t cuda_stream{};
  // this could be a std::array of kernel start/stop pairs possibly.
  // initialization might be a tiny bit hairy.
  CudaEvent xor_kernel_start;
  CudaEvent xor_kernel_stop;
  CudaEvent or_kernel_start;
  CudaEvent or_kernel_stop;

  int sequence_num{};
  bool is_running{false};  // is running (true until results retrieved)
  bool has_run{false};     // has run at least once
  SourceIndex* device_src_indices{};  // allocated in device memory
  std::vector<SourceIndex> src_indices;
};

struct StreamData : public StreamBase {
  static int next_sequence_num() {
    static int sequence_num{};
    return sequence_num++;
  }

  StreamData(int idx, cudaStream_t stream, int stride)
      : StreamBase(idx, stream), num_list_indices(stride) {}

  int num_ready(const IndexStates& indexStates) const;

  int num_done(const IndexStates& indexStates) const;

  int num_compatible(const IndexStates& indexStates) const;

  bool fill_source_indices(IndexStates& idx_states, int max_idx);

  bool fill_source_indices(IndexStates& idx_states) {
    return fill_source_indices(idx_states, num_list_indices);
  }

  void alloc_copy_source_indices(
      [[maybe_unused]] const IndexStates& idx_states);

  auto hasWorkRemaining() const { return !src_indices.empty(); }

  void dump() const;

  int num_list_indices;    // TODO: this doesn't belong here
};  // struct StreamData

}  // namespace cm
