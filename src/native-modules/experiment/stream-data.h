#pragma once

#include "source-index.h"
#include "cuda-types.h"

namespace cm {
  
class IndexStates;

struct StreamBase {
  StreamBase() = delete;

  StreamBase(index_t stream_idx, index_t stride, cudaStream_t stream)
      : stream_idx(stream_idx), stride(stride), cuda_stream(stream) {}

  index_t stream_idx;
  index_t stride;
  cudaStream_t cuda_stream;

  CudaEvent xor_kernel_start{};
  CudaEvent xor_kernel_stop{};
  int sequence_num{};
  bool is_running{false};  // true until results retrieved
  bool has_run{false};     // has run at least once
  SourceIndex* device_src_indices{};     // allocated in device memory
  size_t num_device_src_indices_bytes{}; // size of above buffer
  std::vector<SourceIndex> src_indices;

private:
  int swarm_idx;
};

struct StreamData : public StreamBase {
  static int next_sequence_num() {
    static int sequence_num{};
    return sequence_num++;
  }

  StreamData(index_t idx, index_t stride, cudaStream_t stream)
      : StreamBase{idx, stride, stream} {}

  int num_ready(const IndexStates& indexStates) const;

  int num_done(const IndexStates& indexStates) const;

  int num_compatible(const IndexStates& indexStates) const;

  bool fill_source_indices(IndexStates& idx_states, index_t max_indices);

  bool fill_source_indices(IndexStates& idx_states) {
    return fill_source_indices(idx_states, stride);
  }

  void alloc_copy_source_indices();

  auto hasWorkRemaining() const { return !src_indices.empty(); }

  void dump() const;
};  // struct StreamData

}  // namespace cm
