#pragma once

#include <vector>
#include "source-index.h"
#include "stream-data.h"

namespace cm {
  
class IndexStates;

struct FilterStream : public StreamData {
  FilterStream(index_t idx, index_t stride, cudaStream_t stream)
      : StreamData{idx, stride, stream} {}

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

  SourceIndex* device_src_indices{};     // allocated in device memory
  size_t num_device_src_indices_bytes{}; // size of above buffer
  std::vector<SourceIndex> src_indices;
};  // struct FilterStream

}  // namespace cm
