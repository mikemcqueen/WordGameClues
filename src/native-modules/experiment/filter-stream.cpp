
#include <iostream>
#include "filter-stream.h"
#include "filter-types.h"

namespace cm {

int FilterStream::num_ready(const IndexStates& indexStates) const {
  return indexStates.num_ready(0, stride);
}

int FilterStream::num_done(const IndexStates& indexStates) const {
  return indexStates.num_done(0, stride);
}

int FilterStream::num_compatible(const IndexStates& indexStates) const {
  return indexStates.num_compatible(0, stride);
}

bool FilterStream::fill_source_indices(IndexStates& idx_states, index_t max_indices) {
  // iters hackery (TODO: better comment)
  src_indices.resize(idx_states.has_fill_indices() ? max_indices : 0u);
  for (size_t idx{}; idx < src_indices.size(); ++idx) {
    auto opt_src_idx = idx_states.get_next_fill_idx();
    if (!opt_src_idx.has_value()) {
      src_indices.resize(idx);
      break;
    }
    src_indices.at(idx) = opt_src_idx.value();
  }
  if (log_level(ExtraVerbose)) {
    auto first = src_indices.empty() ? -1 : (int)src_indices.front().listIndex;
    auto last = src_indices.empty() ? -1 : (int)src_indices.back().listIndex;
    std::cerr << "stream " << stream_idx << " filled " << src_indices.size()
              << " of " << max_indices << ", first: " << first
              << ", last: " << last << std::endl;
  }
  return !src_indices.empty();
}

void FilterStream::alloc_copy_source_indices() {
  // alloc source indices
  auto num_bytes = src_indices.size() * sizeof(SourceIndex);
  if (num_device_src_indices_bytes < num_bytes) {
    if (device_src_indices) {
      cuda_free_async(device_src_indices, cuda_stream);
      device_src_indices = nullptr;
    }
    cuda_malloc_async((void**)&device_src_indices, num_bytes, cuda_stream,
        "src_indices");
    num_device_src_indices_bytes = num_bytes;
  }
  // copy source indices
  auto err = cudaMemcpyAsync(device_src_indices, src_indices.data(), num_bytes,
      cudaMemcpyHostToDevice, cuda_stream);
  assert_cuda_success(err, "copy src_indices");
}

void FilterStream::dump() const {
  std::cerr << "kernel " << stream_idx << ", is_running: " << std::boolalpha
            << is_running << ", src_indices: " << src_indices.size()
            << ", stride: " << stride << std::endl;
}

}  // namespace cm
