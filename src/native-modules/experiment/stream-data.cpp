#include <iostream>
#include "stream-data.h"
#include "filter-types.h"

namespace cm {

int StreamData::num_ready(const IndexStates& indexStates) const {
  return indexStates.num_ready(0, num_list_indices);
}

int StreamData::num_done(const IndexStates& indexStates) const {
  return indexStates.num_done(0, num_list_indices);
}

int StreamData::num_compatible(const IndexStates& indexStates) const {
  return indexStates.num_compatible(0, num_list_indices);
}

bool StreamData::fill_source_indices(IndexStates& idx_states, int max_idx) {
  // iters hackery (TODO: better comment)
  src_indices.resize(idx_states.has_fill_indices() ? max_idx : 0);
  for (size_t idx{}; idx < src_indices.size(); ++idx) {
    auto opt_src_idx = idx_states.get_next_fill_idx();
    if (!opt_src_idx.has_value()) {
      src_indices.resize(idx);
      break;
    }
    src_indices.at(idx) = opt_src_idx.value();
  }
  if (log_level(ExtraVerbose)) {
    const auto first =
        src_indices.empty() ? -1 : (int)src_indices.front().listIndex;
    const auto last =
        src_indices.empty() ? -1 : (int)src_indices.back().listIndex;
    std::cerr << "stream " << stream_idx << " filled " << src_indices.size()
              << " of " << max_idx << ", first: " << first << ", last: " << last
              << std::endl;
  }
  return !src_indices.empty();
}

void StreamData::alloc_copy_source_indices(
    [[maybe_unused]] const IndexStates& idx_states) {
  cudaError_t err = cudaSuccess;
  auto indices_bytes = src_indices.size() * sizeof(SourceIndex);
  // alloc source indices
  if (!device_src_indices) {
    cuda_malloc_async((void**)&device_src_indices, indices_bytes,  //
        cuda_stream, "src_indices");
  }
  // copy source indices
  err = cudaMemcpyAsync(device_src_indices, src_indices.data(), indices_bytes,
      cudaMemcpyHostToDevice, cuda_stream);
  assert_cuda_success(err, "copy src_indices");
}

void StreamData::dump() const {
  std::cerr << "kernel " << stream_idx << ", is_running: " << std::boolalpha
            << is_running << ", src_indices: " << src_indices.size()
            << ", num_list_indices: " << num_list_indices << std::endl;
}

}  // namespace cm
