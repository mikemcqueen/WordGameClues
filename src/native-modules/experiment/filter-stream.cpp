
#include <cassert>
#include <iostream>
#include "cuda-types.h"
#include "filter.cuh"
#include "filter-stream.h"
#include "filter-types.h"

namespace cm {

extern __constant__ FilterStreamData::Device stream_data[kMaxStreams];

void FilterStreamData::Device::init(FilterStream& stream, FilterData& mfd) {
  if (!stream.host.device_initialized) {
    assert(stream.host.global_idx() < kMaxStreams);
    src_idx_list = nullptr;
    num_src_idx = 0;
    alloc_buffers(mfd, stream.cuda_stream);
    cuda_copy_to_symbol(stream.host.global_idx(), stream.cuda_stream);
    stream.host.device_initialized = true;
  }
}

void FilterStreamData::Device::alloc_buffers(FilterData& mfd,
    cudaStream_t stream) {
  const auto [grid_size, _] = get_filter_kernel_grid_block_sizes();
  if (mfd.device_xor.num_unique_variations) {
    const auto num_bytes = mfd.device_xor.num_unique_variations * grid_size
        * sizeof(index_t);
    cuda_malloc_async((void**)&xor_src_compat_uv_indices, num_bytes, stream,
        "xor_src_compat_uv_indices");
  }
  if (mfd.device_or.num_unique_variations) {
    // NB: these have the potential to grow large as num_variations grow
    const auto num_bytes = mfd.device_or.num_unique_variations * grid_size
        * sizeof(index_t);
    cuda_malloc_async((void**)&or_xor_compat_uv_indices, num_bytes, stream,
        "or_xor_compat_uv_indices");
  }
#if 1  // !ONE_ARRAY
  const auto max_uv = std::max(mfd.device_or.num_unique_variations,
      mfd.device_xor.num_unique_variations);
  if (max_uv) {
    const auto num_bytes = max_uv * grid_size * sizeof(index_t);
    cuda_malloc_async((void**)&variations_compat_results, num_bytes, stream,
        "variations_compat_results");
  }
  mfd.device_xor.variations_results_per_block = max_uv;
#endif
  // TODO: retarded name. num_src_compat_results or, is idx_list in host struct?
  if (mfd.device_or.sum_idx_list_sizes) {
    const auto num_bytes = mfd.device_or.sum_idx_list_sizes * grid_size
        * sizeof(result_t);
    cuda_malloc_async((void**)&or_src_bits_compat_results, num_bytes, stream,
        "or_src_bits_compat_results");
  }
}

void FilterStreamData::Device::cuda_copy_to_symbol(index_t idx,
    cudaStream_t stream) {
  // copy device data to constant memory
  const auto num_bytes = sizeof(FilterStreamData::Device);
  const auto err = cudaMemcpyToSymbolAsync(stream_data, this, num_bytes,
      idx * num_bytes, cudaMemcpyHostToDevice, stream);
  assert_cuda_success(err, "copy filter stream data to symbol");
}

//////////
 
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
  host.src_idx_list.resize(idx_states.has_fill_indices() ? max_indices : 0u);
  for (size_t idx{}; idx < host.src_idx_list.size(); ++idx) {
    auto opt_src_idx = idx_states.get_next_fill_idx();
    if (!opt_src_idx.has_value()) {
      host.src_idx_list.resize(idx);
      break;
    }
    host.src_idx_list.at(idx) = opt_src_idx.value();
  }
  if (log_level(ExtraVerbose)) {
    auto first = host.src_idx_list.empty() ? -1 : (int)host.src_idx_list.front().listIndex;
    auto last = host.src_idx_list.empty() ? -1 : (int)host.src_idx_list.back().listIndex;
    std::cerr << "stream " << stream_idx << " filled " << host.src_idx_list.size()
              << " of " << max_indices << ", first: " << first
              << ", last: " << last << std::endl;
  }
  return !host.src_idx_list.empty();
}

void FilterStream::alloc_copy_source_indices() {
  // alloc source indices
  const auto num_src_idx = index_t(host.src_idx_list.size());
  const auto num_bytes = num_src_idx * sizeof(SourceIndex);
  if (device.num_src_idx < num_src_idx) {
    if (device.src_idx_list) {
      cuda_free_async(device.src_idx_list, cuda_stream);
      device.src_idx_list = nullptr;
    }
    cuda_malloc_async((void**)&device.src_idx_list, num_bytes, cuda_stream,
        "device.src_idx_list");
    device.num_src_idx = num_src_idx;
  }
  // copy source indices
  auto err = cudaMemcpyAsync(device.src_idx_list, host.src_idx_list.data(),
      num_bytes, cudaMemcpyHostToDevice, cuda_stream);
  assert_cuda_success(err, "copy device.src_idx_list");
}

void FilterStream::dump() const {
  std::cerr << "kernel " << stream_idx << ", is_running: " << std::boolalpha
            << is_running << ", src_indices: " << host.src_idx_list.size()
            << ", stride: " << stride << std::endl;
}

}  // namespace cm
