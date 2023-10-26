#include <future>
#include <numeric>
#include <string>
#include <unordered_set>
#include <vector>
#include <cuda_runtime.h>
#include "filter.h"
#include "merge-filter-common.h"
#include "merge-filter-data.h"
#include "candidates.h"

namespace {
  
using namespace cm;

static std::vector<std::future<filter_result_t>> filter_futures_;

void checkCudaError(cudaStream_t stream, const char* str) {
  auto err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    fprintf(stderr, "checkError %s: %s", str, cudaGetErrorString(err));
    assert(0);
  }
}

void add_filter_future(std::future<filter_result_t>&& filter_future) {
  filter_futures_.emplace_back(std::move(filter_future));
}

auto* allocCopySources(const CandidateList& candidates) {
  // alloc sources
  const cudaStream_t stream = cudaStreamPerThread;
  cudaError_t err = cudaSuccess;
  auto sources_bytes =
    count_candidates(candidates) * sizeof(SourceCompatibilityData);
  SourceCompatibilityData* device_sources;
  err = cudaMallocAsync((void**)&device_sources, sources_bytes, stream);
  assert((err == cudaSuccess) && "alloc sources");

  // copy sources
  size_t index{};
  for (const auto& candidate : candidates) {
    const auto& src_list = candidate.src_list_cref.get();
    err = cudaMemcpyAsync(&device_sources[index], src_list.data(),
      src_list.size() * sizeof(SourceCompatibilityData), cudaMemcpyHostToDevice,
      stream);
    if (err != cudaSuccess) {
      fprintf(stdout, "copy sources, error: %s", cudaGetErrorString(err));
      throw std::runtime_error("copy sources");
    }
    index += src_list.size();
  }
  return device_sources;
}

void zeroResults(result_t* results, size_t num_results, cudaStream_t stream) {
  cudaError_t err = cudaSuccess;
  auto results_bytes = num_results * sizeof(result_t);
  err = cudaMemsetAsync(results, 0, results_bytes, stream);
  assert((err == cudaSuccess) && "zero results");
}

auto allocResults(size_t num_results) {  // TODO cudaStream_t stream) {
  cudaStream_t stream = cudaStreamPerThread;
  cudaError_t err = cudaSuccess;
  // alloc results
  auto results_bytes = num_results * sizeof(result_t);
  result_t* device_results;
  err = cudaMallocAsync((void**)&device_results, results_bytes, stream);
  assert((err == cudaSuccess) && "alloc results");
  return device_results;
}

auto* allocCopyListStartIndices(const IndexStates& index_states) {
  cudaStream_t stream = cudaStreamPerThread;
  cudaError_t err = cudaSuccess;
  // alloc indices
  auto indices_bytes = index_states.list_start_indices.size() * sizeof(index_t);
  index_t* device_indices;
  err = cudaMallocAsync((void**)&device_indices, indices_bytes, stream);
  assert((err == cudaSuccess) && "alloc list start indices");
  // copy indices
  err = cudaMemcpyAsync(device_indices, index_states.list_start_indices.data(),
    indices_bytes, cudaMemcpyHostToDevice, stream);
  assert((err == cudaSuccess) && "copy list start indices");
  return device_indices;
}

void copy_device_results(std::vector<result_t>& results,
  result_t* device_results, cudaStream_t stream) {
  //
  cudaError_t err = cudaStreamSynchronize(stream);
  assert((err == cudaSuccess) && "sychronize");

  auto results_bytes = results.size() * sizeof(result_t);
  err = cudaMemcpyAsync(results.data(), device_results, results_bytes,
    cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    fprintf(stdout, "copy device results, error: %s", cudaGetErrorString(err));
    assert((err != cudaSuccess) && "copy device results");
  }
  err = cudaStreamSynchronize(stream);
  assert((err == cudaSuccess) && "cudaStreamSynchronize");
}

int run_filter_task(StreamSwarm& streams, int threads_per_block,
  const CandidateList& candidates, IndexStates& idx_states,
  const MergeFilterData& mfd, const SourceCompatibilityData* device_sources,
  result_t* device_results, const index_t* device_list_start_indices,
  std::vector<result_t>& results) {
  //
  using namespace std::chrono;
  int total_compat{};
  int current_stream{-1};
  while (streams.get_next_available(current_stream)) {
    auto& stream = streams.at(current_stream);
    if (!stream.is_running) {
      auto f0 = high_resolution_clock::now();
      if (!stream.fillSourceIndices(idx_states, candidates)) {
        continue;
      }
      if constexpr (0) {
        auto f1 = high_resolution_clock::now();
        auto f_dur = duration_cast<milliseconds>(f1 - f0).count();
        std::cerr << "fill - " << f_dur << "ms" << std::endl;
      }
#if defined(LOGGING)
      std::cerr << "stream " << stream.stream_idx
                << " source_indices: " << stream.source_indices.size()
                << ", ready: " << stream.num_ready(idx_states)
                << std::endl;
#endif
      stream.allocCopy(idx_states);
      run_xor_kernel(stream, threads_per_block, mfd, device_sources,
        device_results, device_list_start_indices);
      continue;
    }

    stream.has_run = true;
    stream.is_running = false;
    copy_device_results(results, device_results, stream.cuda_stream);
    auto k1 = high_resolution_clock::now();
    [[maybe_unused]] auto d_kernel =
      duration_cast<milliseconds>(k1 - stream.start_time).count();

    auto num_compat =
      idx_states.update(stream.source_indices, results, stream.stream_idx);
    total_compat += num_compat;

#if 0 || defined(LOGGING)
    auto num_actual_compat = std::accumulate(results.begin(), results.end(), 0,
      [](int sum, result_t r) { return r ? sum + 1 : sum; });
    std::cerr << " stream " << stream.stream_idx
              << " compat results: " << num_compat
              << ", total: " << total_compat
              << ", actual: " << num_actual_compat << " - " << d_kernel << "ms"
              << std::endl;
#endif
  }
  return total_compat;
}

auto add_incompatible_sources(
  const IndexStates& idx_states, const CandidateList& candidates) {
  //
  int count{};
  for (const auto& data : idx_states.list) {
    if (!data.is_compatible()) {
      for (const auto& src :
        candidates.at(data.sourceIndex.listIndex).src_list_cref.get()) {
        incompatible_sources.push_back(src);
        ++count;
      }
    }
  }
  return count;
}

std::unordered_set<std::string> get_compat_combos(
  const std::vector<result_t>& results,
  const CandidateList& candidate_list) {
  //
  std::unordered_set<std::string> compat_combos{};
  for (size_t i{}; i < candidate_list.size(); ++i) {
    if (results.at(i)) {
      const auto& combos = candidate_list.at(i).combos;
      compat_combos.insert(combos.begin(), combos.end());
    }
  }
  return compat_combos;
}

std::unordered_set<std::string> filter_task(const MergeFilterData& mfd, int sum,
  int threads_per_block, int num_streams, int stride, int iters,
  bool synchronous = false) {
  //
  num_streams = num_streams ? num_streams : 3;
  stride = stride ? stride : 1024;
  iters = iters ? iters : 1;

  using namespace std::chrono;
  [[maybe_unused]] cudaError_t err = cudaSuccess;
  //err = cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 7'500'000);

  auto it = allSumsCandidateData.find(sum);
  assert((it != allSumsCandidateData.end()) && "no candidates for sum");
  const auto& candidates = it->second;
  auto device_sources = allocCopySources(candidates);
  IndexStates idx_states{candidates};
  auto device_list_start_indices = allocCopyListStartIndices(idx_states);
  auto device_results = allocResults(candidates.size());

  stride = std::min((int)candidates.size(), stride);
  StreamSwarm streams(num_streams, stride);

#if defined(LOGGING)
  std::cerr << "sourcelists: " << candidates.size() << ", streams: " << num_streams
            << ", stride: " << stride << std::endl;
#endif
  std::vector<result_t> results(candidates.size());
  for(int i{}; i < iters; ++i) {
    streams.reset();
    idx_states.reset();
    zeroResults(device_results, candidates.size(), cudaStreamPerThread);
    auto t0 = high_resolution_clock::now();

    auto total_compat =
      run_filter_task(streams, threads_per_block, candidates, idx_states, mfd,
        device_sources, device_results, device_list_start_indices, results);

    auto t1 = high_resolution_clock::now();
    auto d_total = duration_cast<milliseconds>(t1 - t0).count();
    std::cerr << "sum(" << sum << ")";
    if (iters > 1) {
      std::cerr << " iter: " << i;
    }
    std::cerr << " total compat: " << total_compat << " of " << candidates.size();
    if (synchronous) {
      auto num_incompat = add_incompatible_sources(idx_states, candidates);
      std::cerr << " (" << num_incompat << " incompatible)";
    }
    std::cerr << " - " << d_total << "ms" << std::endl;
  }

#if 0
  err = cudaFree(device_sources);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device sources, error: %s\n",
      cudaGetErrorString(err));
    throw std::runtime_error("failed to free device sources");
  }
#endif
  return get_compat_combos(results, candidates);
}

// TODO: same as sum_sizes
auto countIndices(const VariationIndicesList& variationIndices) {
  return std::accumulate(variationIndices.begin(), variationIndices.end(), 0,
    [](int total, const auto& indices) {
      total += indices.size();
      return total;
    });
}

#if 0
auto debug_build_compat_lists(int sum) {
  std::vector<IndexList> compat_lists;
  const auto& candidate_data = allSumsCandidateData.find(sum)->second;
  const auto& src_lists = candidate_data.sourceCompatLists;
  for (size_t i{}; i < src_lists.size(); ++i) {
    const auto sz = src_lists.at(i).size();
    if (sz >= compat_lists.size()) {
      compat_lists.resize(sz + 1);
    }
    compat_lists.at(sz).push_back(i);
  }
  return compat_lists;
}
#endif

void debug_dump_compat_lists(const std::vector<IndexList>& compat_lists) {
  for (size_t i{}; i < compat_lists.size(); ++i) {
    const auto& idx_list = compat_lists.at(i);
    if (idx_list.empty()) {
      continue;
    }
    std::cout << "size: " << i << ", count: " << idx_list.size() << std::endl;
  }
}

void dump_src_list(const SourceCompatibilityList& src_list) {
  for (const auto& src : src_list) {
    src.dump();
  }
}

}  // namespace

namespace cm {

void filterCandidatesCuda(const MergeFilterData& mfd, int sum,
  int threads_per_block, int num_streams, int stride, int iters,
  bool synchronous) {
  //
#if 0
  auto compat_lists = debug_build_compat_lists(sum);
  debug_dump_compat_lists(compat_lists);

  const auto& candidate_data = allSumsCandidateData.find(sum)->second;
  auto big_list_idx = compat_lists.back().front();
  std::cout << "list: " << big_list_idx << std::endl;
  const auto& src_list = candidate_data.sourceCompatLists.at(big_list_idx);
  dump_src_list(src_list);
#endif

  if (synchronous) {
    std::promise<std::unordered_set<std::string>> p;
    p.set_value(filter_task(mfd, sum, threads_per_block, num_streams, stride, iters, true));
    add_filter_future(p.get_future());
  } else {
    add_filter_future(std::async(std::launch::async, filter_task, mfd, sum,
      threads_per_block, num_streams, stride, iters, false));
  }
}

filter_result_t get_filter_result() {
  filter_result_t unique_combos;
  std::string results{"results: "};
  int total{-1};
  for (auto& fut : filter_futures_) {
    assert(fut.valid());
    auto combos = fut.get();
    if (total > -1) {
      results.append(", ");
    } else {
      total = 0;
    }
    results.append(std::to_string(combos.size()));
    total += combos.size();
    unique_combos.insert(combos.begin(), combos.end());
  }
  std::cerr << results << ", total: " << total
            << ", unique: " << unique_combos.size() << std::endl;
  return unique_combos;
}

auto markAllXorCompatibleOrSourcesCuda(
  const device::OrSourceData* device_or_sources, unsigned num_or_sources,
  const std::vector<SourceList>& xor_src_lists,
  const std::vector<IndexList>& compat_idx_lists,
  const ComboIndexList& compat_indices)
  -> std::pair<index_t*, unsigned> {
  // TODO: alloc/zero results
  /*
  mark_or_sources_task(device_or_sources, num_or_sources, xor_src_lists,
    compat_idx_lists, combo_indices);
  */
  // TODO: build index list, return index list
  // TODO: in filterCandidatesCuda, take idx_list; in filter_task; alloc/copy device_or_indices
  return std::make_pair(nullptr, 0);
}

[[nodiscard]] SourceCompatibilityData* cuda_allocCopyXorSources(
  const XorSourceList& xorSourceList) {
  //
  auto xorsrc_bytes = xorSourceList.size() * sizeof(SourceCompatibilityData);
  SourceCompatibilityData* device_xorSources = nullptr;
  cudaError_t err = cudaMallocAsync(
    (void**)&device_xorSources, xorsrc_bytes, cudaStreamPerThread);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device xorSources, error: %s\n",
      cudaGetErrorString(err));
    assert(!"failed to allocate device xorSources");
  }
  auto compat_sources = makeCompatibleSources(xorSourceList);
  err = cudaMemcpyAsync(device_xorSources, compat_sources.data(),
    xorsrc_bytes, cudaMemcpyHostToDevice, cudaStreamPerThread);
  if (err != cudaSuccess) {
    fprintf(
      stderr, "copy xorSource to device, error: %s\n", cudaGetErrorString(err));
    assert(!"failed to copy xorSource to device");
  }
  return device_xorSources;
}

[[nodiscard]] std::pair<device::OrSourceData*, unsigned>
cuda_allocCopyOrSources(const OrArgList& orArgList) {
  cudaError_t err = cudaSuccess;
  // build host-side vector of compatible device::OrSourceData that we can
  // blast to device with one copy.
  // TODO: while this method is faster kernel-side, it's also uses more memory,
  // at one index per or_src (maybe index can be reduced to one byte? depends on
  // size/alignment of or_src probably).
  // Or, just blast these out in chunks, per or_arg. Will need to alloc/copy a
  // device_or_arg_sizes and pass num_or_args as well in that case.
  std::vector<device::OrSourceData> or_sources;
  for (unsigned arg_idx{}; arg_idx < orArgList.size(); ++arg_idx) {
    const auto& or_arg = orArgList.at(arg_idx);
    for (const auto& or_src: or_arg.or_src_list) {
      or_sources.emplace_back(device::OrSourceData{or_src.src, arg_idx});
    }
  }
  const auto or_src_bytes = or_sources.size() * sizeof(device::OrSourceData);
  device::OrSourceData* device_or_sources;
  err = cudaMallocAsync(
    (void**)&device_or_sources, or_src_bytes, cudaStreamPerThread);
  assert((err == cudaSuccess) && "alloc device or sources");

  err = cudaMemcpyAsync(device_or_sources, or_sources.data(), or_src_bytes,
    cudaMemcpyHostToDevice, cudaStreamPerThread);
  assert((err == cudaSuccess) && "copy device or sources");

  // TODO: need to store or_sources in MFD or something for async copy lifetime
  err = cudaStreamSynchronize(cudaStreamPerThread);
  assert((err == cudaSuccess) && "sync device or sources");

  return std::make_pair(device_or_sources, or_sources.size());
}

[[nodiscard]] auto cuda_allocCopySentenceVariationIndices(
  const SentenceVariationIndices& sentenceVariationIndices)
  -> device::VariationIndices* {
  // assumption made throughout
  static_assert(sizeof(combo_index_t) == sizeof(index_t) * 2);
  cudaError_t err = cudaSuccess;
  using DeviceVariationIndicesArray =
    std::array<device::VariationIndices, kNumSentences>;
  DeviceVariationIndicesArray device_indices_array;
  for (int s{}; s < kNumSentences; ++s) {
    auto& variation_indices = sentenceVariationIndices.at(s);
    const auto num_variations{variation_indices.size()};
    // clever: n * sizeof(combo_index_t) == 2 * n * sizeof(index_t)
    const auto device_data_bytes =
      (countIndices(variation_indices) + num_variations)
      * sizeof(combo_index_t);
    auto& device_indices = device_indices_array.at(s);
    err = cudaMallocAsync((void**)&device_indices.device_data,
      device_data_bytes, cudaStreamPerThread);
    assert(err == cudaSuccess);

    device_indices.num_variations = num_variations;
    // copy combo indices first, populating offsets and num_combo_indices
    device_indices.combo_indices = &device_indices.device_data[num_variations];
    std::vector<index_t> variation_offsets;
    std::vector<index_t> num_combo_indices;
    size_t offset{};
    for (const auto& combo_indices : variation_indices) {
      variation_offsets.push_back(offset);
      num_combo_indices.push_back(combo_indices.size());
      // NOTE: Async. I'm going to need to preserve
      // sentenceVariationIndices until copy is complete - (kernel
      // execution/synchronize?)
      const auto indices_bytes = combo_indices.size() * sizeof(combo_index_t);
      err = cudaMemcpyAsync(&device_indices.combo_indices[offset],
        combo_indices.data(), indices_bytes, cudaMemcpyHostToDevice,
        cudaStreamPerThread);
      assert(err == cudaSuccess);
      offset += combo_indices.size();
    }
    // copy variation offsets
    device_indices.variation_offsets = (index_t*)device_indices.device_data;
    const auto offsets_bytes = variation_offsets.size() * sizeof(index_t);
    err = cudaMemcpyAsync(device_indices.variation_offsets,
      variation_offsets.data(), offsets_bytes, cudaMemcpyHostToDevice,
      cudaStreamPerThread);
    assert(err == cudaSuccess);
    // copy num combo indices
    device_indices.num_combo_indices =
      &((index_t*)device_indices.device_data)[num_variations];
    const auto num_indices_bytes = num_combo_indices.size() * sizeof(index_t);
    err = cudaMemcpyAsync(device_indices.num_combo_indices,
      num_combo_indices.data(), num_indices_bytes, cudaMemcpyHostToDevice,
      cudaStreamPerThread);
    assert(err == cudaSuccess);
  }
  const auto variation_indices_bytes =
    kNumSentences * sizeof(device::VariationIndices);
  device::VariationIndices* device_variation_indices;
  err = cudaMallocAsync((void**)&device_variation_indices,
    variation_indices_bytes, cudaStreamPerThread);
  assert(err == cudaSuccess);

  err = cudaMemcpyAsync(device_variation_indices,
    device_indices_array.data(), variation_indices_bytes,
    cudaMemcpyHostToDevice, cudaStreamPerThread);
  assert(err == cudaSuccess);

  // TODO: be nice to get rid of this
  // just to be sure, due to lifetime problems of local host-side memory
  // solution: cram stuff into MFD
  err = cudaStreamSynchronize(cudaStreamPerThread);
  assert(err == cudaSuccess);
  return device_variation_indices;
}

}  // namespace cm
