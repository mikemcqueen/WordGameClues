// filter-support.cpp

#include <experimental/scope>
#include <future>
#include <numeric>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include "filter.h"
#include "merge-filter-common.h"
#include "merge-filter-data.h"
#include "candidates.h"
#include "log.h"

namespace {
  
using namespace cm;

using filter_task_result_t = std::pair<std::unordered_set<std::string>,
  std::optional<SourceCompatibilitySet>>;
inline std::vector<std::future<filter_task_result_t>> filter_futures_;

void add_filter_future(std::future<filter_task_result_t>&& filter_future) {
  filter_futures_.emplace_back(std::move(filter_future));
}

[[nodiscard]] auto* cuda_alloc_copy_sources(const CandidateList& candidates,
  int num_sources, const cudaStream_t stream = cudaStreamPerThread) {
  // alloc sources
  cudaError_t err = cudaSuccess;
  auto sources_bytes = num_sources * sizeof(SourceCompatibilityData);
  SourceCompatibilityData* device_sources;
  err = cudaMallocAsync((void**)&device_sources, sources_bytes, stream);
  assert_cuda_success(err, "alloc sources");
  // copy sources
  size_t index{};
  for (const auto& candidate : candidates) {
    const auto& src_list = candidate.src_list_cref.get();
    err = cudaMemcpyAsync(&device_sources[index], src_list.data(),
      src_list.size() * sizeof(SourceCompatibilityData), cudaMemcpyHostToDevice,
      stream);
    assert_cuda_success(err, "copy sources");
    index += src_list.size();
  }
  return device_sources;
}

[[nodiscard]] auto* cuda_alloc_copy_sources(const CandidateList& candidates,
  const cudaStream_t stream = cudaStreamPerThread) {
  return cuda_alloc_copy_sources(
    candidates, count_candidates(candidates), stream);
}

auto cuda_alloc_results(size_t num_results,
  cudaStream_t stream = cudaStreamPerThread) {
  // alloc results
  auto results_bytes = num_results * sizeof(result_t);
  result_t* device_results;
  cudaError_t err = cudaMallocAsync((void**)&device_results, results_bytes, stream);
  assert_cuda_success(err, "alloc results");
  return device_results;
}

void cuda_zero_results(result_t* results, size_t num_results,
  cudaStream_t stream = cudaStreamPerThread) {
  // memset results to zero
  auto results_bytes = num_results * sizeof(result_t);
  cudaError_t err = cudaMemsetAsync(results, 0, results_bytes, stream);
  assert_cuda_success(err, "zero results");
}

void cuda_copy_results(std::vector<result_t>& results,
  result_t* device_results, cudaStream_t stream = cudaStreamPerThread) {
  // sync the kernel
  cudaError_t err = cudaStreamSynchronize(stream);
  assert_cuda_success(err, "copy results sync kernel");
  // copy results
  auto results_bytes = results.size() * sizeof(result_t);
  err = cudaMemcpyAsync(results.data(), device_results, results_bytes,
    cudaMemcpyDeviceToHost, stream);
  assert_cuda_success(err, "copy results memcpy");
  // sync the memcpy
  err = cudaStreamSynchronize(stream);
  assert_cuda_success(err, "copyresults sync memcpy");
}

[[nodiscard]] auto* allocCopyListStartIndices(
  const IndexStates& index_states, cudaStream_t stream = cudaStreamPerThread) {
  // alloc indices
  cudaError_t err = cudaSuccess;
  auto indices_bytes = index_states.list_start_indices.size() * sizeof(index_t);
  index_t* device_indices;
  err = cudaMallocAsync((void**)&device_indices, indices_bytes, stream);
  assert_cuda_success(err, "alloc list start indices");
  // copy indices
  err = cudaMemcpyAsync(device_indices, index_states.list_start_indices.data(),
    indices_bytes, cudaMemcpyHostToDevice, stream);
  assert_cuda_success(err, "copy list start indices");
  return device_indices;
}

auto build_src_lists(
  const CandidateList& candidates, const std::vector<result_t>& results) {
  SourceCompatibilityLists src_lists;
  for (index_t result_idx{}; const auto& candidate : candidates) {
    SourceCompatibilityList src_list;
    for (const auto& src : candidate.src_list_cref.get()) {
      if (results.at(result_idx)) {
        src_list.emplace_back(src);
      }
      ++result_idx;
    }
    // add empty src_list so list indexing stays in since with candidates
    src_lists.emplace_back(std::move(src_list));
  }
  std::cerr << " compat_lists: " << src_lists.size() << std::endl;
  return src_lists;
}

auto make_compatible_src_lists(const CandidateList& candidates,
  SourceCompatibilityData* device_incompatible_sources,
  unsigned num_incompatible_sources) {
  // * alloc/copy candidate sources to device in one flat array
  // * alloc/zero results (flat array bytes same size as above)
  // * run kernel
  // * copy result data from device
  // * return build_src_lists from original candidates & result data
  using namespace std::experimental::fundamentals_v3;
  int num_sources = count_candidates(candidates);
  auto device_sources = cuda_alloc_copy_sources(candidates, num_sources);
  scope_exit free_sources{[device_sources]() { cuda_free(device_sources); }};
  auto device_results = cuda_alloc_results(num_sources);
  scope_exit free_results{[device_results]() { cuda_free(device_results); }};
  cuda_zero_results(device_results, num_sources);

  run_get_compatible_sources_kernel(device_sources, num_sources,
    device_incompatible_sources, num_incompatible_sources, device_results);

  std::vector<result_t> results(num_sources);
  cuda_copy_results(results, device_results);
  std::cerr << " make_compatible: " << util::sum<result_t, unsigned>(results)
            << " of " << results.size() << std::endl;

  return build_src_lists(candidates, results);
}

CandidateList make_compatible_candidates(
  const SourceCompatibilityLists& src_lists,
  const CandidateList& unfiltered_candidates) {
  // TODO dumb, remove, or assert
  if (src_lists.empty()) {
    return {};
  }
  assert(src_lists.size() == unfiltered_candidates.size());
  CandidateList candidates;
  for (size_t i{}; i < src_lists.size(); ++i) {
    const auto& src_list = src_lists.at(i);
    if (!src_list.empty()) {
      candidates.emplace_back(
        std::cref(src_list), unfiltered_candidates.at(i).combos);
    }
  }
  return candidates;
}

//////////

int run_xor_filter_task(StreamSwarm& streams, int threads_per_block,
  const CandidateList& candidates, IndexStates& idx_states,
  const MergeFilterData& mfd, const SourceCompatibilityData* device_src_list,
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
      if (log_level > 1) {
        auto f1 = high_resolution_clock::now();
        auto f_dur = duration_cast<milliseconds>(f1 - f0).count();
        std::cerr << "stream " << stream.stream_idx
                  << " source_indices: " << stream.source_indices.size()
                  << ", ready: " << stream.num_ready(idx_states)
                  << ", filled in " << f_dur << "ms" << std::endl;
      }
      stream.allocCopy(idx_states);
      run_xor_kernel(stream, threads_per_block, mfd, device_src_list,
        device_results, device_list_start_indices);
      continue;
    }

    stream.has_run = true;
    stream.is_running = false;
    cuda_copy_results(results, device_results, stream.cuda_stream);
    auto k1 = high_resolution_clock::now();
    [[maybe_unused]] auto d_kernel =
      duration_cast<milliseconds>(k1 - stream.start_time).count();

    auto num_compat =
      idx_states.update(stream.source_indices, results, stream.stream_idx);
    total_compat += num_compat;

    if (log_level > 1) {
      auto num_actual_compat = std::accumulate(results.begin(), results.end(),
        0, [](int sum, result_t r) { return r ? sum + 1 : sum; });
      std::cerr << " stream " << stream.stream_idx
                << " compat results: " << num_compat
                << ", total: " << total_compat
                << ", actual: " << num_actual_compat << " - " << d_kernel
                << "ms" << std::endl;
    }
  }
  return total_compat;
}

inline int total_incompat{};

auto get_incompatible_sources(
  const IndexStates& idx_states, const CandidateList& candidates) {
  // this code doesn't make sense to me. it's probably right, and i'm dumb.
  SourceCompatibilitySet src_set;
  for (const auto& data : idx_states.list) {
    if (!data.is_compatible()) {
      const auto& src_list =
        candidates.at(data.sourceIndex.listIndex).src_list_cref.get();
      src_set.insert(src_list.begin(), src_list.end());
      total_incompat += src_list.size();
    }
  }
  return src_set;
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

void log_xor_filter_task(int sum, size_t total, int num_compat,
  std::optional<SourceCompatibilitySet> opt_sources,
  std::chrono::milliseconds::rep duration_ms) {
  //
  std::cerr << "sum(" << sum << ")";
#if 0
  if (iters > 1) {
    std::cerr << " iter: " << i;
  }
#endif
  std::cerr << " compat: " << num_compat << " of " << total;
  if (opt_sources.has_value()) {
    std::cerr << ", incompat total: " << total_incompat
              << ", unique: " << opt_sources->size();
  }
  std::cerr << " - " << duration_ms << "ms" << std::endl;
}

//std::unordered_set<std::string>
filter_task_result_t xor_filter_task(const MergeFilterData& mfd, int sum,
  int threads_per_block, int num_streams, int stride, int iters,
  const CandidateList& candidates, bool synchronous) {
  //
  num_streams = num_streams ? num_streams : 3;
  stride = stride ? stride : 1024;
  iters = iters ? iters : 1;

  using namespace std::chrono;
  [[maybe_unused]] cudaError_t err = cudaSuccess;
  // err = cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 7'500'000);

  auto device_src_list = cuda_alloc_copy_sources(candidates);
  IndexStates idx_states{candidates};
  auto device_list_start_indices = allocCopyListStartIndices(idx_states);
  auto device_results = cuda_alloc_results(candidates.size());

  stride = std::min((int)candidates.size(), stride);
  StreamSwarm streams(num_streams, stride);

  if (log_level > 1) {
    std::cerr << "sourcelists: " << candidates.size()
              << ", streams: " << num_streams << ", stride: " << stride
              << std::endl;
  }
  std::vector<result_t> results(candidates.size());
  //  for (int i{}; i < iters; ++i) {
  streams.reset();
  idx_states.reset();
  cuda_zero_results(device_results, candidates.size());
  auto t0 = high_resolution_clock::now();

  auto num_compat =
    run_xor_filter_task(streams, threads_per_block, candidates, idx_states, mfd,
      device_src_list, device_results, device_list_start_indices, results);

  auto t1 = high_resolution_clock::now();
  auto duration = duration_cast<milliseconds>(t1 - t0).count();
  std::optional<SourceCompatibilitySet> opt_incompatible_sources;
  if (synchronous) {
    opt_incompatible_sources = std::move(get_incompatible_sources(idx_states, candidates));
  }
  log_xor_filter_task(
    sum, candidates.size(), num_compat, opt_incompatible_sources, duration);
  //  }

#if 0
  err = cudaFree(device_src_list);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device src_list, error: %s\n",
      cudaGetErrorString(err));
    throw std::runtime_error("failed to free device src_list");
  }
#endif
  return std::make_pair(get_compat_combos(results, candidates),
    std::move(opt_incompatible_sources));
}

/*
const auto& get_candidate_src(
  const CandidateList& candidates, SourceIndex src_idx) {
  return candidates.at(src_idx.listIndex).src_list_cref.get().at(src_idx.index);
}

auto is_compatible_src(const SourceCompatibilityData& src,
  const std::unordered_set<SourceCompatibilityData>& incompatible_sources) {
  //
  for (const auto& bad_src : incompatible_sources) {
    if (bad_src.isAndCompatibleWith(src)) {
      return false;
    }
  }
  return true;
}

SourceCompatibilityLists make_compatible_src_lists(
  const CandidateList& candidates,
  const std::unordered_set<SourceCompatibilityData>& incompatible_sources) {
  //
  if (incompatible_sources.empty()) {
    return {};
  }
  SourceCompatibilityLists src_lists;
  for (const auto& candidate : candidates) {
    SourceCompatibilityList src_list;
    for (const auto& src : candidate.src_list_cref.get()) {
      if (is_compatible_src(src, incompatible_sources)) {
        src_list.push_back(src);
      }
    }
    src_lists.emplace_back(std::move(src_list));
  }
  return src_lists;
}
*/

auto filter_task(const MergeFilterData& mfd, int sum, int threads_per_block,
  int num_streams, int stride, int iters, const CandidateMap& all_candidates,
  bool synchronous = false) {
  using namespace std::chrono;
  // Assumption: only one synchronous task will run at a time that populates
  // this set. Multiple async tasks may run simultaneously afterwards which 
  // access it.
  auto it = all_candidates.find(sum);
  assert((it != all_candidates.end()) && "no candidates for sum");
  const auto& unfiltered_candidates = it->second;
  // compatible_src_lists needs function-scope lifetime
  SourceCompatibilityLists compatible_src_lists;
  CandidateList compatible_candidates;
  if (!synchronous) {
    auto c0 = high_resolution_clock::now();
    compatible_src_lists =
      std::move(make_compatible_src_lists(unfiltered_candidates,
        mfd.device.incompatible_sources, mfd.device.num_incompatible_sources));
    assert(!compatible_src_lists.empty());
    compatible_candidates = std::move(
      make_compatible_candidates(compatible_src_lists, unfiltered_candidates));
    assert(!compatible_candidates.empty());
    auto c1 = high_resolution_clock::now();
    auto c_dur = duration_cast<milliseconds>(c1 - c0).count();
    if (log_level >= 1) {
      std::cerr << "sum(" << sum << ") candidates"
                << ", original: " << count_candidates(unfiltered_candidates)
                << ", compatible: " << count_candidates(compatible_candidates)
                << " - " << c_dur << "ms" << std::endl;
    }
  }
  const auto& candidates =
    synchronous ? unfiltered_candidates : compatible_candidates;
  return xor_filter_task(mfd, sum, threads_per_block, num_streams, stride,
    iters, candidates, synchronous);
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

auto filter_candidates_cuda(const MergeFilterData& mfd, int sum,
  int threads_per_block, int num_streams, int stride, int iters,
  bool synchronous) -> std::optional<SourceCompatibilitySet> {
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
    std::promise<filter_task_result_t> p;
    auto result = filter_task(mfd, sum, threads_per_block, num_streams, stride,
      iters, allSumsCandidateData, true);
    auto opt_incompatible_sources = std::move(result.second.value());
    result.second.reset();
    p.set_value(result);
    add_filter_future(p.get_future());
    return opt_incompatible_sources;
  } else {
    add_filter_future(std::async(std::launch::async, filter_task, mfd, sum,
                                 threads_per_block, num_streams, stride, iters, allSumsCandidateData, false));
    return std::nullopt;
  }
}

filter_result_t get_filter_result() {
  filter_result_t unique_combos;
  std::string results{"results: "};
  int total{-1};
  for (auto& fut : filter_futures_) {
    assert(fut.valid());
    const auto result = fut.get();
    const auto& combos = result.first;
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

auto cuda_markAllXorCompatibleOrSources(const MergeFilterData& mfd)
  -> std::vector<result_t> {
  // alloc/zero results
  auto device_results = cuda_alloc_results(mfd.device.num_or_sources);
  cuda_zero_results(device_results, mfd.device.num_or_sources);
  run_mark_or_sources_kernel(mfd, device_results);
  std::vector<result_t> results(mfd.device.num_or_sources);
  cuda_copy_results(results, device_results);
  auto marked = std::accumulate(results.begin(), results.end(), 0,
    [](int total, result_t val) { return total + val; });
  std::cerr << "  cuda marked " << marked << " of " << mfd.device.num_or_sources
            << std::endl;
  return results;
}

// This "compacts" the device_or_src_list by moving all "compatible" sources
// to the front of the array, and returns the compatible source count.
unsigned move_marked_or_sources(device::OrSourceData* device_or_src_list,
  const std::vector<result_t>& mark_results) {
  //
  size_t dst_idx{};
  size_t src_idx{};
  // skip over any marked (and therefore correctly placed) results at beginning
  for (; (src_idx < mark_results.size()) && mark_results[src_idx];
       ++src_idx, ++dst_idx)
    ;
  // move any remaining marked (and therefore incorrectly placed) results
  for (; src_idx < mark_results.size(); ++src_idx) {
    if (mark_results[src_idx]) {
      cudaError_t err = cudaMemcpyAsync(&device_or_src_list[dst_idx++],
        &device_or_src_list[src_idx], sizeof(device::OrSourceData),
        cudaMemcpyDeviceToDevice);
      assert((err == cudaSuccess) && "move_marked_or_sources memcpy");
    }
  }
  return dst_idx;
}

  /*
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
  */

[[nodiscard]] SourceCompatibilityData* cuda_alloc_copy_sources(
  const SourceCompatibilitySet& sources) {
  //
  auto sources_bytes = sources.size() * sizeof(SourceCompatibilityData);
  SourceCompatibilityData* device_sources = nullptr;
  cudaError_t err = cudaMallocAsync(
    (void**)&device_sources, sources_bytes, cudaStreamPerThread);
  assert_cuda_success(err, "alloc sources");
  for (int idx{}; const auto& src : sources) {
    err = cudaMemcpyAsync(&device_sources[idx++], &src,
      sizeof(SourceCompatibilityData), cudaMemcpyHostToDevice,
      cudaStreamPerThread);
    assert_cuda_success(err, "copy source");
  }
  return device_sources;
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
  std::vector<device::OrSourceData> or_src_list;
  for (unsigned arg_idx{}; arg_idx < orArgList.size(); ++arg_idx) {
    const auto& or_arg = orArgList.at(arg_idx);
    for (const auto& or_src: or_arg.or_src_list) {
      or_src_list.emplace_back(device::OrSourceData{or_src.src, arg_idx});
    }
  }
  const auto or_src_bytes = or_src_list.size() * sizeof(device::OrSourceData);
  device::OrSourceData* device_or_src_list;
  err = cudaMallocAsync(
    (void**)&device_or_src_list, or_src_bytes, cudaStreamPerThread);
  assert((err == cudaSuccess) && "or_sources alloc");

  err = cudaMemcpyAsync(device_or_src_list, or_src_list.data(), or_src_bytes,
    cudaMemcpyHostToDevice, cudaStreamPerThread);
  assert((err == cudaSuccess) && "or_sources memcpy");

  // TODO: need to store or_src_list in MFD or something for async copy lifetime
  err = cudaStreamSynchronize(cudaStreamPerThread);
  assert((err == cudaSuccess) && "or_sources sync");

  return std::make_pair(device_or_src_list, or_src_list.size());
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
