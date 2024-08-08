// filter-support.cpp

#include <experimental/scope>
#include <future>
#include <limits>
#include <numeric>
#include <optional>
#include <semaphore>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include "candidates.h"
#include "cm-precompute.h"
#include "cuda-types.h"
#include "filter.cuh"
#include "filter.h"
#include "merge-filter-common.h"
#include "merge-filter-data.h"
#include "log.h"
#include "util.h"

namespace cm {

namespace {

// types

using filter_task_result_t = std::pair<std::unordered_set<std::string>,
    std::optional<SourceCompatibilitySet>>;

// globals

std::vector<std::future<filter_task_result_t>> filter_futures_;

// functions

void add_filter_future(std::future<filter_task_result_t>&& filter_future) {
  filter_futures_.push_back(std::move(filter_future));
}

[[nodiscard]] auto cuda_alloc_copy_sources(int sum,
    const CandidateList& candidates, size_t num_sources,
    const cudaStream_t stream = cudaStreamPerThread) {
  // alloc sources
  auto sources_bytes = num_sources * sizeof(SourceCompatibilityData);
  SourceCompatibilityData* device_sources{};
  cuda_malloc_async(
      (void**)&device_sources, sources_bytes, stream, "filter sources");
  // copy sources
  size_t idx{};
  for (const auto& candidate : candidates) {
    const auto& src_compat_list = candidate.src_list_cref.get();
    auto err = cudaMemcpyAsync(&device_sources[idx], src_compat_list.data(),
        src_compat_list.size() * sizeof(SourceCompatibilityData),
        cudaMemcpyHostToDevice, stream);
    assert_cuda_success(err, "copy filter sources");
    idx += src_compat_list.size();
  }
  return device_sources;
}

/* unused. nothing wrong with it though.
[[nodiscard]] auto cuda_alloc_copy_sources(const CandidateList& candidates,
  const cudaStream_t stream = cudaStreamPerThread) {
return cuda_alloc_copy_sources(
    candidates, get_num_candidate_sources(candidates), stream);
}
*/

template <typename T = result_t>
void cuda_copy_results(std::vector<T>& results, T* device_results,
    cudaStream_t stream = cudaStreamPerThread) {
  // sync the kernel
  cudaError_t err = cudaStreamSynchronize(stream);
  assert_cuda_success(err, "copy results sync kernel");
  // copy results
  auto results_bytes = results.size() * sizeof(T);
  err = cudaMemcpyAsync(results.data(), device_results, results_bytes,
      cudaMemcpyDeviceToHost, stream);
  assert_cuda_success(err, "copy results memcpy");
  // sync the memcpy
  CudaEvent temp(stream);
  temp.synchronize();
}

template <typename T = result_t>
auto cuda_copy_results(T* device_results, unsigned num_results,
    cudaStream_t stream = cudaStreamPerThread) {
  std::vector<T> results(num_results);
  cuda_copy_results(results, device_results, stream);
  return results;
}

#if 0
[[nodiscard]] auto allocCopyListStartIndices(const IndexStates& idx_states,
    cudaStream_t stream = cudaStreamPerThread) {
  // alloc indices
  auto indices_bytes = idx_states.list_start_indices().size() * sizeof(index_t);
  index_t* device_indices;
  cudaError_t err{};
  cuda_malloc_async((void**)&device_indices, indices_bytes, stream,
      "filter start_indices");  // cl-format
  // copy indices
  err = cudaMemcpyAsync(device_indices, idx_states.list_start_indices().data(),
      indices_bytes, cudaMemcpyHostToDevice, stream);
  assert_cuda_success(err, "copy list start indices");
  return device_indices;
}
#endif

auto get_compatible_sources_results(int sum,
    const SourceCompatibilityData* device_sources, unsigned num_sources,
    const UsedSources::SourceDescriptorPair* device_incompatible_src_desc_pairs,
    unsigned num_src_desc_pairs) {
  const auto stream = cudaStreamPerThread;
  CudaEvent alloc_event;
  auto device_results = cuda_alloc_results(num_sources, stream,  //
      "get_compat_sources results");
  cuda_zero_results(device_results, num_sources);
  CudaEvent start_event;
  run_get_compatible_sources_kernel(device_sources, num_sources,
    device_incompatible_src_desc_pairs, num_src_desc_pairs, device_results);
  // probably sync always is correct/easiest thing to do here. sync'ing only
  // when logging is wrong, esp. if semaphore is introduced at calling site.
  CudaEvent stop_event;
  stop_event.synchronize();
  if (log_level(Verbose)) {
    if (log_level(Ludicrous)) {
      auto results = cuda_copy_results(device_results, num_sources);
      auto num_compat = std::accumulate(results.begin(), results.end(), 0u,
          [](unsigned sum, result_t r) { return sum + r; });
      fprintf(stderr, " actual compat: %d\n", num_compat);
    }
    auto alloc_duration = start_event.elapsed(alloc_event);
    auto kernel_duration = stop_event.elapsed(start_event);
    std::cerr << " " << sum
              << ": get_compatible_sources kernel - alloc: " << alloc_duration
              << "ms, kernel: " << kernel_duration << "ms" << std::endl;
  }
  return device_results;
}

//////////

// TODO: host-filter.cpp/host.h
namespace host {

/*
#define MAX_OR_ARGS 20
int incompatible_or_arg_counts[MAX_OR_ARGS] = {0};

void show_incompat_or_args(unsigned num_or_args) {
  // std::cerr << "HOST or-compatible sources: " << num_or_arg_compat << std::endl;
  std::cerr << "HOST incompatible or_args:\n";
  auto max_or_args = std::min(MAX_OR_ARGS, (int)num_or_args);
  for (int i{}; i < max_or_args; ++i) {
    std::cerr << " arg" << i << ": " << incompatible_or_arg_counts[i]
              << std::endl;
  }
}

void update_incompat_or_args(size_t num_or_args, bool incompat_or_args[]) {
  auto max_or_args = std::min(MAX_OR_ARGS, (int)num_or_args);
  for (int i{}; i < max_or_args; ++i) {
    if (incompat_or_args[i]) {
      incompatible_or_arg_counts[i]++;
    }
  }
}

auto is_OR_compatible(
    const SourceCompatibilityData& source, const MergeFilterData& mfd) {
  bool incompat_or_args[MAX_OR_ARGS] = {false};
  size_t num_compat_or_args{};
  int or_arg_idx{};
  for (const auto& or_arg: mfd.host_or.arg_list) {
    bool compat_arg{};
    for (const auto& or_src: or_arg.src_list_cref.get()) {
      // TODO: ignoring "marked" is_xor_compat flag for now
      if (or_src.isOrCompatibleWith(source)) {
        compat_arg = true;
      } else {
        incompat_or_args[or_arg_idx] = true;
      }
    }
    if (compat_arg) {
      ++num_compat_or_args;
    }
    ++or_arg_idx;
  }
  const auto num_or_args = mfd.host_or.arg_list.size();
  bool compat = num_compat_or_args == num_or_args;
  if (!compat) {
    update_incompat_or_args(num_or_args, incompat_or_args);
  }
  return compat;
}
*/

auto is_XOR_compatible(const SourceCompatibilityData& source,
    const std::vector<SourceList>& xor_src_lists) {
  size_t num_compat_lists{};
  for (const auto& xor_src_list : xor_src_lists) {
    for (const auto& xor_src : xor_src_list) {
      if (source.isXorCompatibleWith(xor_src)) {
        ++num_compat_lists;
        break;
      }
    }
  }
  return num_compat_lists == xor_src_lists.size();
}

void xor_filter(int sum, const StreamData& stream, const FilterData& mfd) {
  // const IndexStates& idx_states
  const auto& candidates = get_candidates(sum);
  std::vector<result_t> results(candidates.size());
  const auto& xor_src_lists = mfd.host_xor.src_lists;
  std::cerr << "HOST xor_src_lists: " << xor_src_lists.size()
            << ", xor_sources: " << xor_src_lists.at(0).size() << std::endl;
  size_t num_compat{};
  size_t num_xor_compat{};
  for (auto src_idx : stream.src_indices) {
    const auto& src_list = candidates.at(src_idx.listIndex).src_list_cref.get();
    const auto& source = src_list.at(src_idx.index);
    if (is_XOR_compatible(source, xor_src_lists)) {
      ++num_xor_compat;
      if (1) {// is_OR_compatible(source, mfd)) {
        ++num_compat;
      }
    }
  }
  std::cerr << "HOST compatible sources " << num_compat << " of "
            << stream.src_indices.size() << std::endl;
  std::cerr << "HOST xor-compatible sources: " << num_xor_compat << std::endl;
  // show_incompat_or_args(mfd.host_or.arg_list.size());
}

}  // namespace host

void log_fill_indices(int sum, const StreamData& stream,
    const IndexStates& idx_states, long duration) {
  if (log_level(Ludicrous)) {
    std::cerr << " " << sum << ": stream " << stream.stream_idx
              << " source_indices: " << stream.src_indices.size()
              << ", ready: " << stream.num_ready(idx_states) << ", filled in "
              << duration << "us" << std::endl;
  }
}

void log_filter_kernel(int sum, const StreamData& stream,
    const std::vector<result_t>& results, unsigned num_compat,
    unsigned total_compat) {
  if (log_level(ExtraVerbose)) {
    auto xor_kernel_duration =
        stream.xor_kernel_stop.synchronize(stream.xor_kernel_start);
    std::cerr << " " << sum << ": stream " << stream.stream_idx
              << " xor_kernel " << xor_kernel_duration << "ms" << std::endl;
  }
  if (log_level(Ludicrous)) {
    auto num_actual_compat = std::accumulate(results.begin(), results.end(), 0,
        [](int sum, result_t r) { return r ? sum + 1 : sum; });
    std::cerr << " " << sum << ": stream " << stream.stream_idx
              << " compat results: " << num_compat
              << ", total: " << total_compat
              << ", actual: " << num_actual_compat << std::endl;
  }
}

//
// This is the innermost filter kernel function wrapper.
//
// Divide device_sources up into chunks, and run multiple xor_kernels
// concurrently on different streams, with each stream processing one
// chunk at a time.
//
// Returns a pair<int, int>, consisting of the number of processed sources
// and the number of xor-compatible sources. (TODO: compatible with what)
//
auto run_concurrent_filter_kernels(int sum, StreamSwarm& streams,
    int threads_per_block, IndexStates& idx_states, const FilterData& mfd,
    const SourceCompatibilityData* device_src_list,
    const result_t* device_compat_src_results, result_t* device_results,
    const index_t* device_start_indices, std::vector<result_t>& results) {
  using namespace std::chrono;
  int total_processed{};
  int total_compat{};
  int current_stream{-1};
  while (streams.get_next_available(current_stream)) {
    auto& stream = streams.at(current_stream);
    if (!stream.is_running) {
      auto t = util::Timer::start_timer();
      if (!stream.fill_source_indices(idx_states)) continue;
      t.stop();
      log_fill_indices(sum, stream, idx_states, t.microseconds());
      stream.alloc_copy_source_indices(idx_states);
      run_filter_kernel(threads_per_block, stream, mfd, device_src_list,
          device_compat_src_results, device_results, device_start_indices);
      continue;
    }
    stream.has_run = true;
    stream.is_running = false;
    cuda_copy_results(results, device_results, stream.cuda_stream);
    auto num_compat = idx_states.update(stream, results);
    total_compat += num_compat;
    total_processed += stream.src_indices.size();
    log_filter_kernel(sum, stream, results, num_compat, total_compat);
    if (log_level(Ludicrous)) {
      //host::xor_filter(sum, stream, mfd);
    }
  }
  return std::make_pair(total_processed, total_compat);
}

std::unordered_set<std::string> get_compat_combos(
    const CandidateList& candidates, const std::vector<result_t>& results,
    int num_processed) {
  std::unordered_set<std::string> compat_combos{};
  auto num_candidates = std::min(candidates.size(), (size_t)num_processed);
  for (size_t i{}; i < num_candidates; ++i) {
    if (results.at(i)) {
      const auto& combos = candidates.at(i).combos;
      compat_combos.insert(combos.begin(), combos.end());
    }
  }
  return compat_combos;
}

void log_filter_sources(int sum, int num_processed, int num_compat,
    int num_results, const SourceCompatibilitySet& incompat_sources,
    int num_incompat_sources, std::chrono::milliseconds::rep duration_ms) {
  // diagnostic to clue me in
  if (num_processed < num_results) {
    std::cerr << "INFO: num_processed(" << num_processed << ") < num_results ("
             << num_results << ")\n";
  }
  // TODO: I don't understand this. when can num_processed ever be smaller?
  auto num_src_lists = std::min(num_processed, num_results);
  auto cc = get_candidate_counts(sum);
  // assert(cc.num_candidates == num_src_lists); // "often true" but..
  std::cerr << "sum(" << sum << ")"
            << " compat src_lists: " << num_compat  //
            << " of " << num_src_lists
            << ", sources processed: " << num_processed  //
            << " of " << cc.num_sources                  //
            << " - " << duration_ms << "ms" << std::endl;
  if (incompat_sources.size()) {
    std::cerr << "       incompat total: " << num_incompat_sources
              << ", unique: " << incompat_sources.size() << std::endl;
  }
}

//
// This is the intermediate filter "task" function, which takes a device
// array of sources to filter.
//
// Additional preparation and post-processing surrounding the call to
// run_xor_filter_task.
//
// Preparation:
//
// * Determine the # of streams and stride, and create a StreamSwarm object.
// * Create and initialize an IndexStates object from all available
//   candidate sources.
// * Allocate and copy "list start indices" from the IndexStates object.
// * Allocate results buffer.
//
// Post-processing:
//
// * If call is synchronous (i.e., sum == 2), populate the incompat_sources
//   set. This set is used to speed up subsequent async sums.
// * Create a set of compatible combos from compatible candidate src_lists.
//
// Returns a pair<compat_combo_string_set, incompat_sources_set>
//
filter_task_result_t filter_sources(const FilterData& mfd,
    const SourceCompatibilityData* device_sources,
    const result_t* device_compat_src_results, const CandidateList& candidates,
    int sum, int threads_per_block, int num_streams, int stride, int iters,
    bool synchronous) {
  using namespace std::experimental::fundamentals_v3;
  using namespace std::chrono;

  // TODO: FIXMENOW: streams = 3
  num_streams = num_streams ? num_streams : 1;
  stride = stride ? stride : candidates.size();
  iters = iters ? iters : 1;

  const auto stream = cudaStreamPerThread;
  IndexStates idx_states{candidates};
  auto device_start_indices = idx_states.alloc_copy_start_indices(stream);
  scope_exit free_indices{
      [device_start_indices]() { cuda_free(device_start_indices); }};
  auto device_results =
      cuda_alloc_results(candidates.size(), stream, "xor_filter results");
  scope_exit free_results{[device_results]() { cuda_free(device_results); }};

  // TODO: num_streams should be min'd here too
  // stride = std::min((int)candidates.size(), stride);
  StreamSwarm streams(num_streams, stride);
  std::vector<result_t> results(candidates.size());
  cuda_zero_results(device_results, results.size());
  if (synchronous || log_level(Verbose)) {
    std::cerr << " " << sum << ": src_lists: " << candidates.size()
              << ", streams: " << num_streams << ", stride: " << stride
              << std::endl;
  }
  auto t = util::Timer::start_timer();
  auto [num_processed, num_compat] = run_concurrent_filter_kernels(sum,
      streams, threads_per_block, idx_states, mfd, device_sources,
      device_compat_src_results, device_results, device_start_indices, results);
  t.stop();
  SourceCompatibilitySet incompat_sources;
  int num_incompat_sources{};
  if (synchronous) {
    num_incompat_sources =
        idx_states.get_incompatible_sources(candidates, incompat_sources);
  }
  log_filter_sources(sum, num_processed, num_compat, results.size(),
      incompat_sources, num_incompat_sources, t.count());
  return std::make_pair(
      std::move(get_compat_combos(candidates, results, num_processed)),
      std::move(incompat_sources));
}

void log_copy_sources(int sum, int num_sources, bool synchronous,
    const CudaEvent& copy_start) {  //
  if ((synchronous && log_level(Normal))
      || (!synchronous && log_level(Verbose))) {
    CudaEvent copy_stop;
    auto copy_duration = copy_stop.synchronize(copy_start);
    auto free_mem = cuda_get_free_mem() / 1'000'000;
    std::cerr << " " << sum << ": alloc_copy_sources(" << num_sources << ")"
              << ", free: " << free_mem << "MB"
              << " - " << copy_duration << "ms" << std::endl;
  }
}

//
// This is the outermost filter "task" function.
//
// Preparation prior to calling xor_filter_task:
//
// * Determine the candidate sources for the supplied sum.
// * Allocate and copy candidate sources.
// * If call is asynchronous (i.e. sum > 2), run the compat_sources_kernel to
//   populate a result_t array which represents src_list indices that are not
//   incompatible (i.e. compatible) with the incompat_sources that resulted
//   from the first synchronous call (sum=2). This is a significant source
//   filtering optimization.
//
// Returns the result of xor_filter_task with no additional post-processing.
//
auto filter_task(const FilterData& mfd, int sum,
    int threads_per_block, int num_streams, int stride, int iters,
    bool synchronous = false) {
  using namespace std::chrono;
  using namespace std::experimental::fundamentals_v3;

  auto& candidates = get_candidates(sum);
  auto num_sources = get_num_candidate_sources(candidates);
  CudaEvent copy_start;
  auto device_sources = cuda_alloc_copy_sources(sum, candidates, num_sources);
  scope_exit free_sources{[device_sources]() { cuda_free(device_sources); }};
  log_copy_sources(sum, num_sources, synchronous, copy_start);
  result_t* device_compat_src_results{};
  scope_exit free_results{
      [device_compat_src_results]() { cuda_free(device_compat_src_results); }};
  if (!synchronous) {
    device_compat_src_results = get_compatible_sources_results(sum,
        device_sources, num_sources, mfd.device_xor.incompat_src_desc_pairs,
        mfd.host_xor.incompat_src_desc_pairs.size());
  }
  auto filter_result = filter_sources(mfd, device_sources,
      device_compat_src_results, candidates, sum, threads_per_block,
      num_streams, stride, iters, synchronous);
  // free host memory associated with candidates
  clear_candidates(sum);
  return filter_result;
}

/*
auto async_filter_task(FilterData& mfd, int sum,
    int threads_per_block, int num_streams, int stride, int iters) {
  static std::counting_semaphore<2> semaphore(2);  // TODO: not great
  semaphore.acquire();
  auto filter_result = filter_task(
      std::move(mfd), sum, threads_per_block, num_streams, stride, iters);
  semaphore.release();
  return filter_result;
}
*/

/*
// TODO: same as sum_sizes
auto countIndices(const VariationIndicesList& variationIndices) {
  return std::accumulate(variationIndices.begin(), variationIndices.end(), 0u,
      [](size_t total, const auto& indices) {
        return total + indices.size();
      });
}
*/

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

}  // anonymous namespace

auto filter_candidates_cuda(const FilterData& mfd, int sum,
    int threads_per_block, int num_streams, int stride, int iters,
    bool synchronous) -> std::optional<SourceCompatibilitySet> {
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
    auto result = filter_task(std::cref(mfd), sum, threads_per_block,
        num_streams, stride, iters, true);
    auto opt_incompatible_sources = std::move(result.second.value());
    result.second.reset();
    p.set_value(result); // TODO: std::move
    add_filter_future(p.get_future());
    return opt_incompatible_sources;
  } else {
    add_filter_future(
        std::async(std::launch::async, filter_task, std::cref(mfd), sum,
            threads_per_block, num_streams, stride, iters, false));
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

[[nodiscard]] auto cuda_alloc_copy_source_descriptor_pairs(
    const std::vector<UsedSources::SourceDescriptorPair>& src_desc_pairs,
    cudaStream_t stream) -> UsedSources::SourceDescriptorPair* {
  auto pairs_bytes =
      src_desc_pairs.size() * sizeof(UsedSources::SourceDescriptorPair);
  UsedSources::SourceDescriptorPair* device_src_desc_pairs{};
  cuda_malloc_async((void**)&device_src_desc_pairs, pairs_bytes, stream,
      "src_desc_pairs");  // cl-format
  auto err = cudaMemcpyAsync(device_src_desc_pairs, src_desc_pairs.data(),
      pairs_bytes, cudaMemcpyHostToDevice, stream);
  assert_cuda_success(err, "copy src_desc_pairs");
  return device_src_desc_pairs;
}

[[nodiscard]] auto cuda_allocCopySentenceVariationIndices(
    const SentenceXorVariationIndices& svi,
    cudaStream_t stream) -> device::XorVariationIndices* {
  // assumption made throughout
  static_assert(sizeof(fat_index_t) == sizeof(index_t) * 2);
  using DeviceVariationIndicesArray =
      std::array<device::XorVariationIndices, kNumSentences>;
  DeviceVariationIndicesArray device_indices_array;
  cudaError_t err{};
  for (int s{}; s < kNumSentences; ++s) {
    const auto& variation_indices = svi.at(s);
    const auto num_variations{variation_indices.size()};
    // clever: n * sizeof(fat_index_t) == 2 * n * sizeof(index_t)
    const auto device_data_bytes =
        (util::sum_sizes(variation_indices) + num_variations)
        * sizeof(fat_index_t);
    auto& device_indices = device_indices_array.at(s);
    cuda_malloc_async((void**)&device_indices.device_data, device_data_bytes,
        stream, "variation_indices");
    device_indices.num_variations = num_variations;
    // copy combo indices first, populating offsets and num_combo_indices
    device_indices.indices = &device_indices.device_data[num_variations];
    std::vector<index_t> variation_offsets;
    std::vector<index_t> num_indices;
    for (size_t offset{}; const auto& fat_idx_list : variation_indices) {
      variation_offsets.push_back(offset);
      num_indices.push_back(fat_idx_list.size());
      const auto indices_bytes = fat_idx_list.size() * sizeof(fat_index_t);
      err = cudaMemcpyAsync(&device_indices.indices[offset],
          fat_idx_list.data(), indices_bytes, cudaMemcpyHostToDevice, stream);
      assert_cuda_success(err, "copy variation indices");
      offset += fat_idx_list.size();
      assert(offset < std::numeric_limits<index_t>::max());
    }
    // copy variation offsets
    device_indices.variation_offsets = (index_t*)device_indices.device_data;
    const auto offsets_bytes = variation_offsets.size() * sizeof(index_t);
    err = cudaMemcpyAsync(device_indices.variation_offsets,
        variation_offsets.data(), offsets_bytes, cudaMemcpyHostToDevice,
        stream);
    assert_cuda_success(err, "copy variation_offsets");
    // copy num combo indices
    device_indices.num_indices_per_variation =
      &((index_t*)device_indices.device_data)[num_variations];
    const auto num_indices_bytes = num_indices.size() * sizeof(index_t);
    err = cudaMemcpyAsync(device_indices.num_indices_per_variation,
        num_indices.data(), num_indices_bytes, cudaMemcpyHostToDevice, stream);
    assert_cuda_success(err, "copy num_indices_per_variation");
  }
  const auto variation_indices_bytes =
      kNumSentences * sizeof(device::XorVariationIndices);
  device::XorVariationIndices* device_variation_indices;
  cuda_malloc_async((void**)&device_variation_indices, variation_indices_bytes,
      stream, "variation_indices");
  err = cudaMemcpyAsync(device_variation_indices, device_indices_array.data(),
      variation_indices_bytes, cudaMemcpyHostToDevice, stream);
  assert_cuda_success(err, "copy variation_indices");

  // TODO: be nice to get rid of this
  // just to be sure, due to lifetime problems of local host-side memory
  // solution: cram stuff into MFD
  CudaEvent temp(stream);
  temp.synchronize();
  return device_variation_indices;
}

auto get_variation_lengths(
    const std::vector<VariationIndexOffset>& vi_offsets) {
  IndexList lengths;
  for (index_t idx{1u}; idx < vi_offsets.size(); ++idx) {
    lengths.push_back(
        vi_offsets.at(idx).offset - vi_offsets.at(idx - 1).offset);
  }
  return lengths;
}

/*
*/

[[nodiscard]] auto cuda_alloc_copy_variation_indices(
    const SentenceOrVariationIndices& host_svi,
    cudaStream_t stream) -> device::OrVariationIndices* {
  static_assert(sizeof(VariationIndexOffset) == sizeof(index_t) * 2);
  using DeviceSentenceOrVariationIndices =
      std::array<device::OrVariationIndices, kNumSentences>;
  DeviceSentenceOrVariationIndices device_svi;
  cudaError_t err{};
  for (int s{}; s < kNumSentences; ++s) {
    const auto& host_vi = host_svi.at(s);
    const auto num_variations{host_vi.index_offsets.size()};
    auto& device_vi = device_svi.at(s);
    const auto device_data_bytes =  // TODO: device_vi.calc_size(host_vi)
        host_vi.indices.size() * sizeof(index_t) +       // indices
        2 * sizeof(index_t) +                            // num_indices/num_vars
        num_variations * sizeof(VariationIndexOffset) +  // var_index_offets
        num_variations * sizeof(index_t);                // num_idx_per_var
    cuda_malloc_async((void**)&device_vi.device_data, device_data_bytes, stream,
        "or variation_indices");
    device_vi.num_variations = num_variations;

    // copy indices first, populating offsets and num_indices
    // per above static assert: index_t + 2 * index_t = 3 * index_t
    device_vi.indices = &device_vi.device_data[num_variations * 3];
    const auto indices_bytes = host_vi.indices.size() * sizeof(index_t);
    // TODO: questionable destination address
    err = cudaMemcpyAsync(device_vi.indices, host_vi.indices.data(),
        indices_bytes, cudaMemcpyHostToDevice, stream);
    assert_cuda_success(err, "copy or variation indices");

    // copy variation index offsets
    device_vi.variation_index_offsets =
        (VariationIndexOffset*)device_vi.device_data;
    const auto index_offsets_bytes =
        host_vi.index_offsets.size() * sizeof(VariationIndexOffset);
    err = cudaMemcpyAsync(device_vi.variation_index_offsets,
        host_vi.index_offsets.data(), index_offsets_bytes,
        cudaMemcpyHostToDevice, stream);
    assert_cuda_success(err, "copy or variation_index_offsets");

    // copy num indices
    auto lengths = get_variation_lengths(host_vi.index_offsets);
    device_vi.num_indices_per_variation =
        &device_vi.device_data[num_variations * 2];
    const auto num_indices_bytes = lengths.size() * sizeof(index_t);
    err = cudaMemcpyAsync(device_vi.num_indices_per_variation, lengths.data(),
        num_indices_bytes, cudaMemcpyHostToDevice, stream);
    assert_cuda_success(err, "copy or num_indices_per_variation");
  }
  const auto vi_bytes = kNumSentences * sizeof(device::OrVariationIndices);
  device::OrVariationIndices* device_variation_indices;
  cuda_malloc_async((void**)&device_variation_indices, vi_bytes, stream,  //
      "or variation_indices");
  err = cudaMemcpyAsync(device_variation_indices, device_svi.data(), vi_bytes,
      cudaMemcpyHostToDevice, stream);
  assert_cuda_success(err, "copy or variation_indices");

  // TODO: be nice to get rid of this
  // just to be sure, due to lifetime problems of local host-side memory
  // solution: cram stuff into MFD
  CudaEvent temp(stream);
  temp.synchronize();
  return device_variation_indices;
}

void alloc_copy_compat_indices(FilterData::HostOr& host,
    FilterData::DeviceOr& device, cudaStream_t stream) {
  if (host.compat_indices.empty()) return;
  const auto indices_bytes = host.compat_indices.size() * sizeof(fat_index_t);
  cuda_malloc_async((void**)&device.compat_indices, indices_bytes, stream,  //
      "filter or_compat_indices");
  auto err = cudaMemcpyAsync(device.compat_indices, host.compat_indices.data(),
      indices_bytes, cudaMemcpyHostToDevice, stream);
  assert_cuda_success(err, "copy or_compat_indices");
  device.num_compat_indices = host.compat_indices.size();
}

void alloc_copy_filter_indices(FilterData& mfd,
    const UsedSources::VariationsList& or_variations_list,
    cudaStream_t stream) {
  assert(!mfd.host_xor.compat_indices.empty()); // arbitrary
  util::LogDuration ld("alloc_copy_filter_indices", Verbose);
  alloc_copy_start_indices(mfd.host_xor, mfd.device_xor, stream);
  auto xor_vi = buildSentenceVariationIndices(mfd.host_xor.src_lists,
      mfd.host_xor.compat_idx_lists, mfd.host_xor.compat_indices);
  mfd.device_xor.variation_indices =
    cuda_allocCopySentenceVariationIndices(xor_vi, stream);

  if (!mfd.host_or.compat_indices.empty()) {
    alloc_copy_start_indices(mfd.host_or, mfd.device_or, stream);
    /*
    auto or_vi = build_variation_indices(or_variations_list,  //
        mfd.host_or.compat_indices);
    mfd.device_or.variation_indices =
        cuda_alloc_copy_variation_indices(or_vi, stream);
    */
    alloc_copy_compat_indices(mfd.host_or, mfd.device_or, stream);
  } else {
    // TOOD: cuda_free(), assuming we reset_pointers somewhere earlier.
    mfd.device_or.reset_pointers();
  }
}

void alloc_copy_filter_data(FilterData& mfd,
    const UsedSources::VariationsList& or_variations_list,
    cudaStream_t stream) {
  alloc_copy_filter_indices(mfd, or_variations_list, stream);

  const auto xor_bytes = sizeof(FilterData::DeviceXor::Base);
  cudaError_t err{};
  cuda_malloc_async((void**)&mfd.device_xor_data, xor_bytes, stream,  //
      "filter data");
  err = cudaMemcpyAsync(mfd.device_xor_data, &mfd.device_xor, xor_bytes,  //
      cudaMemcpyHostToDevice);
  assert_cuda_success(err, "copy filter data");

  const auto or_bytes = sizeof(FilterData::DeviceOr);
  cuda_malloc_async((void**)&mfd.device_or_data, or_bytes, stream,  //
      "filter data");
  err = cudaMemcpyAsync(mfd.device_or_data, &mfd.device_or, or_bytes,  //
      cudaMemcpyHostToDevice);
  assert_cuda_success(err, "copy filter data");
  std::cerr << "OR variation indices: " << mfd.device_or.num_variation_indices
            << std::endl;
}

}  // namespace cm
