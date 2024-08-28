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
#include "unique-variations.h"
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

[[nodiscard]] auto cuda_alloc_copy_sources(
    const SourceCompatibilityList& src_compat_list, const cudaStream_t stream) {
  // alloc sources
  auto num_bytes = src_compat_list.size() * sizeof(SourceCompatibilityData);
  SourceCompatibilityData* device_sources{};
  cuda_malloc_async((void**)&device_sources, num_bytes, stream,  //
      "filter sources");
  // copy sources
  auto err = cudaMemcpyAsync(device_sources, src_compat_list.data(), num_bytes,
      cudaMemcpyHostToDevice, stream);
  assert_cuda_success(err, "copy filter sources");
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
void cuda_copy_results(
    std::vector<T>& results, T* device_results, cudaStream_t stream) {
  // sync the kernel
  cudaError_t err = cudaStreamSynchronize(stream);
  assert_cuda_success(err, "copy filter results sync kernel");
  // copy results
  auto num_bytes = results.size() * sizeof(T);
  err = cudaMemcpyAsync(results.data(), device_results, num_bytes,
      cudaMemcpyDeviceToHost, stream);
  assert_cuda_success(err, "copy filter results");
  // sync the memcpy
  CudaEvent temp(stream);
  temp.synchronize();
}

template <typename T = result_t>
auto cuda_copy_results(
    T* device_results, unsigned num_results, cudaStream_t stream) {
  std::vector<T> results(num_results);
  cuda_copy_results(results, device_results, stream);
  return results;
}

auto get_compatible_sources_results(int sum,
    const SourceCompatibilityData* device_sources, unsigned num_sources,
    const UsedSources::SourceDescriptorPair* device_incompatible_src_desc_pairs,
    int num_src_desc_pairs) {
  const auto stream = cudaStreamPerThread;
  CudaEvent alloc_event;
  auto device_results = cuda_alloc_results(num_sources, stream,  //
      "get_compat_sources results");
  cuda_zero_results(device_results, num_sources, stream);
  CudaEvent start_event;
  run_get_compatible_sources_kernel(device_sources, num_sources,
    device_incompatible_src_desc_pairs, num_src_desc_pairs, device_results);
  // probably sync always is correct/easiest thing to do here. sync'ing only
  // when logging is wrong, esp. if semaphore is introduced at calling site.
  CudaEvent stop_event;
  stop_event.synchronize();
  if (log_level(Verbose)) {
    if (log_level(Ludicrous)) {
      auto results = cuda_copy_results(device_results, num_sources, stream);
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
*/

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
  int total_processed{};
  int total_compat{};
  int current_stream{-1};
  while (streams.get_next_available(current_stream)) {
    auto& stream = streams.at(current_stream);
    if (!stream.is_running) {
      // start kernel on stream
      auto t = util::Timer::start_timer();
      if (stream.fill_source_indices(idx_states)) {
        t.stop();
        log_fill_indices(sum, stream, idx_states, t.microseconds());
        stream.alloc_copy_source_indices(idx_states);
        run_filter_kernel(threads_per_block, stream, mfd, device_src_list,
            device_compat_src_results, device_results, device_start_indices);
      }
    } else {
      // kernel on stream has finished
      stream.has_run = true;
      stream.is_running = false;
      cuda_copy_results(results, device_results, stream.cuda_stream);
      auto num_compat = idx_states.update(stream, results);
      total_compat += num_compat;
      total_processed += int(stream.src_indices.size());
      log_filter_kernel(sum, stream, results, num_compat, total_compat);
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
    const result_t* device_compat_src_results,
    const UniqueVariations* device_uv, std::vector<IndexList>& idx_lists,
    const FilterParams& params, cudaStream_t stream) {
  using namespace std::experimental::fundamentals_v3;
  using namespace std::chrono;

  // TODO: FIXMENOW: streams = 3
  auto num_streams = params.num_streams ? params.num_streams : 1;
  auto stride = params.stride ? params.stride : int(idx_lists.size());

  // IndexStates idx_states{candidates};
  // get num_src_lists before moving idx_lists
  const auto num_src_lists = idx_lists.size();
  const auto num_results = num_src_lists;
  IndexStates idx_states{std::move(idx_lists)};
  auto device_start_indices = idx_states.alloc_copy_start_indices(stream);
  scope_exit free_indices{
      [device_start_indices]() { cuda_free(device_start_indices); }};

  auto device_results = cuda_alloc_results(num_results, stream, "filter results");
  scope_exit free_results{[device_results]() { cuda_free(device_results); }};
  cuda_zero_results(device_results, num_results, stream);
  // TODO: num_streams should be min'd here too
  // stride = std::min((int)candidates.size(), stride);
  if (params.synchronous || log_level(Verbose)) {
    std::cerr << " " << params.sum << ": src_lists: " << num_src_lists
              << ", streams: " << num_streams << ", stride: " << stride
              << std::endl;
  }
  StreamSwarm streams(num_streams, stride);
  std::vector<result_t> results(num_results);
  auto t = util::Timer::start_timer();
  auto [num_processed, num_compat] = run_concurrent_filter_kernels(params.sum,
      streams, params.threads_per_block, idx_states, mfd, device_sources,
      device_compat_src_results, device_results, device_start_indices, results);
  t.stop();
  SourceCompatibilitySet incompat_sources;
  int num_incompat_sources{};
  const auto& candidates = get_candidates(params.sum);
  if (params.synchronous) {
    incompat_sources =
        idx_states.get_incompatible_sources(candidates, &num_incompat_sources);
  }
  log_filter_sources(params.sum, num_processed, num_compat, int(results.size()),
      incompat_sources, num_incompat_sources, t.count());
  return std::make_pair(get_compat_combos(candidates, results, num_processed),
      std::move(incompat_sources));
}

void log_copy_sources(size_t num_sources, const FilterParams& params,
    const CudaEvent& copy_start) {  //
  if ((params.synchronous && log_level(Normal))
      || (!params.synchronous && log_level(Verbose))) {
    CudaEvent copy_stop;
    auto copy_duration = copy_stop.synchronize(copy_start);
    auto free_mem = cuda_get_free_mem() / 1'000'000;
    std::cerr << " " << params.sum << ": alloc_copy_sources(" << num_sources
              << ")" << ", free: " << free_mem << "MB"
              << " - " << copy_duration << "ms" << std::endl;
  }
}

auto alloc_copy_unique_variations(
    const std::vector<UniqueVariations>& unique_variations, cudaStream_t stream,
    std::string_view tag = "unique_variations") {
  assert(!unique_variations.empty());
  const auto num_bytes =
      unique_variations.size() * sizeof(UniqueVariations);
  UniqueVariations* device_uv{};
  cuda_malloc_async((void**)&device_uv, num_bytes, stream, tag);
  auto err = cudaMemcpyAsync(device_uv, unique_variations.data(), num_bytes,
      cudaMemcpyHostToDevice, stream);
  assert_cuda_success(err, "copy unique_variations");
  return device_uv;
}

//
// This is the outermost filter "task" function.
//
// Preparation prior to calling xor_filter_task:
//
// NOTE: changes have been made to this, related to unique_variations.
//
// * Determine the candidate sources for the supplied sum.
// * Allocate and copy candidate sources.
// * If call is asynchronous (i.e. sum > 2), run the compat_sources_kernel
// to
//   populate a result_t array which represents src_list indices that are
//   not incompatible (i.e. compatible) with the incompat_sources that
//   resulted from the first synchronous call (sum=2). This is a significant
//   source filtering optimization.
//
// Returns the result of filter_sources with no additional post-processing.
//
// NOTE params copy by value is necessary for async launch.
//
auto filter_task(const FilterData& mfd, FilterParams params) {
  using namespace std::chrono;
  using namespace std::experimental::fundamentals_v3;

  const auto stream = cudaStreamPerThread;
  const auto& candidates = get_candidates(params.sum);
  CudaEvent copy_start;
  auto idx_lists = make_variations_sorted_idx_lists(candidates);
  const auto num_sources = util::sum_sizes(idx_lists);
  UniqueVariations* device_uv{};
  scope_exit free_uv{[device_uv]() { cuda_free(device_uv); }};
  SourceCompatibilityData* device_sources{};
  scope_exit free_sources{[device_sources]() { cuda_free(device_sources); }};
  {
    const auto src_compat_list = make_src_compat_list(candidates, idx_lists);
    std::cerr << "src_compat_list: " << src_compat_list.size() << std::endl;
    device_sources = cuda_alloc_copy_sources(src_compat_list, stream);
    const auto unique_variations = make_unique_variations(src_compat_list);
    device_uv = alloc_copy_unique_variations(unique_variations, stream,  //
        "sources unique_variations");
  }
  log_copy_sources(num_sources, params, copy_start);
  result_t* device_compat_src_results{};
  scope_exit free_results{
      [device_compat_src_results]() { cuda_free(device_compat_src_results); }};
  if (!params.synchronous) {
    device_compat_src_results = get_compatible_sources_results(params.sum,
        device_sources, num_sources, mfd.device_xor.incompat_src_desc_pairs,
        int(mfd.host_xor.incompat_src_desc_pairs.size()));
  }
  auto filter_result = filter_sources(mfd, device_sources,
      device_compat_src_results, device_uv, idx_lists, params, stream);
  // free host memory associated with candidates
  clear_candidates(params.sum);
  return filter_result;
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

auto make_source_descriptor_pairs(
    const SourceCompatibilitySet& incompatible_sources) {
  std::vector<UsedSources::SourceDescriptorPair> src_desc_pairs;
  src_desc_pairs.reserve(incompatible_sources.size());
  for (const auto& src: incompatible_sources) {
    src_desc_pairs.push_back(src.usedSources.get_source_descriptor_pair());
  }
  return src_desc_pairs;
}

[[nodiscard]] auto cuda_allocCopySentenceVariationIndices(
    const SentenceVariationIndices& svi,
    cudaStream_t stream) -> device::VariationIndices* {
  using DeviceVariationIndicesArray =
      std::array<device::VariationIndices, kNumSentences>;
  DeviceVariationIndicesArray device_indices_array;
  cudaError_t err{};
  for (int s{}; s < kNumSentences; ++s) {
    const auto& variation_idx_lists = svi.at(s);
    const auto num_variations{variation_idx_lists.size()};
    const auto num_indices{util::sum_sizes(variation_idx_lists)};
    const auto device_data_bytes =
        ((num_variations * 2) + num_indices) * sizeof(index_t);
    auto& device_indices = device_indices_array.at(s);
    cuda_malloc_async((void**)&device_indices.device_data, device_data_bytes,
        stream, "variation_indices device_data");
    device_indices.num_indices = num_indices; // probably unused
    device_indices.num_variations = index_t(num_variations);
    // copy combo indices first, populating offsets and num_indices
    device_indices.indices = &device_indices.device_data[num_variations * 2];
    IndexList offsets_list;
    IndexList num_indices_list;
    for (size_t offset{}; const auto& idx_list : variation_idx_lists) {
      offsets_list.push_back(index_t(offset));
      num_indices_list.push_back(index_t(idx_list.size()));
      const auto indices_bytes = idx_list.size() * sizeof(index_t);
      err = cudaMemcpyAsync(&device_indices.indices[offset],
          idx_list.data(), indices_bytes, cudaMemcpyHostToDevice, stream);
      assert_cuda_success(err, "copy variation indices");
      assert(offset + idx_list.size() < std::numeric_limits<index_t>::max());
      offset += idx_list.size();
    }
    assert(num_indices_list.size() == offsets_list.size());
    // copy variation offsets
    device_indices.variation_offsets = device_indices.device_data;
    const auto offsets_bytes = offsets_list.size() * sizeof(index_t);
    err = cudaMemcpyAsync(device_indices.variation_offsets,
        offsets_list.data(), offsets_bytes, cudaMemcpyHostToDevice,
        stream);
    assert_cuda_success(err, "copy variation_offsets");
    // copy num combo indices
    device_indices.num_indices_per_variation =
        &device_indices.device_data[num_variations];
    const auto num_indices_bytes = num_indices_list.size() * sizeof(index_t);
    err = cudaMemcpyAsync(device_indices.num_indices_per_variation,
        num_indices_list.data(), num_indices_bytes, cudaMemcpyHostToDevice,
        stream);
    assert_cuda_success(err, "copy num_indices_per_variation");
  }
  const auto variation_indices_bytes =
      kNumSentences * sizeof(device::VariationIndices);
  device::VariationIndices* device_variation_indices;
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

void alloc_copy_start_indices(MergeData::Host& host,
    FilterData::DeviceCommon& device, cudaStream_t stream) {
  auto src_list_start_indices = make_start_indices(host.src_lists);
  device.src_list_start_indices = cuda_alloc_copy_start_indices(
      src_list_start_indices, stream);  // "src_list_start_indices"
  auto idx_list_start_indices = make_start_indices(host.compat_idx_lists);
  device.idx_list_start_indices = cuda_alloc_copy_start_indices(
      idx_list_start_indices, stream);  // "idx_list_start_indices"
}

void alloc_copy_compat_indices(FilterData::HostCommon& host,
    FilterData::DeviceCommon& device, cudaStream_t stream) {
  assert(!host.compat_indices.empty());
  const auto indices_bytes = host.compat_indices.size() * sizeof(fat_index_t);
  cuda_malloc_async((void**)&device.compat_indices, indices_bytes, stream,  //
      "filter or.compat_indices");
  auto err = cudaMemcpyAsync(device.compat_indices, host.compat_indices.data(),
      indices_bytes, cudaMemcpyHostToDevice, stream);
  assert_cuda_success(err, "copy or.compat_indices");
  device.num_compat_indices = int(host.compat_indices.size());
}

void alloc_copy_unique_variations(FilterData::HostCommon& host,
    FilterData::DeviceCommon& device, cudaStream_t stream,
    std::string_view tag) {
  device.unique_variations =
      alloc_copy_unique_variations(host.unique_variations, stream, tag);
  device.num_unique_variations = int(host.unique_variations.size());
}

}  // anonymous namespace

auto filter_candidates_cuda(const FilterData& mfd,
    const FilterParams& params) -> std::optional<SourceCompatibilitySet> {
  if (params.synchronous) {
    std::promise<filter_task_result_t> p;
    auto result = filter_task(std::cref(mfd), params);
    auto opt_incompatible_sources = std::move(result.second.value());
    result.second.reset();
    p.set_value(std::move(result));
    add_filter_future(p.get_future());
    return opt_incompatible_sources;
  } else {
    add_filter_future(
        std::async(std::launch::async, filter_task, std::cref(mfd), params));
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
    total += int(combos.size());
    unique_combos.insert(combos.begin(), combos.end());
  }
  std::cerr << results << ", total: " << total
            << ", unique: " << unique_combos.size() << std::endl;
  return unique_combos;
}

void set_incompatible_sources(FilterData& mfd,
    const SourceCompatibilitySet& incompat_sources, cudaStream_t stream) {
  // empty set technically possible; disallowed here as a canary
  assert(!incompat_sources.empty());
  assert(mfd.host_xor.incompat_src_desc_pairs.empty());
  assert(!mfd.device_xor.incompat_src_desc_pairs);

  mfd.host_xor.incompat_src_desc_pairs =
      std::move(make_source_descriptor_pairs(incompat_sources));
  mfd.device_xor.incompat_src_desc_pairs =
      cuda_alloc_copy_source_descriptor_pairs(
          mfd.host_xor.incompat_src_desc_pairs, stream);
}

void alloc_copy_filter_indices(FilterData& mfd, cudaStream_t stream) {
  assert(!mfd.host_xor.compat_indices.empty()); // arbitrary
  util::LogDuration ld("alloc_copy_filter_indices", Verbose);
  alloc_copy_start_indices(mfd.host_xor, mfd.device_xor, stream);
  alloc_copy_compat_indices(mfd.host_xor, mfd.device_xor, stream);
  alloc_copy_unique_variations(mfd.host_xor, mfd.device_xor, stream,  //
      "xor unique_variations");
  // NOTE i'm not sure variation indices are required anymore if
  // unique_variations works for XOR. something to look into.
  auto xor_vi = buildSentenceVariationIndices(mfd.host_xor.src_lists,
      mfd.host_xor.compat_idx_lists, mfd.host_xor.compat_indices);
  mfd.device_xor.variation_indices =
    cuda_allocCopySentenceVariationIndices(xor_vi, stream);

  if (!mfd.host_or.compat_indices.empty()) {
    alloc_copy_start_indices(mfd.host_or, mfd.device_or, stream);
    alloc_copy_compat_indices(mfd.host_or, mfd.device_or, stream);
    alloc_copy_unique_variations(mfd.host_or, mfd.device_or, stream,  //
        "or unique_variations");
  } else {
    // TODO: cuda_free(), assuming we reset_pointers somewhere earlier.
    mfd.device_or.reset_pointers();
  }

  // these all get allocated within the run_kernel function because they
  // are dependent on grid_size.
  mfd.device_xor.variations_compat_results = nullptr;
  mfd.device_xor.variations_scan_results = nullptr;
  mfd.device_xor.compat_src_uv_indices = nullptr;
  mfd.device_xor.compat_or_uv_indices = nullptr;
  mfd.device_or.src_compat_results = nullptr;
}

}  // namespace cm
