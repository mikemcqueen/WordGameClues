// filter-support.cpp

#include <experimental/scope>
#include <functional>
#include <future>
#include <limits>
#include <numeric>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include "candidates.h"
#include "clue-manager.h"
#include "cm-precompute.h"
#include "cuda-types.h"
#include "filter.cuh"
#include "filter.h"
#include "filter-types.h"
#include "merge-filter-common.h"
#include "merge-filter-data.h"
#include "stream-swarm.h"
#include "unique-variations.h"
#include "log.h"
#include "util.h"

// CompletionTracker
#include <bitset>
#include <condition_variable>
#include <mutex>

// TEST
#include "cm-hash.h"

namespace cm {

namespace {

// types

using filter_task_result_t = std::unordered_set<std::string>;

// globals

std::vector<std::future<filter_task_result_t>> filter_futures_;

StreamSwarmPool<FilterSwarm> swarm_pool_(1);

cudaStream_t sources_stream_{};

//
  
struct CompletionTracker {
  void complete(int sum) {
    assert(sum >= 2);
    auto pos = sum - 2;
    assert(!bits_.test(pos));
    std::lock_guard lock(mutex_);
    bits_.set(pos);
    cv_.notify_all();
  }

  void wait_for_all_prior(int sum) {
    assert(sum >= 2);
    auto pos = sum - 2;
    if (!pos) return;
    unsigned long mask = (1ul << pos) - 1;
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this, mask]() { return this->test_mask(mask); });
  }

private:
  bool test_mask(unsigned long mask) {
    return (bits_.to_ulong() & mask) == mask;
  }

  std::mutex mutex_;
  std::condition_variable cv_;
  std::bitset<kMaxSums> bits_;
};

CompletionTracker source_copy_tracker_;

// functions

void add_filter_future(std::future<filter_task_result_t>&& filter_future) {
  filter_futures_.push_back(std::move(filter_future));
}

auto get_prior_sum_source_list(int sum) {
  // NOTE: unncessary copy of primary source list introduced when splitting
  // this function out of cuda_alloc_copy_sources.
  auto src_list = (sum == 2)
      ? clue_manager::get_primary_unique_clue_source_list()
      : clue_manager::make_unique_clue_source_list(sum - 1);
  return src_list;
}

[[nodiscard]] auto cuda_alloc_copy_sources(
    const SourceCompatibilityList& src_list, const cudaStream_t stream) {
  // alloc sources
  const auto num_bytes = src_list.size() * sizeof(SourceCompatibilityData);
  SourceCompatibilityData* device_sources{};
  cuda_malloc_async((void**)&device_sources, num_bytes, stream,
      "filter sources");
  // copy sources
  auto err = cudaMemcpyAsync(device_sources, src_list.data(), num_bytes,
      cudaMemcpyHostToDevice, stream);
  assert_cuda_success(err, "copy filter sources");
  return device_sources;
}

void copy_prior_sum_sources_pointer(int sum,
    const SourceCompatibilityData* device_sources, cudaStream_t stream) {
  // copy pointer to constant memory
  const auto ptr_size = sizeof(SourceCompatibilityData*);
  const auto err = cudaMemcpyToSymbolAsync(sources_data, &device_sources,
      ptr_size, (sum - 1) * ptr_size, cudaMemcpyHostToDevice, stream);
  assert_cuda_success(err, "copy prior sum sources pointer");
}

auto cuda_alloc_copy_prior_sum_sources(int sum, cudaStream_t stream) {
  const auto& src_list = get_prior_sum_source_list(sum);
  const auto device_sources = cuda_alloc_copy_sources(src_list, stream);
  copy_prior_sum_sources_pointer(sum, device_sources, stream);
  return src_list.size();
}

//
  
auto get_unique_clue_source_descriptor(index_t src_idx) {
  return clue_manager::get_unique_clue_source(1, src_idx)
      .usedSources.get_source_descriptor();
}

auto make_source_descriptor_pairs(
    const CompatSourceIndicesSet& src_indices_set) {
  std::vector<SourceDescriptorPair> src_desc_pairs;
  src_desc_pairs.reserve(src_indices_set.size());
  for (const auto src_indices : src_indices_set) {
    // passing index() only as get_unique() only works for primary indices.
    src_desc_pairs.emplace_back(
        get_unique_clue_source_descriptor(src_indices.first().index()),
        get_unique_clue_source_descriptor(src_indices.second().index()));
  }
  return src_desc_pairs;
}

[[nodiscard]] auto cuda_alloc_copy_source_descriptor_pairs(
    const std::vector<SourceDescriptorPair>& src_desc_pairs,
    cudaStream_t stream) -> SourceDescriptorPair* {
  auto pairs_bytes = src_desc_pairs.size() * sizeof(SourceDescriptorPair);
  SourceDescriptorPair* device_src_desc_pairs{};
  cuda_malloc_async((void**)&device_src_desc_pairs, pairs_bytes, stream,
      "src_desc_pairs");
  auto err = cudaMemcpyAsync(device_src_desc_pairs, src_desc_pairs.data(),
      pairs_bytes, cudaMemcpyHostToDevice, stream);
  assert_cuda_success(err, "copy src_desc_pairs");
  return device_src_desc_pairs;
}

//

[[nodiscard]] auto cuda_alloc_copy_source_indices(
    const CompatSourceIndicesList& src_indices, const cudaStream_t stream) {
  // alloc source indices
  auto num_bytes = src_indices.size() * sizeof(CompatSourceIndices);
  CompatSourceIndices* device_src_indices{};
  cuda_malloc_async((void**)&device_src_indices, num_bytes, stream,
      "source indices");
  // copy source indices
  auto err = cudaMemcpyAsync(device_src_indices, src_indices.data(), num_bytes,
      cudaMemcpyHostToDevice, stream);
  assert_cuda_success(err, "copy source indices");
  return device_src_indices;
}

template <typename T = result_t>
void cuda_copy_results(
    std::vector<T>& results, T* device_results, cudaStream_t stream) {
  // sync the kernel
  CudaEvent event(stream);
  event.synchronize();
  // copy results
  auto num_bytes = results.size() * sizeof(T);
  auto err = cudaMemcpyAsync(results.data(), device_results, num_bytes,
      cudaMemcpyDeviceToHost, stream);
  assert_cuda_success(err, "copy filter results");
  // sync the memcpy
  event.record(stream);
  event.synchronize();
}

template <typename T = result_t>
auto cuda_copy_results(
    T* device_results, size_t num_results, cudaStream_t stream) {
  std::vector<T> results(num_results);
  cuda_copy_results(results, device_results, stream);
  return results;
}

auto get_compatible_sources_results(int sum,
    const CompatSourceIndices* device_src_indices, size_t num_src_indices,
    const SourceDescriptorPair* device_incompat_src_desc_pairs,
    size_t num_src_desc_pairs, cudaStream_t sync_stream, cudaStream_t stream) {
  CudaEvent alloc_event(stream);
  auto device_results = cuda_alloc_results(num_src_indices, stream,
      "get_compat_sources results");
  cuda_zero_results(device_results, num_src_indices, stream);
  CudaEvent start_event(stream);
  run_get_compatible_sources_kernel(device_src_indices, num_src_indices,
      device_incompat_src_desc_pairs, num_src_desc_pairs, device_results,
      sync_stream, stream);
  // probably sync always is correct/easiest thing to do here. sync'ing only
  // when logging is wrong, esp. if semaphore is introduced at calling site.
  CudaEvent stop_event(stream);
  stop_event.synchronize();
  if (log_level(Verbose)) {
    if (log_level(Ludicrous)) {
      auto results = cuda_copy_results(device_results, num_src_indices, stream);
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

void log_fill_indices(int sum, const FilterStream& stream,
    const IndexStates& idx_states, long duration) {
  if (log_level(Ludicrous)) {
    std::cerr << " " << sum << ": stream " << stream.stream_idx
              << " num_src_idx: " << stream.host.src_idx_list.size()
              << ", ready: " << stream.num_ready(idx_states) << ", filled in "
              << duration << "us" << std::endl;
  }
}

void log_filter_kernel(int sum, const FilterStream& stream,
    const std::vector<result_t>& results, unsigned num_compat,
    unsigned total_compat) {
  if (log_level(ExtraVerbose)) {
    auto kernel_duration = stream.kernel_stop.synchronize(stream.kernel_start);
    std::cerr << " " << sum << ": stream " << stream.stream_idx
              << " xor_kernel " << kernel_duration << "ms" << std::endl;

    if (log_level(Ludicrous)) {
      auto num_actual_compat = std::accumulate(results.begin(), results.end(),
          0, [](int sum, result_t r) { return r ? sum + 1 : sum; });
      std::cerr << " " << sum << ": stream " << stream.stream_idx
                << " compat results: " << num_compat
                << ", total: " << total_compat
                << ", actual: " << num_actual_compat << std::endl;
    }
  }
}

//
// This is the innermost filter kernel function wrapper.
//
// Divide device_src_indices into chunks, and run multiple filter_kernels
// concurrently on different streams, with each stream processing one
// chunk at a time.
//
// Returns a pair<int, int>, consisting of the number of processed sources
// and the number of compatible sources.
//
auto run_concurrent_filter_kernels(int sum, FilterSwarm& swarm,
    int threads_per_block, IndexStates& idx_states, result_t* device_results,
    std::vector<result_t>& results) {
  // would like to get rid of this for sure. only the first stream in the
  // swarm will auto-sync with any prior alloc/zero/copies on that stream.
  if (swarm.num_streams() > 1) {
    cudaStreamSynchronize(swarm.at(0).cuda_stream);
  }
  int total_processed{};
  int total_compat{};
  int stream_idx{-1};
  while (swarm.get_next_available(stream_idx)) {
    auto& stream = swarm.at(stream_idx);
    if (!stream.is_running) {
      // start kernel on stream
      auto t = util::Timer::start_timer();
      if (stream.fill_source_index_list(idx_states)) {
        t.stop();
        log_fill_indices(sum, stream, idx_states, t.microseconds());
        stream.alloc_copy_source_index_list();
        run_filter_kernel(threads_per_block, swarm.host.swarm_idx(), stream,
            device_results);
      }
    } else {
      // kernel on stream has finished
      stream.has_run = true;
      stream.is_running = false;
      cuda_copy_results(results, device_results, stream.cuda_stream);
      auto num_compat = idx_states.update(stream, results);
      total_compat += num_compat;
      total_processed += int(stream.host.src_idx_list.size());
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
    int num_results,
    const CompatSourceIndicesSet& incompat_src_indices,
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
  if (incompat_src_indices.size()) {
    std::cerr << "       incompat total: " << num_incompat_sources
              << ", unique(what?): " << incompat_src_indices.size() << std::endl;
  }
}

// This is the intermediate filter "task" function, which takes a device
// array of compat_src_indices to filter.
//
// Additional preparation and post-processing surrounding the call to
// run_xor_filter_task.
//
// TODO: THIS HAS CHANGED A BIT, UPDATE COMMENTS
//
// Preparation:
//
// * Determine the # of streams and stride, and create a FilterSwarm object.
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
std::pair<filter_task_result_t, std::optional<CompatSourceIndicesSet>>
filter_sources(FilterSwarm& swarm, const FilterParams& params,
    std::vector<IndexList>& idx_lists) {
  using namespace std::experimental::fundamentals_v3;
  const auto stream = swarm.at(0).cuda_stream;
  const auto num_streams = params.num_streams ? params.num_streams : 1;
  const auto num_src_lists = idx_lists.size();
  const auto num_results = num_src_lists;
  const auto stride = params.stride ? params.stride : int(num_src_lists);
  // NB: get num_src_lists, above, before moving idx_lists
  IndexStates idx_states{std::move(idx_lists)};
  // TDOO: FilterStreamData, for each stream
  auto device_results = cuda_alloc_results(num_results, stream,
      "filter results");
  scope_exit free_results{[device_results, stream]() {  //
    cuda_free_async(device_results, stream);
  }};
  // TODO: I'm not sure this is necessary
  cuda_zero_results(device_results, num_results, stream);
  if (params.synchronous || log_level(Verbose)) {
    std::cerr << " " << params.sum << ": src_lists: " << num_src_lists
              << ", streams: " << swarm.num_streams() << ", stride: " << stride
              << std::endl;
  }

#if 1
  cuda_memory_dump("before run_concurrent");
#endif

  // TODO: FilterStreamData::Host, resize_results()
  std::vector<result_t> results(num_results);
  // TODO: this is dumb
  swarm.init(num_streams, stride);
  auto t = util::Timer::start_timer();
  const auto [num_processed, num_compat] =  //
      run_concurrent_filter_kernels(params.sum, swarm, params.threads_per_block,
          idx_states, device_results, results);
  t.stop();
  CompatSourceIndicesSet incompat_src_indices;
  int num_incompat_sources{};
  const auto& candidates = get_candidates(params.sum);
  if (params.synchronous) {
    incompat_src_indices =
        idx_states.get_incompatible_sources(candidates, &num_incompat_sources);
  }
  log_filter_sources(params.sum, num_processed, num_compat, int(results.size()),
      incompat_src_indices, num_incompat_sources, t.count());
  return std::make_pair(get_compat_combos(candidates, results, num_processed),
      std::move(incompat_src_indices));
}

//

auto copy_prior_sum_sources(const FilterParams& params, cudaStream_t stream) {
  size_t num_sources{};
  if (params.copy_all_prior_sources) {
    // copy sources for all prior sums except immediately prior
    for (int sum{3}; sum < params.sum; ++sum) {
      num_sources += cuda_alloc_copy_prior_sum_sources(sum, stream);
      source_copy_tracker_.complete(sum);
    }
  }
  // copy sources for immediately prior sum
  num_sources += cuda_alloc_copy_prior_sum_sources(params.sum, stream);
  // indicate that our immediately prior sum source copy is initiated
  source_copy_tracker_.complete(params.sum);
  return num_sources;
}

// this is called after sum==2 filter_kernel is complete, and before any
// subsequent-sum kernel has started.
void set_incompat_sources(FilterData& mfd,
    const CompatSourceIndicesSet& incompat_src_indices, cudaStream_t stream) {
  // empty set technically possible; disallowed here as a canary
  assert(!incompat_src_indices.empty());
  assert(mfd.host_xor.incompat_src_desc_pairs.empty());
  assert(!mfd.device_xor.incompat_src_desc_pairs);

  mfd.host_xor.incompat_src_desc_pairs =
      std::move(make_source_descriptor_pairs(incompat_src_indices));
  mfd.device_xor.incompat_src_desc_pairs =
      cuda_alloc_copy_source_descriptor_pairs(
          mfd.host_xor.incompat_src_desc_pairs, stream);
}

void log_copy_sources(const FilterParams& params, size_t num_sources,
    const CudaEvent& copy_start, const CudaEvent& copy_stop) {
  if (log_level(Verbose)) {
    auto copy_duration = copy_stop.synchronize(copy_start);
    std::cerr << " " << params.sum << ": alloc_copy_sources: " << num_sources
              << " - " << copy_duration << "ms\n";
  }
}

void log_make_indices(const FilterParams& params,
    const std::vector<IndexList>& idx_lists,
    const CompatSourceIndicesList& src_indices, const util::Timer& make_timer) {
  if (log_level(Verbose)) {
    const auto num_src_indices = util::sum_sizes(idx_lists);
    std::cerr << " " << params.sum
              << ": make_variations_sorted_idx_lists: " << idx_lists.size()
              << " : " << num_src_indices << std::endl;
    std::cerr << " " << params.sum
              << ": make_compat_src_indices: " << src_indices.size()
              << " - combined " << make_timer.count() << "ms\n";
    assert((num_src_indices == src_indices.size()) && "src_indices mismatch");
  }
}

void log_copy_indices(const FilterParams& params, size_t num_src_indices,
    const CudaEvent& copy_start, const CudaEvent& copy_stop) {
  if ((params.synchronous && log_level(Normal))
      || (!params.synchronous && log_level(Verbose))) {
    auto copy_duration = copy_stop.synchronize(copy_start);
    auto free_mem = cuda_get_free_mem() / 1'000'000;
    std::cerr << " " << params.sum << ": alloc_copy_src_indices("
              << num_src_indices << ")" << ", free: " << free_mem << "MB"
              << " - " << copy_duration << "ms" << std::endl;
  }
}

// This is the outermost filter "task" function.
//
// Preparation prior to calling xor_filter_task:
//
// TOOD: changes have been made to this, related to unique_variations
//       and candidates containing compat_src_indices vs. sources.
//
// * Determine the candidate sources for the supplied sum.
// * Allocate and copy candidate sources.
// * If call is asynchronous (i.e. sum > 2), run the compat_sources_kernel to
//   populate a result_t array which represents src_list indices that are not
//   incompatible (i.e. compatible) with the incompat_sources that resulted
//   from the first synchronous call (sum = 2). This is a significant source
//   filtering optimization.
// * Allocate all FilterSwarmData
//
// Returns the result of filter_sources with no additional post-processing.
//
// NOTE: params copy by value is necessary for async launch.
//
filter_task_result_t filter_task(FilterData& mfd, FilterParams params) {
  using namespace std::experimental::fundamentals_v3;  // scope_exit
  // The logic in this function isn't segregated very well. we enter here on a
  // separate thread-per-sum. the goal of each thread is to perform the
  // following in order:
  //
  // 1. Initiate fulfillment of the device-data dependency (sources for prior
  //    sum) for subsequent-sum threads as quickly as possible, because they
  //    are all blocked until this data is available. All of these
  //    alloc/copies use the same stream shared across all threads, so that
  //    there is only one "dependent" stream to sync with before launching a
  //    kernel.

  auto stream = sources_stream_;
  CudaEvent copy_start(stream);
  auto num_sources = copy_prior_sum_sources(params, stream);
  CudaEvent copy_stop(stream);
  log_copy_sources(params, num_sources, copy_start, copy_stop);

  // 2. Perform as much CPU work as possible for data that is only used by
  // this
  //    sum's kernel, before blocking for one of the available stream swarms
  //    from the swarm pool. Once we take ownership of a swarm, we should be
  //    in GPU mode.

  // generate compat source indices.
  const auto& candidates = get_candidates(params.sum);
  auto t_make = util::Timer::start_timer();
  auto idx_lists = make_variations_sorted_idx_lists(candidates);
  auto src_indices = make_compat_source_indices(candidates, idx_lists);
  t_make.stop();
  log_make_indices(params, idx_lists, src_indices, t_make);

  // 3. Wait for all prior sum source copies to be initiated

  source_copy_tracker_.wait_for_all_prior(params.sum);

  // 4. Acquire swarm

  // Even though we don't really use the "swarm" feature until filter kernel,
  // acquire here in order to limit the number of concurrent kernels
  // (including get_compatible_results) to the # of swarms.
  auto& swarm = swarm_pool_.acquire();
  swarm.ensure_streams(1);

  //
  // NOTE: we switch streams here! weee! because reasons!
  //
  stream = swarm.at(0).cuda_stream;
  copy_start.record(stream);
  const auto device_src_indices = cuda_alloc_copy_source_indices(src_indices,
      stream);
  copy_stop.record(stream);
  log_copy_indices(params, src_indices.size(), copy_start, copy_stop);

  result_t* device_compat_src_results{};
  if (!params.synchronous) {
    device_compat_src_results = get_compatible_sources_results(params.sum,
        device_src_indices, src_indices.size(),
        mfd.device_xor.incompat_src_desc_pairs,
        mfd.host_xor.incompat_src_desc_pairs.size(), sources_stream_, stream);
    // proactively relieve some memory pressure 
    src_indices.clear();
    src_indices.shrink_to_fit();
  } else {
    // TOOD: hate this. the stream story is a mess here. fix.
    cudaStreamSynchronize(sources_stream_);
  }

  // TODO: magic, but also wrong-headed. we need to init every stream in the 
  // swarm. I would like swarm to call stream.init() when it constructs the
  // object, but unfortunately I need access to MFD.
  // stream here is questionable
  swarm.device.update(swarm.host.swarm_idx(), device_src_indices,
      device_compat_src_results, stream);
  swarm.at(0).init(mfd);

  auto filter_result = filter_sources(swarm, params, idx_lists);

  if (params.synchronous) {
    assert(filter_result.second.has_value());
    set_incompat_sources(mfd, filter_result.second.value(), sources_stream_);
  }
  swarm_pool_.release(swarm);

  // free host memory associated with candidates
  clear_candidates(params.sum);
  return std::move(filter_result.first);
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
  cuda_malloc_async((void**)&device.compat_indices, indices_bytes, stream,
      "or.compat_indices");
  auto err = cudaMemcpyAsync(device.compat_indices, host.compat_indices.data(),
      indices_bytes, cudaMemcpyHostToDevice, stream);
  assert_cuda_success(err, "copy or.compat_indices");
  device.num_compat_indices = int(host.compat_indices.size());
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

void alloc_copy_unique_variations(FilterData::HostCommon& host,
    FilterData::DeviceCommon& device, cudaStream_t stream,
    std::string_view tag) {
  device.unique_variations =
      alloc_copy_unique_variations(host.unique_variations, stream, tag);
  device.num_unique_variations = int(host.unique_variations.size());
}

void alloc_copy_common_filter_data(FilterData::HostCommon& host,
    FilterData::DeviceCommon& device, cudaStream_t stream,
    std::string_view tag) {
  alloc_copy_start_indices(host, device, stream);
  alloc_copy_compat_indices(host, device, stream);
  alloc_copy_unique_variations(host, device, stream,
      std::string(tag) + ".unique_variations");
}

void alloc_copy_filter_data(FilterData& mfd, cudaStream_t stream) {
  assert(!mfd.host_xor.compat_indices.empty()); // arbitrary
  util::LogDuration ld("alloc_copy_filter_data", Verbose);
  alloc_copy_common_filter_data(mfd.host_xor, mfd.device_xor, stream, "xor");
  if (!mfd.host_or.compat_indices.empty()) {
    alloc_copy_common_filter_data(mfd.host_or, mfd.device_or, stream, "or");
  } else {
    // TODO: cuda_free(), assuming we reset_pointers somewhere earlier.
    mfd.device_or.reset_pointers();
  }
}

}  // anonymous namespace

void filter_candidates_cuda(FilterData& mfd, const FilterParams& params) {
  if (params.synchronous) {
    std::promise<filter_task_result_t> p;
    p.set_value(filter_task(mfd, params));
    add_filter_future(p.get_future());
  } else {
    add_filter_future(std::async(std::launch::async, filter_task, std::ref(mfd),
        params));
  }
}

filter_result_t get_filter_result() {
  filter_result_t unique_combos;
  std::string result_str{"results: "};
  int total{-1};
  for (auto& fut : filter_futures_) {
    assert(fut.valid());
    const auto combos = fut.get();
    if (total > -1) {
      result_str.append(", ");
    } else {
      total = 0;
    }
    result_str.append(std::to_string(combos.size()));
    total += int(combos.size());
    unique_combos.insert(combos.begin(), combos.end());
  }
  std::cerr << result_str << ", total: " << total
            << ", unique: " << unique_combos.size() << std::endl;
  return unique_combos;
}

void filter_init(FilterData& mfd) {
  assert(!sources_stream_);
  auto err = cudaStreamCreate(&sources_stream_);
  assert_cuda_success(err, "create sources_stream");
  alloc_copy_filter_data(mfd, sources_stream_);
  copy_filter_data_to_symbols(mfd, sources_stream_);
}

void filter_cleanup() {
  auto err = cudaStreamDestroy(sources_stream_);
  assert_cuda_success(err, "destroy sources_stream");
  swarm_pool_.destroy_all_streams();
}

}  // namespace cm
