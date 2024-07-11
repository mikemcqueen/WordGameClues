// filter-support.cpp

#include <experimental/scope>
#include <future>
#include <numeric>
#include <optional>
#include <semaphore>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include "filter.h"
#include "merge-filter-common.h"
#include "merge-filter-data.h"
#include "candidates.h"
#include "cuda-types.h"
#include "log.h"

namespace {
  
using namespace cm;

using filter_task_result_t = std::pair<std::unordered_set<std::string>,
    std::optional<SourceCompatibilitySet>>;

// global
inline std::vector<std::future<filter_task_result_t>> filter_futures_;

void add_filter_future(std::future<filter_task_result_t>&& filter_future) {
  filter_futures_.emplace_back(std::move(filter_future));
}

[[nodiscard]] auto cuda_alloc_copy_sources(const CandidateList& candidates,
    int num_sources, const cudaStream_t stream = cudaStreamPerThread) {
  // alloc sources
  auto sources_bytes = num_sources * sizeof(SourceCompatibilityData);
  SourceCompatibilityData* device_sources;
  cudaError_t err{};
  //  auto err = cudaMallocAsync((void**)&device_sources, sources_bytes, stream);
  //  assert_cuda_success(err, "alloc sources");
  cuda_malloc_async((void**)&device_sources, sources_bytes, stream, "filter sources");
  // copy sources
  size_t index{};
  for (const auto& candidate : candidates) {
    const auto& src_list = candidate.src_list_cref.get();
    err = cudaMemcpyAsync(&device_sources[index], src_list.data(),
        src_list.size() * sizeof(SourceCompatibilityData),
        cudaMemcpyHostToDevice, stream);
    assert_cuda_success(err, "copy sources");
    index += src_list.size();
  }
  return device_sources;
}

[[nodiscard]] auto cuda_alloc_copy_sources(const CandidateList& candidates,
    const cudaStream_t stream = cudaStreamPerThread) {
  return cuda_alloc_copy_sources(
      candidates, count_candidates(candidates), stream);
}

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
  //auto err = cudaMallocAsync((void**)&device_indices, indices_bytes, stream);
  //assert_cuda_success(err, "alloc list start indices");
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
  auto device_results = cuda_alloc_results<compat_src_result_t>(
      num_sources, stream, "get_compatible_sources results");
  cuda_zero_results<compat_src_result_t>(device_results, num_sources);
  CudaEvent start_event;
  run_get_compatible_sources_kernel(device_sources, num_sources,
    device_incompatible_src_desc_pairs, num_src_desc_pairs, device_results);
  // probably sync always is correct/easiest thing to do here. sync'ing only
  // when logging is wrong, esp. if semaphore is introduced at calling site.
  CudaEvent stop_event;
  stop_event.synchronize();
  if (log_level(Verbose)) {
    if (log_level(Ludicrous)) {
      auto results =
          cuda_copy_results<compat_src_result_t>(device_results, num_sources);
      auto num_compat = std::accumulate(results.begin(), results.end(), 0u,
          [](unsigned sum, compat_src_result_t r) { return sum + r; });
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
  for (const auto& or_arg: mfd.host.or_arg_list) {
    bool compat_arg{};
    for (const auto& or_src: or_arg.or_src_list) {
      // TODO: ignoring "marked" is_xor_compat flag for now
      if (or_src.src.isOrCompatibleWith(source)) {
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
  const auto num_or_args = mfd.host.or_arg_list.size();
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

void xor_filter(int sum, const StreamData& stream, const MergeFilterData& mfd) {
  // const IndexStates& idx_states
  const auto& all_candidates = allSumsCandidateData;
  const auto it = all_candidates.find(sum);
  assert((it != all_candidates.end()) && "no candidates for sum");
  const auto& candidates = it->second;
  std::vector<result_t> results(candidates.size());
  const auto& xor_src_lists = mfd.host.xor_src_lists;
  std::cerr << "HOST xor_src_lists: " << xor_src_lists.size()
            << ", xor_sources: " << xor_src_lists.at(0).size() << std::endl;
  size_t num_compat{};
  size_t num_xor_compat{};
  for (auto src_idx : stream.src_indices) {
    const auto& src_list = candidates.at(src_idx.listIndex).src_list_cref.get();
    const auto& source = src_list.at(src_idx.index);
    if (is_XOR_compatible(source, xor_src_lists)) {
      ++num_xor_compat;
      if (is_OR_compatible(source, mfd)) {
        ++num_compat;
      }
    }
  }
  std::cerr << "HOST compatible sources " << num_compat << " of "
            << stream.src_indices.size() << std::endl;
  std::cerr << "HOST xor-compatible sources: " << num_xor_compat << std::endl;
  show_incompat_or_args(mfd.host.or_arg_list.size());
}

}  // namespace host

void log_fill(int sum, const StreamData& stream, const IndexStates& idx_states,
    long duration) {
  if (log_level(Ludicrous)) {
    std::cerr << " " << sum << ": stream " << stream.stream_idx
              << " source_indices: " << stream.src_indices.size()
              << ", ready: " << stream.num_ready(idx_states) << ", filled in "
              << duration << "us" << std::endl;
  }
}

void log_xor_kernel(int sum, const StreamData& stream,
    const std::vector<result_t>& results, unsigned num_compat,
    unsigned total_compat) {
  if (log_level(ExtraVerbose)) {
    auto kernel_duration = stream.kernel_stop.synchronize(stream.kernel_start);
    std::cerr << " " << sum << ": stream " << stream.stream_idx
              << " xor_kernel took " << kernel_duration << "ms" << std::endl;
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
// This is the innermost filter "task" function.
//
// Divide device_sources up into chunks, and run multiple xor_kernels
// concurrently on different streams, with each stream processing one
// chunk at a time.
//
// Returns a pair<int, int>, consisting of the number of processed sources
// and the number of xor-compatible sources. (TODO: compatible with what)
//
auto run_xor_filter_task(int sum, StreamSwarm& streams, int threads_per_block,
    IndexStates& idx_states, const MergeFilterData& mfd,
    const SourceCompatibilityData* device_sources,
    const compat_src_result_t* device_compat_src_results,
    result_t* device_results, const index_t* device_start_indices,
    std::vector<result_t>& results) {
  using namespace std::chrono;
  int num_processed{};
  int total_compat{};
  int current_stream{-1};
  while (streams.get_next_available(current_stream)) {
    auto& stream = streams.at(current_stream);
    if (!stream.is_running) {
      auto t = util::Timer::start_timer();
      if (!stream.fill_source_indices(idx_states)) {
        continue;
      }
      t.stop();
      log_fill(sum, stream, idx_states, t.microseconds());
      // stream.copy_sources_start.record();
      stream.alloc_copy_source_indices(idx_states);
      // stream.copy_sources_stop.record();
      run_xor_kernel(stream, threads_per_block, mfd, device_sources,
          device_compat_src_results, device_results, device_start_indices);
      continue;
    }
    stream.has_run = true;
    stream.is_running = false;
    cuda_copy_results(results, device_results, stream.cuda_stream);
    auto num_compat =
        idx_states.update(stream.src_indices, results, stream.stream_idx);
    total_compat += num_compat;
    num_processed += stream.src_indices.size();
    log_xor_kernel(sum, stream, results, num_compat, total_compat);
    if (log_level(Ludicrous)) {
      host::xor_filter(sum, stream, mfd);
    }
    // FIXMENOW debug: stop after first kernel
    // break;
  }
  return std::make_pair(num_processed, total_compat);
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

void log_xor_filter_task(int sum, int num_processed, int num_compat,
    int num_results, const SourceCompatibilitySet& incompat_sources,
    int num_incompat_sources, std::chrono::milliseconds::rep duration_ms) {
  auto num_src_lists = std::min(num_processed, num_results);
  std::cerr << "sum(" << sum << ")"
            << " compatible src_lists: " << num_compat << " of "
            << num_src_lists << ", sources processed: " << num_processed;
  if (incompat_sources.size()) {
    std::cerr << ", incompat total: " << num_incompat_sources
              << ", unique: " << incompat_sources.size();
  }
  std::cerr << " - " << duration_ms << "ms" << std::endl;
}

//
// This is the intermediate filter "task" function.
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
// * If call is synchronous (i.e., sum == 2), populate the
// incompatible_sources
//   set. This set is used to speed up subsequent async sums.
// * Create a set of compatible combos from compatible candidate sourcelists.
//
// Returns a pair<compat_combo_string_set, incompat_sources_set>
//
filter_task_result_t xor_filter_task(const MergeFilterData& mfd,
    const SourceCompatibilityData* device_sources,
    const compat_src_result_t* device_compat_src_results,
    const CandidateList& candidates, int sum, int threads_per_block,
    int num_streams, int stride, int iters, bool synchronous) {
  using namespace std::experimental::fundamentals_v3;
  using namespace std::chrono;
  // err = cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 7'500'000);

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
  if ((synchronous && log_level(Normal))
      || (!synchronous && log_level(Verbose))) {
    std::cerr << " " << sum << ": src_lists: " << candidates.size()
              << ", streams: " << num_streams << ", stride: " << stride
              << std::endl;
  }
  auto t = util::Timer::start_timer();
  auto [num_processed, num_compat] = run_xor_filter_task(sum, streams,
      threads_per_block, idx_states, mfd, device_sources,
      device_compat_src_results, device_results, device_start_indices, results);
  t.stop();
  cuda_memory_dump();
  SourceCompatibilitySet incompat_sources;
  int num_incompat_sources{};
  if (synchronous) {
    num_incompat_sources =
        idx_states.get_incompatible_sources(candidates, incompat_sources);
  }
  log_xor_filter_task(sum, num_processed, num_compat, results.size(),
      incompat_sources, num_incompat_sources, t.count());

  return std::make_pair(get_compat_combos(candidates, results, num_processed),
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
// * If call is not synchronous (i.e. sum > 2), run the compat_sources_kernel
//   in order to determine which of the sources are compatible with the set of
//   incompatible_sources that resulted from the first synchronous call
//   (sum=2). (TODO: fudgy language, add why is this done).
//
// Returns the result of xor_filter_task with no additional post-processing.
//
auto filter_task(const MergeFilterData& mfd, int sum, int threads_per_block,
    int num_streams, int stride, int iters,  // CandidateMap& all_candidates,
    bool synchronous = false) {
  using namespace std::chrono;
  using namespace std::experimental::fundamentals_v3;
  auto& all_candidates = allSumsCandidateData;
  auto it = all_candidates.find(sum);
  assert((it != all_candidates.end()) && "no candidates for sum");
  auto& candidates = it->second;
  scope_exit free_candidates{[sum, candidates]() mutable {
    candidates.clear();
    clear_candidates(sum);
  }};
  auto num_sources = count_candidates(candidates);
  CudaEvent copy_start;
  auto device_sources = cuda_alloc_copy_sources(candidates, num_sources);
  scope_exit free_sources{[device_sources]() { cuda_free(device_sources); }};
  log_copy_sources(sum, num_sources, synchronous, copy_start);
  compat_src_result_t* device_compat_src_results{};
  scope_exit free_results{[device_compat_src_results]() {
    if (device_compat_src_results)
      cuda_free(device_compat_src_results);
  }};
  if (!synchronous) {
    // static std::counting_semaphore<2> compat_src_semaphore(2);
    // compat_src_semaphore.acquire();
    device_compat_src_results = get_compatible_sources_results(sum,
        device_sources, num_sources, mfd.device.incompatible_src_desc_pairs,
        mfd.device.num_incompatible_sources);
    // compat_src_semaphore.release();
  }
  // static std::counting_semaphore<2> filter_semaphore(4);
  // filter_semaphore.acquire();
  auto filter_result = xor_filter_task(mfd, device_sources,
      device_compat_src_results, candidates, sum, threads_per_block,
      num_streams, stride, iters, synchronous);
  // filter_semaphore.release();
  return filter_result;
}

auto async_filter_task(const MergeFilterData& mfd, int sum,
    int threads_per_block, int num_streams, int stride, int iters) {
  static std::counting_semaphore<2> semaphore(2);  // TODO: not great
  semaphore.acquire();
  auto filter_result =
      filter_task(mfd, sum, threads_per_block, num_streams, stride, iters);
  semaphore.release();
  return filter_result;
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
        iters, /*allSumsCandidateData,*/ true);
    auto opt_incompatible_sources = std::move(result.second.value());
    result.second.reset();
    p.set_value(result);
    add_filter_future(p.get_future());
    return opt_incompatible_sources;
  } else {
    add_filter_future(std::async(std::launch::async, filter_task, mfd, sum,
        threads_per_block, num_streams, stride, iters, false));
    return std::nullopt;
  }
}

filter_result_t get_filter_result(const MergeFilterData& mfd) {
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
  if (log_level(Ludicrous)) {
    show_or_arg_counts(mfd.host.or_arg_list.size());
  }
  return unique_combos;
}

auto cuda_markAllXorCompatibleOrSources(const MergeFilterData& mfd)
    -> std::vector<result_t> {
  // alloc/zero results
  const auto stream = cudaStreamPerThread;
  auto device_results = cuda_alloc_results(mfd.device.num_or_sources, stream,
      "mark_or_sources results");  // cl-format
  cuda_zero_results(device_results, mfd.device.num_or_sources);
  run_mark_or_sources_kernel(mfd, device_results);
  auto results = cuda_copy_results(device_results, mfd.device.num_or_sources);
  if (log_level(Verbose)) {
    auto num_marked = std::accumulate(results.begin(), results.end(), 0,
        [](int total, result_t val) { return total + val; });
    std::cerr << " marked " << num_marked << " of " << mfd.device.num_or_sources
              << " or sources" << std::endl;
  }
  return results;
}

// This "compacts" the device_or_src_list by moving all "compatible" sources
// to the front of the array, and returns the compatible source count.
unsigned move_marked_or_sources(device::OrSourceData* device_or_src_list,
    const std::vector<result_t>& mark_results) {
  size_t dst_idx{};
  size_t src_idx{};
  // skip over any marked (and therefore correctly placed) results at beginning
  for (; (src_idx < mark_results.size()) && mark_results[src_idx];
       ++src_idx, ++dst_idx)
    ;
  // move any remaining marked (and therefore incorrectly placed) results
  for (; src_idx < mark_results.size(); ++src_idx) {
    if (mark_results[src_idx]) {
      auto err = cudaMemcpyAsync(&device_or_src_list[dst_idx++],
          &device_or_src_list[src_idx], sizeof(device::OrSourceData),
          cudaMemcpyDeviceToDevice);
      assert_cuda_success(err, "move_marked_or_sources memcpy");
    }
  }
  return dst_idx;
}

[[nodiscard]] UsedSources::SourceDescriptorPair*
cuda_alloc_copy_source_descriptor_pairs(
    const std::vector<UsedSources::SourceDescriptorPair>& src_desc_pairs) {
  auto pairs_bytes =
      src_desc_pairs.size() * sizeof(UsedSources::SourceDescriptorPair);
  UsedSources::SourceDescriptorPair* device_src_desc_pairs{};
  const auto stream = cudaStreamPerThread;
  cudaError_t err{};
  // cudaError_t err = cudaMallocAsync(
  //    (void**)&device_src_desc_pairs, pairs_bytes, stream);
  // assert_cuda_success(err, "alloc src_desc_pairs");
  cuda_malloc_async((void**)&device_src_desc_pairs, pairs_bytes, stream,
      "src_desc_pairs");  // cl-format
  err = cudaMemcpyAsync(device_src_desc_pairs, src_desc_pairs.data(),
      pairs_bytes, cudaMemcpyHostToDevice, stream);
  assert_cuda_success(err, "copy src_desc_pairs");
  return device_src_desc_pairs;
}

[[nodiscard]] std::pair<device::OrSourceData*, unsigned>
cuda_allocCopyOrSources(const OrArgList& orArgList) {
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
    for (const auto& or_src : or_arg.or_src_list) {
      or_src_list.emplace_back(device::OrSourceData{or_src.src, arg_idx});
    }
  }
  const auto or_src_bytes = or_src_list.size() * sizeof(device::OrSourceData);
  device::OrSourceData* device_or_src_list;
  const auto stream = cudaStreamPerThread;
  cudaError_t err{};
  // auto err = cudaMallocAsync(
  //     (void**)&device_or_src_list, or_src_bytes, stream);
  // assert_cuda_success(err, "alloc or_sources");
  cuda_malloc_async((void**)&device_or_src_list, or_src_bytes, stream,
      "filter or_src_list");  // cl-format
  err = cudaMemcpyAsync(device_or_src_list, or_src_list.data(), or_src_bytes,
      cudaMemcpyHostToDevice, stream);
  assert_cuda_success(err, "copy or_src_list");
  return std::make_pair(device_or_src_list, or_src_list.size());
}

[[nodiscard]] auto cuda_allocCopySentenceVariationIndices(
    const SentenceVariationIndices& sentenceVariationIndices,
    cudaStream_t stream) -> device::VariationIndices* {
  // assumption made throughout
  static_assert(sizeof(combo_index_t) == sizeof(index_t) * 2);
  using DeviceVariationIndicesArray =
      std::array<device::VariationIndices, kNumSentences>;
  DeviceVariationIndicesArray device_indices_array;
  cudaError_t err{};
  for (int s{}; s < kNumSentences; ++s) {
    auto& variation_indices = sentenceVariationIndices.at(s);
    const auto num_variations{variation_indices.size()};
    // clever: n * sizeof(combo_index_t) == 2 * n * sizeof(index_t)
    const auto device_data_bytes =
        (countIndices(variation_indices) + num_variations)
        * sizeof(combo_index_t);
    auto& device_indices = device_indices_array.at(s);
    // auto err = cudaMallocAsync((void**)&device_indices.device_data,
    //     device_data_bytes, stream);
    // assert_cuda_success(err, "alloc device data");
    cuda_malloc_async((void**)&device_indices.device_data, device_data_bytes,
        stream, "variation_indices");
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
          stream);
      assert_cuda_success(err, "copy combo_indices");
      offset += combo_indices.size();
    }
    // copy variation offsets
    device_indices.variation_offsets = (index_t*)device_indices.device_data;
    const auto offsets_bytes = variation_offsets.size() * sizeof(index_t);
    err = cudaMemcpyAsync(device_indices.variation_offsets,
        variation_offsets.data(), offsets_bytes, cudaMemcpyHostToDevice,
        stream);
    assert_cuda_success(err, "copy variation_offsets");
    // copy num combo indices
    device_indices.num_combo_indices =
      &((index_t*)device_indices.device_data)[num_variations];
    const auto num_indices_bytes = num_combo_indices.size() * sizeof(index_t);
    err = cudaMemcpyAsync(device_indices.num_combo_indices,
        num_combo_indices.data(), num_indices_bytes, cudaMemcpyHostToDevice,
        stream);
    assert_cuda_success(err, "copy combo_indices");
  }
  const auto variation_indices_bytes =
      kNumSentences * sizeof(device::VariationIndices);
  device::VariationIndices* device_variation_indices;
  // auto err = cudaMallocAsync((void**)&device_variation_indices,
  //     variation_indices_bytes, stream);
  // assert_cuda_success(err, "alloc variation_indices");
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

}  // namespace cm
