#include <numeric>
#include <string>
#include <unordered_set>
#include <vector>
#include <cuda_runtime.h>
#include "filter.h"
#include "candidates.h"

namespace {
  
using namespace cm;

static std::vector<std::future<filter_result_t>> filter_futures_;

void add_filter_future(std::future<filter_result_t>&& filter_future) {
  filter_futures_.emplace_back(std::move(filter_future));
}

auto count(const SourceCompatibilityLists& sources) {
  size_t num{};
  for (const auto& sourceList : sources) {
    num += sourceList.size();
  }
  return num;
}

auto* allocCopySources(const SourceCompatibilityLists& sources) {
  // alloc sources
  const cudaStream_t stream = cudaStreamPerThread;
  cudaError_t err = cudaSuccess;
  auto sources_bytes = count(sources) * sizeof(SourceCompatibilityData);
  SourceCompatibilityData* device_sources;
  err = cudaMallocAsync((void**)&device_sources, sources_bytes, stream);
  assert((err == cudaSuccess) && "alloc sources");

  // copy sources
  size_t index{};
  for (const auto& sourceList : sources) {
    err = cudaMemcpyAsync(&device_sources[index], sourceList.data(),
      sourceList.size() * sizeof(SourceCompatibilityData),
      cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
      fprintf(stdout, "copy sources, error: %s", cudaGetErrorString(err));
      throw std::runtime_error("copy sources");
    }
    index += sourceList.size();
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
  IndexStates& idx_states, const SourceCompatibilityData* device_sources,
  result_t* device_results, const index_t* device_list_start_indices,
  std::vector<result_t>& results) {
  //
  using namespace std::chrono;
  int total_compat{};
  int current_stream{-1};
  while (streams.get_next_available(current_stream)) {
    auto& stream = streams.at(current_stream);
    if (!stream.is_running) {
      if (!stream.fillSourceIndices(idx_states)) {
        continue;
      }
#if defined(LOGGING)
      std::cerr << "stream " << stream.stream_idx
                << " source_indices: " << stream.source_indices.size()
                << ", ready: " << stream.num_ready(idx_states)
                << std::endl;
#endif
      stream.allocCopy(idx_states);
      run_xor_kernel(stream, threads_per_block, device_sources,
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

#if defined(LOGGING)
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

std::unordered_set<std::string> get_compat_combos(
  const std::vector<result_t>& results,
  const OneSumCandidateData& candidate_data) {
  //
  std::unordered_set<std::string> compat_combos{};
  //int flat_index{};
  for (size_t i{}; i < candidate_data.sourceCompatLists.size(); ++i) {
    //const auto& src_list = candidate_data.sourceCompatLists.at(i);
    if (results.at(i)) {
      const auto& combos = candidate_data.indexComboListMap.at(i);
      compat_combos.insert(combos.begin(), combos.end());
    }
  }
  return compat_combos;
}

std::unordered_set<std::string> filter_task(
  int sum, int threads_per_block, int num_streams, int stride, int iters) {
  //
  num_streams = num_streams ? num_streams : 3;
  stride = stride ? stride : 1024;
  iters = iters ? iters : 1;

  using namespace std::chrono;
  [[maybe_unused]] cudaError_t err = cudaSuccess;
  err = cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 7'500'000);

  const auto& candidate_data = allSumsCandidateData.find(sum)->second;
  const auto& src_lists = candidate_data.sourceCompatLists;
  auto device_sources = allocCopySources(src_lists);
  IndexStates idx_states{src_lists};
  auto device_list_start_indices = allocCopyListStartIndices(idx_states);
  auto device_results = allocResults(src_lists.size());

  stride = std::min((int)src_lists.size(), stride);
  StreamSwarm streams(num_streams, stride);

#if defined(LOGGING)
  std::cerr << "sourcelists: " << src_lists.size() << ", streams: " << num_streams
            << ", stride: " << stride << std::endl;
#endif
  std::vector<result_t> results(src_lists.size());
  for(int i{}; i < iters; ++i) {
    streams.reset();
    idx_states.reset();
    zeroResults(device_results, src_lists.size(), cudaStreamPerThread);
    auto t0 = high_resolution_clock::now();

    auto total_compat = run_filter_task(streams, threads_per_block, idx_states,
      device_sources, device_results, device_list_start_indices, results);

    auto t1 = high_resolution_clock::now();
    auto d_total = duration_cast<milliseconds>(t1 - t0).count();
    std::cerr << "sum(" << sum << ")";
    if (iters > 1) {
      std::cerr << " iter: " << i;
    }
    std::cerr << " total compat: " << total_compat << " of " << src_lists.size()
              << " - " << d_total << "ms" << std::endl;
  }

#if 0
  err = cudaFree(device_sources);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device sources, error: %s\n",
      cudaGetErrorString(err));
    throw std::runtime_error("failed to free device sources");
  }
#endif
  return get_compat_combos(results, candidate_data);
}

// TODO: same as sum_sizes
auto countIndices(const VariationIndicesList& variationIndices) {
  return std::accumulate(variationIndices.begin(), variationIndices.end(), 0,
    [](int total, const auto& indices) {
      total += indices.size();
      return total;
    });
}

}  // anonymous namespace



namespace cm {

void filterCandidatesCuda(
  int sum, int threads_per_block, int num_streams, int stride, int iters) {
  add_filter_future(std::async(std::launch::async, filter_task, sum,
    threads_per_block, num_streams, stride, iters));
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
  // blast to device with one copy
  std::vector<device::OrSourceData> or_sources;
  for (unsigned arg_idx{}; arg_idx < orArgList.size(); ++arg_idx) {
    const auto& or_arg = orArgList.at(arg_idx);
    for (const auto& or_src: or_arg.orSourceList) {
      if (or_src.xorCompatible) {
        or_sources.emplace_back(device::OrSourceData{or_src.source, arg_idx});
      }
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

  return std::make_pair(device_or_sources, or_sources.size());
}

[[nodiscard]] auto cuda_allocCopySentenceVariationIndices(
  const SentenceVariationIndices& sentenceVariationIndices)
  -> device::VariationIndices* {
  //
  cudaError_t err = cudaSuccess;
  using DeviceVariationIndicesArray =
    std::array<device::VariationIndices, kNumSentences>;
  DeviceVariationIndicesArray deviceVariationIndicesArray;
  for (int s{}; s < kNumSentences; ++s) {
    auto& variation_indices = sentenceVariationIndices.at(s);
    // 2 * size to account for leading variation_offsets and num_src_indices
    const auto device_data_bytes =
      (countIndices(variation_indices) + (2 * variation_indices.size()))
      * sizeof(index_t);
    auto& deviceVariationIndices = deviceVariationIndicesArray.at(s);
    err = cudaMallocAsync((void**)&deviceVariationIndices.device_data,
      device_data_bytes, cudaStreamPerThread);
    assert(err == cudaSuccess);

    // const static int terminator = -1;
    std::vector<index_t> variation_offsets;
    std::vector<index_t> num_src_indices;
    const auto num_variations{variation_indices.size()};
    deviceVariationIndices.num_variations = num_variations;
    // copy src indices first, populating offsets and num_src_indices
    deviceVariationIndices.src_indices =
      &deviceVariationIndices.device_data[num_variations * 2];
    size_t offset{};
    int v{};
    for (const auto& indices : variation_indices) {
      variation_offsets.push_back(offset);
      num_src_indices.push_back(indices.size());
      std::cout << "sentence " << s << ", variation " << v++
                << ", indices: " << indices.size() << std::endl;
      // NOTE: Async. I'm going to need to preserve
      // sentenceVariationIndices until copy is complete - (kernel
      // execution/synchronize?)
      const auto indices_bytes = indices.size() * sizeof(index_t);
      err = cudaMemcpyAsync(&deviceVariationIndices.src_indices[offset],
        indices.data(), indices_bytes, cudaMemcpyHostToDevice,
        cudaStreamPerThread);
      assert(err == cudaSuccess);
      offset += indices.size();
#if 0
      err = cudaMemcpyAsync(&deviceVariationIndices.src_indices[offset],
        &terminator, sizeof(terminator), cudaMemcpyHostToDevice, cudaStreamPerThread);
      assert(err == cudaSuccess);
      offset += 1;
#endif
    }
    // copy variation offsets
    deviceVariationIndices.variation_offsets =
      deviceVariationIndices.device_data;
    const auto offsets_bytes = variation_offsets.size() * sizeof(index_t);
    err = cudaMemcpyAsync(deviceVariationIndices.variation_offsets,
      variation_offsets.data(), offsets_bytes, cudaMemcpyHostToDevice,
      cudaStreamPerThread);
    assert(err == cudaSuccess);
    // copy num src indices
    deviceVariationIndices.num_src_indices =
      &deviceVariationIndices.device_data[num_variations];
    const auto num_src_indices_bytes = num_src_indices.size() * sizeof(index_t);
    err = cudaMemcpyAsync(deviceVariationIndices.num_src_indices,
      num_src_indices.data(), num_src_indices_bytes, cudaMemcpyHostToDevice,
      cudaStreamPerThread);
    assert(err == cudaSuccess);
  }
  //  const auto sentenceVariationIndices_bytes =
  //    kNumSentences * sizeof(device::VariationIndices);
  const auto variation_indices_bytes =
    kNumSentences * sizeof(device::VariationIndices);
  device::VariationIndices* device_variation_indices;
  err = cudaMallocAsync((void**)&device_variation_indices,
    variation_indices_bytes, cudaStreamPerThread);
  assert(err == cudaSuccess);

  err = cudaMemcpyAsync(device_variation_indices,
    deviceVariationIndicesArray.data(), variation_indices_bytes,
    cudaMemcpyHostToDevice, cudaStreamPerThread);
  assert(err == cudaSuccess);

  // just to be sure, due to lifetime problems of local host-side memory
  err = cudaStreamSynchronize(cudaStreamPerThread);
  assert(err == cudaSuccess);
  return device_variation_indices;
}

// TODO: only required until all clues are converted to sentences
[[nodiscard]] index_t* cuda_allocCopyXorSourceIndices(
  const std::vector<index_t> xor_src_indices) {
  //
  cudaError_t err = cudaSuccess;
  const auto indices_bytes = xor_src_indices.size() * sizeof(index_t);
  index_t* device_xor_src_indices;
  err = cudaMallocAsync(
    (void**)&device_xor_src_indices, indices_bytes, cudaStreamPerThread);
  assert(err == cudaSuccess);

  err = cudaMemcpyAsync(device_xor_src_indices, xor_src_indices.data(),
    indices_bytes, cudaMemcpyHostToDevice, cudaStreamPerThread);
  assert(err == cudaSuccess);
  return device_xor_src_indices;
}

}  // namespace cm
