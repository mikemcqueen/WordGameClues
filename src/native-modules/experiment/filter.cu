// filter.cu

#include <algorithm>
#include <chrono>
#include <exception>
#include <future>
#include <limits>
#include <numeric>
#include <optional>
#include <thread>
#include <tuple>
#include <cuda_runtime.h>
#include "candidates.h"
#include "filter-types.h"

//#define STREAM_LOG

namespace {

using namespace cm;

template <typename T> __device__ __forceinline__ T load(const T* addr) {
  return *(const volatile T*)addr;
}

template <typename T> __device__ __forceinline__ void store(T* addr, T val) {
  *(volatile T*)addr = val;
}

__device__ __host__ auto isSourceXORCompatibleWithAnyXorSource(
  const SourceCompatibilityData& source, const XorSource* xorSources,
  size_t numXorSources) {
  bool compatible = true;
  for (size_t i{}; i < numXorSources; ++i) {
    compatible = source.isXorCompatibleWith(xorSources[i]);
    if (compatible) {
      break;
    }
  }
  return compatible;
}

#if 0
__device__ bool first = true;

__device__ void check_source(
  const SourceCompatibilityData& source, const device::OrSourceData* or_sources) {
  //
  if (source.usedSources.hasVariation(3)
      && (source.usedSources.getVariation(3) == 0)
      && source.usedSources.getBits().test(UsedSources::getFirstBitIndex(3))
      && source.legacySourceBits.test(1)) {
    printf("---match---\n");
    source.dump(nullptr, true);
    const auto& or_src = or_args[0].or_sources[0].source;
    bool xor_compat = source.isXorCompatibleWith(or_src);
    bool and_compat = source.isAndCompatibleWith(or_src);
    printf("xor: %d, and: %d\n", xor_compat, and_compat);
  }
}
#endif

__device__ bool is_source_or_compatibile(const SourceCompatibilityData& source,
  const unsigned num_or_args,
  const device::OrSourceData* __restrict__ or_sources,
  const unsigned num_or_sources) {
  //
  extern __shared__ volatile result_t or_arg_results[];

  // ASSUMPTION: # of --or args will always be smaller than block size.
  if (threadIdx.x < num_or_args) {
    // don't think store required here, even without volatile, because __sync
    or_arg_results[threadIdx.x] = (result_t)0;
  }
  __syncthreads();
  for (unsigned or_chunk{}; or_chunk * blockDim.x < num_or_sources;
       ++or_chunk) {
    //__syncthreads();
    const auto or_idx = or_chunk * blockDim.x + threadIdx.x;
    // TODO: if this works without sync in loop, i can possibly move this
    // conditional to loop definition 
    if (or_idx < num_or_sources) {
      const auto& or_src = or_sources[or_idx];
      if (source.isOrCompatibleWith(or_src.source)) {
        //store(&or_arg_results[or_src.or_arg_idx], (result_t)1);
        // don't think store required here, even without volatile, because __sync
        or_arg_results[or_src.or_arg_idx] = (result_t)1;
      }
    }
  }
  // i could safely initialize reduce_idx to 16 I think (max 32 --or args)
  for (int reduce_idx = blockDim.x / 2; reduce_idx > 0; reduce_idx /= 2) {
    __syncthreads();
    if ((threadIdx.x < reduce_idx) && (reduce_idx + threadIdx.x < num_or_args)) {
      // g++ has deprecated += on volatile destination;
      or_arg_results[threadIdx.x] =
        or_arg_results[threadIdx.x] + or_arg_results[reduce_idx + threadIdx.x];
    }
  }
  if (!threadIdx.x) {
    const auto compat_with_all = or_arg_results[threadIdx.x] == num_or_args;
    return compat_with_all;
  }
  return false;
}

__device__ bool is_source_xor_or_compatible(
  const SourceCompatibilityData& source,
  const SourceCompatibilityData* __restrict__ xor_sources,
  const unsigned num_xor_sources, const unsigned num_or_args,
  const device::OrSourceData* __restrict__ or_sources,
  const unsigned num_or_sources) {
  //
  __shared__ bool is_xor_compat;
  __shared__ bool is_or_compat;

  if (!threadIdx.x) {
    store(&is_xor_compat, false);
    store(&is_or_compat, false);
  }
  // for each xor_source (one thread per xor_source)
  for (unsigned xor_chunk{}; xor_chunk * blockDim.x < num_xor_sources;
       ++xor_chunk) {
    __syncthreads();
    const auto xor_idx = xor_chunk * blockDim.x + threadIdx.x;
    if (xor_idx < num_xor_sources) {
      if (source.isXorCompatibleWith(xor_sources[xor_idx])) {
        // TODO: Do I need volatile when read/writing shrd mem?
        // TODO: unnecessary store due to __sync
        store(&is_xor_compat, true);
      }
    }
    __syncthreads();
    // if source is not XOR compatible with any --xor sources
    // TODO: unnecessary load due to __sync
    if (!load(&is_xor_compat)) {
      continue;
    }
    if (num_or_args > 0) {
      // source must also be OR compatible with at least one source
      // of each or_arg
      if (is_source_or_compatibile(
            source, num_or_args, or_sources, num_or_sources)) {
        // TODO: unnecessary store due to __sync
        store(&is_or_compat, true);
      }
      __syncthreads();
      // TODO: unnecessary load due to __sync
      if (!load(&is_or_compat)) {
        if (!threadIdx.x) {
          // reset is_xor_compat. sync will happen at loop entrance.
          // TODO: unnecessary store due to __sync
          store(&is_xor_compat, false);
        }
        continue;
      }
    }
    return true;
  }
  return false;
}

// one block per source
__global__ void xor_kernel_new(
  const SourceCompatibilityData* __restrict__ sources,
  const unsigned num_sources,
  const SourceCompatibilityData* __restrict__ xor_sources,
  const unsigned num_xor_sources, const unsigned num_or_args,
  const device::OrSourceData* __restrict__ or_sources,
  const unsigned num_or_sources, const SourceIndex* __restrict__ source_indices,
  const index_t* __restrict__ list_start_indices, result_t* results,
  int stream_idx) {
  //
  //  extern __shared__ result_t or_arg_results[];
  // for each source (one block per source)
  for (unsigned idx{blockIdx.x}; idx < num_sources; idx += gridDim.x) {
    const auto src_idx = source_indices[idx];
    const auto flat_index =
      list_start_indices[src_idx.listIndex] + src_idx.index;
    auto& source = sources[flat_index];
    auto& result = results[src_idx.listIndex];

    __syncthreads();
    if (is_source_xor_or_compatible(source, xor_sources, num_xor_sources,
          num_or_args, or_sources, num_or_sources)) {
      // check_source(source, or_args);
      if (!threadIdx.x) {
        store(&result, (result_t)1);
      }
    }
  }
}

void run_xor_kernel(StreamData& stream, int threads_per_block,
  const SourceCompatibilityData* device_sources, result_t* device_results,
  const index_t* device_list_start_indices) {
  //
  int num_sm;
  cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0);
  auto threads_per_sm = 2048;
  auto block_size = threads_per_block ? threads_per_block : 1024;
  auto blocks_per_sm = threads_per_sm / block_size;
  //  assert(blocks_per_sm * block_size == threads_per_sm);
  auto grid_size = num_sm * blocks_per_sm;  // aka blocks per grid
  auto shared_bytes = PCD.orArgList.size() * sizeof(result_t);
  // enforce assumption in is_source_or_compatible()
  assert(PCD.orArgList.size() < block_size);

  stream.is_running = true;
  stream.sequence_num = StreamData::next_sequence_num();
  stream.start_time = std::chrono::high_resolution_clock::now();
  dim3 grid_dim(grid_size);
  dim3 block_dim(block_size);
  cudaStreamSynchronize(cudaStreamPerThread);
  xor_kernel_new<<<grid_dim, block_dim, shared_bytes, stream.cuda_stream>>>(
    device_sources, stream.source_indices.size(), PCD.device_xorSources,
    PCD.xorSourceList.size(), PCD.orArgList.size(), PCD.device_or_sources,
    PCD.num_or_sources, stream.device_source_indices, device_list_start_indices,
    device_results, stream.stream_idx);

#if 0 || defined(STREAM_LOG)
  std::cerr << "stream " << stream.stream_idx
            << " started with " << grid_size << " blocks"
            << " of " << block_size << " threads"
          //<< " starting, sequence: " << stream.sequence_num
            << std::endl;
#endif
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

// TODO: stream.get_results()? just to call out to free function?  maybe
/*
auto getKernelResults(StreamData& stream, result_t* device_results) {
  return copy_device_results(
    device_results, stream.num_src_lists, stream.stream);
}
*/

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

auto flat_index(
  const SourceCompatibilityLists& sources, const SourceIndex src_idx) {
  uint32_t flat{};
  for (size_t i{}; i < src_idx.listIndex; ++i) {
    flat += sources.at(i).size();
  }
  return flat + src_idx.index;
}

void check(
  const SourceCompatibilityLists& sources, index_t list_index, index_t index) {
  constexpr const auto logging = true;
  if constexpr (logging) {
    SourceIndex src_idx{list_index, index};
    char idx_buf[32];
    char buf[64];
    snprintf(buf, sizeof(buf), "%s, flat: %d", src_idx.as_string(idx_buf),
      flat_index(sources, src_idx));
    auto& source = sources.at(list_index).at(index);
    source.dump(buf);
    //int compat_index{-1};
    auto compat = isSourceXORCompatibleWithAnyXorSource(
      source, PCD.xorSourceList.data(), PCD.xorSourceList.size());
    std::cerr << "compat: " << compat
              << std::endl;  //<< " (" << compat_index << ")"
  }
}

auto makeCompatibleSources(const SourceList& sources) {
  std::vector<SourceCompatibilityData> compat_sources;
  for (const auto& src : sources) {
    compat_sources.push_back(src);
  }
  return compat_sources;
}

auto countIndices(const VariationIndicesList& variationIndices) {
  return std::accumulate(variationIndices.begin(), variationIndices.end(), 0,
    [](int total, const auto& indices) {
      total += indices.size();
      return total;
    });
}

void dump_xor(int index) {
  const XorSourceList& xorSources = PCD.xorSourceList;
  auto host_index = index;
  const auto& src = xorSources.at(host_index);
  char buf[32];
  snprintf(buf, sizeof(buf), "xor: device(%d) host(%d)", index, host_index);
  src.dump(buf);
}

int run_filter_task(StreamSwarm& streams, int threads_per_block,
  IndexStates& idx_states, const SourceCompatibilityData* device_sources,
  result_t* device_results, const index_t* device_list_start_indices,
  std::vector<result_t>& results) {
  //
  using namespace std::chrono;
  int total_compat{};
  int current_stream{-1};
  int actual_num_compat{};
  while (streams.get_next_available(current_stream)) {
    auto& stream = streams.at(current_stream);
    if (!stream.is_running) {
      if (!stream.fillSourceIndices(idx_states)) {
        continue;
      }
#if 0
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
    auto d_kernel = duration_cast<milliseconds>(k1 - stream.start_time).count();

    auto num_compat =
      idx_states.update(stream.source_indices, results, stream.stream_idx);
    total_compat += num_compat;

#if 0
    auto num_actual_compat = std::accumulate(results.begin(), results.end(), 0,
      [](int sum, result_t r) { return r ? sum + 1 : sum; });
    std::cerr << " stream " << stream.stream_idx
              << " compat results: " << num_compat
              << ", total: " << total_compat
              << ", actual: " << num_actual_compat
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
  int flat_index{};
  for (size_t i{}; i < candidate_data.sourceCompatLists.size(); ++i) {
    const auto& src_list = candidate_data.sourceCompatLists.at(i);
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
  //  StreamData::init(kernels, stride);

#if 0
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

static std::vector<std::future<filter_result_t>> filter_futures_;

void add_filter_future(std::future<filter_result_t>&& filter_future) {
  filter_futures_.emplace_back(std::move(filter_future));
}

}  // anonymous namespace

namespace cm {

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

void filterCandidatesCuda(
  int sum, int threads_per_block, int num_streams, int stride, int iters) {
  add_filter_future(std::async(std::launch::async, filter_task, sum,
    threads_per_block, num_streams, stride, iters));
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
    auto& variationIndices = sentenceVariationIndices.at(s);
    // 2 * size to account for one -1 indices terminator per variation
    const auto device_data_bytes =
      (countIndices(variationIndices) + (2 * variationIndices.size()))
      * sizeof(int);
    auto& deviceVariationIndices = deviceVariationIndicesArray.at(s);
    err = cudaMalloc(
      (void**)&deviceVariationIndices.device_data, device_data_bytes);
    assert(err == cudaSuccess);

    const static int terminator = -1;
    std::vector<int> variationOffsets;
    const auto num_variations{variationIndices.size()};
    deviceVariationIndices.variationOffsets =
      deviceVariationIndices.device_data;
    deviceVariationIndices.num_variations = num_variations;
    deviceVariationIndices.sourceIndices =
      &deviceVariationIndices.device_data[num_variations];
    size_t offset{};
    for (const auto& indices : variationIndices) {
      variationOffsets.push_back(offset);
      // NOTE: Async. I'm going to need to preserve sentenceVariationIndices
      // until copy is complete - (kernel execution/synhronize?)
      const auto indices_bytes = indices.size() * sizeof(int);
      err = cudaMemcpyAsync(&deviceVariationIndices.sourceIndices[offset],
        indices.data(), indices_bytes, cudaMemcpyHostToDevice);
      assert(err == cudaSuccess);
      offset += indices.size();
      err = cudaMemcpyAsync(&deviceVariationIndices.sourceIndices[offset],
        &terminator, sizeof(terminator), cudaMemcpyHostToDevice);
      assert(err == cudaSuccess);
      offset += 1;
    }
    const auto variationOffsets_bytes = variationOffsets.size() * sizeof(int);
    err = cudaMemcpyAsync(deviceVariationIndices.variationOffsets,
      variationOffsets.data(), variationOffsets_bytes, cudaMemcpyHostToDevice);
    assert(err == cudaSuccess);
  }
  //  const auto sentenceVariationIndices_bytes =
  //    kNumSentences * sizeof(device::VariationIndices);
  const auto variationIndices_bytes =
    kNumSentences * sizeof(device::VariationIndices);
  device::VariationIndices* device_variationIndices;
  err = cudaMalloc((void**)&device_variationIndices, variationIndices_bytes);
  assert(err == cudaSuccess);

  err =
    cudaMemcpyAsync(device_variationIndices, deviceVariationIndicesArray.data(),
      variationIndices_bytes, cudaMemcpyHostToDevice);
  assert(err == cudaSuccess);

  return device_variationIndices;
}

}  // namespace cm
