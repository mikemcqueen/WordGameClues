// filter.cu

#include <algorithm>
#include <chrono>
#include <exception>
#include <numeric>
#include <cuda_runtime.h>
#include "candidates.h"

namespace {
  using namespace cm;

  /*
__device__ auto isSourceORCompatibleWithAnyOrSource(
  const SourceCompatibilityData& source, const OrSourceList& orSourceList)
{
  auto compatible = false;
  for (const auto& orSource : orSourceList) {
    // skip any sources that were already determined to be XOR incompatible
    // or AND compatible with --xor sources.
    if (!orSource.xorCompatible || orSource.andCompatible) continue;
    compatible = source.isOrCompatibleWith(orSource.source);
    if (compatible) break;
  }
  return compatible;
};

__device__ auto isSourceCompatibleWithEveryOrArg(
  const SourceCompatibilityData& source, const OrArgDataList& orArgDataList)
{
  auto compatible = true; // if no --or sources specified, compatible == true
  for (const auto& orArgData : orArgDataList) {
    // TODO: skip calls to here if container.compatible = true  which may have
    // been determined in Precompute phase @ markAllANDCompatibleOrSources()
    // and skip the XOR check as well in this case.
    compatible = isSourceORCompatibleWithAnyOrSource(source,
     orArgData.orSourceList);
    if (!compatible) break;
  }
  return compatible;
}
  */

  /*
//__device__
auto isSourceXORCompatibleWithAnyXorSource(
  const SourceCompatibilityData& source, const XorSourceList& xorSourceList,
  const std::vector<int>& indices)
{
  bool compatible = true; // empty list == compatible
  for (auto index : indices) {
    const auto& xorSource = xorSourceList[index];
    compatible = source.isXorCompatibleWith(xorSource, false);
    if (compatible) break;
  }
  return compatible;
};

//__device__
auto isSourceXORCompatibleWithAnyXorSource(
  const SourceCompatibilityData& source, const XorSourceList& xorSourceList,
  const std::array<VariationIndicesMap, kNumSentences>& variationIndicesMaps)
{
  for (auto s = 0; s < kNumSentences; ++s) {
    auto variation = source.usedSources.variations[s];
    const auto& map = variationIndicesMaps[s];
    if ((variation < 0) || (map.size() == 1)) continue;
    if (auto it = map.find(variation); it != map.end()) {
      if (isSourceXORCompatibleWithAnyXorSource(source, xorSourceList,
        it->second))
      {
        return true;
      }
    }
    if (auto it = map.find(-1); it != map.end()) {
      if (isSourceXORCompatibleWithAnyXorSource(source, xorSourceList,
        it->second))
      {
        return true;
      }
    }
    return false;
  }
  return isSourceXORCompatibleWithAnyXorSource(source, xorSourceList,
    variationIndicesMaps[1].at(-1)); // hack: we know "sentence" 2 doesn't exist
}
  */

  /*
__host__ __device__ bool isAnySourceCompatibleWithUseSources(
  const SourceCompatibilityData *source, int numCompatData,
  const XorSource* xorSources, size_t numXorSources)
{
  if (!numCompatData) return true;
  auto compatible = false;
  for (auto i = 0; i < numCompatData; ++i) {
    compatible = isSourceXORCompatibleWithAnyXorSource(source[i],
      xorSources, numXorSources); // , PCD.variationIndicesMaps);
    // if there were --xor sources specified, and none are compatible with the
    // current source, no further compatibility checking is necessary; continue
    // to next source.
    if (!compatible) continue;
    // TODO
    //compatible = isSourceCompatibleWithEveryOrArg(source[i],
    //  PCD.orArgDataList);
    if (compatible) break;
  }
  return compatible;
};
  */
  __host__ __device__ auto isSourceXORCompatibleWithAnyXorSource(
    const SourceCompatibilityData& source,
    const XorSource* xorSources, size_t numXorSources,
    int* outIndex = nullptr)
  {
    bool compatible = true; // empty list == compatible
    for (auto i = 0u; i < numXorSources; ++i) {
      compatible = source.isXorCompatibleWith(xorSources[i], false);
      if (compatible) {
        if (outIndex) *outIndex = i;
        break;
      }
    }
    return compatible;
  }

  using result_t = uint8_t;

  __global__
  void kernel(const SourceCompatibilityData* sources, size_t num_sources,
    const XorSource* xorSources, size_t num_xorSources, 
    const int* source_indices, /*size_t num_source_indices,*/ result_t* results)
  {
    auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_sources) return;
    bool compat = isSourceXORCompatibleWithAnyXorSource(
        sources[source_indices[index]], xorSources, num_xorSources);
    results[index] = compat ? 1 : 0;
  }

  using ResultList = std::vector<result_t>;

  struct SourceIndex {
    int listIndex{};
    int index{};
    int flat_index;
  };

  struct IndexStates {
    enum class State {
      ready, compatible, done
    };

    struct Data {
      constexpr auto ready_state() const { return state == State::ready; }

      SourceIndex sourceIndex;
      State state = State::ready;
    };
    
    IndexStates() = delete;
    IndexStates(const SourceCompatibilityLists& sources) {
      list.resize(sources.size());
      std::for_each(list.begin(), list.end(), [idx = 0](Data& data) mutable {
        data.sourceIndex.listIndex = idx++;
      });
      int flat_index{};
      for (const auto& sourceList: sources) {
        flat_indices.push_back(flat_index);
        flat_index += sourceList.size();
      }
    }

    auto flat_index(int list_index) const {
      return flat_indices.at(list_index);
    }

    auto num_in_state(int first, int count, State state) const {
      int total{};
      for (int i{}; i < count; ++i) {
        if (list.at(first + i).state == state) total++;
      }
      return total;
    }

    auto num_ready(int first, int count) const {
      return num_in_state(first, count, State::ready);
    }

    auto num_done(int first, int count) const {
      return num_in_state(first, count, State::done);
    }

    auto num_compatible(int first, int count) const {
      return num_in_state(first, count, State::compatible);
    }

    auto update(const std::vector<SourceIndex>& sourceIndices,
      const ResultList& results, const SourceCompatibilityLists& sources)
    {
      int num_compatible{};
      for (size_t i{}; i < sourceIndices.size(); ++i) {
        const auto result = results.at(i);
        assert(result == 0 || result == 1); // sanity check
        auto& data = list.at(sourceIndices.at(i).listIndex);
        assert(data.sourceIndex.listIndex == sourceIndices.at(i).listIndex);
        // this should only ever happen if the number of lists in "ready"
        // state was less than minimum stride (we doubled up sources from
        // one or more lists).
        if (!data.ready_state()) continue;
        if (result) {
          data.state = State::compatible;
          num_compatible++;
        } else {
          auto sourcelist_size = (int)sources.at(data.sourceIndex.listIndex).size();
          if (++data.sourceIndex.index >= sourcelist_size) {
            data.state = State::done;
          }
        }
      }
      return num_compatible;
    }
    
    std::vector<Data> list;
    std::vector<int> flat_indices;
  }; // struct IndexStates

  //////////

  std::vector<cudaStream_t> streams;
  
  // the pointers in this are allocated in device memory
  struct KernelData {
  private:
    static const int num_cores = 1280;
    static const int min_workitems = 2 * num_cores;
    static const int max_workitems = 2 * num_cores;

  public:
    //constexpr
    static int getNumStreams(size_t num_sources) {
      return std::min(24ul, num_sources / min_workitems + 1);
    }

    static void init(std::vector<KernelData>& dataVec, int num_sourcelists) {
      const int stride = num_sourcelists / dataVec.size();
      assert((stride < max_workitems) && "not supported (but could be)");
      int start_index{};
      for (auto i{ 0u }; i < dataVec.size(); ++i) {
        auto& data = dataVec.at(i);
        data.list_start_index = start_index;
        int remain = num_sourcelists - start_index;
        data.source_indices.resize(remain < stride ? remain : stride);
        // this is necessary because source_indices.size() may change, but the
        // number of list_indices this kernel is concerned with remains constant
        data.num_list_indices = data.source_indices.size();
        start_index += stride;
        if (i >= streams.size()) {
          cudaStream_t stream;
          cudaError_t err = cudaStreamCreate(&stream);
          assert(err == cudaSuccess);
          streams.push_back(stream);
        }
        data.stream_index = i;
        data.stream = streams[i];
      }
    }

    //

    int num_ready(const IndexStates& indexStates) const {
      return indexStates.num_ready(list_start_index, num_list_indices);
    }

    int num_done(const IndexStates& indexStates) const {
      return indexStates.num_done(list_start_index, num_list_indices);
    }

    int num_compatible(const IndexStates& indexStates) const {
      return indexStates.num_compatible(list_start_index, num_list_indices);
    }

    bool fillSourceIndices(const IndexStates& indexStates) {
      auto num_ready = indexStates.num_ready(list_start_index, num_list_indices);
      std::set<int> list_indices{}; // logging
      if (num_ready) {
        auto num_indices = num_ready;
        if (num_ready < num_list_indices) num_indices = min_workitems;
        // TODO: should probably be a percentage, not a fixed #
        if (num_ready < 250) num_indices = max_workitems;
        source_indices.resize(num_indices);
        for (int source_index{}; source_index < num_indices; /* nothing */) {
          auto any{ false };
          for (int list_index{}; list_index < num_list_indices; ++list_index) {
            const auto& indexState =
              indexStates.list.at(list_start_index + list_index);
            if (indexState.ready_state()) {
              source_indices.at(source_index++) = indexState.sourceIndex;
              list_indices.insert(indexState.sourceIndex.listIndex); // logging
              any = true;
              if (source_index >= num_indices) break;
            }
          }
          if (!any) {
            source_indices.resize(source_index);
            break;
          }
        }
      } else {
        source_indices.resize(0);
      }
      if (source_indices.empty()) {
      #ifdef DEBUG
        std::cerr << "  fill " << stream_index << ": empty " << std::endl;
      #endif
        return false;
      }
      #if 1 || defined(DEBUG)
      std::cerr << "  fill " << stream_index << ":"
                << " added " << source_indices.size() << " sources"
                << " from " << list_indices.size() << " sourcelists"
                << " (" << list_start_index << " - "
                << list_start_index + num_list_indices - 1 << ")"
                << std::endl;
      #endif
      return true;
    }

    void allocCopy(const IndexStates& indexStates) {
      cudaError_t err = cudaSuccess;

      //auto num_sources = source_indices.size();
      //assert(num_sources > 0);
      // alloc source indices
      if (!device_source_indices) {
        auto sources_bytes = max_workitems * sizeof(int);
        err = cudaMallocAsync((void **)&device_source_indices, sources_bytes,
          stream);
        if (err != cudaSuccess) {
          fprintf(stderr, "Failed to allocate stream %d source indices"
            ", error: %s\n", stream_index, cudaGetErrorString(err));
          throw std::runtime_error("failed to allocate source indices");
        }
      }

      std::vector<int> flat_indices;
      flat_indices.reserve(min_workitems);
      for (const auto& sourceIndex: source_indices) {
        flat_indices.push_back(
          indexStates.flat_index(sourceIndex.listIndex) + sourceIndex.index);
      }
      // copy (flat) source indices
      err = cudaMemcpyAsync(device_source_indices, flat_indices.data(),
        flat_indices.size() * sizeof(int), cudaMemcpyHostToDevice, stream);
      if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy stream %d source indices, error: %s\n",
          stream_index, cudaGetErrorString(err));
        throw std::runtime_error("failed to copy source indices");
      }
      
      // alloc results
      if (!device_results) {
        auto results_bytes = max_workitems * sizeof(result_t);
        err = cudaMallocAsync((void **)&device_results, results_bytes, stream);
        if (err != cudaSuccess) {
          fprintf(stderr, "Failed to allocate stream %d results, error: %s\n",
            stream_index, cudaGetErrorString(err));
          throw std::runtime_error("failed to allocate results");
        }
      }
    }

    auto hasWorkRemaining() const {
      return !source_indices.empty();
    }

    //    SourceCompatibilityData* device_sources = nullptr;
    int num_sources;      // # of entries in device_sources
    int list_start_index; // starting index in SourceCompatibiliityLists
    int num_list_indices; // # of above list entries we are concerned with
    int stream_index;
    bool running = false;
    result_t *device_results = nullptr;
    cudaStream_t stream = nullptr;
    std::vector<SourceIndex> source_indices;
    int* device_source_indices = nullptr;
  }; // struct KernelData

  //////////

  bool getRunningComplete(const std::vector<KernelData>& kernelVec,
    int& index)
  {
    // it would be better here to start at index+1 and wrap
    for (auto i{ 0u }; i < kernelVec.size(); ++i) {
      const auto& kd = kernelVec[i];
      if (kd.running && (cudaSuccess == cudaStreamQuery(kd.stream))) {
        index = i;
        return true;
      }
    }
    return false;
  }

  bool getNextWithWorkRemaining(const std::vector<KernelData>& kernelVec,
    int& current)
  {
    if (!getRunningComplete(kernelVec, current)) {
      bool wrapped = false;
      do {
        if (++current >= (int)kernelVec.size()) {
          current = 0;
          if (wrapped) return false;
          wrapped = true;
        }
      } while (!kernelVec.at(current).hasWorkRemaining());
    }
    return true;
  }

  void runKernel(KernelData& kd, const SourceCompatibilityData* device_sources) {
    auto num_sources = kd.source_indices.size();
    int threadsPerBlock = 32;
    int blocksPerGrid = (num_sources + threadsPerBlock - 1) / threadsPerBlock;
    kd.running = true;
    kernel<<<blocksPerGrid, threadsPerBlock, 0, kd.stream>>>(device_sources,
      num_sources, PCD.device_xorSources, PCD.xorSourceList.size(),
      kd.device_source_indices, /*kd.source_indices.size(),*/ kd.device_results);

#if 0 || defined(DEBUG)
    fprintf(stderr, "  kernel %d launched with %d blocks of %d threads...\n",
      kd.stream_index, blocksPerGrid, threadsPerBlock);
#endif
  }

  auto getKernelResults(KernelData& kernel) {
    auto num_sources = kernel.source_indices.size();
    ResultList results(num_sources);
    auto results_bytes = num_sources * sizeof(result_t);
    cudaStreamSynchronize(kernel.stream);
    cudaError_t err = cudaMemcpyAsync(results.data(), kernel.device_results,
      results_bytes, cudaMemcpyDeviceToHost, kernel.stream);
    kernel.running = false;
    if (err != cudaSuccess) {
      fprintf(stderr, "Failed to copy results from device -> host, error: %s\n",
              cudaGetErrorString(err));
      throw std::runtime_error("failed to copy results from device -> host");
    }
    return results;
  }

  void showAllNumReady(const std::vector<KernelData>& kernels,
    const IndexStates& indexStates)
  {
    for (auto& k: kernels) {
      std::cerr << "  kernel " << k.stream_index << ": " 
                << k.num_ready(indexStates) << std::endl;
    }
  }

  auto count(const SourceCompatibilityLists& sources) {
    size_t num{};
    for (const auto& sourceList: sources) {
      num += sourceList.size();
    }
    return num;
  }

  auto* allocCopySources(const SourceCompatibilityLists& sources) {
    // alloc sources
    cudaError_t err = cudaSuccess;
    auto sources_bytes = count(sources) * sizeof(SourceCompatibilityData);
    SourceCompatibilityData* device_sources;
    err = cudaMalloc((void **)&device_sources, sources_bytes);
    if (err != cudaSuccess) {
      fprintf(stderr, "Failed to allocate sources, error: %s\n",
        cudaGetErrorString(err));
      throw std::runtime_error("failed to allocate sources");
    }

    // copy sources
    size_t index{};
    for (const auto& sourceList: sources) {
      err = cudaMemcpy(&device_sources[index], sourceList.data(),
        sourceList.size() * sizeof(SourceCompatibilityData),
        cudaMemcpyHostToDevice);
      if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy sourcelist, error: %s\n",
          cudaGetErrorString(err));
        throw std::runtime_error("failed to copy sourcelist");
      }
      index += sourceList.size();
    }
    return device_sources;
  }

  void printCompatRecord(const std::vector<std::vector<int>>& compat_record) {
    int total{};
    for (size_t i{}; i < compat_record.size(); ++i) {
      const auto& counts = compat_record.at(i);
      std::cerr << "stream " << i << ":";
      for (auto count: counts) {
        std::cerr << " " << count;
        total += count;
      }
      std::cerr << std::endl;
    }
    std::cerr << "total: " << total << std::endl;
  }
} // anonymous namespace

namespace cm {

void filterCandidatesCuda(int sum) {
  using namespace std::chrono;

  std::cerr << "++filterCandidatesCuda" << std::endl;

  const auto& sources = allSumsCandidateData.find(sum)->second
    .sourceCompatLists;
  auto device_sources = allocCopySources(sources);

  //constexpr
  const int num_streams = KernelData::getNumStreams(sources.size());
  std::vector<KernelData> kernels(num_streams);
  KernelData::init(kernels, sources.size());
  std::cerr << "using " << num_streams << " streams" << std::endl;

  std::vector<std::vector<int>> compat_record(num_streams);
  IndexStates indexStates{ sources };
  //auto first{ true };
  int total_compatible{};
  int current_kernel = -1;
  while (getNextWithWorkRemaining(kernels, current_kernel)) {
    auto& kd = kernels.at(current_kernel);
    if (!kd.running) {
      if (kd.fillSourceIndices(indexStates)) {
        // TODO: move alloc to separate func outside loop
        // consider copying all source data on stream0, 
        // and only copy indices array here
        kd.allocCopy(indexStates);
        runKernel(kd, device_sources);
      }
      continue;
    }

    auto t0 = high_resolution_clock::now();
    auto results = getKernelResults(kd);
    auto t1 = high_resolution_clock::now();
    auto d = duration_cast<milliseconds>(t1 - t0).count();

    auto num_compatible = indexStates.update(kd.source_indices, results, sources);
    total_compatible += num_compatible;

    compat_record.at(kd.stream_index).push_back(num_compatible);

    #ifdef DEBUG
    std::cerr << "  kernel " << current_kernel << " done"
      //<< ", done: " << kd.num_done(indexStates)
      //<< ", compatible reported: " << num_compatible
      //<< " actual:" << kd.num_compatible(indexStates)
      //<< ", total compatible: " << total_compatible
      //<< ", remaining: " << kd.num_ready(indexStates)
      << " - " << d << "ms" << std::endl;
      #endif
    #ifdef DEBUG
    assert(kd.num_list_indices == kd.num_ready(indexStates) +
      kd.num_compatible(indexStates) + kd.num_done(indexStates));
    #endif
  }
  printCompatRecord(compat_record);
  std::cerr << "total compatible: " << total_compatible << " of "
            << sources.size() << std::endl;

  /*
  //  auto num_sources = count(compatLists);
  auto source_bytes = num_sources * sizeof(SourceCompatibilityData);

  auto ac0 = high_resolution_clock::now();
  // begin alloc_copy 

  // end alloc-copy
  auto ac1 = high_resolution_clock::now();
  auto dur_ac = duration_cast<milliseconds>(ac1 - ac0).count();
  std::cerr << "  alloc/copy " << compatLists.size() << " compatLists"
            << " (" << num_sources << ") done - " << dur_ac << "ms"
            << std::endl;
  */

//#define IMMEDIATE_RESULTS
#ifdef IMMEDIATE_RESULTS
  std::vector<result_t> results;
  results.resize(num_source);
  err = cudaMemcpy(results.data(), device_results, results_bytes,
                   cudaMemcpyDeviceToHost, stream);

  auto& indexComboListMap = allSumsCandidateData.find(sum)->second
    .indexComboListMap;
  int num_compat_combos{};
  int num_compat_sourcelists{};
  index = 0;
  int list_index{};
  for (const auto& compatList: compatLists) {
    int result_index{ index };
    for (const auto& source: compatList) {
      if (results.at(result_index)) {
        ++num_compat_sourcelists;
        num_compat_combos += indexComboListMap.at(list_index).size();
        break;
      }
      result_index++;
    }
    index += compatList.size();
    ++list_index;
  }
  int num_compat_results = std::accumulate(results.cbegin(), results.cend(), 0,
    [](int num_compatible, result_t result) mutable {
      if (result) num_compatible++;
      return num_compatible;
    });
  std::cerr << "  results: " << results.size()
    << ", compat results: " << num_compat_results
    << ", compat sourcelists: " << num_compat_sourcelists
    << ", compat combos: " << num_compat_combos
    << std::endl;

  err = cudaFree(device_results);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device results (error code %s)!\n",
      cudaGetErrorString(err));
    throw std::runtime_error("failed to free device results");
  }
#endif // IMMEDIATE_RESULTS

#if 0
  err = cudaFree(device_sources);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device sources, error: %s\n",
      cudaGetErrorString(err));
    throw std::runtime_error("failed to free device sources");
  }
#endif

  std::cerr << "--filterCandidatesCuda" << std::endl;
}

[[nodiscard]]
XorSource* cuda_allocCopyXorSources(const XorSourceList& xorSourceList,
  const std::vector<int> sortedIndices)
{
  auto xorSources_bytes = xorSourceList.size() * sizeof(XorSource);
  XorSource *device_xorSources = nullptr;
  cudaError_t err = cudaMalloc((void **)&device_xorSources, xorSources_bytes);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device xorSources, error: %s\n",
      cudaGetErrorString(err));
    throw std::runtime_error("failed to allocate device xorSources");
  }
  for (size_t i{}; i < sortedIndices.size(); ++i) {
    err = cudaMemcpyAsync(&device_xorSources[i],
      &xorSourceList.at(sortedIndices[i]), sizeof(XorSource),
      cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      fprintf(stderr, "Failed to copy xorSources host -> device, error: %s\n",
              cudaGetErrorString(err));
      throw std::runtime_error("failed to copy xorSources host -> device");
    }
  }
  return device_xorSources;
}

auto countIndices(const VariationIndicesList& variationIndices) {
  return std::accumulate(variationIndices.cbegin(), variationIndices.cend(), 0,
    [](int total, const auto& indices) {
      total += indices.size();
      return total;
    });
}

[[nodiscard]]
auto cuda_allocSentenceVariationIndices(
  const SentenceVariationIndices& sentenceVariationIndices)
  -> device::VariationIndices*
{
  cudaError_t err = cudaSuccess;
  using SentenceDeviceVariationIndices =
    std::array<device::VariationIndices, kNumSentences>;
  SentenceDeviceVariationIndices sentenceDeviceVariationIndices;
  for (int s{}; s < kNumSentences; ++s) {
    auto& variationIndices = sentenceVariationIndices.at(s);
    // 2 * size to account for one -1 indices terminator per variation
    const auto device_data_bytes = (countIndices(variationIndices) +
      (2 * variationIndices.size())) * sizeof(int);
    auto& deviceVariationIndices = sentenceDeviceVariationIndices.at(s);
    err = cudaMalloc((void **)&deviceVariationIndices.device_data,
      device_data_bytes);
    assert(err == cudaSuccess);

    const static int terminator = -1;
    std::vector<int> variationOffsets;
    const auto num_variations{ variationIndices.size() };
    deviceVariationIndices.variationOffsets = deviceVariationIndices.device_data;
    deviceVariationIndices.num_variations = num_variations;
    deviceVariationIndices.sourceIndices =
      &deviceVariationIndices.device_data[num_variations];
    size_t offset{};
    for (const auto& indices: variationIndices) {
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
  const auto sentenceVariationIndices_bytes =
    kNumSentences * sizeof(device::VariationIndices);
  device::VariationIndices* device_variationIndices;
  err = cudaMalloc((void **)&device_variationIndices,
    sentenceVariationIndices_bytes);
  assert(err == cudaSuccess);

  err = cudaMemcpyAsync(device_variationIndices,
    sentenceDeviceVariationIndices.data(), sentenceVariationIndices_bytes,
    cudaMemcpyHostToDevice);
  assert(err == cudaSuccess);

  return device_variationIndices;
}

} // namespace cm
