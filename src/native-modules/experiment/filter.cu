// filter.cu

#include <algorithm>
#include <chrono>
#include <exception>
#include <numeric>
#include <optional>
#include <thread>
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

  __device__
  auto isSourceXORCompatibleWithAnyXorSource(
    const SourceCompatibilityData& source, const XorSource* xorSources,
    const int* indices, int* compat_index = nullptr, int* reason = nullptr)
  {
    auto compatible{ false }; // important. explicit compatibility required
    for (int i{}; indices[i] > -1; ++i) {
      const auto& xorSource = xorSources[indices[i]];
      compatible = source.isXorCompatibleWith(xorSource, false, reason);
      if (compatible) {
        if (compat_index) *compat_index = indices[i];
        break;
      }
    }
    return compatible;
  }

  __device__ __host__
  auto isSourceXORCompatibleWithAnyXorSource(
    const SourceCompatibilityData& source, const XorSource* xorSources,
    size_t numXorSources, int* compat_index = nullptr, int* reason = nullptr)
  {
    bool compatible = true;
    for (size_t i{}; i < numXorSources; ++i) {
      compatible = source.isXorCompatibleWith(xorSources[i], false, reason);
      if (compatible) {
        if (compat_index) *compat_index = i;
        break;
      }
    }
    return compatible;
  }

  __device__
  auto isSourceXORCompatibleWithAnyXorSource(
    const SourceCompatibilityData& source,
    const XorSource* xorSources, size_t num_xorSources,
    const device::VariationIndices* sentenceVariationIndices,
    int* compat_index = nullptr, int* reason = nullptr)
  {
    for (int s{}; s < kNumSentences; ++s) {
      auto variation = source.usedSources.variations.at(s) + 1;
      const auto& variationIndices = sentenceVariationIndices[s];
      if (!variation || (!variationIndices.num_variations)) {
        continue;
      }
      { // anonymous block
        const auto offset = variationIndices.variationOffsets[variation];
        const auto* indices = &variationIndices.sourceIndices[offset];
        if (isSourceXORCompatibleWithAnyXorSource(source, xorSources,
          indices, compat_index, reason)) return true;
      }
      { // anonymous block
        const auto offset = variationIndices.variationOffsets[0];
        const auto* indices = &variationIndices.sourceIndices[offset];
        if (isSourceXORCompatibleWithAnyXorSource(source, xorSources,
          indices, compat_index, reason)) return true;
      }
      // The idea here is that once we test compatibility with all sources
      // that match the variation of any single sentence of the supplied
      // source, (including with those sources that have no variation for
      // that sentence), we're done. Compatible source(s) were either found
      // for that sentence, or not. We use sentence-compatibility as a shortcut
      // for source-compatibility.
      // TODO: there is an optimization possible here; we could iterate over
      // all sentences first, to identify the sentence with the smallest
      // variation indices list, and test that one, instead of the first one
      // we find.
      return false;
    }
    return isSourceXORCompatibleWithAnyXorSource(source, xorSources,
      num_xorSources, compat_index, reason);
  }

  using result_t = uint8_t;

  __shared__ int compat_index;
  __shared__ int reason;

  __global__
  void xorKernel(const SourceCompatibilityData* sources, size_t num_sources,
    const XorSource* xorSources, size_t num_xorSources,
    const device::VariationIndices* sentenceVariationIndices,
    const int* source_indices, int stream_index, result_t* results)
  {
    constexpr const auto logging = false;

    auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_sources) return;
    //int compat_index{ -1 }, reason{ -1 };
    
    auto debug{ false };
    if constexpr (logging) {
      if ((stream_index == 0) && (index == 48)) {
        reason = 100;
        debug = true;
      }
    }

    bool compat = isSourceXORCompatibleWithAnyXorSource(
      sources[source_indices[index]], xorSources, num_xorSources,
      sentenceVariationIndices, debug ? &compat_index : nullptr,
      debug ? &reason : nullptr);
    results[index] = compat ? 1 : 0;

    if constexpr (logging) {
      if (debug) {
        const auto& source = sources[source_indices[index]];
        const auto& us = source.usedSources;
        printf("KERNEL: stream %d, index: %d, src_index: %d"
               ", v1: %d (%d), v3: %d (%d)"
               ", firstLegacySrc: %d"
               ", compat: %d, compat_index %d, reason %d\n",
               stream_index, index, source_indices[index],
               us.getVariation(1), us.countSources(1),
               us.getVariation(3), us.countSources(3),
               source.getFirstLegacySource(),
               compat, compat_index, reason);
        if (compat) {
          assert(compat_index > -1);
          const auto& xor_source = xorSources[compat_index];
          const auto& xor_us = xor_source.usedSources;
          printf("  Xor[%d]: v1: %d (%d), v3: %d (%d), firstLegacySrc: %d\n",
                 compat_index,
                 xor_us.getVariation(1), xor_us.countSources(1),
                 xor_us.getVariation(3), xor_us.countSources(3),
                 xor_source.getFirstLegacySource());
        }
      }
    }
  }

  using ResultList = std::vector<result_t>;

  struct SourceIndex {
    int listIndex{};
    int index{};

    bool operator<(const SourceIndex& rhs) const {
      return (listIndex < rhs.listIndex) || (index < rhs.index);
    }
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
      list.resize(sources.size()); // i.e. "num_sourcelists"
      std::for_each(list.begin(), list.end(), [idx = 0](Data& data) mutable {
        data.sourceIndex.listIndex = idx++;
      });
      for (int flat_index{}; const auto& sourceList: sources) {
        list_sizes.push_back(sourceList.size());
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
      const ResultList& results, const SourceCompatibilityLists& sources,
      int stream_index) // for logging
    {
      constexpr static const bool logging = false;
      std::set<SourceIndex> compat_src_indices; // for logging
      int num_compatible{};
      for (size_t i{}; i < sourceIndices.size(); ++i) {
        const auto result = results.at(i);
        assert(result == 0 || result == 1); // sanity check
        auto& indexState = list.at(sourceIndices.at(i).listIndex);
        // no longer (always) true, as indexState.index is incremented at
        // fill() time.
        //assert(indexState.sourceIndex.listIndex ==
        //  sourceIndices.at(i).listIndex);
        if constexpr (logging) {
          const auto si = sourceIndices.at(i);
          if ((stream_index == 0) && (si.index == 1)
              && ((si.listIndex == 209) || (si.listIndex == 1174)))
          {
            std::cerr << "stream " << stream_index << " "
                      << si.listIndex << ":" << si.index
                      << ", results index: " << i
                      << ", compat: " << std::boolalpha << (result == 1)
                      << ", ready: " << indexState.ready_state()
                      << std::endl;
          }
        }
        // this should only ever happen if the number of lists in "ready"
        // state was less than minimum stride (we doubled up sources from
        // one or more lists).
        if (!indexState.ready_state()) continue;
        if (result) {
          indexState.state = State::compatible;
          num_compatible++;
          if constexpr (logging) compat_src_indices.insert(sourceIndices.at(i));
        }
        else {
          // index was incremented when we grabbed it in fill(). if it was
          // incremented past length, mark it done now.
          auto sourcelist_size =
            (int)sources.at(indexState.sourceIndex.listIndex).size();
          if (indexState.sourceIndex.index >= sourcelist_size) {
            indexState.state = State::done;
          }
        }
      }
      if constexpr (logging) {
        if (compat_src_indices.size() && (compat_src_indices.size() < 200)) {
          std::cerr << "stream " << stream_index << " update:";
          for (const auto& src_index: compat_src_indices) {
            std::cerr << " " << src_index.listIndex << ":" << src_index.index;
          }
          std::cerr << std::endl;
        }
      }
      return num_compatible;
    }

    auto get(int list_index) const {
      return list.at(list_index);
    }

    auto get_and_increment_index(int list_index) -> std::optional<SourceIndex> {
      auto& data = list.at(list_index);
      if (data.ready_state() &&
         (data.sourceIndex.index < list_sizes.at(list_index)))
      {
        // capture and return value before increment
        auto capture = std::make_optional(data.sourceIndex);
        ++data.sourceIndex.index;
        return capture;
      }
      return std::nullopt;
    }
    
    std::vector<Data> list;
    std::vector<int> flat_indices;
    std::vector<int> list_sizes;
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
        // number of list_indices this stream is concerned with is constant
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

    auto fillSourceIndices(IndexStates& indexStates, int num_indices) {
      constexpr const auto dupe_checking = true;
      std::set<std::string> dupe_check_indices{}; // debugging
      std::set<int> list_indices_used{};
      source_indices.resize(num_indices);
      for (int i{}; i < num_indices; /* nothing */) {
        auto any{ false };
        for (int list_offset{}; list_offset < num_list_indices; ++list_offset) {
          const auto list_index = list_start_index + list_offset;
          const auto opt_src_index =
            indexStates.get_and_increment_index(list_index);
          if (opt_src_index.has_value()) {
            const auto src_index = opt_src_index.value();
            source_indices.at(i++) = src_index;
            list_indices_used.insert(src_index.listIndex);

            if constexpr (dupe_checking) {
              char buf[32];
              snprintf(buf, sizeof(buf), "%d:%d", src_index.listIndex,
                src_index.index);
              std::string str_index{ buf };
              if (!dupe_check_indices.insert(str_index).second) {
                std::cerr << "stream " << stream_index << ": duplicate index: "
                          << str_index << std::endl;
              }
            }

            any = true;
            if (i >= num_indices) break;
          }
        }
        if (!any) {
          source_indices.resize(i);
          break;
        }
      }
      return list_indices_used.size();
    }

    bool fillSourceIndices(IndexStates& indexStates) {
      constexpr static const auto logging = false;
      auto num_ready = indexStates.num_ready(list_start_index,
        num_list_indices);
      int num_sourcelists{};
      if (num_ready) {
        auto num_indices = num_ready;
        if (num_ready < num_list_indices) num_indices = min_workitems;
        // TODO: should probably be a percentage, not a fixed #
        if (num_ready < 250) num_indices = max_workitems;
        num_sourcelists = fillSourceIndices(indexStates, num_indices);
      } else {
        source_indices.resize(0);
      }
      if (source_indices.empty()) {
        if constexpr (logging) {
          std::cerr << "  fill " << stream_index << ": empty " << std::endl;
        }
        return false;
      }
      if constexpr (logging) {
        std::cerr << "  fill " << stream_index << ":"
                  << " added " << source_indices.size() << " sources"
                  << " from " << num_sourcelists << " sourcelists"
                  << " (" << list_start_index << " - "
                  << list_start_index + num_list_indices - 1 << ")"
                  << std::endl;
      }
      return true;
    }

    void allocCopy(const IndexStates& indexStates) {
      cudaError_t err = cudaSuccess;
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

    int list_start_index; // starting index in SourceCompatibiliityLists
    int num_list_indices; // # of above list entries we are concerned with
    int stream_index;
    bool has_run{ false };
    bool running{ false };
    result_t *device_results{ nullptr };
    cudaStream_t stream{ nullptr };
    std::vector<SourceIndex> source_indices;
    int* device_source_indices{ nullptr };
  }; // struct KernelData

  //////////

  bool anyWithWorkRemaining(const std::vector<KernelData>& kernelVec) {
    for (const auto& kernel : kernelVec) {
      if (kernel.hasWorkRemaining()) return true;
    }
    return false;
  }

  bool anyIdleWithWorkRemaining(const std::vector<KernelData>& kernelVec,
    int& index)
  {
    for (size_t i{}; i < kernelVec.size(); ++i) {
      const auto& kernel = kernelVec[i];
      if (!kernel.running && kernel.hasWorkRemaining()) {
        index = i;
        return true;
      }
    }
    return false;
  }

  bool anyRunningComplete(const std::vector<KernelData>& kernelVec,
    int& index)
  {
    // it would be better here to start at index+1 and wrap
    for (size_t i{}; i < kernelVec.size(); ++i) {
      const auto& kernel = kernelVec[i];
      if (kernel.running && (cudaSuccess == cudaStreamQuery(kernel.stream))) {
        index = i;
        return true;
      }
    }
    return false;
  }

  bool getNextWithWorkRemaining(const std::vector<KernelData>& kernelVec,
    int& current)
  {
    using namespace std::chrono_literals;

    // First priority: ensure all streams have started at least once
    if (++current >= (int)kernelVec.size()) {
      current = 0;
    } else if (!kernelVec[current].has_run) {
      return true;
    }

    // Second priority: run any idle (non-running) stream with work remaining
    if (anyIdleWithWorkRemaining(kernelVec, current)) {
      return true;
    }

    // There are no idle streams with work remaining, but there may be running
    // streams with work remaining. Put another way, are there any streams
    // (at all) with work remaining? If not, we're done.
    if (!anyWithWorkRemaining(kernelVec)) {
      return false;
    }

    // There are running streams with work remaining. Wait for one to complete.
    while (!anyRunningComplete(kernelVec, current)) {
      // TODO events
      std::this_thread::sleep_for(10ms);
      //std::this_thread::yield();
    }
    return true;
  }

  void runKernel(KernelData& kernel,
    const SourceCompatibilityData* device_sources)
  {
    auto num_sources = kernel.source_indices.size();
    int threadsPerBlock = 32;
    int blocksPerGrid = (num_sources + threadsPerBlock - 1) / threadsPerBlock;
    kernel.has_run = true;
    kernel.running = true;
    xorKernel<<<blocksPerGrid, threadsPerBlock, 0, kernel.stream>>>(
      device_sources, num_sources,
      PCD.device_xorSources, PCD.xorSourceList.size(),
      PCD.device_sentenceVariationIndices, kernel.device_source_indices,
      kernel.stream_index, kernel.device_results);

#if 0 || defined(DEBUG)
    fprintf(stderr, "  kernel %d launched with %d blocks of %d threads...\n",
      kernel.stream_index, blocksPerGrid, threadsPerBlock);
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

  void check(const SourceCompatibilityLists& sources, int list_index,
    int index)
  {
    constexpr const auto logging = false;
    if constexpr (logging) {
      char buf[32];
      snprintf(buf, sizeof(buf), "%d:%d", list_index, index);
      auto& source = sources.at(list_index).at(index);
      source.dump(buf);
      int compat_index{ -1 };
      auto compat = isSourceXORCompatibleWithAnyXorSource(source,
        PCD.xorSourceList.data(), PCD.xorSourceList.size(), &compat_index);
      std::cerr << "compat: " << compat << " (" << compat_index << ")"
                << std::endl;
    }
  }

  void dump(const XorSourceList& xorSources,
    const std::vector<int>& xorSourceIndices, int index)
  {
    auto host_index = xorSourceIndices.at(index);
    const auto& src = xorSources.at(host_index);
    char buf[32];
    snprintf(buf, sizeof(buf), "xor: device(%d) host(%d)", index, host_index);
    src.dump(buf);
  }
} // anonymous namespace

namespace cm {

void filterCandidatesCuda(int sum) {
  using namespace std::chrono;

  std::cerr << "++filterCandidatesCuda" << std::endl;

  const auto& sources = allSumsCandidateData.find(sum)->second
    .sourceCompatLists;
  auto device_sources = allocCopySources(sources);

  /*
  check(sources, 209, 1);
  dump(PCD.xorSourceList, PCD.xorSourceIndices, 0);
  check(sources, 229, 1);
  check(sources, 232, 1);
  check(sources, 233, 1);
  check(sources, 234, 1);
  check(sources, 235, 1);
  check(sources, 236, 1);
  check(sources, 1174, 1);
  */

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
    auto& kernel = kernels.at(current_kernel);
    if (!kernel.running) {
      if (kernel.fillSourceIndices(indexStates)) {
        // TODO: move alloc to separate func outside loop
        // consider copying all source data on stream0, 
        // and only copy indices array here
        kernel.allocCopy(indexStates);
        runKernel(kernel, device_sources);
      }
      continue;
    }

    auto t0 = high_resolution_clock::now();
    auto results = getKernelResults(kernel);
    auto t1 = high_resolution_clock::now();
    auto d = duration_cast<milliseconds>(t1 - t0).count();

    auto num_compatible = indexStates.update(kernel.source_indices, results, sources,
      kernel.stream_index);
    total_compatible += num_compatible;

    compat_record.at(kernel.stream_index).push_back(num_compatible);

#if 1 || defined(DEBUG)
    std::cerr << "  kernel " << current_kernel << " done"
      << ", done: " << kernel.num_done(indexStates)
      //<< ", compat reported: " << num_compatible
      << ", compat actual:" << kernel.num_compatible(indexStates)
      //<< ", total compatible: " << total_compatible
      << ", remaining: " << kernel.num_ready(indexStates)
      << " - " << d << "ms" << std::endl;
#endif
    #ifdef DEBUG
    assert(kernel.num_list_indices == kernel.num_ready(indexStates) +
      kernel.num_compatible(indexStates) + kernel.num_done(indexStates));
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
auto cuda_allocCopySentenceVariationIndices(
  const SentenceVariationIndices& sentenceVariationIndices)
  -> device::VariationIndices*
{
  cudaError_t err = cudaSuccess;
  using DeviceVariationIndicesArray =
    std::array<device::VariationIndices, kNumSentences>;
  DeviceVariationIndicesArray deviceVariationIndicesArray;
  for (int s{}; s < kNumSentences; ++s) {
    auto& variationIndices = sentenceVariationIndices.at(s);
    // 2 * size to account for one -1 indices terminator per variation
    const auto device_data_bytes = (countIndices(variationIndices) +
      (2 * variationIndices.size())) * sizeof(int);
    auto& deviceVariationIndices = deviceVariationIndicesArray.at(s);
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
  const auto variationIndices_bytes =
    kNumSentences * sizeof(device::VariationIndices);
  device::VariationIndices* device_variationIndices;
  err = cudaMalloc((void **)&device_variationIndices, variationIndices_bytes);
  assert(err == cudaSuccess);

  err = cudaMemcpyAsync(device_variationIndices,
    deviceVariationIndicesArray.data(), variationIndices_bytes,
    cudaMemcpyHostToDevice);
  assert(err == cudaSuccess);

  return device_variationIndices;
}

} // namespace cm
