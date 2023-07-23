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

  using result_t = int32_t;

  //__shared__ int compat_index;
  //__shared__ int reason;

  __global__
  void xorKernel(const SourceCompatibilityData* sources, size_t num_sources,
    const XorSource* xorSources, size_t num_xorSources,
    const device::VariationIndices* sentenceVariationIndices,
    const int* source_indices, int stream_index, result_t* results,
    int special)
  {
    constexpr const auto logging = false;

    auto index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_sources) return;
    int compat_index{ -1 }, reason{ -1 };
    auto debug{ true };

    const auto& source = sources[source_indices[index]];
    bool compat = isSourceXORCompatibleWithAnyXorSource(
      source, xorSources, num_xorSources,
      sentenceVariationIndices, &compat_index,
      debug ? &reason : nullptr);
    if (compat) assert(compat_index > -1); // probably slowish
    results[index] = compat ? compat_index : -1;

    if constexpr (logging) {
      const auto src_index = source_indices[index];
      if (debug) {
        const auto& us = source.usedSources;
        printf("stream %d, index: %d, src_index: %d"
               ", s1 v%d 1st:%d (%d)"
               ", s3 v%d 1st:%d (%d)"
               ", legacy 1st:%d (%d)"
               ", compat: %d, compat_index %d, reason %d\n",
               stream_index, index, src_index,
               us.getVariation(1), us.getFirstSource(1), us.countSources(1),
               us.getVariation(3), us.getFirstSource(3), us.countSources(3),
               source.getFirstLegacySource(), source.countLegacySources(),
               compat, compat_index, reason);
        if (compat) {
          const auto& xor_source = xorSources[compat_index];
          const auto& xor_us = xor_source.usedSources;
          printf("  Xor[%d]: s1 v%d 1st:%d (%d), s3 v%d 1st:%d (%d)"
                 ", legacy 1st:%d (%d)\n",
                 compat_index, xor_us.getVariation(1), xor_us.getFirstSource(1),
                 xor_us.countSources(1), xor_us.getVariation(3),
                 xor_us.getFirstSource(3), xor_us.countSources(3),
                 xor_source.getFirstLegacySource(),
                 xor_source.countLegacySources());
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
      
    auto list_size(int list_index) const {
      return list_sizes.at(list_index);
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
      ResultList& results, const SourceCompatibilityLists& sources,
      int stream_index) // for logging
    {
      constexpr static const bool logging = false;
      std::set<SourceIndex> compat_src_indices; // for logging
      int num_compatible{};
      for (size_t i{}; i < sourceIndices.size(); ++i) {
        const auto result = results.at(i);
        const auto src_index = sourceIndices.at(i);
        auto& indexState = list.at(src_index.listIndex);
        // no longer (always) true, as indexState.index is incremented at
        // fill() time.
        //assert(indexState.sourceIndex.listIndex == src_index.listIndex);
        if constexpr (logging) {
          if (src_index.index == -1) {// && other conditions
            std::cerr << "stream " << stream_index << " "
                      << src_index.listIndex << ":" << src_index.index
                      << ", results index: " << i
                      << ", result: " << result
                      << ", compat: " << std::boolalpha << (result > -1)
                      << ", ready: " << indexState.ready_state()
                      << std::endl;
          }
        }
        // this should only ever happen if the number of lists in "ready"
        // state was less than minimum stride (we doubled up sources from
        // one or more lists).
        if (!indexState.ready_state()) {
          // for debugging purposes, set these "duplicate results for same
          // sentence" results to -1, so we can later determine the exact
          // set of "first matched soruces".
          results.at(i) = -1;
          continue;
        }
        if (result > -1) {
          indexState.state = State::compatible;
          num_compatible++;
          if constexpr (logging) compat_src_indices.insert(src_index);
        }
        else {
          // if this is the result for the last source in a sourcelist,
          // mark the list (indexState) as done.
          // note that doing it this way *will* put a dependency on the
          // order in which we process sources within a list (currently,
          // in-order), but presumably there'd be some sourceIndexIndices
          // abomination that we could use to determine the *actual* index.
          auto sourcelist_size = (int)sources.at(src_index.listIndex).size();
          if (src_index.index >= sourcelist_size) {
            indexState.state = State::done;
          }
        }
      }
      if constexpr (logging) {
        if (compat_src_indices.size()) {// && (compat_src_indices.size() < 200)) {
          //std::cerr << "stream " << stream_index << " update:";
          for (const auto& src_index: compat_src_indices) {
            std::cout << "" << src_index.listIndex << ":" << src_index.index
                      << std::endl;
          }
          //std::cerr << std::endl;
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
  int host_special{};
  
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
      int leftovers = num_sourcelists % dataVec.size();
      assert((stride < max_workitems) && "not supported (but could be)");
      int start_index{};
      for (size_t i{}; i < dataVec.size(); ++i) {
        auto& data = dataVec.at(i);
        data.list_start_index = start_index;
        int share_of_leftovers = leftovers ? (--leftovers, 1) : 0;
        //if (leftovers) --leftovers;
        // this separate accounting of "number of indices" is necessary because
        // source_indices.size() may change, but the number of list_indices this
        // stream is concerned with is constant
        data.num_list_indices = stride + share_of_leftovers;
        data.source_indices.resize(data.num_list_indices);
        start_index += data.num_list_indices;
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
      constexpr const auto dupe_checking = false;

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
            assert(src_index.listIndex == list_index);

            // DEBUGGGGGGGGGG
            if (src_index.listIndex == -1) {
              host_special = i;
            }
            const auto flat_index = indexStates.flat_index(src_index.listIndex) +
              src_index.index;
            if (src_index.listIndex == -1) {
              //const auto list_size = indexStates.flat_index(src_index.listIndex + 1) -
              //indexStates.flat_index(src_index.listIndex);
              std::cerr << "flat_index: " << flat_index << " = "
                        << src_index.listIndex << ":" << src_index.index
                        << ", list size: " << indexStates.list_size(src_index.listIndex)
                        << std::endl;
            }

            source_indices.at(i++) = src_index;
            // TODO this is slow and only used for logging and I don't like it.
            // figure it out.
            //list_indices_used.insert(src_index.listIndex);

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
      return 8008135; // list_indices_used.size();
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
      /*
      constexpr static bool debug_indices = true;
      if constexpr (debug_indices) {
        // alloc indices (debugging)
        if (!device_compat_indices) {
          auto indices_bytes = max_workitems * sizeof(int);
          err = cudaMallocAsync((void **)&device_compat_indices, indices_bytes, stream);
          if (err != cudaSuccess) {
            fprintf(stderr, "Failed to allocate stream %d indices, error: %s\n",
              stream_index, cudaGetErrorString(err));
            throw std::runtime_error("failed to allocate results");
          }
        }
      }
      */
    }

    auto hasWorkRemaining() const {
      return !source_indices.empty();
    }

    int list_start_index; // starting index in SourceCompatibiliityLists
    int num_list_indices; // # of above list entries we are concerned with
    int stream_index;
    bool has_run{ false };
    bool running{ false };
    int* device_source_indices{ nullptr }; // in
    result_t *device_results{ nullptr }; // out
    cudaStream_t stream{ nullptr };
    std::vector<SourceIndex> source_indices;  // .size() == num_results
    //result_t *device_compat_indices{ nullptr }; 
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
      kernel.stream_index, kernel.device_results, host_special);

#if 0 || defined(DEBUG)
    fprintf(stderr, "  kernel %d launched with %d blocks of %d threads...\n",
      kernel.stream_index, blocksPerGrid, threadsPerBlock);
#endif
  }

  // todo: kernel.getResults()
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

  void check(const SourceCompatibilityLists& sources, int list_index, int index)
  {
    constexpr const auto logging = true;
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

  const auto& sources = allSumsCandidateData.find(sum)->second
    .sourceCompatLists;
  auto device_sources = allocCopySources(sources);

  auto t0 = high_resolution_clock::now();
  const int num_streams = KernelData::getNumStreams(sources.size());
  std::vector<KernelData> kernels(num_streams);
  KernelData::init(kernels, sources.size());
  std::cerr << "using " << num_streams << " streams" << std::endl;

  std::set<int> compat_indices;
  std::vector<std::vector<int>> compat_record(num_streams);
  IndexStates indexStates{ sources };
  int total_compatible{};
  int current_kernel{ -1 };
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

    auto r0 = high_resolution_clock::now();
    auto results = getKernelResults(kernel);
    auto r1 = high_resolution_clock::now();
    auto d_results = duration_cast<milliseconds>(r1 - r0).count();

    auto num_compatible = indexStates.update(kernel.source_indices, results,
      sources, kernel.stream_index);
    total_compatible += num_compatible;

    #if 0
    for (size_t r{}; r < kernel.source_indices.size(); ++r) {
      const auto result = results.at(r);
      if (result > -1) compat_indices.insert(result);
    }
    compat_record.at(kernel.stream_index).push_back(num_compatible);
    #endif

#if 1 || defined(DEBUG)
    std::cerr << "  kernel " << current_kernel << " done"
      //<< ", done: " << kernel.num_done(indexStates)
      //<< ", compat reported: " << num_compatible
      //<< ", compat actual:" << kernel.num_compatible(indexStates)
      //<< ", total compatible: " << total_compatible
      //<< ", remaining: " << kernel.num_ready(indexStates)
      << " - " << d_results << "ms" << std::endl;
#endif
    #ifdef DEBUG
    assert(kernel.num_list_indices == kernel.num_ready(indexStates) +
      kernel.num_compatible(indexStates) + kernel.num_done(indexStates));
    #endif
  }
  auto t1 = high_resolution_clock::now();
  auto d_kernel = duration_cast<milliseconds>(t1 - t0).count();

  //printCompatRecord(compat_record);
  std::cerr << "total compatible: " << total_compatible << " of "
            << sources.size() << " - " << d_kernel << "ms"
            << std::endl;
  #if 0
  for (auto index: compat_indices) {
    std::cout << index << std::endl;
  }
  #endif

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
