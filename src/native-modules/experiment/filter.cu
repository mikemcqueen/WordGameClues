// filter.cu

#include <algorithm>
#include <chrono>
#include <exception>
//#include <format>
#include <limits>
#include <numeric>
#include <optional>
#include <thread>
#include <tuple>
#include <cuda_runtime.h>
#include "candidates.h"
#include "source-counts.h"

//#define STREAM_LOG

namespace {
  using namespace cm;

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
  using index_t = uint32_t;

  struct SourceIndex {
    index_t listIndex{};
    index_t index{};

    bool operator<(const SourceIndex& rhs) const {
      return (listIndex < rhs.listIndex) || (index < rhs.index);
    }

    constexpr const char* as_string(char *buf) const {
      sprintf(buf, "%d:%d", listIndex, index);
      return buf;
    }
  };

  __device__ uint8_t atomicAddUint8(uint8_t* address, uint8_t val) {
    size_t long_address_modulo = (size_t) address & 3;
    auto* base_address = (unsigned int*) ((char*) address - long_address_modulo);
    unsigned int long_val = (unsigned int) val << (8 * long_address_modulo);
    unsigned int long_old = atomicAdd((unsigned int *)base_address, long_val);

    if (long_address_modulo == 3) {
      // the first 8 bits of long_val represent the char value,
      // hence the first 8 bits of long_old represent its previous value.
      return (char) (long_old >> 24);
    } else {
      // bits that represent the char value within long_val
      unsigned int mask = 0x000000ff << (8 * long_address_modulo);
      unsigned int masked_old = long_old & mask;
      // isolate the bits that represent the char value within long_old, add the long_val to that,
      // then re-isolate by excluding bits that represent the char value
      unsigned int overflow = (masked_old + long_val) & ~mask;
      if (overflow) {
        atomicSub(base_address, overflow);
      }
      return (char) (masked_old >> 8 * long_address_modulo);
    }
  }

  __device__ unsigned device_xor_kernel_idx = 0;

  __host__ void prepareXorKernel() {
    static const unsigned idx_zero = 0;
    cudaError_t err = cudaMemcpyToSymbol(device_xor_kernel_idx, &idx_zero,
      sizeof(idx_zero));
    //std::cerr << cudaGetErrorString(err) << std::endl;
    assert((err == cudaSuccess) && "copy idx zero to device");
  }
 
  __device__ bool updateXorKernelIdx(unsigned new_idx) {
    const auto old_expected = device_xor_kernel_idx;
    auto old_actual = atomicCAS(&device_xor_kernel_idx, old_expected, new_idx);
    #if 0
    if (old_actual == old_expected) {
      printf("CAS success\n");
    } else {
      printf("update to %d failed, old: actual: %d, expected: %d\n",
             new_idx, old_actual, old_expected);
    }
    #endif
    return old_actual == old_expected;
  }

  __device__ auto once = false;

  __global__
  void xorKernel(const SourceCompatibilityData* __restrict__ sources,
    const unsigned num_sources, const XorSource* __restrict__ xor_sources,
    const unsigned num_xor_sources,
    const device::VariationIndices* sentenceVariationIndices,
    const SourceIndex* __restrict__ source_indices,
    const index_t* __restrict__ list_start_indices,
    result_t* results, int stream_index, int special)
  {
    extern __shared__ uint16_t shared[];
    uint16_t* block_results = shared;
    SourceCompatibilityData* fast_source =
      (SourceCompatibilityData *)&shared[blockDim.x];

    // should only happen with very low xor_sources count 
    // 64 * 63 + 63 = 4095 
    const auto thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    if (thread_id >= num_xor_sources) return;

    auto reason{ -1 };
    auto once_only = true;

    // zero-out results
    const auto threads_per_grid = gridDim.x * blockDim.x;
    for (int start_idx = blockIdx.x * blockDim.x;
      start_idx + threadIdx.x < num_sources;
      start_idx += threads_per_grid)
    {
      results[start_idx + threadIdx.x] = 0;
    }
    __syncthreads();

    // for each source
    for (unsigned idx{}; idx < num_sources; ++idx) {
      const auto src_idx = source_indices[idx];
      const auto flat_index = 
        list_start_indices[src_idx.listIndex] + src_idx.index;
      const auto& source = sources[flat_index];

      // for each xor_source
      for (int start_idx = blockIdx.x * blockDim.x;
        start_idx + threadIdx.x < num_xor_sources;
        start_idx += threads_per_grid)
      {
        block_results[threadIdx.x] = 0;
        // a solution was found for the current src_idx, bail from inner loop
        if (idx < device_xor_kernel_idx) break;

        auto xor_src_idx = start_idx + threadIdx.x;
        if (source.isXorCompatibleWith(xor_sources[xor_src_idx],
          false, &reason))
        {
          block_results[threadIdx.x] = 1;
          #if 1
          if (!src_idx.index && (src_idx.listIndex == 50)) {
            // || (src_idx.listIndex == 1907) || (src_idx.listIndex == 1908))) {
            printf("MATCH %d %d:%d, xor %d, flat %d\n",
                   idx, src_idx.listIndex, src_idx.index,
                   xor_src_idx, flat_index);
          }
          #endif
        }
        #if 1
        if (//(src_idx.index == 2) && ((src_idx.listIndex == 142) && (xor_src_idx >= 1390) && (xor_src_idx <= 1400)))
            (src_idx.index == 0) && ((src_idx.listIndex >= 50) && (src_idx.listIndex <= 50)))
              /* && (xor_src_idx >= 37645) && (xor_src_idx == 37648)))*/
        {
          if (!once_only || !once) {
            once = true;
            __syncthreads();
            if (1) { // !once_only || !threadIdx.x) {
              if (0) {
                printf("%d\n", xor_src_idx);
              } else {
                /*
                  printf("%d:%d compat: %d, idx: %d, xor_src_idx: %d, flat_index: %d\n",
                  src_idx.listIndex, src_idx.index, block_results[threadIdx.x],
                  idx, xor_src_idx, flat_index);
                */
                printf("%d %d:%d, xor %d, flat: %d, thread: %d\n", idx, src_idx.listIndex, src_idx.index,
                       xor_src_idx, flat_index, threadIdx.x);
                char big_buf[256];
                char smol_buf[32];
                char id_buf[32];
                source.dump(src_idx.as_string(id_buf), true, big_buf, smol_buf);
                //xor_sources[xor_src_idx].dump("37648", true);
              }
            }
          }
        }
        #endif
        #if 1
        __syncthreads();
        if ((src_idx.listIndex == 50) && (src_idx.index == 0)) {
          int count{};
          for (int i{}; i < blockDim.x; ++i) {
            if (block_results[i]) {
              ++count;
            }
          }
          if (count) {
            printf("blk: %d, thrd: %d, cnt %d\n", blockIdx.x, threadIdx.x, count);
          }
        }
        #endif

        for (int reduce_idx = blockDim.x / 2; reduce_idx > 0; reduce_idx /= 2) {
          __syncthreads();
          if (threadIdx.x < reduce_idx) {
            #if 0
            if ((blockIdx.x == 72) && !threadIdx.x &&
                (block_results[threadIdx.x] || block_results[reduce_idx + threadIdx.x])) {
              printf("blk: %d, thrd: %d, %d\n", blockIdx.x, reduce_idx + threadIdx.x,
                     block_results[reduce_idx + threadIdx.x]);
            }
            #endif
            block_results[threadIdx.x] += block_results[reduce_idx + threadIdx.x];
          }
        }
        if (!threadIdx.x && block_results[threadIdx.x]) {
          #if 0
          if (((src_idx.listIndex == 50)))
            // || (src_idx.listIndex == 1907) || (src_idx.listIndex == 1908)))
          {
            printf("incr %d:%d @ idx %d, thread: %d, block, %d: block_result: %d\n",
                   src_idx.listIndex, src_idx.index, idx,
                   threadIdx.x, blockIdx.x,
                   block_results[threadIdx.x]);
          }
          #endif
          // NOTE: data-write race, safe?
          results[idx] = 1;
          //atomicAddUint8(&results[idx], 1);
          //updateXorKernelIdx(idx + 1);
          break;
        }
      }
    }
  }

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
      for (index_t list_start_index{}; const auto& sourceList: sources) {
        list_sizes.push_back(sourceList.size());
        list_start_indices.push_back(list_start_index);
        list_start_index += (index_t)sourceList.size();
      }
    }
    
    uint32_t flat_index(SourceIndex src_index) const {
      return list_start_indices.at(src_index.listIndex) + src_index.index;
    }
      
    auto list_size(int list_index) const {
      return list_sizes.at(list_index);
    }
      
    auto num_in_state(int first, int count, State state) const {
      int total{};
      for (int i{}; i < count; ++i) {
        if (list.at(first + i).state == state) {
          ++total;
        }
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
      std::vector<result_t>& results, const SourceCompatibilityLists& sources,
      int stream_index) // for logging
    {
      constexpr static const bool logging = false;
      std::set<SourceIndex> compat_src_indices; // for logging
      int num_compatible{};
      int num_done{};
      for (size_t i{}; i < sourceIndices.size(); ++i) {
        const auto src_index = sourceIndices.at(i);
        auto& indexState = list.at(src_index.listIndex);
        const auto result = results.at(i);

        if constexpr (logging) {
          if (0) { // src_index.index == 0) {// && other conditions
            std::cout << "stream " << stream_index << ", "
                      << src_index.listIndex << ":" << src_index.index
                      << ", results index: " << i
                      << ", result: " << unsigned(result)
                      << ", compat: " << std::boolalpha << bool(result)
                      << ", ready: " << indexState.ready_state()
                      << std::endl;
          }
        }
        if (!indexState.ready_state()) {
          // for debugging purposes, set duplicate results for the same
          // sourcelist" results to 0, so we can later determine the exact
          // set of "first matched sources".
          results.at(i) = 0;
          continue;
        }
        if (result > 0) {
          indexState.state = State::compatible;
          ++num_compatible;
          if constexpr (logging) compat_src_indices.insert(src_index);
        }
        else {
          // if this is the result for the last source in a sourcelist,
          // mark the list (indexState) as done.
          auto sourcelist_size = sources.at(src_index.listIndex).size();
          if (src_index.index >= sourcelist_size) {
            indexState.state = State::done;
          }
        }
      }
      if constexpr (logging) {
        if (compat_src_indices.size()) {
          for (const auto& src_index: compat_src_indices) {
            std::cout << src_index.listIndex << ":" << src_index.index
                      << std::endl;
          }
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
    std::vector<uint32_t> list_start_indices;
    std::vector<uint32_t> list_sizes;
  }; // struct IndexStates

  //////////

  std::vector<cudaStream_t> streams;
  int host_special{};
  
  // the pointers in this are allocated in device memory
  struct KernelData {
  private:
    static const auto num_cores = 1280;
    static const auto max_chunks = 20ul;

    static auto calc_stride(const int num_sourcelists) {
      return num_sourcelists;
      /*
      const auto num_chunks = num_sourcelists / max_workitems() + 1;
      assert((num_chunks < max_chunks) && "chunks not supported (but could be)");
      const auto stride = num_sourcelists / num_chunks;
      assert((stride < max_workitems()) && "stride not supported (but could be)");
      return stride;
      */
    }

  public:
    static auto calc_num_streams(const size_t num_sourcelists) {
      const auto num_strides = num_sourcelists / calc_stride(num_sourcelists);
      return std::min(20ul, num_strides + num_strides / 2 + 1);
    }

    static void init(std::vector<KernelData>& kernelVec,
      const size_t num_sourcelists)
    {
      const auto stride = calc_stride(num_sourcelists);
      auto num_primary_streams = num_sourcelists / stride;
      int leftovers = num_sourcelists % stride;
      int start_index{};
      for (size_t i{}; i < kernelVec.size(); ++i) {
        auto& kernel = kernelVec.at(i);
        if (i < num_primary_streams) {
          kernel.list_start_index = start_index;
          int share_of_leftovers = leftovers ? (--leftovers, 1) : 0;
          // this separate accounting of "number of indices" is necessary
          // because source_indices.size() may change, but the number of
          // lists that this stream is concerned with stays constant.
          kernel.num_list_indices = stride + share_of_leftovers;
          start_index += kernel.num_list_indices;
          kernel.source_indices.resize(kernel.num_list_indices);
          
        } else {
          // mark all "extra" streams as unattachle, with no work remaining
          kernel.is_attachable = false;
          kernel.source_indices.resize(0);
        }
        if (i >= streams.size()) {
          cudaStream_t stream;
          cudaError_t err = cudaStreamCreate(&stream);
          assert((err == cudaSuccess) && "failed to create stream");
          streams.push_back(stream);
        }
        kernel.stream_index = i;
        kernel.stream = streams[i];
      }
    }

    static int min_workitems(int override = 0) {
      static int the_min_workitems = num_cores + (num_cores / 2);
      if (override) the_min_workitems = override;
      return the_min_workitems;
    }

    static int max_workitems(int override = 0) {
      static int the_max_workitems = 15000; // 2 * num_cores;
      if (override) the_max_workitems = override;
      return the_max_workitems;
    }

    static int next_sequence_num() {
      static int sequence_num{};
      return sequence_num++;
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
      //std::set<int> list_indices_used{};
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
      // TODO: I don't think this is necessary
      auto num_ready = indexStates.num_ready(list_start_index,
        num_list_indices);
      int num_sourcelists{};
      if (num_ready) {
        num_sourcelists = fillSourceIndices(indexStates, num_list_indices);
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

    void allocCopy([[maybe_unused]] const IndexStates& indexStates) {
      cudaError_t err = cudaSuccess;
      auto indices_bytes = source_indices.size() * sizeof(SourceIndex); // TODO: max_workitems()
      // alloc source indices
      if (!device_source_indices) {
        err = cudaMallocAsync((void **)&device_source_indices, indices_bytes,
          stream);
        assert((err == cudaSuccess) && "allocate source indices");
      }

      /*
      std::vector<index_t> flat_indices;
      flat_indices.reserve(source_indices.size());
      for (const auto& src_index: source_indices) {
        flat_indices.push_back(indexStates.flat_index(src_index));
      }
      */

      // copy source indices
      err = cudaMemcpyAsync(device_source_indices, source_indices.data(),
        indices_bytes, cudaMemcpyHostToDevice, stream);
      assert((err == cudaSuccess) && "failed to copy source indices");
      
      // alloc results
      if (!device_results) {
        auto results_bytes = source_indices.size() * sizeof(result_t); // max_workitems()
        err = cudaMallocAsync((void **)&device_results, results_bytes, stream);
        assert((err == cudaSuccess) && "failed to allocate results");
      }
    }

    auto hasWorkRemaining() const {
      return !source_indices.empty();
    }

    void attach(struct KernelData& kernel) {
      assert(!is_running && !is_attached() && kernel.is_running);
      num_attached = 0;
      attached_to = kernel.stream_index;
      is_attachable = false;
      has_run = true;
      list_start_index = kernel.list_start_index;
      num_list_indices = kernel.num_list_indices;
      // signal work remaining
      source_indices.resize(1);

      kernel.num_attached++;

#ifdef STREAM_LOG
      std::cerr << "stream " << stream_index << " attaching to stream "
                << kernel.stream_index << std::endl;
#endif
    }

    bool is_attached() const { return attached_to > -1; }

    int list_start_index; // starting index in SourceCompatibiliityLists
    int num_list_indices; // # of above list entries we are concerned with
    int stream_index{ -1 };
    int num_attached{};
    int sequence_num{};
    int attached_to{ -1 };      // stream_index of stream we're attached to
    bool is_attachable{ true }; // can be attached to 
    bool is_running{ false };   // is running (may be complete; output not retrieved)
    bool has_run{ false };      // has run at least once
    SourceIndex* device_source_indices{ nullptr }; // in
    result_t *device_results{ nullptr }; // out
    cudaStream_t stream{ nullptr };
    std::vector<SourceIndex> source_indices;  // .size() == num_results
    //result_t *device_compat_indices{ nullptr }; 
  }; // struct KernelData

  //////////

  struct ValueIndex {
    int value{};
    int index{ -1 };
  };

  auto anyWithWorkRemaining(const std::vector<KernelData>& kernelVec,
    bool attachable_only = false) -> std::optional<int>
  {
    // NOTE: list initialization
    ValueIndex fewest_attached = { std::numeric_limits<int>::max() };
    for (size_t i{}; i < kernelVec.size(); ++i) {
      const auto& kernel = kernelVec[i];
      if (kernel.hasWorkRemaining() &&
        (!attachable_only || kernel.is_attachable))
      {
        if (kernel.num_attached < fewest_attached.value) {
          fewest_attached.value = kernel.num_attached;
          fewest_attached.index = i;
          if (!fewest_attached.value) break;
        }
      }
    }
    if (fewest_attached.index > -1) {
      return std::make_optional(fewest_attached.index);
    }
    return std::nullopt;
  }

  bool anyIdleWithWorkRemaining(const std::vector<KernelData>& kernelVec,
    int& index)
  {
    for (size_t i{}; i < kernelVec.size(); ++i) {
      const auto& kernel = kernelVec[i];
      if (!kernel.is_running && kernel.hasWorkRemaining()) {
        index = i;
        return true;
      }
    }
    return false;
  }

  auto anyReadyToAttach(const std::vector<KernelData>& kernelVec)
    -> std::optional<int>
  {
    for (size_t i{}; i < kernelVec.size(); ++i) {
      const auto& kernel = kernelVec[i];
      if (!kernel.is_running && !kernel.is_attached() &&
          !kernel.hasWorkRemaining())
      {
        return std::make_optional(i);
      }
    }
    return std::nullopt;
  }

  // TODO: std::optional, and above here
  bool anyRunningComplete(const std::vector<KernelData>& kernelVec,
    int& index)
  {
    // NOTE: list initialization
    ValueIndex lowest = { std::numeric_limits<int>::max() };
    for (size_t i{}; i < kernelVec.size(); ++i) {
      const auto& kernel = kernelVec[i];
      if (kernel.is_running && (cudaSuccess == cudaStreamQuery(kernel.stream))) {
        if (kernel.sequence_num < lowest.value) {
          lowest.value = kernel.sequence_num;
          lowest.index = i;
        }
      }
    }
    if (lowest.index > -1) {
      index = lowest.index;
      return true;
    }
    return false;
  }

  bool getNextWithWorkRemaining(std::vector<KernelData>& kernelVec,
    int& current)
  {
    using namespace std::chrono_literals;

    // First: ensure all primary streams have started at least once
    if (++current >= (int)kernelVec.size()) {
      current = 0;
    } else {
      const auto& kernel = kernelVec[current];
      if (!kernel.is_running && !kernel.has_run && kernel.hasWorkRemaining()) {
        return true;
      }
    }

    // Second: process results for any "running" stream that has completed
    if (anyRunningComplete(kernelVec, current)) {
      return true;
    }

    // Third: run any idle (non-running) stream with work remaining
    if (anyIdleWithWorkRemaining(kernelVec, current)) {
      return true;
    }

    // There is no idle stream that has work remaining. Is there an attachable
    // running stream that has work remaining?
    std::optional<int> opt_attachable = anyWithWorkRemaining(kernelVec, true);
    if (opt_attachable.has_value()) {
      // Is there an idle stream that can attach to it? If so, attach it.
      std::optional<int> opt_attach = anyReadyToAttach(kernelVec);
      if (opt_attach.has_value()) {
        kernelVec.at(opt_attach.value())
          .attach(kernelVec.at(opt_attachable.value()));
        current = opt_attach.value();
        return true;
      }
    } else {
      // There is no idle stream, and no attachable running stream that has work
      // remaining. Is there any stream with work remaining? If not, we're done.
      if (!anyWithWorkRemaining(kernelVec).has_value()) {
        return false;
      }
    }

    // Wait for one to complete.
    while (!anyRunningComplete(kernelVec, current)) {
      // TODO events
      std::this_thread::sleep_for(10ms);
      //std::this_thread::yield();
    }
    return true;
  }

  void runKernel(KernelData& kernel,
    int threads_per_block,
    const SourceCompatibilityData* device_sources,
    const index_t* device_list_start_indices)
  {
    auto num_sources = kernel.source_indices.size();
    auto num_sm = 10;
    auto threads_per_sm = 2048;
    auto block_size = threads_per_block ? threads_per_block : 128;
    auto blocks_per_sm = threads_per_sm / block_size;
    assert(blocks_per_sm * block_size == threads_per_sm);
    
    //auto blocksPerGrid = (num_sources + threadsPerBlock - 1) / threadsPerBlock;
    auto grid_size = num_sm * blocks_per_sm; // aka blocks per grid
    auto shared_bytes = block_size * sizeof(uint16_t) +
      sizeof(SourceCompatibilityData);
    kernel.is_running = true;
    kernel.sequence_num = KernelData::next_sequence_num();
    dim3 grid_dim(grid_size);
    dim3 block_dim(block_size);
    prepareXorKernel();
    xorKernel<<<grid_dim, block_dim, shared_bytes, kernel.stream>>>(
      device_sources, num_sources,
      PCD.device_xorSources, PCD.xorSourceList.size(),
      PCD.device_sentenceVariationIndices, kernel.device_source_indices,
      device_list_start_indices, kernel.device_results, kernel.stream_index,
      host_special);

#if 1 || defined(STREAM_LOG)
    std::cerr << "stream " << kernel.stream_index
              << " started with " << grid_size << " blocks"
              << " of " << block_size << " threads"
             //<< " starting, sequence: " << kernel.sequence_num
              << std::endl;
#endif
  }

  // todo: kernel.getResults()
  auto getKernelResults(KernelData& kernel) {
    cudaError_t err = cudaStreamSynchronize(kernel.stream);
    if (err != cudaSuccess) {
      std::cerr << "Failed to synchronize, error: "
                << cudaGetErrorString(err) << std::endl;
      assert((err == cudaSuccess) && "sychronize");
    }

    auto num_sources = kernel.source_indices.size();
    std::vector<result_t> results(num_sources);
    auto results_bytes = num_sources * sizeof(result_t);
    #if 1
    err = cudaMemcpyAsync(results.data(), kernel.device_results,
      results_bytes, cudaMemcpyDeviceToHost, kernel.stream);
    #else
    err = cudaMemcpy(results.data(), kernel.device_results,
      results_bytes, cudaMemcpyDeviceToHost);
    #endif
    if (err != cudaSuccess) {
      std::cerr << "Failed to copy device results, error: "
                << cudaGetErrorString(err) << std::endl;
      assert(!"failed to copy results from device -> host");
    }
    kernel.is_running = false;
    return results;
  }

  void showAllNumReady(const std::vector<KernelData>& kernels,
    const IndexStates& indexStates)
  {
    for (auto& k: kernels) {
      std::cerr << "  stream " << k.stream_index << ": " 
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
    assert((err == cudaSuccess) && "failed to allocate sources");

    // copy sources
    size_t index{};
    for (const auto& sourceList: sources) {
      auto sourceIndices = cm::getSortedSourceIndices(sourceList, false);
      if (sourceIndices.size()) {
        for (size_t i{}; i < sourceIndices.size(); ++i) {
          const auto& src = sourceList.at(sourceIndices.at(i));
          err = cudaMemcpy(&device_sources[index++], &src,
            sizeof(SourceCompatibilityData), cudaMemcpyHostToDevice);
          assert((err == cudaSuccess) && "failed to copy source");
        }
      } else {
        err = cudaMemcpy(&device_sources[index], sourceList.data(),
          sourceList.size() * sizeof(SourceCompatibilityData),
          cudaMemcpyHostToDevice);
        assert((err == cudaSuccess) && "failed to copy sources");
        index += sourceList.size();
      }
    }
    return device_sources;
  }
 
  auto* allocCopyListStartIndices(const IndexStates& index_states) {
    // alloc flat indices
    cudaError_t err = cudaSuccess;
    auto indices_bytes =
      index_states.list_start_indices.size() * sizeof(index_t);
    index_t* device_indices;
    err = cudaMalloc((void **)&device_indices, indices_bytes);
    assert((err == cudaSuccess) && "failed to alloc list start indices");
    err = cudaMemcpy(device_indices, index_states.list_start_indices.data(),
      indices_bytes, cudaMemcpyHostToDevice);
    assert((err == cudaSuccess) && "failed to copy list start indices");
    return device_indices;
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

  int median(std::vector<int>& a) {
    const int n = a.size();
    if (!(n % 2)) {
      std::nth_element(a.begin(), a.begin() + n / 2, a.end());
      std::nth_element(a.begin(), a.begin() + (n - 1) / 2, a.end());
      return (a[(n - 1) / 2] + a[n / 2]) / 2;
    }
    std::nth_element(a.begin(), a.begin() + n / 2, a.end());
    return a[n / 2];
  }

  auto sources_stats(const SourceCompatibilityLists& sources) {
    std::vector<int> sizes;
    sizes.reserve(sources.size());
    size_t mode{}, sum{};
    for (const auto& src_list: sources) {
      const auto size = src_list.size();
      if (size > mode) {
        mode = size;
      }
      sum += size;
      sizes.push_back(size);
    }
    // mean, median, mode
    return std::make_tuple(sum / sources.size(), median(sizes), mode);
  }

  auto flat_index(const SourceCompatibilityLists& sources,
    const SourceIndex src_idx)
  {
    uint32_t flat{};
    for (size_t i{}; i < src_idx.listIndex; ++i) {
      flat += sources.at(i).size();
    }
    return flat + src_idx.index;
  }

  void check(const SourceCompatibilityLists& sources, 
    index_t list_index, index_t index)
  {
    constexpr const auto logging = true;
    if constexpr (logging) {
      SourceIndex src_idx{ list_index, index };
      char idx_buf[32];
      char buf[64];
      snprintf(buf, sizeof(buf), "%s, flat: %d", src_idx.as_string(idx_buf),
               flat_index(sources, src_idx));
      auto& source = sources.at(list_index).at(index);
      source.dump(buf);
      int compat_index{ -1 };
      auto compat = isSourceXORCompatibleWithAnyXorSource(source,
        PCD.xorSourceList.data(), PCD.xorSourceList.size(), &compat_index);
      std::cerr << "compat: " << compat << " (" << compat_index << ")"
                << std::endl;
    }
  }

  void dump_xor(int index) {
    const XorSourceList& xorSources = PCD.xorSourceList;
      //const std::vector<int>& /*xorSourceIndices*/, 
    auto host_index = index; //xorSourceIndices.at(index);
    const auto& src = xorSources.at(host_index);
    char buf[32];
    snprintf(buf, sizeof(buf), "xor: device(%d) host(%d)", index, host_index);
    src.dump(buf);
  }
} // anonymous namespace

namespace cm {

void filterCandidatesCuda(int sum, int threads_per_block, int num_streams,
    int workitems)
{
  using namespace std::chrono;

  const auto& sources = allSumsCandidateData.find(sum)->second
    .sourceCompatLists;
  auto device_sources = allocCopySources(sources);
  IndexStates indexStates{ sources };
  auto device_list_start_indices = allocCopyListStartIndices(indexStates);

  check(sources, 50, 0);
  check(sources, 51, 0);
  //dump_xor(37648);

  KernelData::max_workitems(workitems);
  if (!num_streams) num_streams = 2; // KernelData::calc_num_streams(sources.size());
  std::vector<KernelData> kernels(num_streams);
  KernelData::init(kernels, sources.size());
  std::cerr << "sourcelists: " << sources.size()
            << ", streams: " << num_streams
            << std::endl;

  //std::set<int> compat_indices;
  std::vector<std::vector<int>> compat_record(num_streams);
  
  int total_compatible{};
  int current_kernel{}; // { -1 };
  //
  auto t0 = high_resolution_clock::now();
  for (;;) {
    auto& kernel = kernels.at(current_kernel);
    if (!kernel.fillSourceIndices(indexStates)) break;
    std::cerr << "stream " << kernel.stream_index
              << " source_indices: " << kernel.source_indices.size()
              << ", ready: " << kernel.num_ready(indexStates)
              << std::endl;
    kernel.allocCopy(indexStates);
    //assert(kernel.source_indices.size() <= KernelData::max_workitems());
    auto k0 = high_resolution_clock::now();
    if (!kernel.is_running) {
      runKernel(kernel, threads_per_block, device_sources,
        device_list_start_indices);
    }
    auto r0 = high_resolution_clock::now();

    auto results = getKernelResults(kernel);

    auto r1 = high_resolution_clock::now();
    auto d_results = duration_cast<milliseconds>(r1 - r0).count();
    auto k1 = high_resolution_clock::now();
    auto d_kernel = duration_cast<milliseconds>(k1 - k0).count();

    auto num_compatible = indexStates.update(kernel.source_indices,
      results, sources, kernel.stream_index);
    std::cerr << "stream " << kernel.stream_index
              << " compat results: " << num_compatible << std::endl;
    total_compatible += num_compatible;
  }
  auto t1 = high_resolution_clock::now();
  //printCompatRecord(compat_record);
  auto d_total = duration_cast<milliseconds>(t1 - t0).count();
  std::cerr << "total compatible: " << total_compatible
            << " of " << sources.size()
            << " - " << d_total << "ms"
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
  assert();

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
