// filter.cu

#include <algorithm>
#include <chrono>
#include <exception>
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

  template<typename T>
  __device__ T load(const T* addr) {
    return *(const volatile T*)addr;
  }

  template<typename T>
  __device__ void store(T* addr, T val) {
    *(volatile T*)addr = val;
  }

  __global__
  void xorKernel(const SourceCompatibilityData* __restrict__ sources,
    const unsigned num_sources, const XorSource* __restrict__ xor_sources,
    const unsigned num_xor_sources,
    const SourceIndex* __restrict__ source_indices,
    const index_t* __restrict__ list_start_indices,
    result_t* results, int stream_idx)
  {
    // should only happen with very low xor_sources count 
    // 64 * 63 + 63 = 4095 
    //const auto thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    // I question the validity of this.
    // should it not be > max(num_sources, num_xor_sources)
    // does this check matter? (performance)
    //if (thread_id >= num_xor_sources) return;

    // TODO: memset_async before call?
    // zero-out results
    const auto threads_per_grid = gridDim.x * blockDim.x;
    /*
    for (int start_idx = blockIdx.x * blockDim.x;
      start_idx + threadIdx.x < num_sources;
      start_idx += threads_per_grid)
    {
      results[start_idx + threadIdx.x] = 0;
    }
    __syncthreads();
    */

    // for each source
    for (unsigned idx{}; idx < num_sources; ++idx) {
      const auto src_idx = source_indices[idx];
      const auto flat_index = list_start_indices[src_idx.listIndex] +
        src_idx.index;
      const auto& source = sources[flat_index];
      auto& result = results[src_idx.listIndex];

      // for each xor_source
      for (int start_idx = blockIdx.x * blockDim.x;
        start_idx + threadIdx.x < num_xor_sources;
        start_idx += threads_per_grid)
      {
        if (load(&result)) break;
        if (source.isXorCompatibleWith(xor_sources[start_idx + threadIdx.x],
          false))
        {
          store(&result, (uint8_t)1);
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
    
    index_t flat_index(SourceIndex src_index) const {
      return list_start_indices.at(src_index.listIndex) + src_index.index;
    }
      
    auto list_size(index_t list_index) const {
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
 
    auto update(const std::vector<SourceIndex>& src_indices,
      const std::vector<result_t>& results,
      int stream_idx) // for logging
    {
      //std::cerr << "update" << std::endl;

      constexpr static const bool logging = false;
      //std::set<SourceIndex> compat_src_indices; // for logging
      int num_compatible{};
      int num_done{};
      for (size_t i{}; i < src_indices.size(); ++i) {
        const auto src_idx = src_indices.at(i);
        auto& idx_state = list.at(src_idx.listIndex);
        const auto result = results.at(src_idx.listIndex);

        if constexpr (logging) {
          if (0) { // src_idx.index == 0) {// && other conditions
            std::cout << "stream " << stream_idx << ", "
                      << src_idx.listIndex << ":" << src_idx.index
                      << ", results index: " << i
                      << ", result: " << unsigned(result)
                      << ", compat: " << std::boolalpha << bool(result)
                      << ", ready: " << idx_state.ready_state()
                      << std::endl;
          }
        }
        if (!idx_state.ready_state()) {
          continue;
        }
        if (result > 0) {
          idx_state.state = State::compatible;
          ++num_compatible;
          //if constexpr (logging) compat_src_indices.insert(src_idx);
        }
        else if (src_idx.index == list_sizes.at(src_idx.listIndex) - 1) {
          // if this is the result for the last source in a sourcelist,
          // mark the list (indexState) as done.
          idx_state.state = State::done;
          ++num_done;
        }
      }
      #if 0
      if (compat_src_indices.size()) {
        for (const auto& src_idx: compat_src_indices) {
          std::cout << src_idx.listIndex << ":" << src_idx.index
                    << std::endl;
        }
      }
      #endif
      #if 0
      std::cerr << "stream " << stream_idx
                << " update, total: " << src_indices.size()
                << ", compat: " << num_compatible
                << ", done: " << num_done << std::endl;
      #endif
      return num_compatible;
    }

    auto get(index_t list_index) const {
      return list.at(list_index);
    }

    auto get_and_increment_index(index_t list_index)
      -> std::optional<SourceIndex>
    {
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
    
    int num_lists() const {
      return list.size();
    }

    auto get_next_fill_idx() {
      auto fill_idx = next_fill_idx;
      if (++next_fill_idx >= num_lists()) next_fill_idx = 0;
      return fill_idx;
    }

    bool done{ false };
    int next_fill_idx{ 0 };
    std::vector<Data> list;
    std::vector<uint32_t> list_start_indices;
    std::vector<uint32_t> list_sizes;
  }; // struct IndexStates

  //////////

  std::vector<cudaStream_t> streams;
  
  // the pointers in this are allocated in device memory
  struct KernelData {
  private:
    using hr_time_point_t = decltype(std::chrono::high_resolution_clock::now());
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
       size_t num_sourcelists, size_t stride)
    {
      stride = std::min(num_sourcelists, stride);
      for (size_t i{}; i < kernelVec.size(); ++i) {
        auto& kernel = kernelVec.at(i);
        kernel.num_src_lists = num_sourcelists;
        kernel.num_list_indices = stride;
        kernel.source_indices.resize(kernel.num_list_indices);
        if (i >= streams.size()) {
          cudaStream_t stream;
          cudaError_t err = cudaStreamCreate(&stream);
          assert((err == cudaSuccess) && "failed to create stream");
          streams.push_back(stream);
        }
        kernel.stream_idx = i;
        kernel.stream = streams[i];
      }
    }

    static int next_sequence_num() {
      static int sequence_num{};
      return sequence_num++;
    }

    //

    int num_ready(const IndexStates& indexStates) const {
      return indexStates.num_ready(0, num_list_indices);
    }

    int num_done(const IndexStates& indexStates) const {
      return indexStates.num_done(0, num_list_indices);
    }

    int num_compatible(const IndexStates& indexStates) const {
      return indexStates.num_compatible(0, num_list_indices);
    }

    auto fillSourceIndices(IndexStates& idx_states, int max_idx) {
      //std::cerr << "starting next_fill_idx: " << idx_states.next_fill_idx << std::endl;
      source_indices.resize(idx_states.done ? 0 : max_idx);
      for (int idx{}; !idx_states.done && (idx < max_idx);) {
        auto num_skipped_idx{ 0 }; // how many idx were skipped in a row
        // breaks in loop logic necessary to keep fill_idx in sync
        for (auto list_idx = idx_states.get_next_fill_idx(); /*nada*/;
          list_idx = idx_states.get_next_fill_idx())
        {
          const auto opt_src_idx =
            idx_states.get_and_increment_index(list_idx);
          if (opt_src_idx.has_value()) {
            const auto src_idx = opt_src_idx.value();
            assert(src_idx.listIndex == list_idx);
            source_indices.at(idx++) = src_idx;
            if (idx >= max_idx) break;
            num_skipped_idx = 0;
          } else if (++num_skipped_idx >= idx_states.num_lists()) {
            // we've skipped over the entire list (with index overlap)
            // and haven't consumed any indices. nothing left to do.
            idx_states.done = true;
            source_indices.resize(idx);
            break;
          }
        }
      }
      #if 0
      std::cerr << "ending next_fill_idx: " << idx_states.next_fill_idx << std::endl;
      std::cerr << "stream " << stream_idx
                << " filled " << source_indices.size()
                << " of " << max_idx
                << ", first = " << (source_indices.empty() ? -1 : (int)source_indices.front().listIndex)
                << ", last = " << (source_indices.empty() ? -1 : (int)source_indices.back().listIndex)
                << ", done: " << std::boolalpha << idx_states.done
                << std::endl;
      #endif
      return !source_indices.empty();
    }

    bool fillSourceIndices(IndexStates& idx_states) {
      return fillSourceIndices(idx_states, num_list_indices);
    }

    void allocCopy([[maybe_unused]] const IndexStates& idx_states) {
      cudaError_t err = cudaSuccess;
      auto indices_bytes = source_indices.size() * sizeof(SourceIndex);
      // alloc source indices
      if (!device_source_indices) {
        err = cudaMallocAsync((void **)&device_source_indices, indices_bytes,
          stream);
        assert((err == cudaSuccess) && "allocate source indices");
      }

      /*
      std::vector<index_t> flat_indices;
      flat_indices.reserve(source_indices.size());
      for (const auto& src_idx: source_indices) {
        flat_indices.push_back(idx_states.flat_index(src_idx));
      }
      */

      // copy source indices
      err = cudaMemcpyAsync(device_source_indices, source_indices.data(),
        indices_bytes, cudaMemcpyHostToDevice, stream);
      assert((err == cudaSuccess) && "copy source indices");
      
      // alloc results
      /*
      if (!device_results) {
        auto results_bytes = source_indices.size() * sizeof(result_t); // max_workitems()
        err = cudaMallocAsync((void **)&device_results, results_bytes, stream);
        assert((err == cudaSuccess) && "failed to allocate results");
      }
      */
    }

    auto hasWorkRemaining() const {
      return !source_indices.empty();
    }

    /*
    void attach(struct KernelData& kernel) {
      assert(!is_running && !is_attached() && kernel.is_running);
      num_attached = 0;
      attached_to = kernel.stream_idx;
      is_attachable = false;
      has_run = true;
      list_start_index = kernel.list_start_index;
      num_list_indices = kernel.num_list_indices;
      // signal work remaining
      source_indices.resize(1);

      kernel.num_attached++;

#ifdef STREAM_LOG
      std::cerr << "stream " << stream_idx << " attaching to stream "
                << kernel.stream_idx << std::endl;
#endif
    }
    
    bool is_attached() const { return attached_to > -1; }
    */

    void dump() const {
      std::cerr << "kernel " << stream_idx
                << ", is_running: " << std::boolalpha << is_running
                << ", src_idx: " << source_indices.size()
                << std::endl;
    }

    //int list_start_index; // starting index in SourceCompatibiliityLists
    int num_src_lists; // total # of sourcelists (== # of device_results) (doesn't belong here)
    int num_list_indices;
    int stream_idx{ -1 };
    //int num_attached{};
    int sequence_num{};
    //int attached_to{ -1 };      // stream_idx of stream we're attached to
    //bool is_attachable{ true }; // can be attached to 
    bool is_running{ false };   // is running (may be complete; output not retrieved)
    bool has_run{ false };      // has run at least once
    SourceIndex* device_source_indices{ nullptr }; // in
    //result_t *device_results{ nullptr }; // out
    cudaStream_t stream{ nullptr };
    std::vector<SourceIndex> source_indices;  // .size() == num_results
    hr_time_point_t start_time;
    //result_t *device_compat_indices{ nullptr }; 
  }; // struct KernelData

  //////////

  struct ValueIndex {
    int value{};
    int index{ -1 };
  };

  auto anyWithWorkRemaining(const std::vector<KernelData>& kernelVec)
    -> std::optional<int>
  {
    for (size_t i{}; i < kernelVec.size(); ++i) {
      const auto& kernel = kernelVec[i];
      if (kernel.hasWorkRemaining()) {
        return std::make_optional(i);
      }
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

  /*
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
  */

  // TODO: std::optional, and above here
  bool anyRunningComplete(const std::vector<KernelData>& kernelVec,
    int& index)
  {
    ValueIndex lowest = { std::numeric_limits<int>::max() };
    for (size_t i{}; i < kernelVec.size(); ++i) {
      const auto& kernel = kernelVec[i];
      if (kernel.is_running &&
        (cudaSuccess == cudaStreamQuery(kernel.stream)))
      {
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

  bool get_next_available(std::vector<KernelData>& kernelVec, int& current) {
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
      //std::cerr << "RunningComplete: " << current << std::endl;
      return true;
    }

    // Third: run any idle (non-running) stream with work remaining
    if (anyIdleWithWorkRemaining(kernelVec, current)) {
      //std::cerr << "IdleWorkRemaining: " << current << std::endl;
      return true;
    }

    // There is no idle stream that has work remaining. Is there an attachable
    // running stream that has work remaining?
    /*
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
    */
    // There is no idle stream, and no attachable running stream that has work
    // remaining. Is there any stream with work remaining? If not, we're done.
    if (!anyWithWorkRemaining(kernelVec).has_value()) {
      //std::cerr << "!anyWithWorkRemaining" << std::endl;
      return false;
    }
    //}

    // Wait for one to complete.
    while (!anyRunningComplete(kernelVec, current)) {
      std::this_thread::sleep_for(5ms);
    }
    //std::cerr << "waitedToComplete: " << current << std::endl;
    return true;
  }

  void runKernel(KernelData& kernel,
    int threads_per_block,
    const SourceCompatibilityData* device_sources,
    result_t* device_results,
    const index_t* device_list_start_indices)
  {
    auto num_sm = 10;
    auto threads_per_sm = 2048;
    auto block_size = threads_per_block ? threads_per_block : 128;
    auto blocks_per_sm = threads_per_sm / block_size;
    assert(blocks_per_sm * block_size == threads_per_sm);
    auto grid_size = num_sm * blocks_per_sm; // aka blocks per grid
    auto shared_bytes = 0; // block_size * sizeof(block_result_t);

    kernel.is_running = true;
    kernel.sequence_num = KernelData::next_sequence_num();
    kernel.start_time = std::chrono::high_resolution_clock::now();
    dim3 grid_dim(grid_size);
    dim3 block_dim(block_size);
    xorKernel<<<grid_dim, block_dim, shared_bytes, kernel.stream>>>(
      device_sources, kernel.source_indices.size(),
      PCD.device_xorSources, PCD.xorSourceList.size(),
      kernel.device_source_indices,
      device_list_start_indices, device_results, kernel.stream_idx);

    #if 0 || defined(STREAM_LOG)
    std::cerr << "stream " << kernel.stream_idx
              << " started with " << grid_size << " blocks"
              << " of " << block_size << " threads"
             //<< " starting, sequence: " << kernel.sequence_num
              << std::endl;
    #endif
  }

  // todo: kernel.getResults()
  auto getKernelResults(KernelData& kernel, result_t* device_results) {
    cudaError_t err = cudaStreamSynchronize(kernel.stream);
    if (err != cudaSuccess) {
      std::cerr << "Failed to synchronize, error: "
                << cudaGetErrorString(err) << std::endl;
      assert((err == cudaSuccess) && "sychronize");
    }

    // TODO this could go into kernelData
    std::vector<result_t> results(kernel.num_src_lists);
    auto results_bytes = kernel.num_src_lists * sizeof(result_t);
    err = cudaMemcpyAsync(results.data(), device_results, 
      results_bytes, cudaMemcpyDeviceToHost, kernel.stream);
    if (err != cudaSuccess) {
      std::cerr << "copy device results, error: "
                << cudaGetErrorString(err) << std::endl;
      assert(!"copy results from device");
    }
    err = cudaStreamSynchronize(kernel.stream);
    assert((err == cudaSuccess) && "cudaStreamSynchronize");
    kernel.is_running = false;
    return results;
  }

  void showAllNumReady(const std::vector<KernelData>& kernels,
    const IndexStates& idx_states)
  {
    for (auto& k: kernels) {
      std::cerr << "  stream " << k.stream_idx << ": " 
                << k.num_ready(idx_states) << std::endl;
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
    cudaStream_t stream = cudaStreamPerThread;
    cudaError_t err = cudaSuccess;
    auto sources_bytes = count(sources) * sizeof(SourceCompatibilityData);
    SourceCompatibilityData* device_sources;
    err = cudaMallocAsync((void **)&device_sources, sources_bytes, stream);
    assert((err == cudaSuccess) && "failed to allocate sources");

    // copy sources
    size_t index{};
    for (const auto& sourceList: sources) {
      auto sourceIndices = cm::getSortedSourceIndices(sourceList, false);
      if (sourceIndices.size()) {
        for (size_t i{}; i < sourceIndices.size(); ++i) {
          const auto& src = sourceList.at(sourceIndices.at(i));
          err = cudaMemcpyAsync(&device_sources[index++], &src,
            sizeof(SourceCompatibilityData), cudaMemcpyHostToDevice, stream);
          assert((err == cudaSuccess) && "failed to copy source");
        }
      } else {
        err = cudaMemcpyAsync(&device_sources[index], sourceList.data(),
          sourceList.size() * sizeof(SourceCompatibilityData),
          cudaMemcpyHostToDevice, stream);
        assert((err == cudaSuccess) && "failed to copy sources");
        index += sourceList.size();
      }
    }
    return device_sources;
  }
 
  auto allocZeroResults(size_t num_results) { // TODO cudaStream_t stream) {
    cudaStream_t stream = cudaStreamPerThread;
    cudaError_t err = cudaSuccess;
    // alloc results
    auto results_bytes = num_results * sizeof(result_t);
    result_t* device_results;
    err = cudaMallocAsync((void **)&device_results, results_bytes, stream);
    assert((err == cudaSuccess) && "alloc results");
    err = cudaMemsetAsync(device_results, results_bytes, 0, stream);
    assert((err == cudaSuccess) && "zero results");
    return device_results;
  }

  auto* allocCopyListStartIndices(const IndexStates& index_states) {
    cudaStream_t stream = cudaStreamPerThread;
    cudaError_t err = cudaSuccess;
    // alloc indices
    auto indices_bytes =
      index_states.list_start_indices.size() * sizeof(index_t);
    index_t* device_indices;
    err = cudaMallocAsync((void **)&device_indices, indices_bytes, stream);
    assert((err == cudaSuccess) && "alloc list start indices");
    // copy indices
    err = cudaMemcpyAsync(device_indices, index_states.list_start_indices.data(),
      indices_bytes, cudaMemcpyHostToDevice, stream);
    assert((err == cudaSuccess) && "copy list start indices");
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
  int stride)
{
  using namespace std::chrono;
  cudaError_t err = cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 7'500'000);

  const auto& sources = allSumsCandidateData.find(sum)->second
    .sourceCompatLists;
  auto device_sources = allocCopySources(sources);
  IndexStates idx_states{ sources };
  auto device_list_start_indices = allocCopyListStartIndices(idx_states);

  if (!num_streams) num_streams = 1;
  //KernelData::max_workitems(workitems);
  std::vector<KernelData> kernels(num_streams);
  stride = stride ? stride : 5000;
  KernelData::init(kernels, sources.size(), stride);
  std::cerr << "sourcelists: " << sources.size()
            << ", streams: " << num_streams
            << std::endl;

  auto device_results = allocZeroResults(sources.size());

  //std::set<int> compat_indices;
  std::vector<std::vector<int>> compat_record(num_streams);
  
  int total_compatible{};
  int current_kernel{ -1 };
  int actual_num_compat{};
  //
  auto t0 = high_resolution_clock::now();
  while (get_next_available(kernels, current_kernel)) {
    auto& kernel = kernels.at(current_kernel);
    if (!kernel.is_running) {
      if (!kernel.fillSourceIndices(idx_states)) {
        kernel.is_running = false;
        continue;
      }
      #if 0
      std::cerr << "stream " << kernel.stream_idx
                << " source_indices: " << kernel.source_indices.size()
                << ", ready: " << kernel.num_ready(idx_states)
                << std::endl;
      #endif
      kernel.allocCopy(idx_states);
      //assert(kernel.source_indices.size() <= KernelData::max_workitems());
      runKernel(kernel, threads_per_block, device_sources, device_results,
        device_list_start_indices);
      continue;
    }

    kernel.has_run = true;
    kernel.is_running = false;

    auto r0 = high_resolution_clock::now();

    auto results = getKernelResults(kernel, device_results);

    auto r1 = high_resolution_clock::now();
    auto d_results = duration_cast<milliseconds>(r1 - r0).count();
    auto k1 = high_resolution_clock::now();
    auto d_kernel = duration_cast<milliseconds>(k1 - kernel.start_time).count();

    auto num_compatible = idx_states.update(kernel.source_indices,
      results, kernel.stream_idx);
    #if 0
    actual_num_compat = std::accumulate(results.begin(), results.end(), 0,
      [](int sum, result_t r) { return r ? sum + 1 : sum; });
    std::cerr << "stream " << kernel.stream_idx
              << " compat results: " << num_compatible
              << " actual: " << actual_num_compat
              << " - results: " << d_results << "ms"
              << ", kernel: " << d_kernel << "ms"
              << std::endl;
    #endif
    total_compatible += num_compatible;
  }
  auto t1 = high_resolution_clock::now();
  #if 0
  for (size_t i{}; i < kernels.size(); ++i) {
    kernels.at(i).dump();
  }
  #endif
  auto d_total = duration_cast<milliseconds>(t1 - t0).count();
  std::cerr << "total compatible: " << total_compatible
            << ", actual: " << actual_num_compat
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
