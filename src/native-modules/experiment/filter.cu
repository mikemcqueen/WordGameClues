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

//#define STREAM_LOG

namespace {

using namespace cm;

__device__ auto isSourceORCompatibleWithAnyOrSource(
  const SourceCompatibilityData& source, const OrSourceData* or_sources,
  unsigned num_or_sources) {
  //
  auto compatible = false;
  for (unsigned i{}; i < num_or_sources; ++i) {
    // skip any sources that were already determined to be XOR incompatible
    // or AND compatible with --xor sources.
    // Wait, what? why not "&& !andCompatible" ?
    const auto& or_src = or_sources[i];
    if (!or_src.xorCompatible || or_src.andCompatible)
      continue;
    compatible = source.isOrCompatibleWith(or_src.source);
    if (compatible)
      break;
  }
  return compatible;
};

__device__ auto isSourceCompatibleWithEveryOrArg(
  const SourceCompatibilityData& source, const device::OrArgData* or_args,
  unsigned num_or_args) {
  //
  auto compatible = true;  // if no --or sources specified, compatible == true
  for (unsigned i{}; i < num_or_args; ++i) {
    // TODO: skip calls to here if container.compatible = true  which may have
    // been determined in Precompute phase @ markAllANDCompatibleOrSources()
    // and skip the XOR check as well in this case.
    const auto& or_arg = or_args[i];
    compatible = isSourceORCompatibleWithAnyOrSource(
      source, or_arg.or_sources, or_arg.num_or_sources);
    if (!compatible)
      break;
  }
  return compatible;
}

__device__ __host__ auto isSourceXORCompatibleWithAnyXorSource(
  const SourceCompatibilityData& source, const XorSource* xorSources,
  size_t numXorSources, int* compat_index = nullptr, int* reason = nullptr) {
  bool compatible = true;
  for (size_t i{}; i < numXorSources; ++i) {
    compatible = source.isXorCompatibleWith(xorSources[i], false, reason);
    if (compatible) {
      if (compat_index)
        *compat_index = i;
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

  constexpr const char* as_string(char* buf) const {
    sprintf(buf, "%d:%d", listIndex, index);
    return buf;
  }
};

template <typename T> __device__ __forceinline__ T load(const T* addr) {
  return *(const volatile T*)addr;
}

template <typename T> __device__ __forceinline__ void store(T* addr, T val) {
  *(volatile T*)addr = val;
}

//__device__ int xor_compat[1024]; // # of blocks

// one block per source
__global__ void xor_kernel_new(
  const SourceCompatibilityData* __restrict__ sources,
  const unsigned num_sources,
  const SourceCompatibilityData* __restrict__ xor_sources,
  const unsigned num_xor_sources, const device::OrArgData* __restrict__ or_args,
  const unsigned num_or_args, const SourceIndex* __restrict__ source_indices,
  const index_t* __restrict__ list_start_indices, result_t* results,
  int stream_idx) {
  //
  __shared__ bool shrd_xor_compat;
  auto& is_xor_compat = shrd_xor_compat; // xor_compat[blockIdx.x];
  // for each source (one block per source)
  for (unsigned idx{blockIdx.x}; idx < num_sources; idx += gridDim.x) {
    const auto src_idx = source_indices[idx];
    const auto flat_index =
      list_start_indices[src_idx.listIndex] + src_idx.index;
    auto& source = sources[flat_index];
    auto& result = results[src_idx.listIndex];

    __syncthreads();
    if (!threadIdx.x) {
      store(&is_xor_compat, false);
    }
    // for each xor_source (one thread per xor_source)
    for (unsigned xor_chunk{}; xor_chunk * blockDim.x < num_xor_sources;
         ++xor_chunk) {
      __syncthreads();
      const auto xor_idx = xor_chunk * blockDim.x + threadIdx.x;
      if (xor_idx < num_xor_sources) {
        if (source.isXorCompatibleWith(xor_sources[xor_idx])) {
          // TODO: Do I need volatile when read/writing shrd mem?
          store(&is_xor_compat, true);
        }
      }
      __syncthreads();
      if (!load(&is_xor_compat)) {
        continue;
      }
      if (isSourceCompatibleWithEveryOrArg(source, or_args, num_or_args)) {
        if (!threadIdx.x) {
          store(&result, (result_t)1);
        }
        break;
      }
    }
  }
}

// one block per source
__global__ void xor_kernel(const SourceCompatibilityData* __restrict__ sources,
  const unsigned num_sources,
  const SourceCompatibilityData* __restrict__ xor_sources,
  const unsigned num_xor_sources,
  const SourceIndex* __restrict__ source_indices,
  const index_t* __restrict__ list_start_indices, result_t* results,
  int stream_idx) {
  // for each source (one block per source)
  for (unsigned idx{blockIdx.x}; idx < num_sources; idx += gridDim.x) {
    const auto src_idx = source_indices[idx];
    const auto flat_index =
      list_start_indices[src_idx.listIndex] + src_idx.index;
    const auto& source = sources[flat_index];
    auto& result = results[src_idx.listIndex];

    // for each xor_source (one thread per xor_source)
    for (unsigned xor_idx{threadIdx.x}; xor_idx < num_xor_sources;
         xor_idx += blockDim.x) {
      if (source.isXorCompatibleWith(xor_sources[xor_idx])) {
        store(&result, (uint8_t)1);
      }
      __syncthreads();
      // TODO: try that warp-broadcast thing here maybe
      if (load(&result))
        break;
    }
  }
}

struct IndexStates {
  enum class State
  {
    ready,
    compatible,
    done
  };

  struct Data {
    constexpr auto ready_state() const {
      return state == State::ready;
    }

    void reset() {
      sourceIndex.index = 0;
      state = State::ready;
    }

    SourceIndex sourceIndex;
    State state = State::ready;
  };

  IndexStates() = delete;
  IndexStates(const SourceCompatibilityLists& sources) {
    list.resize(sources.size());  // i.e. "num_sourcelists"
    std::for_each(list.begin(), list.end(),
      [idx = 0](Data& data) mutable { data.sourceIndex.listIndex = idx++; });
    for (index_t list_start_index{}; const auto& sourceList : sources) {
      list_sizes.push_back(sourceList.size());
      list_start_indices.push_back(list_start_index);
      list_start_index += (index_t)sourceList.size();
    }
  }

  void reset() {
    std::for_each(list.begin(), list.end(),
      [](Data& data) mutable { data.reset(); });
    next_fill_idx = 0;
    done = false;
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
    [[maybe_unused]] int stream_idx)  // for logging
  {
    constexpr static const bool logging = false;
    int num_compatible{};
    int num_done{};
    for (size_t i{}; i < src_indices.size(); ++i) {
      const auto src_idx = src_indices.at(i);
      auto& idx_state = list.at(src_idx.listIndex);
      const auto result = results.at(src_idx.listIndex);
      if (!idx_state.ready_state()) {
        continue;
      }
      if (result > 0) {
        idx_state.state = State::compatible;
        ++num_compatible;
      } else if (src_idx.index == list_sizes.at(src_idx.listIndex) - 1) {
        // if this is the result for the last source in a sourcelist,
        // mark the list (indexState) as done.
        idx_state.state = State::done;
        ++num_done;
      }
    }
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
    -> std::optional<SourceIndex> {
    auto& data = list.at(list_index);
    if (data.ready_state()
        && (data.sourceIndex.index < list_sizes.at(list_index))) {
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
    if (++next_fill_idx >= num_lists())
      next_fill_idx = 0;
    return fill_idx;
  }

  bool done{false};
  int next_fill_idx{0};
  std::vector<Data> list;
  std::vector<uint32_t> list_start_indices;
  std::vector<uint32_t> list_sizes;
};  // struct IndexStates

  //////////

std::vector<cudaStream_t> streams;

// the pointers in this are allocated in device memory
struct KernelData {
private:
  using hr_time_point_t = decltype(std::chrono::high_resolution_clock::now());
  static const auto num_cores = 1280;
  static const auto max_chunks = 20ul;

public:
  static void init(
    std::vector<KernelData>& kernelVec, size_t num_src_lists, size_t stride) {
    stride = std::min(num_src_lists, stride);
    for (size_t i{}; i < kernelVec.size(); ++i) {
      auto& kernel = kernelVec.at(i);
      kernel.num_src_lists = num_src_lists;
      kernel.num_list_indices = stride;
      if (i >= streams.size()) {
        cudaStream_t stream;
        cudaError_t err = cudaStreamCreate(&stream);
        assert((err == cudaSuccess) && "failed to create stream");
        streams.push_back(stream);
      }
      kernel.stream_idx = i;
      kernel.stream = streams[i];
    }
    reset(kernelVec);
  }

  static void reset(std::vector<KernelData>& kernels) {
    for (auto& kernel : kernels) {
      kernel.source_indices.resize(kernel.num_list_indices);
      kernel.is_running = false;
      kernel.has_run = false;
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
    source_indices.resize(idx_states.done ? 0 : max_idx);
    for (int idx{}; !idx_states.done && (idx < max_idx);) {
      auto num_skipped_idx{0};  // how many idx were skipped in a row
      // this loop logic is funky and brittle, but intentional
      for (auto list_idx = idx_states.get_next_fill_idx(); /*nada*/;
           list_idx = idx_states.get_next_fill_idx()) {
        const auto opt_src_idx = idx_states.get_and_increment_index(list_idx);
        if (opt_src_idx.has_value()) {
          const auto src_idx = opt_src_idx.value();
          assert(src_idx.listIndex == list_idx);
          source_indices.at(idx++) = src_idx;
          if (idx >= max_idx)
            break;
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
      err =
        cudaMallocAsync((void**)&device_source_indices, indices_bytes, stream);
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
  }

  auto hasWorkRemaining() const {
    return !source_indices.empty();
  }

  void dump() const {
    std::cerr << "kernel " << stream_idx << ", is_running: " << std::boolalpha
              << is_running << ", source_indices: " << source_indices.size()
              << ", num_list_indices: " << num_list_indices
              << std::endl;
  }

  int num_src_lists;  // total # of sourcelists (== # of device_results)
                      // TODO: doesn't belong here
  int num_list_indices;
  int stream_idx{-1};
  int sequence_num{};
  bool is_running{false};  // is running (true until results retrieved)
  bool has_run{false};     // has run at least once
  SourceIndex* device_source_indices{nullptr};  // in
  cudaStream_t stream{nullptr};
  std::vector<SourceIndex> source_indices;  // hasWorkRemaining = (size() > 0)
  hr_time_point_t start_time;
};  // struct KernelData

//////////

struct ValueIndex {
  int value{};
  int index{-1};
};

auto anyWithWorkRemaining(const std::vector<KernelData>& kernelVec)
  -> std::optional<int> {
  for (size_t i{}; i < kernelVec.size(); ++i) {
    const auto& kernel = kernelVec[i];
    if (kernel.hasWorkRemaining()) {
      return std::make_optional(i);
    }
  }
  return std::nullopt;
}

bool anyIdleWithWorkRemaining(
  const std::vector<KernelData>& kernelVec, int& index) {
  for (size_t i{}; i < kernelVec.size(); ++i) {
    const auto& kernel = kernelVec[i];
    if (!kernel.is_running && kernel.hasWorkRemaining()) {
      index = i;
      return true;
    }
  }
  return false;
}

// TODO: std::optional, and above here
bool anyRunningComplete(const std::vector<KernelData>& kernelVec, int& index) {
  ValueIndex lowest = {std::numeric_limits<int>::max()};
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
    return true;
  }

  // Third: run any idle (non-running) stream with work remaining
  if (anyIdleWithWorkRemaining(kernelVec, current)) {
    return true;
  }

  // There is no idle stream, and no attachable running stream that has work
  // remaining. Is there any stream with work remaining? If not, we're done.
  if (!anyWithWorkRemaining(kernelVec).has_value()) {
    return false;
  }

  // Wait for one to complete.
  while (!anyRunningComplete(kernelVec, current)) {
    std::this_thread::sleep_for(5ms);
  }
  return true;
}

void run_xor_kernel(KernelData& kernel, int threads_per_block,
  const SourceCompatibilityData* device_sources, result_t* device_results,
  const index_t* device_list_start_indices) {
  //
  auto num_sm = 10;
  auto threads_per_sm = 2048;
  auto block_size = threads_per_block ? threads_per_block : 1024;
  auto blocks_per_sm = threads_per_sm / block_size;
  //  assert(blocks_per_sm * block_size == threads_per_sm);
  auto grid_size = num_sm * blocks_per_sm;  // aka blocks per grid
  auto shared_bytes = 0;

  kernel.is_running = true;
  kernel.sequence_num = KernelData::next_sequence_num();
  kernel.start_time = std::chrono::high_resolution_clock::now();
  dim3 grid_dim(grid_size);
  dim3 block_dim(block_size);
  cudaStreamSynchronize(cudaStreamPerThread);
  xor_kernel_new<<<grid_dim, block_dim, shared_bytes, kernel.stream>>>(
    device_sources, kernel.source_indices.size(), PCD.device_xorSources,
    PCD.xorSourceList.size(), PCD.device_or_args, PCD.orArgList.size(),
    kernel.device_source_indices, device_list_start_indices, device_results,
    kernel.stream_idx);

#if 0 || defined(STREAM_LOG)
  std::cerr << "stream " << kernel.stream_idx
            << " started with " << grid_size << " blocks"
            << " of " << block_size << " threads"
          //<< " starting, sequence: " << kernel.sequence_num
            << std::endl;
#endif
}

auto copy_device_results(
  result_t* device_results, int num_results, cudaStream_t stream) {
  //
  cudaError_t err = cudaStreamSynchronize(stream);
  assert((err == cudaSuccess) && "sychronize");

  std::vector<result_t> results(num_results);
  auto results_bytes = num_results * sizeof(result_t);
  err = cudaMemcpyAsync(results.data(), device_results, results_bytes,
    cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    fprintf(stdout, "copy device results, error: %s", cudaGetErrorString(err));
    assert((err != cudaSuccess) && "copy device results");
  }
  err = cudaStreamSynchronize(stream);
  assert((err == cudaSuccess) && "cudaStreamSynchronize");
  return results;
}

// TODO: kernel.get_results()? just to call out to free function?  maybe
auto getKernelResults(KernelData& kernel, result_t* device_results) {
  return copy_device_results(
    device_results, kernel.num_src_lists, kernel.stream);
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
    int compat_index{-1};
    auto compat = isSourceXORCompatibleWithAnyXorSource(source,
      PCD.xorSourceList.data(), PCD.xorSourceList.size(), &compat_index);
    std::cerr << "compat: " << compat << " (" << compat_index << ")"
              << std::endl;
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

int run_filter_task(std::vector<KernelData>& kernels, int threads_per_block,
  IndexStates& idx_states, const SourceCompatibilityData* device_sources,
  result_t* device_results, const index_t* device_list_start_indices) {
  //
  using namespace std::chrono;
  int total_compat{};
  int current_kernel{-1};
  int actual_num_compat{};
  while (get_next_available(kernels, current_kernel)) {
    auto& kernel = kernels.at(current_kernel);
    if (!kernel.is_running) {
      if (!kernel.fillSourceIndices(idx_states)) {
        continue;
      }
#if 0
      std::cerr << "stream " << kernel.stream_idx
                << " source_indices: " << kernel.source_indices.size()
                << ", ready: " << kernel.num_ready(idx_states)
                << std::endl;
#endif
      kernel.allocCopy(idx_states);
      run_xor_kernel(kernel, threads_per_block, device_sources, device_results,
        device_list_start_indices);
      continue;
    }

    kernel.has_run = true;
    kernel.is_running = false;

    auto results = getKernelResults(kernel, device_results);
    auto k1 = high_resolution_clock::now();
    auto d_kernel = duration_cast<milliseconds>(k1 - kernel.start_time).count();

    auto num_compat =
      idx_states.update(kernel.source_indices, results, kernel.stream_idx);
    total_compat += num_compat;

#if 0
    auto num_actual_compat = std::accumulate(results.begin(), results.end(), 0,
      [](int sum, result_t r) { return r ? sum + 1 : sum; });
    std::cerr << " stream " << kernel.stream_idx
              << " compat results: " << num_compat
              << ", total: " << total_compat
              << ", actual: " << num_actual_compat
              << std::endl;
#endif
  }
  return total_compat;
}

int filter_task(
  int sum, int threads_per_block, int num_streams, int stride, int iters) {
  //
  using namespace std::chrono;
  [[maybe_unused]] cudaError_t err = cudaSuccess;
  err = cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 7'500'000);

  const auto& sources =
    allSumsCandidateData.find(sum)->second.sourceCompatLists;
  auto device_sources = allocCopySources(sources);
  IndexStates idx_states{sources};
  auto device_list_start_indices = allocCopyListStartIndices(idx_states);

  if (!num_streams)
    num_streams = 1;
  std::vector<KernelData> kernels(num_streams);
  stride = stride ? stride : 1024;
  KernelData::init(kernels, sources.size(), stride);

#if 0
  std::cerr << "sourcelists: " << sources.size() << ", streams: " << num_streams
            << ", stride: " << stride << std::endl;
#endif

  auto device_results = allocResults(sources.size());

  if (!iters)
    iters = 1;
  for(int i{}; i < iters; ++i) {
    KernelData::reset(kernels);
    idx_states.reset();
    zeroResults(device_results, sources.size(), cudaStreamPerThread);

#if 0
    auto results =
      copy_device_results(device_results, sources.size(), cudaStreamPerThread);
    auto num_compat_results = std::accumulate(results.begin(), results.end(), 0,
      [](int sum, result_t r) { return r ? sum + 1 : sum; });
    std::cerr << " START"
              << " compat results: " << num_compat_results << std::endl;
#endif

    auto t0 = high_resolution_clock::now();
    auto total_compat = run_filter_task(kernels, threads_per_block, idx_states,
      device_sources, device_results, device_list_start_indices);
    auto t1 = high_resolution_clock::now();
    auto d_total = duration_cast<milliseconds>(t1 - t0).count();
    std::cerr << "sum(" << sum << ")";
    if (iters > 1) {
      std::cerr << " iter: " << i;
    }
    std::cerr << " total compat: " << total_compat << " of " << sources.size()
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
  return 0;
}

static std::vector<std::future<int>> filter_futures_;

void add_filter_future(std::future<int>&& filter_future) {
  filter_futures_.emplace_back(std::move(filter_future));
}

}  // anonymous namespace

namespace cm {

int get_filter_results() {
  for (auto& fut : filter_futures_) {
    if (fut.valid())
      fut.get();
  }
  return 0;
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

[[nodiscard]] device::OrArgData* cuda_allocCopyOrArgs(
  const OrArgList& orArgList) {
  //
  cudaError_t err = cudaSuccess;
  // host-side vector of device::orArgData that each contain pointers
  // to device memory, that we can blast to device with one copy
  std::vector<device::OrArgData> or_args;
  or_args.reserve(orArgList.size());
  for (size_t i{}; i < orArgList.size(); ++i) {
    const auto& or_arg = orArgList.at(i);
    const auto& or_src_list = or_arg.orSourceList;
    auto or_src_bytes = or_src_list.size() * sizeof(OrSourceData);
    OrSourceData* device_or_sources;
    err = cudaMallocAsync(
      (void**)&device_or_sources, or_src_bytes, cudaStreamPerThread);
    assert((err == cudaSuccess) && "alloc device or sources");

    err = cudaMemcpyAsync(device_or_sources, or_src_list.data(),
      or_src_bytes, cudaMemcpyHostToDevice, cudaStreamPerThread);
    assert((err == cudaSuccess) && "copy device or sources");
    or_args.emplace_back(device::OrArgData{
        device_or_sources, (unsigned)or_src_list.size(), or_arg.compatible});
  }
  auto or_arg_bytes = or_args.size() * sizeof(device::OrArgData);
  device::OrArgData* device_or_args;
  err =
    cudaMallocAsync((void**)&device_or_args, or_arg_bytes, cudaStreamPerThread);
  assert((err == cudaSuccess) && "alloc device or args");

  err = cudaMemcpyAsync(device_or_args, or_args.data(), or_arg_bytes,
    cudaMemcpyHostToDevice, cudaStreamPerThread);
  assert((err == cudaSuccess) && "copy device or args");

  return device_or_args;
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
