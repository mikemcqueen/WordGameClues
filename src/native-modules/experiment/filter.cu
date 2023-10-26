// filter.cu

#include <algorithm>
#include <chrono>
#include <exception>
#include <future>
#include <limits>
#include <numeric>
#include <optional>
#include <span>
#include <thread>
#include <tuple>
#include <utility> // pair
#include <cuda_runtime.h>
#include "candidates.h"
#include "filter.h"
#include "merge-filter-data.h"

//#define LOGGING

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

__device__ index_t get_combo_index(
  index_t flat_idx, const ComboIndexSpanPair& idx_spans) {
  //
  if (flat_idx < idx_spans.first.size()) {
    return idx_spans.first[flat_idx];
  }
  flat_idx -= idx_spans.first.size();
  assert(flat_idx < idx_spans.second.size());
  return idx_spans.second[flat_idx];
}

// Test if a source is XOR -and- OR compatible with supplied xor/or sources.
__device__ bool is_source_xor_or_compatible(
  const SourceCompatibilityData& source,
  const SourceCompatibilityData* __restrict__ xor_src_lists,
  const index_t* __restrict__ xor_src_list_start_indices,
  const index_t* __restrict__ xor_idx_lists,
  const index_t* __restrict__ xor_idx_list_start_indices,
  const index_t* __restrict__ xor_idx_list_sizes,
  const unsigned num_idx_lists,
  const ComboIndexSpanPair& idx_spans,
  const unsigned num_or_args,
  const device::OrSourceData* __restrict__ or_sources,
  const unsigned num_or_sources) {
  //
  __shared__ bool is_xor_compat;
  __shared__ bool is_or_compat;

  if (!threadIdx.x) {
    store(&is_xor_compat, false);
    store(&is_or_compat, false);
  }
  // NOTE: chunk-indexing as used here is necessary for syncthreads() to work
  //   at least on SM_6 hardware (GTX1060), where *all threads* in the block
  //   must execute the synchthreads() call. In later architectures, those
  //   restrictions may be relaxed, but possibly only for "completely exited
  //   (the kernel)" threads, which wouldn't be relevant here anyway (because
  //   we're in a function called from within a loop in a parent kernel).
  //
  //   Therefore, the following is not an equivalent replacement:
  //
  //   for (unsigned flat_idx{threadIdx.x}; flat_idx < num_xor_sources;
  //      flat_idx += blockDim.x) {
  //
  const auto num_indices = idx_spans.first.size() + idx_spans.second.size();
  const auto chunk_size = blockDim.x;// * num_idx_lists;
  const auto chunk_max = num_indices;// * num_idx_lists;
  // TODO MAYBE: num_idx_lists threads per xor_source (one thread per idx_list)
  // CURRENT: one thread per xor_source
  for (unsigned chunk_idx{}; chunk_idx * chunk_size < chunk_max; ++chunk_idx) {
    __syncthreads();
    // "flat" index, i.e. not yet indexed into appropriate idx_spans array
    const auto flat_idx = chunk_idx * chunk_size + threadIdx.x;
    //(threadIdx.x % num_idx_lists);
    if (flat_idx < num_indices) {
      auto combo_idx = get_combo_index(flat_idx, idx_spans);
      auto all_compat{true};
      for (int list_idx{(int)num_idx_lists - 1}; list_idx >= 0; --list_idx) {
        const auto idx_list =
          &xor_idx_lists[xor_idx_list_start_indices[list_idx]];
        const auto idx_list_size = xor_idx_list_sizes[list_idx];
        const auto xor_src_idx = idx_list[combo_idx % idx_list_size];
        const auto xor_src_list = &xor_src_lists[xor_src_list_start_indices[list_idx]];
        if (!source.isXorCompatibleWith(xor_src_list[xor_src_idx])) {
          all_compat = false;
          break;
        }
        combo_idx /= idx_list_size;
      }
      if (all_compat) {
        is_xor_compat = true;
      }
    }
    __syncthreads();
    // if source is not XOR compatible with any --xor sources
    if (!is_xor_compat) {
      continue;
    }
    if (num_or_args > 0) {
      // source must also be OR compatible with at least one source
      // of each or_arg
      if (is_source_or_compatibile(
            source, num_or_args, or_sources, num_or_sources)) {
        is_or_compat = true;
      }
      __syncthreads();
      if (!is_or_compat) {
        if (!threadIdx.x) {
          // reset is_xor_compat. sync will happen at loop entrance.
          is_xor_compat = false;
        }
        continue;
      }
    }
    return true;
  }
  return false;
}

__device__ bool variation_indices_shown = false;

__device__ void print_variation_indices(
  const device::VariationIndices* __restrict__ variation_indices) {
  //
  if (variation_indices_shown) return;
  printf("device:\n");
  for (int s{}; s < kNumSentences; ++s) {
    const auto& vi = variation_indices[s];
    for (index_t v{}; v < vi.num_variations; ++v) {
      const auto n = vi.num_combo_indices[v];
      printf("sentence %d, variation %d, indices: %d\n", s, v, n);
    }
  }
  variation_indices_shown = true;
}

__device__ auto get_smallest_src_index_spans(
  const SourceCompatibilityData& source,
  const device::VariationIndices* __restrict__ variation_indices) {
  //
  index_t fewest_indices{std::numeric_limits<index_t>::max()};
  int sentence_with_fewest{-1};
  for (int s{}; s < kNumSentences; ++s) {
    const auto& vi = variation_indices[s];
    // skip sentences for which there are no xor_sources with a primary clue
    if (!vi.num_variations) {
      continue;
    }
    const auto variation = source.usedSources.variations[s] + 1;
    // skip sources that have no primary clue from this sentence
    if (!variation) {
      continue;
    }
    const auto num_indices =
      vi.num_combo_indices[0] + vi.num_combo_indices[variation];
    if (num_indices < fewest_indices) {
      fewest_indices = num_indices;
      sentence_with_fewest = s;
      if (!num_indices) {
        break;
      }
    }
  }
  if (!fewest_indices) {
    return std::make_pair(ComboIndexSpan{}, ComboIndexSpan{});
  }
  assert(sentence_with_fewest > -1);
  const auto variation =
    source.usedSources.variations[sentence_with_fewest] + 1;
  const auto& vi = variation_indices[sentence_with_fewest];
  return std::make_pair(
    vi.get_index_span(0), vi.get_index_span(variation));
}

__global__ void xor_kernel_new(
  const SourceCompatibilityData* __restrict__ sources,
  const unsigned num_sources,
  const SourceCompatibilityData* __restrict__ xor_src_lists,
  const index_t* __restrict__ xor_src_list_start_indices,
  const index_t* __restrict__ xor_idx_lists,
  const index_t* __restrict__ xor_idx_list_start_indices,
  const index_t* __restrict__ xor_idx_list_sizes,
  const unsigned num_idx_lists,
  const device::VariationIndices* __restrict__ variation_indices,
  const unsigned num_or_args,
  const device::OrSourceData* __restrict__ or_sources,
  const unsigned num_or_sources, const SourceIndex* __restrict__ source_indices,
  const index_t* __restrict__ list_start_indices,
  result_t* __restrict__ results, int stream_idx) {

#if 0
  if (!threadIdx.x && !blockIdx.x && !stream_idx) {
    print_variation_indices(variation_indices);
  }
#endif

  // for each source (one block per source)
  for (unsigned idx{blockIdx.x}; idx < num_sources; idx += gridDim.x) {
    const auto src_idx = source_indices[idx];
    const auto flat_index =
      list_start_indices[src_idx.listIndex] + src_idx.index;
    auto& source = sources[flat_index];
    auto& result = results[src_idx.listIndex];
    auto idx_spans = get_smallest_src_index_spans(source, variation_indices);
#if 0
    if (!threadIdx.x && !stream_idx) {
      printf("block %d, first: %d, second: %d, total: %d\n", blockIdx.x,
        (int)idx_spans.first.size(), (int)idx_spans.second.size(),
        (int)(idx_spans.first.size() + idx_spans.second.size()));
    }
#endif
    __syncthreads();
    if (!(idx_spans.first.size() + idx_spans.second.size())) {
      continue;
    }
    if (is_source_xor_or_compatible(source, xor_src_lists,
          xor_src_list_start_indices, xor_idx_lists, xor_idx_list_start_indices,
          xor_idx_list_sizes, num_idx_lists, idx_spans, num_or_args, or_sources,
          num_or_sources)) {
      if (!threadIdx.x) {
        // TODO: store probably not necessary
        store(&result, (result_t)1);
      }
    }
  }
}

auto flat_index(
  const SourceCompatibilityLists& sources, const SourceIndex src_idx) {
  uint32_t flat{};
  for (size_t i{}; i < src_idx.listIndex; ++i) {
    flat += sources.at(i).size();
  }
  return flat + src_idx.index;
}

#if 0
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
    auto compat = isSourceXORCompatibleWithAnyXorSource(
      source, MFD.xorSourceList.data(), MFD.xorSourceList.size());
    std::cerr << "compat: " << compat
              << std::endl;
  }
}

void dump_xor(int index) {
  const XorSourceList& xorSources = MFD.xorSourceList;
  auto host_index = index;
  const auto& src = xorSources.at(host_index);
  char buf[32];
  snprintf(buf, sizeof(buf), "xor: device(%d) host(%d)", index, host_index);
  src.dump(buf);
}
#endif

}  // namespace

namespace cm {

void run_xor_kernel(StreamData& stream, int threads_per_block,
  const MergeFilterData& mfd, const SourceCompatibilityData* device_sources,
  result_t* device_results, const index_t* device_list_start_indices) {
  //
  int num_sm;
  cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0);
  auto threads_per_sm = 2048;
  auto block_size = threads_per_block ? threads_per_block : 1024;
  auto blocks_per_sm = threads_per_sm / block_size;
  // assert(blocks_per_sm * block_size == threads_per_sm);
  auto grid_size = num_sm * blocks_per_sm;  // aka blocks per grid
  auto shared_bytes = mfd.host.num_or_args * sizeof(result_t);
  // enforce assumption in is_source_or_compatible()
  assert(mfd.host.num_or_args < block_size);

  stream.is_running = true;
  stream.sequence_num = StreamData::next_sequence_num();
  stream.start_time = std::chrono::high_resolution_clock::now();
  dim3 grid_dim(grid_size);
  dim3 block_dim(block_size);
  // ensure any async alloc/copies are complete on thread stream
  cudaError_t err = cudaStreamSynchronize(cudaStreamPerThread);
  if (err != cudaSuccess) {
    fprintf(stdout, "sync before kernel, error: %s", cudaGetErrorString(err));
    assert((err == cudaSuccess) && "sync before kernel");
  }
  xor_kernel_new<<<grid_dim, block_dim, shared_bytes, stream.cuda_stream>>>(
    device_sources, stream.source_indices.size(), mfd.device.src_lists,
    mfd.device.src_list_start_indices, mfd.device.idx_lists,
    mfd.device.idx_list_start_indices, mfd.device.idx_list_sizes,
    mfd.host.compat_idx_lists.size(), mfd.device.variation_indices, mfd.host.num_or_args,
    mfd.device.or_sources, mfd.device.num_or_sources, stream.device_source_indices,
    device_list_start_indices, device_results, stream.stream_idx);

#if defined(LOGGING)
  std::cerr << "stream " << stream.stream_idx
            << " started with " << grid_size << " blocks"
            << " of " << block_size << " threads"
            << std::endl;
#endif
}

}  // namespace cm
