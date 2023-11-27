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

// Test if the supplied source contains both of the primary sources described
// by any of the supplied source descriptor pairs.
__device__ bool source_contains_any_descriptor_pair(
  const SourceCompatibilityData& source,
  const UsedSources::SourceDescriptorPair* __restrict__ src_desc_pairs,
  const unsigned num_src_desc_pairs) {
  //
  __shared__ bool contains_both;
  if (!threadIdx.x) {
    contains_both = false;
  }
  __syncthreads();
  // one thread per src_desc_pair
  for (unsigned idx{}; idx * blockDim.x < num_src_desc_pairs; ++idx) {
    const auto pair_idx = idx * blockDim.x + threadIdx.x;
    if (pair_idx < num_src_desc_pairs) {
      if (source.usedSources.has(src_desc_pairs[pair_idx])) {
        contains_both = true;
      }
    }
    __syncthreads();
    if (contains_both) {
      return true;
    }
  }
  return false;
}

// note that this really only makes sense when running a single sum, not
// concurrent kernels for a range. for range support, i could allocate 
// and pass a separate per-sum global.
constexpr const bool kLogCompatSources = false;

__device__ unsigned int device_num_compat_sources = 0;
__device__ unsigned int device_num_blocks = 0;

__global__ void get_compatible_sources_kernel(
  const SourceCompatibilityData* __restrict__ sources,
  const unsigned num_sources,
  const UsedSources::SourceDescriptorPair* __restrict__
    incompatible_src_desc_pairs,
  const unsigned num_src_desc_pairs,
  compat_src_result_t* __restrict__ results) {
  // one block per source
  for (unsigned idx{blockIdx.x}; idx < num_sources; idx += gridDim.x) {
    const auto& source = sources[idx];
    // I don't understand why this is required, but fails synccheck and
    // produces wobbly results without.
    __syncthreads();
    if (!source_contains_any_descriptor_pair(
          source, incompatible_src_desc_pairs, num_src_desc_pairs)) {
      if (!threadIdx.x) {
        results[idx] = 1;
        if constexpr (kLogCompatSources) {
          atomicInc(&device_num_compat_sources, num_sources);
        }
      }
    }
    if constexpr (kLogCompatSources) {
      if (!threadIdx.x) {
        atomicInc(&device_num_blocks, gridDim.x);
      }
    }
  }
}

// "with all or sources"
__device__ bool is_source_OR_compatibile(const SourceCompatibilityData& source,
  const unsigned num_or_args,
  const device::OrSourceData* __restrict__ or_sources,
  const unsigned num_or_sources) {
  //
  extern __shared__ volatile result_t or_arg_results[];
  // ASSUMPTION: # of --or args will always be smaller than block size.
  if (threadIdx.x < num_or_args) {
    // store not required here, even without volatile, because __sync
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
      if (source.isOrCompatibleWith(or_src.src)) {
        // store not required here, because volatile
        or_arg_results[or_src.or_arg_idx] = (result_t)1;
      }
    }
  }
  // i could safely initialize reduce_idx to 16 I think (max 32 --or args)
  for (int reduce_idx = blockDim.x / 2; reduce_idx > 0; reduce_idx /= 2) {
    __syncthreads();
    if ((threadIdx.x < reduce_idx)
        && (reduce_idx + threadIdx.x < num_or_args)) {
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

// Test if a source is XOR compatible with sources identified by the
// supplied combo_idx.
// TODO: shouldn't combo_idx be a uint64_t ?
__device__ bool is_source_XOR_compatible(const SourceCompatibilityData& source,
  index_t combo_idx, const SourceCompatibilityData* __restrict__ xor_src_lists,
  const index_t* __restrict__ xor_src_list_start_indices,
  const index_t* __restrict__ xor_idx_lists,
  const index_t* __restrict__ xor_idx_list_start_indices,
  const index_t* __restrict__ xor_idx_list_sizes, unsigned num_idx_lists) {
  //
  for (int list_idx{(int)num_idx_lists - 1}; list_idx >= 0; --list_idx) {
    const auto idx_list = &xor_idx_lists[xor_idx_list_start_indices[list_idx]];
    const auto idx_list_size = xor_idx_list_sizes[list_idx];
    const auto xor_src_idx = idx_list[combo_idx % idx_list_size];
    // const auto& here and no address-of intead?
    const auto xor_src_list =
      &xor_src_lists[xor_src_list_start_indices[list_idx]];
    if (!source.isXorCompatibleWith(xor_src_list[xor_src_idx])) {
      return false;
    }
    combo_idx /= idx_list_size;
  }
  return true;
}

// Test if a source is XOR -and- OR compatible with supplied xor/or sources.
__device__ bool is_source_XOR_and_OR_compatible(
  const SourceCompatibilityData& source,
  const SourceCompatibilityData* __restrict__ xor_src_lists,
  const index_t* __restrict__ xor_src_list_start_indices,
  const index_t* __restrict__ xor_idx_lists,
  const index_t* __restrict__ xor_idx_list_start_indices,
  const index_t* __restrict__ xor_idx_list_sizes, const unsigned num_idx_lists,
  const ComboIndexSpanPair& idx_spans, const unsigned num_or_args,
  const device::OrSourceData* __restrict__ or_sources,
  const unsigned num_or_sources) {
  //
  __shared__ bool is_xor_compat;
  __shared__ bool is_or_compat;

  if (!threadIdx.x) {
    // store not necessary because syncthreads
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
  const auto chunk_size = blockDim.x;
  const auto chunk_max = num_indices;
  // one thread per xor_source
  for (unsigned chunk_idx{}; chunk_idx * chunk_size < chunk_max; ++chunk_idx) {
    // TODO: i imagine i can move this synchthreads to after shared memory
    // initialization, and add one to the end, which might be slightly more
    // performant. This might even be wrong (apparently sanitary though)
    __syncthreads();
    // "flat" index, i.e. not yet indexed into appropriate idx_spans array
    const auto flat_idx = chunk_idx * chunk_size + threadIdx.x;
    if (flat_idx < num_indices) {
      const auto combo_idx = get_combo_index(flat_idx, idx_spans);
      if (is_source_XOR_compatible(source, combo_idx, xor_src_lists,
            xor_src_list_start_indices, xor_idx_lists,
            xor_idx_list_start_indices, xor_idx_list_sizes, num_idx_lists)) {
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
      if (is_source_OR_compatibile(
            source, num_or_args, or_sources, num_or_sources)) {
        is_or_compat = true;
      }
      __syncthreads();
      if (!is_or_compat) {
        if (!threadIdx.x) {
          // reset is_xor_compat. sync will happen at loop entrance.
          is_xor_compat = false;
        }
        // todo: add here, move at loop entrance
        // __syncthreads();
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
  if (variation_indices_shown)
    return;
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

// variation_indices is an optimization. it allows us to restrict comparisons
// of a candidate source to only those xor_sources that have the same (or no)
// variation for each sentence - since a variation mismatch will alway result
// in comparison failure.
//
// individual xor_src indices are potentially (and often) duplicated in the
// variation_indices lists. for example, if a particular compound xor_src has
// variations S1:V1 and S2:V3, its xor_src_idx will appear in both of those
// corresponding variation_indices lists.
//
// because of this, we actually need only compare a candidate source with the
// xor_sources corresponding to the variation_indices that have the same (or
// no) variation for a *single* sentence.
//
// the question then becomes, which sentence/variation should we choose?
//
// answer: the one with the fewest indices! (resulting in the fewest
// compares). that's what this function determines.
//
__device__ auto get_smallest_src_index_spans(
  const SourceCompatibilityData& source,
  const device::VariationIndices* __restrict__ variation_indices) {
  //
  index_t fewest_indices{std::numeric_limits<index_t>::max()};
  int sentence_with_fewest{-1};
  for (int s{}; s < kNumSentences; ++s) {
    const auto& vi = variation_indices[s];
    // if there are no xor_sources that contain a primary source from this
    // sentence, skip it. (as it would result in num_indices == all_indices)
    if (!vi.num_variations) {
      continue;
    }
    const auto variation = source.usedSources.variations[s] + 1;
    // if the candidate source has no primary source from this sentence,
    // skip it. (same reason as above)
    if (!variation) {
      continue;
    }
    // sum the xor_src indices that have no variation (index 0), and the same
    // variation as the candidate source, for this sentence. remember the
    // smallest sum, and sentence.
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
  // if the candidate source sentence variation with the fewest number of
  // compatible (same or no variation) xor_sources is zero, it means the
  // following conditions are met:
  // * the candidate source contains a primary source from this sentence
  // * every xor_source both:
  //   a) contains a primary source from this sentence, and
  //   b) has a different variation for this sentence than that of the
  //      candidate source.
  // as a result, we can skip all xor-compares for the candidate source
  // as we know they will all fail due to variation mismatch.
  if (!fewest_indices) {
    return std::make_pair(ComboIndexSpan{}, ComboIndexSpan{});
  }
  assert(sentence_with_fewest > -1);
  const auto variation =
    source.usedSources.variations[sentence_with_fewest] + 1;
  const auto& vi = variation_indices[sentence_with_fewest];
  return std::make_pair(vi.get_index_span(0), vi.get_index_span(variation));
}

__global__ void xor_kernel_new(
  const SourceCompatibilityData* __restrict__ src_list,
  const unsigned num_sources,
  const SourceCompatibilityData* __restrict__ xor_src_lists,
  const index_t* __restrict__ xor_src_list_start_indices,
  const index_t* __restrict__ xor_idx_lists,
  const index_t* __restrict__ xor_idx_list_start_indices,
  const index_t* __restrict__ xor_idx_list_sizes, const unsigned num_idx_lists,
  const device::VariationIndices* __restrict__ variation_indices,
  const unsigned num_or_args,
  const device::OrSourceData* __restrict__ or_src_list,
  const unsigned num_or_sources, const SourceIndex* __restrict__ source_indices,
  const index_t* __restrict__ list_start_indices,
  const compat_src_result_t* __restrict__ compat_src_results,
  result_t* __restrict__ results, int stream_idx) {
  // for each source (one block per source)
  for (unsigned idx{blockIdx.x}; idx < num_sources; idx += gridDim.x) {
    const auto src_idx = source_indices[idx];
    const auto flat_idx =
      list_start_indices[src_idx.listIndex] + src_idx.index;
    __syncthreads();
    if (compat_src_results && !compat_src_results[flat_idx]) {
      continue;
    }
    const auto& source = src_list[flat_idx];
    auto idx_spans = get_smallest_src_index_spans(source, variation_indices);
    //__syncthreads();
    if (!(idx_spans.first.size() + idx_spans.second.size())) {
      continue;
    }
    if (is_source_XOR_and_OR_compatible(source, xor_src_lists,
          xor_src_list_start_indices, xor_idx_lists, xor_idx_list_start_indices,
          xor_idx_list_sizes, num_idx_lists, idx_spans, num_or_args, or_src_list,
          num_or_sources)) {
      if (!threadIdx.x) {
        // TODO: store probably not necessary
        //store(&result, (result_t)1);
        results[src_idx.listIndex] = 1;
      }
    }
  }
}

// Test if a source is XOR compatible with ANY of the provided xor sources.
__device__ bool is_source_XOR_compatible_with_any(
  const SourceCompatibilityData& source,
  const ComboIndexSpanPair& idx_spans,
  const SourceCompatibilityData* __restrict__ xor_src_lists,
  const index_t* __restrict__ xor_src_list_start_indices,
  const index_t* __restrict__ xor_idx_lists,
  const index_t* __restrict__ xor_idx_list_start_indices,
  const index_t* __restrict__ xor_idx_list_sizes,
  const unsigned num_idx_lists) {
  //
  __shared__ bool is_xor_compat;
  if (!threadIdx.x) {
    store(&is_xor_compat, false);
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

  // TODO: not sure all the syncthreads are necessary here.

  const auto num_indices = idx_spans.first.size() + idx_spans.second.size();
  const auto chunk_size = blockDim.x;
  const auto chunk_max = num_indices;
  // one thread per xor_source
  for (unsigned chunk_idx{}; chunk_idx * chunk_size < chunk_max; ++chunk_idx) {
    __syncthreads();
    // "flat" index, i.e. not yet indexed into appropriate idx_spans array
    const auto flat_idx = chunk_idx * chunk_size + threadIdx.x;
    if (flat_idx < num_indices) {
      auto combo_idx = get_combo_index(flat_idx, idx_spans);
      if (is_source_XOR_compatible(source, combo_idx, xor_src_lists,
            xor_src_list_start_indices, xor_idx_lists,
            xor_idx_list_start_indices, xor_idx_list_sizes, num_idx_lists)) {
        is_xor_compat = true;
      }
    }
    __syncthreads();
    if (is_xor_compat) {
      return true;
    }
  }
  return false;
}

// TODO: syncthreads required?
__global__ void mark_or_sources_kernel(
  const SourceCompatibilityData* __restrict__ xor_src_lists,
  const index_t* __restrict__ xor_src_list_start_indices,
  const index_t* __restrict__ xor_idx_lists,
  const index_t* __restrict__ xor_idx_list_start_indices,
  const index_t* __restrict__ xor_idx_list_sizes, const unsigned num_idx_lists,
  const device::VariationIndices* __restrict__ variation_indices,
  const unsigned num_or_args,
  const device::OrSourceData* __restrict__ or_src_list,
  const unsigned num_or_sources, result_t* __restrict__ results) {
  // for each or_source (one block per source)
  for (unsigned idx{blockIdx.x}; idx < num_or_sources; idx += gridDim.x) {
    const auto& or_source = or_src_list[idx];
    auto& result = results[idx];
    auto idx_spans =
      get_smallest_src_index_spans(or_source.src, variation_indices);
    __syncthreads();
    if (idx_spans.first.size() + idx_spans.second.size()) {
      if (is_source_XOR_compatible_with_any(or_source.src, idx_spans,
            xor_src_lists, xor_src_list_start_indices, xor_idx_lists,
            xor_idx_list_start_indices, xor_idx_list_sizes, num_idx_lists)) {
        if (!threadIdx.x) {
          result = 1;
        }
      }
    }
  }
}

#if 0
auto flat_index(
  const SourceCompatibilityLists& src_list, const SourceIndex src_idx) {
  uint32_t flat{};
  for (size_t i{}; i < src_idx.listIndex; ++i) {
    flat += src_list.at(i).size();
  }
  return flat + src_idx.index;
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

void check(
  const SourceCompatibilityLists& src_list, index_t list_index, index_t index) {
  constexpr const auto logging = true;
  if constexpr (logging) {
    SourceIndex src_idx{list_index, index};
    char idx_buf[32];
    char buf[64];
    snprintf(buf, sizeof(buf), "%s, flat: %d", src_idx.as_string(idx_buf),
      flat_index(src_list, src_idx));
    auto& source = src_list.at(list_index).at(index);
    source.dump(buf);
    auto compat = isSourceXORCompatibleWithAnyXorSource(
      source, MFD.xorSourceList.data(), MFD.xorSourceList.size());
    std::cerr << "compat: " << compat << std::endl;
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

void run_get_compatible_sources_kernel(
  const SourceCompatibilityData* device_sources, unsigned num_sources,
  const UsedSources::SourceDescriptorPair* device_src_desc_pairs,
  unsigned num_src_desc_pairs, compat_src_result_t* device_results) {
  //
  int num_sm;
  cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0);
  auto threads_per_sm = 2048;
  auto block_size = 1024; // aka threads per block
  auto blocks_per_sm = threads_per_sm / block_size;
  assert(blocks_per_sm * block_size == threads_per_sm);
  auto grid_size = num_sm * blocks_per_sm;  // aka blocks per grid
  auto shared_bytes = 0;

  dim3 grid_dim(grid_size);
  dim3 block_dim(block_size);
  // ensure any async alloc/copies are complete on thread stream
  cudaStream_t stream = cudaStreamPerThread;
  cudaError_t err = cudaSuccess;
  //  err = cudaStreamSynchronize(cudaStreamPerThread);
  //  assert_cuda_success(err, "run_get_compatible_sources_kernel sync-before");
  get_compatible_sources_kernel<<<grid_dim, block_dim, shared_bytes, stream>>>(
    device_sources, num_sources, device_src_desc_pairs, num_src_desc_pairs,
    device_results);
  if constexpr(kLogCompatSources) {
    //    CudaEvent stop(stream);
    //    stop.synchronize();
    /*
    unsigned int num_compat_sources;
    err = cudaMemcpyFromSymbol(&num_compat_sources, device_num_compat_sources,
      sizeof(num_compat_sources));
    assert_cuda_success(err, "cudaMemcpyFromSymbol - num_compat_sources");
    unsigned int num_blocks;
    err =
      cudaMemcpyFromSymbol(&num_blocks, device_num_blocks, sizeof(num_blocks));
    assert_cuda_success(err, "cudaMemcpyFromSymbol - num_blocks");
    fprintf(stderr, " atomic compat: %d, blocks: %d\n",
      num_compat_sources, num_blocks);
    */
  }
}

void run_xor_kernel(StreamData& stream, int threads_per_block,
  const MergeFilterData& mfd, const SourceCompatibilityData* device_src_list,
  const compat_src_result_t* device_compat_src_results, result_t* device_results,
  const index_t* device_list_start_indices) {
  //
  int num_sm;
  cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0);
  auto threads_per_sm = 2048;
  auto block_size = threads_per_block ? threads_per_block : 1024;
  auto blocks_per_sm = threads_per_sm / block_size;
  // assert(blocks_per_sm * block_size == threads_per_sm);
  auto grid_size = num_sm * blocks_per_sm;  // aka blocks per grid
  auto shared_bytes = mfd.host.or_arg_list.size() * sizeof(result_t);
  // enforce assumption in is_source_OR_compatible()
  assert(mfd.host.or_arg_list.size() < block_size);

  stream.is_running = true;
  stream.sequence_num = StreamData::next_sequence_num();
  dim3 grid_dim(grid_size);
  dim3 block_dim(block_size);
  // ensure any async alloc/copies are complete on thread stream
  cudaError_t err = cudaStreamSynchronize(cudaStreamPerThread);
  assert_cuda_success(err, "run_xor_kernel sync");
  stream.kernel_start.record();
  xor_kernel_new<<<grid_dim, block_dim, shared_bytes, stream.cuda_stream>>>(
    device_src_list, stream.source_indices.size(), mfd.device.src_lists,
    mfd.device.src_list_start_indices, mfd.device.idx_lists,
    mfd.device.idx_list_start_indices, mfd.device.idx_list_sizes,
    mfd.host.compat_idx_lists.size(), mfd.device.variation_indices,
    mfd.host.or_arg_list.size(), mfd.device.or_src_list,
    mfd.device.num_or_sources, stream.device_source_indices,
    device_list_start_indices, device_compat_src_results, device_results,
    stream.stream_idx);
  stream.kernel_stop.record();

  if constexpr (0) {
    std::cerr << "stream " << stream.stream_idx << " started with " << grid_size
              << " blocks of " << block_size << " threads" << std::endl;
  }
}

void run_mark_or_sources_kernel(
  const MergeFilterData& mfd, result_t* device_results) {
  int num_sm;
  cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0);
  auto threads_per_sm = 2048;
  auto block_size = 1024;
  auto blocks_per_sm = threads_per_sm / block_size;
  auto grid_size = num_sm * blocks_per_sm;  // aka blocks per grid
  auto shared_bytes = 0;

  dim3 grid_dim(grid_size);
  dim3 block_dim(block_size);
  cudaStream_t stream = cudaStreamPerThread;
  //cudaStreamSynchronize(cudaStreamPerThread);
  mark_or_sources_kernel<<<grid_dim, block_dim, shared_bytes, stream>>>(
    mfd.device.src_lists, mfd.device.src_list_start_indices,
    mfd.device.idx_lists, mfd.device.idx_list_start_indices,
    mfd.device.idx_list_sizes, mfd.host.compat_idx_lists.size(),
    mfd.device.variation_indices, mfd.host.or_arg_list.size(),
    mfd.device.or_src_list, mfd.device.num_or_sources, device_results);
}

}  // namespace cm
