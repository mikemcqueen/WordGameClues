// filter.cu

#include <algorithm>
#include <chrono>
#include <exception>
#include <future>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <span>
#include <thread>
#include <tuple>
#include <utility> // pair
#include <cuda_runtime.h>
#include "filter.cuh"
#include "filter-types.h"
#include "merge-filter-data.h"

//#define LOGGING

namespace cm {

namespace {

/*
__device__ bool variation_indices_shown = false;

__device__ void print_variation_indices(
    const device::VariationIndices* __restrict__ variation_indices) {
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
*/

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
    const UsedSources::
        SourceDescriptorPair* __restrict__ incompatible_src_desc_pairs,
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

#ifdef OR_ARG_COUNTS
#define MAX_OR_ARGS 20
__device__ unsigned device_incompatible_or_arg_counts[MAX_OR_ARGS] = {0};
__device__ unsigned device_num_compat_or_args = 0;

__device__ void update_or_arg_counts(
    volatile result_t* or_arg_results, unsigned num_or_args) {
  __syncthreads();
  if (!threadIdx.x) {
    auto max_or_args = std::min(MAX_OR_ARGS, (int)num_or_args);
    bool compatible = true;
    for (int i{}; i < max_or_args; ++i) {
      if (!load(&or_arg_results[i])) {
        atomicInc(&device_incompatible_or_arg_counts[i], 1'000'000);
        compatible = false;
      }
    }
    if (compatible) {
      atomicInc(&device_num_compat_or_args, 1'000'000);
    }
  }
}
#endif

// is_source_OR_compatible with at least one source from each or arg
// TODO:
// Just thinking, but if different sources from the same sentence have
// different variations, don't we need to take that into account?
// i.e. if Arg1 has source in S1:V1 and Arg2 has source in S1:V2, that
// seems a bit whackabilly. Need more brains on it.
__device__ bool is_source_OR_compatible(const SourceCompatibilityData& source,
    const unsigned num_or_args,
    const device::OrSourceData* __restrict__ or_arg_sources,
    const unsigned num_or_arg_sources) {
  extern __shared__ result_t or_arg_results[];
  // ASSUMPTION: # of --or args will always be smaller than block size.
  if (threadIdx.x < num_or_args) {
    or_arg_results[threadIdx.x] = 0;
  }
  const auto chunk_size = blockDim.x;
  const auto chunk_max = num_or_arg_sources;
  for (unsigned chunk_idx{}; chunk_idx * chunk_size < chunk_max; ++chunk_idx) {
    __syncthreads();
    const auto or_arg_src_idx = chunk_idx * chunk_size + threadIdx.x;
    if (or_arg_src_idx < num_or_arg_sources) {
      const auto& or_src = or_arg_sources[or_arg_src_idx];
      // NB! order of a,b in a.isOrCompat(b) here matters!
      if (or_src.src.isOrCompatibleWith(source)) {
        // store not required here, because volatile
        or_arg_results[or_src.or_arg_idx] = 1;
        // TODO: FIXMENOW?
        //store(&or_arg_results[or_src.or_arg_idx], (result_t)1);
      }
    }
  }
#if OR_ARG_COUNTS
    // FIXMENOW: wrap in log_level
    update_or_arg_counts(or_arg_results, num_or_args);
#endif
#if 0
  // parallel reduction (sum)
  // i could safely initialize reduce_idx to 16 I think (max 32 --or args)
  for (int reduce_idx = blockDim.x / 2; reduce_idx > 0; reduce_idx /= 2) {
    __syncthreads();
    if ((threadIdx.x < reduce_idx)
        && (reduce_idx + threadIdx.x < num_or_args)) {
      // g++ has deprecated += on volatile destination;
      or_arg_results[threadIdx.x] = or_arg_results[threadIdx.x]
                                    + or_arg_results[reduce_idx + threadIdx.x];
    }
  }
#endif
    __syncthreads();
  if (!threadIdx.x) {
#if 1
    bool compat_with_all{true};
    for (int i{}; i < num_or_args; ++i) {
      if (!or_arg_results[i]) {
        compat_with_all = false;
        break;
      }
    }
#else
    const auto compat_with_all = or_arg_results[threadIdx.x] == num_or_args;
#endif
    return compat_with_all;
  }
  return false;
}

  /*

    if (num_or_args > 0) {
      // source must also be OR compatible with at least one source
      // of each or_arg
      if (is_source_OR_compatible(
              source, num_or_args, or_arg_sources, num_or_arg_sources)) {
        is_or_compat = true;
      }
      __syncthreads();
      if (!is_or_compat) {
        continue;
      }
    }
  */

__device__ combo_index_t get_combo_index(
    index_t flat_idx, const ComboIndexSpanPair& idx_spans) {
  if (flat_idx < idx_spans.first.size()) {
    return idx_spans.first[flat_idx];
  }
  flat_idx -= idx_spans.first.size();
  assert(flat_idx < idx_spans.second.size());
  return idx_spans.second[flat_idx];
}

// Test if a source is XOR compatible with the XOR source at the index
// identified by the supplied index lists and combo index.
__device__ bool is_source_XOR_compatible(const SourceCompatibilityData& source,
    combo_index_t combo_idx,
    const SourceCompatibilityData* __restrict__ xor_src_lists,
    const index_t* __restrict__ xor_src_list_start_indices,
    const index_t* __restrict__ xor_idx_lists,
    const index_t* __restrict__ xor_idx_list_start_indices,
    const index_t* __restrict__ xor_idx_list_sizes, unsigned num_idx_lists) {

  for (int list_idx{(int)num_idx_lists - 1}; list_idx >= 0; --list_idx) {
    const auto idx_list = &xor_idx_lists[xor_idx_list_start_indices[list_idx]];
    const auto idx_list_size = xor_idx_list_sizes[list_idx];
    // TODO: mod operator is slow? also mixing scalar unit types
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

__device__ unsigned device_num_xor_compat = 0;

// Test if a source is XOR compatible with any of the supplied xor sources.
__device__ bool is_source_XOR_compatible_with_any(
    const SourceCompatibilityData& source,
    const SourceCompatibilityData* __restrict__ xor_src_lists,
    const index_t* __restrict__ xor_src_list_start_indices,
    const index_t* __restrict__ xor_idx_lists,
    const index_t* __restrict__ xor_idx_list_start_indices,
    const index_t* __restrict__ xor_idx_list_sizes,
    const unsigned num_idx_lists, const ComboIndexSpanPair& idx_spans) {
  __shared__ bool is_xor_compat;
  if (!threadIdx.x) is_xor_compat = false;
  // Chunk-indexing required on older GPUs. The following is not an equivalent
  // replacement:
  //
  // for (unsigned flat_idx{threadIdx.x}; flat_idx < num_xor_sources;
  //      flat_idx += blockDim.x) {
  //
  const auto num_indices = idx_spans.first.size() + idx_spans.second.size();
  const auto chunk_size = blockDim.x;
  const auto chunk_max = num_indices;
  // one thread per xor_source
  for (unsigned chunk_idx{}; chunk_idx * chunk_size < chunk_max; ++chunk_idx) {
    __syncthreads();
    if (is_xor_compat) return true;
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
#if 0
    if (!threadIdx.x) {
      atomicInc(&device_num_xor_compat, 1'000'000);
    }
#endif
  }
  return false;
}

struct SmallestSpansResult {
  bool skip;
  ComboIndexSpanPair spans;
};
using smallest_spans_result_t = SmallestSpansResult;
// probably bad alternative: std::optional<ComboIndexSpanPair>

// variation_indices is an optimization. it allows us to restrict comparisons
// of a candidate source to only those xor_sources that have the same (or no)
// variation for each sentence - since a variation mismatch will alway result
// in comparison failure.
//
// individual xor_src indices are potentially (and often) duplicated in the
// variation_indices lists. for example, if a particular compound xor_src has
// variations S1:V1 and S2:V3, its xor_src_idx will appear in the variation
// indices lists for both of sentences 1 and 2.
//
// because of this, we only need to compare a candidate source with the
// xor_sources that have the same (or no) variations for a *single* sentence.
//
// the question then becomes, which sentence should we choose?
//
// the answer: the one with the fewest indices! (which results in the fewest
// comparisons). that's what this function determines.
//
__device__ smallest_spans_result_t get_smallest_src_index_spans(
    const SourceCompatibilityData& source,
    const device::VariationIndices* __restrict__ variation_indices) {
  index_t fewest_indices{std::numeric_limits<index_t>::max()};
  int sentence_with_fewest{-1};
  for (int s{}; s < kNumSentences; ++s) {
    // if there are no xor_sources that contain a primary source from this
    // sentence, skip it. (it would result in num_indices == all_indices which
    // is the worst case).
    const auto& vi = variation_indices[s];
    if (!vi.num_variations) continue;

    // if the candidate source has no primary source from this sentence, skip
    // it. (same reason as above).
    const auto variation = source.usedSources.variations[s] + 1;
    if (!variation) continue;

    // sum the xor_src indices that have no variation (index 0), with those
    // that have the same variation as the candidate source, for this sentence.
    // remember the sentence with the smallest sum.
    const auto num_indices =
        vi.num_combo_indices[0] + vi.num_combo_indices[variation];
    if (num_indices < fewest_indices) {
      fewest_indices = num_indices;
      sentence_with_fewest = s;
      if (!fewest_indices) break;
    }
  }
  if (sentence_with_fewest < 0) {
    // there are no sentences from which both the candidate source and any
    // xor_source contain a primary source. we can skip xor-compat checks
    // since they will all succeed.
    return {true};
  }
  if (!fewest_indices) {
    // both the candidate source and all xor_sources contain a primary source
    // from sentence_with_fewest, but all xor_sources use a different variation
    // than the candidate source. we can skip all xor-compat checks since they
    // will all fail due to variation mismatch.
    return {true};
  }
  const auto variation =
      source.usedSources.variations[sentence_with_fewest] + 1;
  const auto& vi = variation_indices[sentence_with_fewest];
  return {false,  //
      std::make_pair(vi.get_index_span(0), vi.get_index_span(variation))};
}

__device__ unsigned device_wtfbbq = 0;

__global__ void xor_kernel_new(
    const SourceCompatibilityData* __restrict__ src_list,
    const unsigned num_sources,
    const SourceCompatibilityData* __restrict__ xor_src_lists,
    const index_t* __restrict__ xor_src_list_start_indices,
    const index_t* __restrict__ xor_idx_lists,
    const index_t* __restrict__ xor_idx_list_start_indices,
    const index_t* __restrict__ xor_idx_list_sizes,
    const unsigned num_idx_lists,
    const device::VariationIndices* __restrict__ variation_indices,
    const SourceIndex* __restrict__ src_indices,
    const index_t* __restrict__ src_list_start_indices,
    const compat_src_result_t* __restrict__ compat_src_results,
    result_t* __restrict__ results, int stream_idx) {
  __shared__ bool is_compat;
  if (!threadIdx.x) is_compat = false;
  // for each source (one block per source)
  for (unsigned idx{blockIdx.x}; idx < num_sources; idx += gridDim.x) {
    __syncthreads();
    const auto src_idx = src_indices[idx];
    const auto flat_idx =
        src_list_start_indices[src_idx.listIndex] + src_idx.index;
    if (compat_src_results && !compat_src_results[flat_idx]) continue;
    const auto& source = src_list[flat_idx];
    auto result = get_smallest_src_index_spans(source, variation_indices);
    if (result.skip) continue;
    if (is_source_XOR_compatible_with_any(source, xor_src_lists,
            xor_src_list_start_indices, xor_idx_lists,
            xor_idx_list_start_indices, xor_idx_list_sizes, num_idx_lists,
            result.spans)) {
      is_compat = true;
    }
    __syncthreads();
    if (!is_compat) continue;
    if (!threadIdx.x) {
      results[src_idx.listIndex] = 1;
      is_compat = false;
    }
  }
}

/*
// Test if a source is XOR compatible with ANY of the provided xor sources.
__device__ bool is_source_XOR_compatible_with_any(
    const SourceCompatibilityData& source, const ComboIndexSpanPair& idx_spans,
    const SourceCompatibilityData* __restrict__ xor_src_lists,
    const index_t* __restrict__ xor_src_list_start_indices,
    const index_t* __restrict__ xor_idx_lists,
    const index_t* __restrict__ xor_idx_list_start_indices,
    const index_t* __restrict__ xor_idx_list_sizes,
    const unsigned num_idx_lists) {

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

__global__ void mark_or_sources_kernel(
    const SourceCompatibilityData* __restrict__ xor_src_lists,
    const index_t* __restrict__ xor_src_list_start_indices,
    const index_t* __restrict__ xor_idx_lists,
    const index_t* __restrict__ xor_idx_list_start_indices,
    const index_t* __restrict__ xor_idx_list_sizes,
    const unsigned num_idx_lists,
    const device::VariationIndices* __restrict__ variation_indices,
    const unsigned num_or_args,
    const device::OrSourceData* __restrict__ or_src_list,
    const unsigned num_or_sources, result_t* __restrict__ results) {
  // for each or_source (one block per source)
  for (unsigned idx{blockIdx.x}; idx < num_or_sources; idx += gridDim.x) {
    __syncthreads();
    const auto& or_source = or_src_list[idx];
    auto result =
        get_smallest_src_index_spans(or_source.src, variation_indices);
    using enum SmallestSpans::ResultCode;
    if (result.code == None) {
      continue;
    }
    if ((result.code == Check)
        && !is_source_XOR_compatible_with_any(or_source.src, result.idx_spans,
            xor_src_lists, xor_src_list_start_indices, xor_idx_lists,
            xor_idx_list_start_indices, xor_idx_list_sizes, num_idx_lists)) {
      continue;
    }
    // result.code is All, or Check and compatibilty check succeeded
    if (!threadIdx.x) {
      results[idx] = 1;
    }
  }
}
*/
  
/*
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

void check(const SourceCompatibilityLists& src_list, index_t list_index,
    index_t index) {
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
*/

}  // anonymous namespace

void run_get_compatible_sources_kernel(
    const SourceCompatibilityData* device_sources, unsigned num_sources,
    const UsedSources::SourceDescriptorPair* device_src_desc_pairs,
    unsigned num_src_desc_pairs, compat_src_result_t* device_results) {
  int num_sm;
  cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0);
  int threads_per_sm;
  cudaDeviceGetAttribute(
      &threads_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor, 0);
  auto block_size = 768;  // aka threads per block
  auto blocks_per_sm = threads_per_sm / block_size;
  assert(blocks_per_sm * block_size == threads_per_sm);
  auto grid_size = num_sm * blocks_per_sm;  // aka blocks per grid
  auto shared_bytes = 0;

  dim3 grid_dim(grid_size);
  dim3 block_dim(block_size);
  // ensure any async alloc/copies are complete on thread stream
  cudaStream_t stream = cudaStreamPerThread;
  cudaError_t err = cudaSuccess;
  // err = cudaStreamSynchronize(stream);
  // assert_cuda_success(err, "run_get_compatible_sources_kernel sync -before");
  get_compatible_sources_kernel<<<grid_dim, block_dim, shared_bytes, stream>>>(
      device_sources, num_sources, device_src_desc_pairs, num_src_desc_pairs,
      device_results);
  if constexpr (kLogCompatSources) {
    //    CudaEvent stop(stream);
    //    stop.synchronize();
    /*
    unsigned int num_compat_sources;
    err = cudaMemcpyFromSymbol(&num_compat_sources,
      device_num_compat_sources, sizeof(num_compat_sources));
    assert_cuda_success(err, "cudaMemcpyFromSymbol - num_compat_sources");
    unsigned int num_blocks;
    err =
      cudaMemcpyFromSymbol(&num_blocks, device_num_blocks,
         sizeof(num_blocks)); assert_cuda_success(err, "cudaMemcpyFromSymbol -
         num_blocks");
    fprintf(stderr, " atomic compat: %d, blocks: %d\n", num_compat_sources,
      num_blocks);
    */
  }
}

void run_xor_kernel(StreamData& stream, int threads_per_block,
    const MergeFilterData& mfd, const SourceCompatibilityData* device_src_list,
    const compat_src_result_t* device_compat_src_results,
    result_t* device_results, const index_t* device_list_start_indices) {

  int num_sm;
  cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0);
  int threads_per_sm;;
  cudaDeviceGetAttribute(&threads_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor, 0);
  auto block_size = threads_per_block ? threads_per_block : 768;
  auto blocks_per_sm = threads_per_sm / block_size;
  assert(blocks_per_sm * block_size == threads_per_sm);
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
      device_src_list, stream.src_indices.size(), mfd.device.src_lists,
      mfd.device.src_list_start_indices, mfd.device.idx_lists,
      mfd.device.idx_list_start_indices, mfd.device.idx_list_sizes,
      mfd.host.compat_idx_lists.size(), mfd.device.variation_indices,
      stream.device_src_indices, device_list_start_indices,
      device_compat_src_results, device_results, stream.stream_idx);
  stream.kernel_stop.record();

  if constexpr (0) {
    std::cerr << "stream " << stream.stream_idx << " started with " << grid_size
              << " blocks of " << block_size << " threads" << std::endl;
  }
}

void show_or_arg_counts([[maybe_unused]] unsigned num_or_args) {
  cudaError_t err{cudaSuccess};

#if 0
  unsigned wtfbbq;
  err = cudaMemcpyFromSymbol(
      &wtfbbq, device_wtfbbq, sizeof(unsigned));
  assert_cuda_success(err, "cudaMemCopyFromSymbol wtfbbq");
  std::cerr << "wtfbbq: " << wtfbbq << std::endl;
#endif

#ifdef OR_ARG_COUNTS
  unsigned num_compat;
  err = cudaMemcpyFromSymbol(
      &num_compat, device_num_compat_or_args, sizeof(unsigned));
  assert_cuda_success(err, "cudaMemCopyFromSymbol num_compat");
  std::cerr << "compatible sources: " << num_compat << std::endl;
#endif

  unsigned num_xor_compat;
  err = cudaMemcpyFromSymbol(
      &num_xor_compat, device_num_xor_compat, sizeof(unsigned));
  assert_cuda_success(err, "cudaMemCopyFromSymbol num_xor_compat");
  std::cerr << "xor-compatible sources: " << num_xor_compat << std::endl;

#ifdef OR_ARG_COUNTS
  unsigned results[MAX_OR_ARGS] = {0};
  auto max_or_args = std::min(MAX_OR_ARGS, (int)num_or_args);
  err = cudaMemcpyFromSymbol(results, device_incompatible_or_arg_counts,
      max_or_args * sizeof(unsigned));
  assert_cuda_success(err, "cudaMemCopyFromSymbol incompatible_or_arg_counts");
  std::cerr << "incompatible or_args:\n";
  for (int i{}; i < max_or_args; ++i) {
    std::cerr << " arg" << i << ": " << results[i] << std::endl;
  }
#endif
}

/*
void run_mark_or_sources_kernel(
    const MergeFilterData& mfd, result_t* device_results) {
  int num_sm;
  cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0);
  int threads_per_sm;;
  cudaDeviceGetAttribute(&threads_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor, 0);
  auto block_size = 768;
  auto blocks_per_sm = threads_per_sm / block_size;
  assert(blocks_per_sm * block_size == threads_per_sm);
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
*/

}  // namespace cm
