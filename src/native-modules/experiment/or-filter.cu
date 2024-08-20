// filter.cu

#include <type_traits>
#include <cuda_runtime.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/copy.h>
#include "combo-maker.h"
#include "merge-filter-data.h"
#include "or-filter.cuh"

#define RESTRICT __restrict__

namespace cm {

extern __constant__ FilterData::DeviceXor xor_data;
extern __constant__ FilterData::DeviceOr or_data;

#ifdef DEBUG_OR_COUNTS
extern __device__ atomic64_t count_or_src_considered;
extern __device__ atomic64_t or_get_compat_idx_clocks;
extern __device__ atomic64_t count_or_xor_variation_compat;
extern __device__ atomic64_t or_xor_variation_compat_clocks;
extern __device__ atomic64_t count_or_src_variation_compat;
extern __device__ atomic64_t or_src_variation_compat_clocks;
extern __device__ atomic64_t count_or_check_src_compat;
extern __device__ atomic64_t or_check_src_compat_clocks;
extern __device__ atomic64_t count_or_src_compat;
#endif

namespace {

// only used for XOR variations currently
__device__ variation_index_t get_one_variation(
    int sentence, fat_index_t combo_idx) {
  for (int list_idx{int(xor_data.num_idx_lists) - 1}; list_idx >= 0; --list_idx) {
    const auto& src = xor_data.get_source(combo_idx, list_idx);
    const auto variation = src.usedSources.variations[sentence];
    if (variation > -1) return variation;
    combo_idx /= xor_data.idx_list_sizes[list_idx];
  }
  return -1;
}

__device__ auto init_variations_compat_results(
    const UsedSources::Variations& xor_variations) {
  __shared__ bool any_compat;
  if (!threadIdx.x) any_compat = false;
  __syncthreads();
  const auto first_results_idx = blockIdx.x * or_data.num_unique_variations;
  for (auto idx{threadIdx.x}; idx < or_data.num_unique_variations;
      idx += blockDim.x) {
    const auto& or_variations = or_data.unique_variations[idx].variations;
    const auto compat =
        UsedSources::are_variations_compatible(xor_variations, or_variations);
    xor_data.variations_compat_results[first_results_idx + idx] =
        compat ? 1 : 0;
    if (compat) any_compat = true;
  }
  __syncthreads();
  return any_compat;
}

__device__ void init_variations_indices(index_t* num_uv_indices) {
  const auto num_items = or_data.num_unique_variations;
  const auto offset = blockIdx.x * num_items;

  // convert flag array (variations_compat_results) to prefix sums
  // (variations_scan_results)
  const auto d_flags = &xor_data.variations_compat_results[offset];
  auto d_scan_output = &xor_data.variations_scan_results[offset];
  thrust::device_ptr<const result_t> d_flags_ptr(d_flags);
  thrust::device_ptr<result_t> d_scan_output_ptr(d_scan_output);
  thrust::exclusive_scan(
      thrust::device, d_flags_ptr, d_flags_ptr + num_items, d_scan_output_ptr);

  // generate indices from flag array + prefix sums
  auto d_indices = &xor_data.unique_variations_indices[offset];
  thrust::device_ptr<index_t> d_indices_ptr(d_indices);
  thrust::counting_iterator<int> count_begin(0);
  thrust::for_each(thrust::device, count_begin, count_begin + num_items,
      [d_flags, d_scan_output, d_indices] __device__(const index_t idx) {
        if (!threadIdx.x && d_flags[idx]) d_indices[d_scan_output[idx]] = idx;
      });
  // compute total set flags
  *num_uv_indices = d_scan_output[num_items - 1] + d_flags[num_items - 1];
}

__device__ auto init_compat_variations(
    const UsedSources::Variations& xor_variations, index_t* num_indices) {
  if (!init_variations_compat_results(xor_variations)) return false;
  init_variations_indices(num_indices);
  return true;
}

// V1 (very slow): linear walk of or_data.variations_compat_results
__device__ __forceinline__ index_t get_OR_compat_idx_linear_results(
    index_t chunk_idx) {
  const auto first_results_idx = blockIdx.x * or_data.num_unique_variations;
  auto desired_idx = chunk_idx * blockDim.x + threadIdx.x;

  for (index_t idx{}; idx < or_data.num_unique_variations; ++idx) {
    if (!xor_data.variations_compat_results[first_results_idx + idx]) continue;
    const auto& uv = or_data.unique_variations[idx];
    if (desired_idx < uv.start_idx + uv.num_indices) {
      return uv.first_compat_idx + (desired_idx - uv.start_idx);
    }
  }
  return or_data.num_compat_indices;
}

// V2 (faster) linear walk of xor_data.variations_indices
__device__ __forceinline__ index_t get_OR_compat_idx_linear_indices(
    index_t chunk_idx, index_t num_uv_indices) {
  const auto first_uvi_idx = blockIdx.x * or_data.num_unique_variations;
  auto desired_idx = chunk_idx * blockDim.x + threadIdx.x;

  for (index_t idx{}; idx < num_uv_indices; ++idx) {
    const auto uv_idx = xor_data.unique_variations_indices[first_uvi_idx + idx];
    const auto& uv = or_data.unique_variations[uv_idx];
    if (desired_idx < uv.start_idx + uv.num_indices) {
      return uv.first_compat_idx + (desired_idx - uv.start_idx);
    }
  }
  return or_data.num_compat_indices;
}

// V3 (fastest) binary search of xor_data.variations_indices
__device__ index_t get_OR_compat_idx_logn(
    index_t chunk_idx, index_t num_uv_indices) {
  const auto desired_idx = chunk_idx * blockDim.x + threadIdx.x;
  auto begin_uvi_idx = blockIdx.x * or_data.num_unique_variations;
  auto end_uvi_idx = begin_uvi_idx + num_uv_indices;
  const auto last_uv_idx = xor_data.unique_variations_indices[end_uvi_idx - 1];
  const auto& last_uv = or_data.unique_variations[last_uv_idx];
  if (desired_idx < last_uv.start_idx + last_uv.num_indices) {
    while (begin_uvi_idx < end_uvi_idx) {
      auto mid_uvi_idx = begin_uvi_idx + (end_uvi_idx - begin_uvi_idx) / 2; // >> 1
      const auto mid_uv_idx = xor_data.unique_variations_indices[mid_uvi_idx];
      const auto& mid_uv = or_data.unique_variations[mid_uv_idx];
      if (desired_idx >= mid_uv.start_idx) {
        if (desired_idx < mid_uv.start_idx + mid_uv.num_indices) {
          return mid_uv.first_compat_idx + (desired_idx - mid_uv.start_idx);
        }
        begin_uvi_idx = mid_uvi_idx + 1;  // right half
      } else {
        end_uvi_idx = mid_uvi_idx;  // left half
      }
    }
  }
  return or_data.num_compat_indices;
}

// Get a block-sized chunk of OR sources and test them for variation-
// compatibililty with the XOR source specified by the supplied
// xor_combo_index, and for OR-compatibility with the supplied source.
// Return true if at least one OR source is compatible.
__device__ auto get_OR_sources_chunk(const SourceCompatibilityData& source,
    unsigned or_chunk_idx, const UsedSources::Variations& xor_variations,
    index_t num_uv_indices) {
  // one thread per compat_idx
  auto begin = clock64();
  //const auto or_compat_idx = get_OR_compat_idx_linear_results(or_chunk_idx);
  //const auto or_compat_idx = get_OR_compat_idx_linear_indices(or_chunk_idx, num_uv_indices);
  const auto or_compat_idx =
      get_OR_compat_idx_logn(or_chunk_idx, num_uv_indices);

  #ifdef CLOCKS
  atomicAdd(&or_get_compat_idx_clocks, clock64() - begin);
  #endif

  if (or_compat_idx >= or_data.num_compat_indices) return false;

  #ifdef DEBUG_OR_COUNTS
  atomicAdd(&count_or_src_considered, 1);
  #endif

  const auto or_combo_idx = or_data.compat_indices[or_compat_idx];
  const auto or_variations = build_variations(or_combo_idx, or_data);
  begin = clock64();
  auto xor_avc =
      UsedSources::are_variations_compatible(xor_variations, or_variations);

  #ifdef CLOCKS
  atomicAdd(&or_xor_variation_compat_clocks, clock64() - begin);
  #endif

  if (!xor_avc) return false;

  #ifdef DEBUG_OR_COUNTS
  atomicAdd(&count_or_xor_variation_compat, 1);
  #endif

  begin = clock64();
  // TODO: can't i use kNumSrcSentences here?
  // current way: build unpacks combo_idx, iterating sentences via merge,
  //              then iterate over sentences once more
  // alt1 way: unpack or_combo_idx, iterating NumSentences at each source,
  //           possible short circuit,no final iteration <<-- fastest
  // alt2 way: build unpacks as above, final iteration = kNumSentences
  auto or_avc = UsedSources::are_variations_compatible(
      source.usedSources.variations, or_variations);

  #ifdef CLOCKS
  atomicAdd(&or_src_variation_compat_clocks, clock64() - begin);
  #endif

  if (!or_avc) return false;

  #ifdef DEBUG_OR_COUNTS
  atomicAdd(&count_or_src_variation_compat, 1);
  #endif

  begin = clock64();
  auto ccr = check_src_compat_results(or_combo_idx, or_data);

  #ifdef CLOCKS
  atomicAdd(&or_check_src_compat_clocks, clock64() - begin);
  #endif

  if (!ccr) return false;
  
  #ifdef DEBUG_OR_COUNTS
  atomicAdd(&count_or_check_src_compat, 1);
  #endif

  if (!is_source_compatible_with_all<tag::OR>(source, or_combo_idx, or_data))
    return false;

  #ifdef DEBUG_OR_COUNTS
  atomicAdd(&count_or_src_compat, 1);
  #endif

  return true;
}

// With the XOR source identified by the supplied xor_combo_idx.
// Walk through OR-sources one block-sized chunk at a time, until we find
// a chunk that contains at least one OR source that is variation-compatible
// with all of the XOR sources identified by the supplied xor_combo_idx, and
// OR-compatible with the supplied source.
__device__ auto is_any_OR_source_compatible(
    const SourceCompatibilityData& source, fat_index_t xor_combo_idx) {
  const auto block_size = blockDim.x;
  __shared__ bool any_or_compat;
  __shared__ UsedSources::Variations xor_variations;
  if (threadIdx.x < xor_variations.size()) {
    if (!threadIdx.x) any_or_compat = false;
    xor_variations[threadIdx.x] = get_one_variation(threadIdx.x, xor_combo_idx);
  }
  __syncthreads();
  index_t num_indices{};
  if (!init_compat_variations(xor_variations, &num_indices)) return false;
  for (unsigned or_chunk_idx{};
      or_chunk_idx * block_size < or_data.num_compat_indices; ++or_chunk_idx) {
    if (get_OR_sources_chunk(source, or_chunk_idx, xor_variations, num_indices)) {
      any_or_compat = true;
    }
    __syncthreads();

    // Or compatibility here is "success" for the supplied source and will
    // result in an exit out of is_compat_loop.
    if (any_or_compat) return true;
  }
  // No OR sources were compatible with both the supplied xor_combo_idx and
  // the supplied source. The next call to this function will be with a new
  // xor_combo_idx.
  return false;
}

// not the fastest function in the world. but keeps GPU busy at least.
__device__ __forceinline__ auto next_xor_result_idx(index_t result_idx) {
  result_t* xor_results = (result_t*)&dynamic_shared[kXorResults];
  const auto block_size = blockDim.x;
  while ((result_idx < block_size) && !xor_results[result_idx])
    result_idx++;
  return result_idx;
}

}  // namespace

// With any XOR results in the specified chunk
__device__ bool is_any_OR_source_compatible(
    const SourceCompatibilityData& source, index_t xor_chunk_idx,
    const FatIndexSpanPair& xor_idx_spans) {
  result_t* xor_results = (result_t*)&dynamic_shared[kXorResults];
  __shared__ bool any_or_compat;
  if (!threadIdx.x) any_or_compat = false;
  const auto block_size = blockDim.x;
  const index_t num_xor_indices =
      xor_idx_spans.first.size() + xor_idx_spans.second.size();
  auto max_results = num_xor_indices - blockDim.x * xor_chunk_idx;
  if (max_results > block_size) max_results = block_size;
  __syncthreads();
  for (index_t xor_results_idx{next_xor_result_idx(0)};
      xor_results_idx < max_results;) {
    const auto xor_flat_idx = get_flat_idx(xor_chunk_idx, xor_results_idx);
    const auto xor_combo_idx = get_xor_combo_index(xor_flat_idx, xor_idx_spans);
    if (is_any_OR_source_compatible(source, xor_combo_idx)) {
      any_or_compat = true;
    }
    __syncthreads();

    #if defined(PRINTF)
    if (threadIdx.x == CHECK_TID) {
      printf("  block: %u num_idx: %u chunk_idx: %u"
             " results_idx: %u max_results: %u compat: %d\n",
          blockIdx.x, num_xor_indices, xor_chunk_idx, xor_results_idx,
          max_results, any_or_compat ? 1 : 0);
    }
    #endif

    // Or compatibility success ends the search for this source and results
    // in an exit out of is_compat_loop.
    if (any_or_compat) return true;
    // Or compatibility failure means we have exhausted all OR source chunks
    // for this XOR result; proceed to next XOR result.
    xor_results_idx = next_xor_result_idx(xor_results_idx + 1);
  }
  // No XOR results in this XOR chunk were compatible. The next call to this
  // function for this block will be with a new XOR chunk.
  return false;
}

}  // namespace cm
