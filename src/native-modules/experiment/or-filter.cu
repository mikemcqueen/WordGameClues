// filter.cu

#include <type_traits>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/copy.h>
#include "combo-maker.h"
#include "merge-filter-data.h"
#include "or-filter.cuh"
#include "mmebitset.h"

#define RESTRICT __restrict__

namespace cm {

namespace cg = cooperative_groups;

/*
extern __constant__ FilterData::DeviceXor xor_data;
extern __constant__ FilterData::DeviceOr or_data;
*/  

#ifdef DEBUG_OR_COUNTS
extern __device__ atomic64_t or_init_compat_variations_clocks;
extern __device__ atomic64_t or_get_compat_idx_clocks;
extern __device__ atomic64_t or_build_variation_clocks;
extern __device__ atomic64_t or_are_variations_compat_clocks;
extern __device__ atomic64_t or_check_src_compat_clocks;
extern __device__ atomic64_t or_is_src_compat_clocks;

extern __device__ atomic64_t count_or_get_compat_idx;
extern __device__ atomic64_t count_or_src_considered;
extern __device__ atomic64_t count_or_src_variation_compat;
extern __device__ atomic64_t count_or_check_src_compat;
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

__device__ auto init_variations_indices(index_t* num_uv_indices) {
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
        if (!threadIdx.x && d_flags[idx]) {
          d_indices[d_scan_output[idx]] = idx;
        }
      });
  // compute total set flags
  *num_uv_indices = d_scan_output[num_items - 1] + d_flags[num_items - 1];
  return true;
}

__device__ auto init_compat_variations(
    const UsedSources::Variations& xor_variations, index_t* num_uv_indices) {
  const auto begin = clock64();
  const auto result = init_variations_compat_results(xor_variations)
                      && init_variations_indices(num_uv_indices);

  #ifdef CLOCKS
  atomicAdd(&or_init_compat_variations_clocks, clock64() - begin);
  #endif

  return result;
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

// V2 (faster) linear walk of xor_data.unique_variations_indices
__device__ __forceinline__ auto get_OR_compat_idx_linear_indices(
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

// V3 (even faster) binary search of xor_data.unique_variations_indices
__device__ auto get_OR_compat_idx_binary_search_uv(
    index_t chunk_idx, index_t num_uv_indices) {
  const auto desired_idx = chunk_idx * blockDim.x + threadIdx.x;
  auto begin_uvi_idx = blockIdx.x * or_data.num_unique_variations;
  auto end_uvi_idx = begin_uvi_idx + num_uv_indices;
  const auto last_uv_idx = xor_data.unique_variations_indices[end_uvi_idx - 1];
  const auto& last_uv = or_data.unique_variations[last_uv_idx];
  if (desired_idx < last_uv.start_idx + last_uv.num_indices) {
    while (begin_uvi_idx < end_uvi_idx) {
      const auto mid_uvi_idx = begin_uvi_idx + (end_uvi_idx - begin_uvi_idx) / 2;
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

__device__ __forceinline__ fat_index_t pack(index_t hi, index_t lo) {
  return (fat_index_t{hi} << 32) | lo;
}

__device__ __forceinline__ index_t loword(fat_index_t idx) {
  return index_t(idx & 0xffffffff);
}

__device__ __forceinline__ index_t hiword(fat_index_t idx) {
  return index_t(idx >> 32);
}

// V4 (fastest) incremental indexing of xor_data.unique_variations_indices
__device__ fat_index_t get_OR_compat_idx_incremental_uv(
    index_t chunk_idx, index_t num_uv_indices) {
  assert(num_uv_indices > 0);
  const auto desired_idx = chunk_idx * blockDim.x + threadIdx.x;
  const auto first_uvi_idx = blockIdx.x * or_data.num_unique_variations;
  const auto last_uv_idx =
      xor_data.unique_variations_indices[first_uvi_idx + num_uv_indices - 1];
  const auto& last_uv = or_data.unique_variations[last_uv_idx];
  if (desired_idx < last_uv.start_idx + last_uv.num_indices) {

#if 0
    if ((dynamic_shared[kSrcFlatIdx] == 5497)
               && (threadIdx.x == blockDim.x - 1)) {
      printf("chunk: %u, starting uvi_idx: %u, num_uvi %u, num_uv: %u, "
             "desired_idx: %u, num_idx: %u\n",
          chunk_idx, dynamic_shared[kOrStartUvIdx], num_uv_indices,
          or_data.num_unique_variations, desired_idx,
          or_data.num_compat_indices);
    }
#endif

    auto uvi_idx = dynamic_shared[kOrStartUvIdx];
    for (; uvi_idx < num_uv_indices; ++uvi_idx) {
      const auto uv_idx =
          xor_data.unique_variations_indices[first_uvi_idx + uvi_idx];
      const auto& uv = or_data.unique_variations[uv_idx];
      if (desired_idx < uv.start_idx) break;
      if (desired_idx < uv.start_idx + uv.num_indices) {
        return pack(uvi_idx,
            uv.first_compat_idx + (desired_idx - uv.start_idx));
      }
    }
    return pack(uvi_idx, or_data.num_compat_indices);
  }
  return pack(num_uv_indices, or_data.num_compat_indices);
}


// Process a chunk of OR sources and test them for variation-
// compatibililty with all of the XOR sources identified by the supplied
// xor_combo_index, and for OR-compatibility with the supplied source.
// Return true if at least one OR source is compatible.
__device__ auto get_OR_sources_chunk(const SourceCompatibilityData& source,
    index_t or_chunk_idx, const UsedSources::Variations& xor_variations,
    index_t num_uv_indices) {

  #ifdef DEBUG_OR_COUNTS
  atomicAdd(&count_or_get_compat_idx, 1);
  #endif

  auto begin = clock64();
  // one thread per compat_idx
  //const auto or_compat_idx =
  //    get_OR_compat_idx_binary_search_uv(or_chunk_idx, num_uv_indices);
  const auto idx_pair =
      get_OR_compat_idx_incremental_uv(or_chunk_idx, num_uv_indices);
  __syncthreads();

  // SLOWWWWW for some reason
  if (threadIdx.x == blockDim.x - 1) {
    dynamic_shared[kOrStartUvIdx] = hiword(idx_pair);
  }

#if 0
  if (dynamic_shared[kSrcFlatIdx] == 5497) {
    if (!threadIdx.x || (threadIdx.x == blockDim.x - 1)) {
      printf("tid %u:, chunk: %u, ending uvi_idx: %u\n", threadIdx.x,
             or_chunk_idx, hiword(idx_pair));
    }
  }
#endif

  const auto or_compat_idx = loword(idx_pair);

  #ifdef CLOCKS
  atomicAdd(&or_get_compat_idx_clocks, clock64() - begin);
  #endif

  if (or_compat_idx >= or_data.num_compat_indices) return false;

  #ifdef DEBUG_OR_COUNTS
  atomicAdd(&count_or_src_considered, 1);
  #endif

  begin = clock64();
  const auto or_combo_idx = or_data.compat_indices[or_compat_idx];
  const auto or_variations = build_variations(or_combo_idx, or_data);

  #ifdef CLOCKS
  atomicAdd(&or_build_variation_clocks, clock64() - begin);
  #endif

  // TODO: can't i use kNumSrcSentences here?
  // current way: build unpacks combo_idx, iterating sentences via merge,
  //  then iterate over sentences once more to compare
  // alt1 way: unpack or_combo_idx, iterating NumSentences while comparing
  //  at each source, possible short circuit, no final iteration <<-- fastest
  // alt2 way: build unpacks as above, final iteration = kNumSentences

  begin = clock64();
  auto or_avc = UsedSources::are_variations_compatible(
      source.usedSources.variations, or_variations);

  #ifdef CLOCKS
  atomicAdd(&or_are_variations_compat_clocks, clock64() - begin);
  #endif

  if (!or_avc) return false;

  #ifdef DEBUG_OR_COUNTS
  atomicAdd(&count_or_src_variation_compat, 1);
  #endif

  begin = clock64();
  auto cscr = check_src_compat_results(or_combo_idx, or_data);

  #ifdef CLOCKS
  atomicAdd(&or_check_src_compat_clocks, clock64() - begin);
  #endif

  if (!cscr) return false;

  #ifdef DEBUG_OR_COUNTS
  atomicAdd(&count_or_check_src_compat, 1);
  #endif

  begin = clock64();
  auto isc =
      is_source_compatible_with_all<tag::OR>(source, or_combo_idx, or_data);


  #ifdef CLOCKS
  atomicAdd(&or_is_src_compat_clocks, clock64() - begin);
  #endif

  if (!isc) return false;

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
__device__ auto is_any_OR_source_compatible_old(
    const SourceCompatibilityData& source, fat_index_t xor_combo_idx) {
  const auto block_size = blockDim.x;
  __shared__ bool any_or_compat;
  __shared__ UsedSources::Variations xor_variations;
  if (threadIdx.x < xor_variations.size()) {
    if (!threadIdx.x) {
      dynamic_shared[kOrStartUvIdx] = 0;
      any_or_compat = false;
    }
    xor_variations[threadIdx.x] = get_one_variation(threadIdx.x, xor_combo_idx);
  }
  __syncthreads();
  index_t num_uv_indices{};
  if (!init_compat_variations(xor_variations, &num_uv_indices)) return false;

  auto tile =
      cg::tiled_partition<32, cg::thread_block>(cg::this_thread_block());

  for (index_t or_chunk_idx{};
      or_chunk_idx * block_size < or_data.num_compat_indices; ++or_chunk_idx) {
    if (get_OR_sources_chunk(source, or_chunk_idx, xor_variations,  //
            num_uv_indices)) {
      any_or_compat = true;
    }
    __syncthreads();

    // Or compatibility here is "success" for the supplied source and will
    // result in an exit out of is_compat_loop.
    if (any_or_compat) return true;
    if (dynamic_shared[kOrStartUvIdx] == num_uv_indices) return false;
  }
  // No OR sources were compatible with both the supplied xor_combo_idx and
  // the supplied source. The next call to this function will be with a new
  // xor_combo_idx.
  return false;
}

// V4 (fastest) incremental indexing of xor_data.unique_variations_indices
__device__ fat_index_t get_OR_compat_idx_incremental_uv2(index_t chunk_idx,
    index_t chunk_size, index_t start_uvi_idx,
    index_t start_src_idx,  // unnecessary?
    index_t num_uv_indices) {
  assert(num_uv_indices > 0);
  const auto desired_idx = chunk_idx * chunk_size + threadIdx.x;
  const auto first_uvi_idx = blockIdx.x * or_data.num_unique_variations;
  const auto last_uv_idx =
      xor_data.unique_variations_indices[first_uvi_idx + num_uv_indices - 1];
  const auto& last_uv = or_data.unique_variations[last_uv_idx];
  if (desired_idx < last_uv.start_idx + last_uv.num_indices) {

#if 0
    if ((dynamic_shared[kSrcFlatIdx] == 5497)
        NO //       && (threadIdx.x == blockDim.x - 1)) {
      printf("chunk: %u, starting uvi_idx: %u, num_uvi %u, num_uv: %u, "
             "desired_idx: %u, num_idx: %u\n",
          chunk_idx, dynamic_shared[kOrStartUvIdx], num_uv_indices,
          or_data.num_unique_variations, desired_idx,
          or_data.num_compat_indices);
    }
#endif

    auto uvi_idx = start_uvi_idx;
    for (; uvi_idx < num_uv_indices; ++uvi_idx) {
      const auto uv_idx =
          xor_data.unique_variations_indices[first_uvi_idx + uvi_idx];
      const auto& uv = or_data.unique_variations[uv_idx];
      if (desired_idx < uv.start_idx) break;
      if (desired_idx < uv.start_idx + uv.num_indices) {
        return pack(uvi_idx,
            uv.first_compat_idx + (desired_idx - uv.start_idx));
      }
    }
    return pack(uvi_idx, or_data.num_compat_indices);
  }
  return pack(num_uv_indices, or_data.num_compat_indices);
}

using warp_tile = cg::thread_block_tile<32, cg::thread_block>;

__device__ int get_OR_sources_chunk(warp_tile tile, index_t num_uv_indices) {
  auto or_idx_buffer = &dynamic_shared[kXorResults + blockDim.x / 4];
  index_t buffer_idx{};
  index_t start_uvi_idx{};
  index_t start_src_idx{};

  or_idx_buffer[tile.thread_rank()] = or_data.num_compat_indices;
  if (tile.thread_rank() == tile.num_threads() - 1) {
    start_uvi_idx = dynamic_shared[kOrStartUvIdx];
    //start_src_idx = dynamic_shared[kOrStartSrcIdx];
  }
  tile.shfl(start_uvi_idx, tile.num_threads() - 1);
  //tile.shfl(start_src_idx, tile.num_threads() - 1);
  index_t chunk_idx{};
  for (; start_uvi_idx < num_uv_indices && buffer_idx < blockDim.x;
      ++chunk_idx) {
    const auto idx_pair = get_OR_compat_idx_incremental_uv2(chunk_idx,
        tile.num_threads(), start_uvi_idx, start_src_idx, num_uv_indices);
    const auto uvi_idx = hiword(idx_pair);
    const auto or_compat_idx = loword(idx_pair);
    /*
    const auto num_remaining =
        std::min(tile.num_threads(), blockDim.x - buffer_idx);
    if (tile.thread_rank() < num_remaining) {
    */
    or_idx_buffer[buffer_idx + tile.thread_rank()] = or_compat_idx;
    //}
    start_uvi_idx = uvi_idx;
    tile.shfl(start_uvi_idx, tile.num_threads() - 1);
    /*
    if ((buffer_idx + num_remaining == blockDim.x)
        && (tile.thread_rank() == tile.num_threads() - 1)) {
      dynamic_shared[kOrStartSrcIdx] = or_compat_idx;
    }
    */
    buffer_idx += tile.num_threads();
  }
  if (tile.thread_rank() == tile.num_threads() - 1) {
    dynamic_shared[kOrStartUvIdx] = start_uvi_idx;
  }
  //try: if (chunk_idx > 1) return 1;
  return or_idx_buffer[tile.thread_rank()] < or_data.num_compat_indices ? 1 : 0;
}

// Process a chunk of OR sources and test them for OR-compatibility with the
// supplied source. Return true if at least one OR source is compatible.
__device__ auto process_OR_sources_chunk(
    const SourceCompatibilityData& source) {
  auto or_idx_buffer = &dynamic_shared[kXorResults + blockDim.x / 4];
  const auto or_compat_idx = or_idx_buffer[threadIdx.x];
  if (or_compat_idx >= or_data.num_compat_indices) return false;

  #ifdef DEBUG_OR_COUNTS
  atomicAdd(&count_or_src_considered, 1);
  #endif

  auto begin = clock64();
  const auto or_combo_idx = or_data.compat_indices[or_compat_idx];
  const auto or_variations = build_variations(or_combo_idx, or_data);

  #ifdef CLOCKS
  atomicAdd(&or_build_variation_clocks, clock64() - begin);
  #endif

  // TODO: can't i use kNumSrcSentences here?
  // current way: build unpacks combo_idx, iterating sentences via merge,
  //  then iterate over sentences once more to compare
  // alt1 way: unpack or_combo_idx, iterating NumSentences while comparing
  //  at each source, possible short circuit, no final iteration <<-- fastest
  // alt2 way: build unpacks as above, final iteration = kNumSentences

  begin = clock64();
  auto or_avc = UsedSources::are_variations_compatible(
      source.usedSources.variations, or_variations);

  #ifdef CLOCKS
  atomicAdd(&or_are_variations_compat_clocks, clock64() - begin);
  #endif

  if (!or_avc) return false;

  #ifdef DEBUG_OR_COUNTS
  atomicAdd(&count_or_src_variation_compat, 1);
  #endif

  begin = clock64();
  auto cscr = check_src_compat_results(or_combo_idx, or_data);

  #ifdef CLOCKS
  atomicAdd(&or_check_src_compat_clocks, clock64() - begin);
  #endif

  if (!cscr) return false;

  #ifdef DEBUG_OR_COUNTS
  atomicAdd(&count_or_check_src_compat, 1);
  #endif

  begin = clock64();
  auto isc =
      is_source_compatible_with_all<tag::OR>(source, or_combo_idx, or_data);


  #ifdef CLOCKS
  atomicAdd(&or_is_src_compat_clocks, clock64() - begin);
  #endif

  if (!isc) return false;

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
  __shared__ bool any_sources;
  __shared__ bool any_or_compat;
  __shared__ UsedSources::Variations xor_variations;
  if (threadIdx.x < xor_variations.size()) {
    if (!threadIdx.x) {
      dynamic_shared[kOrStartUvIdx] = 0;
      any_sources = false;
      any_or_compat = false;
    }
    xor_variations[threadIdx.x] = get_one_variation(threadIdx.x, xor_combo_idx);
  }
  __syncthreads();
  index_t num_uv_indices{};
  if (!init_compat_variations(xor_variations, &num_uv_indices)) return false;

  auto tile =
      cg::tiled_partition<32, cg::thread_block>(cg::this_thread_block());

  for (index_t or_chunk_idx{};
      or_chunk_idx * block_size < or_data.num_compat_indices; ++or_chunk_idx) {

    if (!tile.meta_group_rank()) { 
      any_sources = tile.any(get_OR_sources_chunk(tile, num_uv_indices));
    }
    __syncthreads();
    if (any_sources && process_OR_sources_chunk(source)) {
      any_or_compat = true;
    }
    __syncthreads();
    // Or compatibility here is "success" for the supplied source and will
    // result in an exit out of is_compat_loop.
    if (any_or_compat) return true;
    if (!any_sources || (dynamic_shared[kOrStartUvIdx] == num_uv_indices))
      return false;
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
    const IndexSpanPair& xor_idx_spans) {
  result_t* xor_results = (result_t*)&dynamic_shared[kXorResults];
  __shared__ bool any_or_compat;
  if (!threadIdx.x) any_or_compat = false;
  __syncthreads();
  const auto block_size = blockDim.x;
  const index_t num_xor_indices =
      xor_idx_spans.first.size() + xor_idx_spans.second.size();
  auto max_results = num_xor_indices - blockDim.x * xor_chunk_idx;
  if (max_results > block_size) max_results = block_size;
  for (auto xor_results_idx{next_xor_result_idx(0)};
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
