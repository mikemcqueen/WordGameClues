// filter.cu

#include <type_traits>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
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
extern __device__ atomic64_t or_compute_compat_uv_indices_clocks;
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

__device__ auto compute_compat_OR_uv_indices(const Variations& xor_variations) {
  const auto begin = clock64();
  index_t num_uv_indices{};
  if (compute_variations_compat_results(
          xor_variations, or_data, xor_data.variations_results_per_block)) {
    num_uv_indices = compute_compat_uv_indices(or_data.num_unique_variations,
        xor_data.or_compat_uv_indices, xor_data.variations_results_per_block);
  }

  #ifdef CLOCKS
  atomicAdd(&or_compute_compat_uv_indices_clocks, clock64() - begin);
  #endif

  return num_uv_indices;
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

// incremental indexing of xor_data.unique_variations_indices
__device__ fat_index_t get_OR_compat_idx_incremental_uv(
    index_t chunk_idx, index_t num_uv_indices) {
  assert(num_uv_indices > 0);
  auto desired_idx = chunk_idx * blockDim.x + threadIdx.x;
  const auto first_uvi_idx = blockIdx.x * or_data.num_unique_variations;
  const UniqueVariations* uv{};
  auto uvi_idx = dynamic_shared[kOrStartUvIdx];
  auto start_idx = dynamic_shared[kOrStartSrcIdx];
  for (; uvi_idx < num_uv_indices; ++uvi_idx) {
    const auto or_uv_idx = xor_data.or_compat_uv_indices[first_uvi_idx + uvi_idx];
    uv = &or_data.unique_variations[or_uv_idx];
    if (desired_idx < (uv->num_indices - start_idx)) {
      //src_idx = uv.first_compat_idx + start_idx + desired_idx;
      //uv_num_indices = uv.num_indices;
      break;
    }
    desired_idx -= (uv->num_indices - start_idx);
    start_idx = 0;
  }
  __syncthreads();
  auto result =
      (uvi_idx < num_uv_indices)
          ? pack(uvi_idx, uv->first_compat_idx + start_idx + desired_idx)
          : pack(uvi_idx, or_data.num_compat_indices);
  if (threadIdx.x == blockDim.x - 1) {
    if (uvi_idx < num_uv_indices) {
      start_idx += desired_idx + 1;
      if (start_idx == uv->num_indices) {
        uvi_idx++;
        start_idx = 0;
      }
    }
    // might be technically unnecessary, but it keeps the data clean &
    // consistent for purposes of debugging a new/complex implementation
    if (uvi_idx == num_uv_indices) {
      start_idx = or_data.num_compat_indices;
    }
    dynamic_shared[kOrStartUvIdx] = uvi_idx;
    dynamic_shared[kOrStartSrcIdx] = start_idx;
  }
  return result;
}

// Process a chunk of OR sources and test them for variation-
// compatibililty with all of the XOR sources identified by the supplied
// xor_combo_index, and for OR-compatibility with the supplied source.
// Return true if at least one OR source is compatible.
__device__ auto get_OR_sources_chunk(const SourceCompatibilityData& source,
    const index_t or_chunk_idx, const Variations& xor_variations,
    const index_t num_uv_indices) {

  #ifdef DEBUG_OR_COUNTS
  atomicAdd(&count_or_get_compat_idx, 1);
  #endif

  auto begin = clock64();
  // one thread per compat_idx
  const auto packed_idx = get_OR_compat_idx_incremental_uv(or_chunk_idx, num_uv_indices);
  const auto or_compat_idx = loword(packed_idx);

  #ifdef CLOCKS
  atomicAdd(&or_get_compat_idx_clocks, clock64() - begin);
  #endif

  if (or_compat_idx >= or_data.num_compat_indices) return false;

#ifdef LOGGY
  printf("%u\n", /*  dynamic_shared[kSrcFlatIdx]*/ or_compat_idx);
#endif

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
__device__ auto is_any_OR_source_compatible(
    const SourceCompatibilityData& source, fat_index_t xor_combo_idx) {
  const auto block_size = blockDim.x;
  __shared__ bool any_or_compat;
  __shared__ Variations xor_variations;
  if (threadIdx.x < xor_variations.size()) {
    if (!threadIdx.x) {
      dynamic_shared[kOrStartUvIdx] = 0;
      dynamic_shared[kOrStartSrcIdx] = 0;
      any_or_compat = false;
    }
    xor_variations[threadIdx.x] = get_one_variation(threadIdx.x, xor_combo_idx);
  }
  __syncthreads();
  const auto num_uv_indices = compute_compat_OR_uv_indices(xor_variations);
  if (!num_uv_indices) return false;
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
    if (dynamic_shared[kOrStartUvIdx] == num_uv_indices) break;
  }
  // No OR sources were compatible with both the supplied xor_combo_idx and
  // the supplied source. The next call to this function will be with a new
  // xor_combo_idx.
  return false;
}

using warp_tile = cg::thread_block_tile<32, cg::thread_block>;

// V4 (fastest) incremental indexing of xor_data.unique_variations_indices
__device__ fat_index_t get_OR_compat_idx_incremental_uv2(
    warp_tile tile, index_t chunk_idx, index_t num_uv_indices) {
  assert(num_uv_indices > 0);
  const auto first_uvi_idx = blockIdx.x * or_data.num_unique_variations;
  const UniqueVariations* uv{};
  auto uvi_idx = dynamic_shared[kOrStartUvIdx];
  auto start_idx = dynamic_shared[kOrStartSrcIdx];
  auto desired_idx = chunk_idx * tile.num_threads() + tile.thread_rank();
  for (; uvi_idx < num_uv_indices; ++uvi_idx) {
    const auto or_uv_idx = xor_data.or_compat_uv_indices[first_uvi_idx + uvi_idx];
    uv = &or_data.unique_variations[or_uv_idx];
    if (desired_idx < (uv->num_indices - start_idx)) break;
    desired_idx -= (uv->num_indices - start_idx);
    start_idx = 0;
  }
  auto result =
      (uvi_idx < num_uv_indices)
          ? pack(uvi_idx, uv->first_compat_idx + start_idx + desired_idx)
          : pack(uvi_idx, or_data.num_compat_indices);
  if (tile.thread_rank() == tile.num_threads() - 1) {
    if (uvi_idx < num_uv_indices) {
      start_idx += desired_idx + 1;
      if (start_idx == uv->num_indices) {
        uvi_idx++;
        start_idx = 0;
      }
    }
    // technically unnecessary, just keeps the data clean & consistent
    // for purposes of debugging a complex implementation
    if (uvi_idx == num_uv_indices) {
      start_idx = or_data.num_compat_indices;
    }
    dynamic_shared[kOrStartUvIdx] = uvi_idx;
    dynamic_shared[kOrStartSrcIdx] = start_idx;
  }
  return result;
}

__device__ int get_OR_sources_block(
    warp_tile tile, index_t block_idx, index_t num_uv_indices) {
  auto or_idx_buffer = &dynamic_shared[kXorResults + blockDim.x / 4];
  index_t buffer_idx{};
  index_t chunk_idx = block_idx * tile.meta_group_size();
  for (; buffer_idx < blockDim.x; ++chunk_idx) {
    const auto idx_pair =
        get_OR_compat_idx_incremental_uv2(tile, chunk_idx, num_uv_indices);
    const auto uvi_idx = hiword(idx_pair);
    auto or_compat_idx = loword(idx_pair);
    or_idx_buffer[buffer_idx + tile.thread_rank()] = or_compat_idx;
    // TODO no need to return a packed index pair
    or_compat_idx = tile.shfl(or_compat_idx, tile.num_threads() - 1);
    buffer_idx += tile.num_threads();
    if (or_compat_idx >= or_data.num_compat_indices) break;
  }
  unsigned any_compat{};
  if (!tile.thread_rank()) {
    any_compat = or_idx_buffer[0] < or_data.num_compat_indices ? 1 : 0;
  }
  any_compat = tile.shfl(any_compat, 0);
  if (any_compat) {
    // fill any remaining chunk enrtires in the block with no-op
    for (; buffer_idx < blockDim.x; buffer_idx += tile.num_threads()) {
      or_idx_buffer[buffer_idx + tile.thread_rank()] =
          or_data.num_compat_indices;
    }
    return 1;
  }
  return 0;
}

// Process a chunk of OR sources and test them for OR-compatibility with the
// supplied source. Return true if at least one OR source is compatible.
__device__ auto process_OR_sources_block(
    const SourceCompatibilityData& source) {
  auto or_idx_buffer = &dynamic_shared[kXorResults + blockDim.x / 4];
  const auto or_compat_idx = or_idx_buffer[threadIdx.x];
  if (or_compat_idx >= or_data.num_compat_indices) return false;

#ifdef LOGGY
  printf("%u\n", /*  dynamic_shared[kSrcFlatIdx]*/ or_compat_idx);
#endif

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
__device__ auto is_any_OR_source_compatible_warp(
    const SourceCompatibilityData& source, fat_index_t xor_combo_idx) {
  const auto block_size = blockDim.x;
  __shared__ bool any_sources;
  __shared__ bool any_or_compat;
  __shared__ Variations xor_variations;
  if (threadIdx.x < xor_variations.size()) {
    if (!threadIdx.x) {
      dynamic_shared[kOrStartUvIdx] = 0;
      dynamic_shared[kOrStartSrcIdx] = 0;
      any_sources = false;
      any_or_compat = false;
    }
    xor_variations[threadIdx.x] = get_one_variation(threadIdx.x, xor_combo_idx);
  }
  __syncthreads();
  const auto num_uv_indices = compute_compat_OR_uv_indices(xor_variations);
  if (!num_uv_indices) return false;

  auto tile =
      cg::tiled_partition<32, cg::thread_block>(cg::this_thread_block());

  for (index_t or_block_idx{};
      or_block_idx * block_size < or_data.num_compat_indices; ++or_block_idx) {

    if (!tile.meta_group_rank()) {
      any_sources = tile.any(get_OR_sources_block(tile, or_block_idx, num_uv_indices));
    }
    __syncthreads();
    if (any_sources && process_OR_sources_block(source)) {
      any_or_compat = true;
    }
    __syncthreads();
    // Or compatibility here is "success" for the supplied source and will
    // result in an exit out of is_compat_loop.
    if (any_or_compat) return true;
    if (!any_sources || (dynamic_shared[kOrStartUvIdx] == num_uv_indices)) {
#ifdef LOGGY
      if (!threadIdx.x) {
        printf("early bail, any_sources: %s, start_uvi_idx: %u\n",
            any_sources ? "true" : "false", dynamic_shared[kOrStartUvIdx]);
      }
#endif
      return false;
    }
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
  const auto num_xor_indices =
      index_t(xor_idx_spans.first.size() + xor_idx_spans.second.size());
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

#ifdef LOGGY
    if (!blockIdx.x && !threadIdx.x) printf("----\n");
    __syncthreads();
#endif

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
