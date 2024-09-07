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
extern __device__ atomic64_t or_incompat_xor_clocks;

extern __device__ atomic64_t count_or_get_compat_idx;
extern __device__ atomic64_t count_or_src_considered;
extern __device__ atomic64_t count_or_src_variation_compat;
extern __device__ atomic64_t count_or_check_src_compat;
extern __device__ atomic64_t count_or_src_compat;
extern __device__ atomic64_t count_or_get_compat_idx;
extern __device__ atomic64_t count_or_xor_chunks;
extern __device__ atomic64_t count_or_incompat_xor_chunks;
extern __device__ atomic64_t count_or_incompat_xor_chunk_sources;
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
  if (compute_variations_compat_results(xor_variations, or_data,
          xor_data.or_compat_uv_indices)) {
    num_uv_indices = compute_compat_uv_indices(or_data.num_unique_variations,
        xor_data.or_compat_uv_indices);
  }

  #ifdef CLOCKS
  atomicAdd(&or_compute_compat_uv_indices_clocks, clock64() - begin);
  #endif

  return num_uv_indices;
}

__device__ __forceinline__ fat_index_t pack(const index_t hi, const index_t lo) {
  return (fat_index_t{hi} << 32) | lo;
}

__device__ __forceinline__ index_t loword(const fat_index_t idx) {
  return index_t(idx & 0xffffffff);
}

__device__ __forceinline__ index_t hiword(const fat_index_t idx) {
  return index_t(idx >> 32);
}

// incremental indexing of xor_data.unique_variations_indices
__device__ fat_index_t get_OR_compat_idx_incremental_uv(
    const index_t chunk_idx, const index_t num_uv_indices) {
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
      break;
    }
    desired_idx -= (uv->num_indices - start_idx);
    start_idx = 0;
  }
  auto result = (uvi_idx < num_uv_indices)
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

// TODO: update comment
// With the XOR source identified by the supplied xor_combo_idx.
// Walk through OR-sources one block-sized chunk at a time, until we find
// a chunk that contains at least one OR source that is variation-compatible
// with all of the XOR sources identified by the supplied xor_combo_idx, and
// OR-compatible with the supplied source.
__device__ auto is_any_OR_source_compatible(
    const SourceCompatibilityData& source, const fat_index_t xor_combo_idx) {
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
  // compute_compat will call __sync
  const auto num_uv_indices = compute_compat_OR_uv_indices(xor_variations);
  if (!num_uv_indices) return false;
  for (index_t or_chunk_idx{};
      or_chunk_idx * block_size < or_data.num_compat_indices; ++or_chunk_idx) {
    __syncthreads();
    if (get_OR_sources_chunk(source, or_chunk_idx, xor_variations,
            num_uv_indices)) {
      any_or_compat = true;
    }
    __syncthreads();
    if (any_or_compat) break;
    if (dynamic_shared[kOrStartUvIdx] == num_uv_indices) break;
  }
  // Or compatibility here is "success" for the supplied source and will
  // result in an exit out of is_compat_loop.
  // If no OR sources were compatible, the next call to this function will
  // be with a new xor_combo_idx.
  return any_or_compat;
}

// not the fastest function in the world. but keeps GPU busy at least.
__device__ __forceinline__ auto next_xor_result_idx(index_t result_idx) {
  result_t* xor_results = (result_t*)&dynamic_shared[kXorResults];
  const auto block_size = blockDim.x;
  while ((result_idx < block_size) && !xor_results[result_idx])
    result_idx++;
  return result_idx;
}

__device__ auto get_xor_combo_index(index_t xor_flat_idx) {
  // TODO: really should use num_uv_indices here
  const auto uvi_offset = blockIdx.x * xor_data.num_unique_variations;
  for (index_t uvi_idx{}; uvi_idx < xor_data.num_unique_variations; ++uvi_idx) {
    const auto uv_idx = xor_data.src_compat_uv_indices[uvi_offset + uvi_idx];
    const auto& uv = xor_data.unique_variations[uv_idx];
    if (xor_flat_idx < uv.num_indices) {
      return xor_data.compat_indices[uv.first_compat_idx + xor_flat_idx];
    }
    xor_flat_idx -= uv.num_indices;
  }
  assert(0);
  return 0ul;
}

}  // namespace

// With any XOR results in the specified chunk
__device__ bool is_any_OR_source_compatible(
    const SourceCompatibilityData& source, const index_t xor_chunk_idx) {
  result_t* xor_results = (result_t*)&dynamic_shared[kXorResults];
  __shared__ bool any_or_compat;
  if (!threadIdx.x) any_or_compat = false;
  __syncthreads();
  const auto block_size = blockDim.x;
  const auto num_xor_indices = xor_data.num_compat_indices;

  #ifdef DEBUG_OR_COUNTS
  if (!threadIdx.x) atomicAdd(&count_or_xor_chunks, 1);
  #endif
  auto begin = clock64();

  const auto max_results = block_size;
  index_t num_sources{};
  for (auto xor_results_idx{next_xor_result_idx(0)};
      xor_results_idx < block_size; ++num_sources) {

    const auto xor_flat_idx = get_flat_idx(xor_chunk_idx, xor_results_idx);
    const auto xor_combo_idx = get_xor_combo_index(xor_flat_idx);
    if (is_any_OR_source_compatible(source, xor_combo_idx)) {
      any_or_compat = true;
    }
    __syncthreads();

    // Or compatibility success ends the search for this source and results
    // in an exit out of is_compat_loop.
    if (any_or_compat) return true;

    // Or compatibility failure means we have exhausted all OR source chunks
    // for this XOR result; proceed to next XOR result.
    xor_results_idx = next_xor_result_idx(xor_results_idx + 1);
  }
  // No XOR results in this XOR chunk were compatible. The next call to this
  // function for this block will be with a new XOR chunk.

  #ifdef DEBUG_OR_COUNTS
  if (!threadIdx.x) {
    atomicAdd(&count_or_incompat_xor_chunks, 1);
    atomicAdd(&count_or_incompat_xor_chunk_sources, num_sources);
  }
  #endif

  #ifdef CLOCKS
  atomicAdd(&or_incompat_xor_clocks, clock64() - begin);
  #endif

  return false;
}

}  // namespace cm
