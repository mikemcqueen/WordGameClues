// filter.cu

#include <type_traits>
#include <cuda_runtime.h>
#include "combo-maker.h"
#include "merge-filter-data.h"
#include "or-filter.cuh"

#define RESTRICT __restrict__

namespace cm {

extern __constant__ FilterData::DeviceCommon<fat_index_t> xor_data;
extern __constant__ FilterData::DeviceOr or_data;

#ifdef DEBUG_OR_COUNTS
extern __device__ atomic64_t count_or_src_considered;
extern __device__ atomic64_t count_or_xor_variation_compat;
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

// Get a block-sized chunk of OR sources and test them for variation-
// compatibililty with the XOR source specified by the supplied
// xor_combo_index, and for OR-compatibility with the supplied source.
// Return true if at least one OR source is compatible.
__device__ auto get_OR_sources_chunk(const SourceCompatibilityData& source,
    unsigned or_chunk_idx, const UsedSources::Variations& xor_variations) {
  // one thread per compat_idx
  const auto or_compat_idx = get_flat_idx(or_chunk_idx);
  if (or_compat_idx >= or_data.num_compat_indices) return false;

  #ifdef DEBUG_OR_COUNTS
  atomicAdd(&count_or_src_considered, 1);
  #endif

  const auto or_combo_idx = or_data.compat_indices[or_compat_idx];
  const auto or_variations = build_variations(or_combo_idx, or_data);
  //  if (!check_src_compat_results(xor_combo_idx, xor_data)) return false;
  if (!UsedSources::are_variations_compatible(xor_variations, or_variations))
    return false;

  #ifdef DEBUG_OR_COUNTS
  atomicAdd(&count_or_xor_variation_compat, 1);
  #endif

  if (!UsedSources::are_variations_compatible(source.usedSources.variations,  //
          or_variations))
    return false;

  #ifdef DEBUG_OR_COUNTS
  atomicAdd(&count_or_src_variation_compat, 1);
  #endif

  if (!check_src_compat_results(or_combo_idx, or_data)) return false;

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
  for (unsigned or_chunk_idx{};
      or_chunk_idx * block_size < or_data.num_compat_indices; ++or_chunk_idx) {
    if (get_OR_sources_chunk(source, or_chunk_idx, xor_variations)) {
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
__device__ __forceinline__ auto next_xor_result_idx(unsigned result_idx) {
  result_t* xor_results = (result_t*)&dynamic_shared[kXorResults];
  const auto block_size = blockDim.x;
  while ((result_idx < block_size) && !xor_results[result_idx])
    result_idx++;
  return result_idx;
}

}  // namespace

// With any XOR results in the specified chunk
__device__ bool is_any_OR_source_compatible(
    const SourceCompatibilityData& source, unsigned xor_chunk_idx,
    const FatIndexSpanPair& xor_idx_spans) {
  result_t* xor_results = (result_t*)&dynamic_shared[kXorResults];
  __shared__ bool any_or_compat;
  if (!threadIdx.x) any_or_compat = false;
  const auto block_size = blockDim.x;
  const auto num_xor_indices =
      xor_idx_spans.first.size() + xor_idx_spans.second.size();
  auto max_results = num_xor_indices - blockDim.x * xor_chunk_idx;
  if (max_results > block_size) max_results = block_size;
  __syncthreads();
  for (unsigned xor_results_idx{next_xor_result_idx(0)};
      xor_results_idx < max_results;) {
    const auto xor_flat_idx = get_flat_idx(xor_chunk_idx, xor_results_idx);
    const auto xor_combo_idx = get_xor_combo_index(xor_flat_idx, xor_idx_spans);
    if (is_any_OR_source_compatible(source, xor_combo_idx)) {
      any_or_compat = true;
    }
    __syncthreads();

    #ifdef PRINTF
    if (!threadIdx.x) {
      printf("  block: %u get_next_OR xor_chunk_idx: %u, or_chunk_idx: %u, "
             "xor_results_idx: %lu, compat: %d\n",
          blockIdx.x, xor_chunk_idx, or_chunk_idx, xor_results_idx,
          any_or_compat ? 1 : 0);
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
