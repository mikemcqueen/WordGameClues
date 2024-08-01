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
#include "stream-data.h"
#include "merge-filter-data.h"
#include "util.h"

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
  if (!threadIdx.x) contains_both = false;
  // one thread per src_desc_pair
  for (unsigned idx{}; idx * blockDim.x < num_src_desc_pairs; ++idx) {
    __syncthreads();
    if (contains_both) return true;
    const auto pair_idx = idx * blockDim.x + threadIdx.x;
    if (pair_idx < num_src_desc_pairs) {
      if (source.usedSources.has(src_desc_pairs[pair_idx])) {
        contains_both = true;
      }
    }
  }
  return false;
}

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
      }
    }
  }
}

extern __shared__ unsigned dynamic_shared[];

/*
__device__ bool is_source_OR_compatible_old(const SourceCompatibilityData& source,
    const unsigned num_or_args,
    const device::OrSourceData* __restrict__ or_arg_sources,
    const unsigned num_or_arg_sources) {
  //  extern __shared__ result_t or_arg_results[];
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
        or_arg_results[or_src.or_arg_idx] = 1;
      }
    }
  }
  __syncthreads();
  if (!threadIdx.x) {
    bool compat_with_all{true};
    for (int i{}; i < num_or_args; ++i) {
      if (!or_arg_results[i]) {
        compat_with_all = false;
        break;
      }
    }
    return compat_with_all;
  }
  return false;
}

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

struct SmallestSpansResult {
  bool skip;
  ComboIndexSpanPair spans;
};
using smallest_spans_result_t = SmallestSpansResult;

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

__device__ uint64_t get_flat_idx(
    uint64_t block_idx, unsigned thread_idx = threadIdx.x) {
  return block_idx * uint64_t(blockDim.x) + thread_idx;
}

__device__ auto get_xor_combo_index(
    uint64_t flat_idx, const ComboIndexSpanPair& idx_spans) {
  if (flat_idx < idx_spans.first.size()) return idx_spans.first[flat_idx];
  flat_idx -= idx_spans.first.size();
  assert(flat_idx < idx_spans.second.size());
  return idx_spans.second[flat_idx];
}

__device__ auto get_combo_index(uint64_t flat_idx,
    const index_t* __restrict__ idx_list_sizes, unsigned num_idx_lists) {
  combo_index_t combo_idx{};
  for (int list_idx{int(num_idx_lists) - 1}; list_idx >= 0; --list_idx) {
    const auto idx_list_size = idx_list_sizes[list_idx];
    if (combo_idx) combo_idx *= idx_list_size;
    combo_idx += flat_idx % idx_list_size;
    flat_idx /= idx_list_size;
  }
  return combo_idx;
}

const int kXorChunkIdx = 0;
const int kOrChunkIdx = 1;
const int kXorResultsIdx = 2;
const int kNumSharedIndices = 4;  // 32-bit align

// TODO: templatize this & is_source_XOR_compatible
// Test if a source is OR compatible with the OR source specified by the
// supplied combo index and index lists.
__device__ bool is_source_OR_compatible(const SourceCompatibilityData& source,
    combo_index_t or_combo_idx,
    const SourceCompatibilityData* __restrict__ or_src_lists,
    const index_t* __restrict__ or_src_list_start_indices,
    const index_t* __restrict__ or_idx_lists,
    const index_t* __restrict__ or_idx_list_start_indices,
    const index_t* __restrict__ or_idx_list_sizes, unsigned num_or_idx_lists) {
  for (unsigned list_idx{}; list_idx < num_or_idx_lists; ++list_idx) {
    const auto or_src_list =
        &or_src_lists[or_src_list_start_indices[list_idx]];
    const auto idx_list = &or_idx_lists[or_idx_list_start_indices[list_idx]];
    const auto idx_list_size = or_idx_list_sizes[list_idx];
    const auto or_src_idx = idx_list[or_combo_idx % idx_list_size];
    const auto& or_src = or_src_list[or_src_idx];
    if (!or_src.isOrCompatibleWith(source)) return false;
    or_combo_idx /= idx_list_size;
  }
  return true;
}

// Get the next block-sized chunk of OR sources and test them for variation-
// compatibililty with the XOR source specified by the supplied xor_combo_index,
// and for OR-compatibility with the supplied source.
// Return true if at least one OR source is compatible.
__device__ bool get_next_OR_sources_chunk(const SourceCompatibilityData& source,
    unsigned or_chunk_idx, const combo_index_t xor_combo_idx,
    const SourceCompatibilityData* __restrict__ xor_src_lists,
    const index_t* __restrict__ xor_src_list_start_indices,
    const index_t* __restrict__ xor_idx_lists,
    const index_t* __restrict__ xor_idx_list_start_indices,
    const index_t* __restrict__ xor_idx_list_sizes, unsigned num_idx_lists,
    const SourceCompatibilityData* __restrict__ or_src_lists,
    const index_t* __restrict__ or_src_list_start_indices,
    const index_t* __restrict__ or_idx_lists,
    const index_t* __restrict__ or_idx_list_start_indices,
    const index_t* __restrict__ or_idx_list_sizes,
    const unsigned num_or_idx_lists, const unsigned num_or_sources,
    result_t* __restrict__ or_results) {
  /*
  //  __shared__ bool is_variation_compat;
  //  __shared__ bool is_or_compat;
  if (!threadIdx.x) {
    is_or_compat = false;
    //    is_variation_compat = false;
    or_results[threadIdx.x] = 0;
  }
  __syncthreads();
  */
  const auto or_flat_idx = get_flat_idx(or_chunk_idx);
  if (or_flat_idx < num_or_sources) { // or: if (>=) return false;
    const auto or_combo_idx =
        get_combo_index(or_flat_idx, or_idx_list_sizes, num_or_idx_lists);
#if 0
    if (are_variations_compatible(
            or_combo_idx, or_src_lists, xor_combo_idx, xor_src_lists)) {
#endif
    // is_variation_compat = true;
    // }
    // }

    // TODO: get rid of sync here. and is_variation_compat shared var.
    // if a *specific instance* of an OR-source combo_idx is variation-
    // compatible with XOR-source, we can immediately check it for
    // compatibility with a *specfiic instance* of an OR source combo-idx.
    // In other words, nested if statements here, vs. sync, and only one
    // is_or_compat shared. *I THINK*.

    //__syncthreads();
    // if (is_variation_compat) {
    if (is_source_OR_compatible(source, or_combo_idx, or_src_lists,
            or_src_list_start_indices, or_idx_lists, or_idx_list_start_indices,
            or_idx_list_sizes, num_or_idx_lists)) {
      //is_or_compat = true;
      or_results[threadIdx.x] = 1;
      return true;
    }
  }
  //__syncthreads();
  //  return is_or_compat;
  return false;
}

// for one XOR result
__device__ bool get_next_compatible_OR_sources(
    const SourceCompatibilityData& source, unsigned or_chunk_idx,
    combo_index_t xor_combo_idx,
    const SourceCompatibilityData* __restrict__ xor_src_lists,
    const index_t* __restrict__ xor_src_list_start_indices,
    const index_t* __restrict__ xor_idx_lists,
    const index_t* __restrict__ xor_idx_list_start_indices,
    const index_t* __restrict__ xor_idx_list_sizes, unsigned num_xor_idx_lists,
    const SourceCompatibilityData* __restrict__ or_src_lists,
    const index_t* __restrict__ or_src_list_start_indices,
    const index_t* __restrict__ or_idx_lists,
    const index_t* __restrict__ or_idx_list_start_indices,
    const index_t* __restrict__ or_idx_list_sizes,
    const unsigned num_or_idx_lists, const unsigned num_or_sources,
    const combo_index_t* __restrict__ or_variations,
    const unsigned num_or_variations, result_t* __restrict__ or_results) {
  const auto block_size = blockDim.x;
  unsigned* or_chunk_idx_ptr = &dynamic_shared[kOrChunkIdx];
  __shared__ bool any_or_compat;
  if (!threadIdx.x) any_or_compat = false;
  __syncthreads();
  for (; or_chunk_idx * block_size < num_or_sources; ++or_chunk_idx) {
    if (get_next_OR_sources_chunk(source, or_chunk_idx, xor_combo_idx,
            xor_src_lists, xor_src_list_start_indices, xor_idx_lists,
            xor_idx_list_start_indices, xor_idx_list_sizes, num_xor_idx_lists,
            or_src_lists, or_src_list_start_indices, or_idx_lists,
            or_idx_list_start_indices, or_idx_list_sizes, num_or_idx_lists,
            num_or_sources, or_results)) {
      any_or_compat = true;
    }
    __syncthreads();
    if (any_or_compat) break;
  }
  if (!threadIdx.x) *or_chunk_idx_ptr = or_chunk_idx + 1;
  return any_or_compat;
}

__device__ auto next_xor_result_idx(
    unsigned result_idx, const result_t* __restrict__ xor_results) {
  const auto block_size = blockDim.x;
  while ((result_idx < block_size) && !xor_results[result_idx])
    result_idx++;
  return result_idx;
}

// for all XOR results
__device__ bool get_next_compatible_OR_sources(
    const SourceCompatibilityData& source, unsigned or_chunk_idx,
    unsigned xor_chunk_idx, const result_t* __restrict__ xor_results,
    const SourceCompatibilityData* __restrict__ xor_src_lists,
    const index_t* __restrict__ xor_src_list_start_indices,
    const index_t* __restrict__ xor_idx_lists,
    const index_t* __restrict__ xor_idx_list_start_indices,
    const index_t* __restrict__ xor_idx_list_sizes, unsigned num_xor_idx_lists,
    const ComboIndexSpanPair& idx_spans,
    const SourceCompatibilityData* __restrict__ or_src_lists,
    const index_t* __restrict__ or_src_list_start_indices,
    const index_t* __restrict__ or_idx_lists,
    const index_t* __restrict__ or_idx_list_start_indices,
    const index_t* __restrict__ or_idx_list_sizes,
    const unsigned num_or_idx_lists, const unsigned num_or_sources,
    const combo_index_t* __restrict__ or_variations,
    const unsigned num_or_variations, result_t* __restrict__ or_results) {
  const auto block_size = blockDim.x;
  unsigned* xor_results_idx_ptr = &dynamic_shared[kXorResultsIdx];
  unsigned* or_chunk_idx_ptr = &dynamic_shared[kOrChunkIdx];
  __shared__ bool any_or_compat;
  if (!threadIdx.x) {
    if (!or_chunk_idx) {
      *xor_results_idx_ptr = next_xor_result_idx(0, xor_results);
    }
    any_or_compat = false;
  }
  __syncthreads();
  auto xor_results_idx = *xor_results_idx_ptr;
  //
  // TODO: I think I could use a local here for results_idx and
  // update it at end of loop.
  //
  while (xor_results_idx < block_size) {
    const auto xor_flat_idx = get_flat_idx(xor_chunk_idx, xor_results_idx);
    const auto xor_combo_idx = get_xor_combo_index(xor_flat_idx, idx_spans);
    if (get_next_compatible_OR_sources(source, or_chunk_idx, xor_combo_idx,
            xor_src_lists, xor_src_list_start_indices, xor_idx_lists,
            xor_idx_list_start_indices, xor_idx_list_sizes, num_xor_idx_lists,
            or_src_lists, or_src_list_start_indices, or_idx_lists,
            or_idx_list_start_indices, or_idx_list_sizes, num_or_idx_lists,
            num_or_sources, or_variations, num_or_variations, or_results)) {
      any_or_compat = true;
    }
    __syncthreads();
    // sync failure here? using local for results_idx eliminates problem
    // (and a __syncthreads)
    if (any_or_compat) break; 
    //if (!threadIdx.x) {
      xor_results_idx = next_xor_result_idx(xor_results_idx + 1, xor_results);
      //}
      //__syncthreads();
  }
  if (!threadIdx.x) {
    // If we incremented xor_results_idx beyond the end of the xor_results
    // array at the end of the loop above, manually zero the or_chunk_idx.
    //if (*xor_results_idx >= block_size) *or_chunk_idx_ptr = 0;
    // TODO:
    *xor_results_idx_ptr = (xor_results_idx < block_size) ? xor_results_idx : 0;
  }
  return any_or_compat;
}

// Test if a source is XOR compatible with the XOR source specified by the
// supplied combo index and index lists.
__device__ bool is_source_XOR_compatible(const SourceCompatibilityData& source,
    combo_index_t combo_idx,
    const SourceCompatibilityData* __restrict__ xor_src_lists,
    const index_t* __restrict__ xor_src_list_start_indices,
    const index_t* __restrict__ xor_idx_lists,
    const index_t* __restrict__ xor_idx_list_start_indices,
    const index_t* __restrict__ xor_idx_list_sizes,
    unsigned num_xor_idx_lists) {

  for (unsigned list_idx{}; list_idx < num_xor_idx_lists; ++list_idx) {
    const auto xor_src_list =
        &xor_src_lists[xor_src_list_start_indices[list_idx]];
    const auto idx_list = &xor_idx_lists[xor_idx_list_start_indices[list_idx]];
    const auto idx_list_size = xor_idx_list_sizes[list_idx];
    const auto xor_src_idx = idx_list[combo_idx % idx_list_size];
    if (!source.isXorCompatibleWith(xor_src_list[xor_src_idx])) return false;
    combo_idx /= idx_list_size;
  }
  return true;
}

// Get the next block-sized chunk of XOR sources and test them for 
// compatibility with the supplied source.
// Return true if at least one XOR source is compatible.
__device__ bool get_next_XOR_sources_chunk(
    const SourceCompatibilityData& source, const unsigned xor_chunk_idx,
    const SourceCompatibilityData* __restrict__ xor_src_lists,
    const index_t* __restrict__ xor_src_list_start_indices,
    const index_t* __restrict__ xor_idx_lists,
    const index_t* __restrict__ xor_idx_list_start_indices,
    const index_t* __restrict__ xor_idx_list_sizes,
    const unsigned num_xor_idx_lists, const ComboIndexSpanPair& idx_spans,
    result_t* __restrict__ xor_results) {
  const auto num_xor_indices = idx_spans.first.size() + idx_spans.second.size();
  __shared__ bool is_xor_compat;
  if (!threadIdx.x) {
    is_xor_compat = false;
    xor_results[threadIdx.x] = 0;
  }
  __syncthreads();
  //
  // TODO: i have a sneaking suspicious i could get rid of the shared/syncs
  // in here similar to get_next_OR_sources_chunk
  //
  const auto xor_flat_idx = get_flat_idx(xor_chunk_idx);
  if (xor_flat_idx < num_xor_indices) {
    const auto xor_combo_idx = get_xor_combo_index(xor_flat_idx, idx_spans);
    if (is_source_XOR_compatible(source, xor_combo_idx, xor_src_lists,
            xor_src_list_start_indices, xor_idx_lists,
            xor_idx_list_start_indices, xor_idx_list_sizes,
            num_xor_idx_lists)) {
      is_xor_compat = true;
      xor_results[threadIdx.x] = 1;
    }
  }
  __syncthreads();
  return is_xor_compat;
}

// Loop through block-sized chunks of XOR sources until we find one that 
// contains at leasst one XOR source that is compatibile with the supplied
// source, or until all XOR sources are exhausted.
// Return true if at least one XOR source is compatible.
__device__ bool get_next_compatible_XOR_sources(
    const SourceCompatibilityData& source, unsigned xor_chunk_idx,
    const SourceCompatibilityData* __restrict__ xor_src_lists,
    const index_t* __restrict__ xor_src_list_start_indices,
    const index_t* __restrict__ xor_idx_lists,
    const index_t* __restrict__ xor_idx_list_start_indices,
    const index_t* __restrict__ xor_idx_list_sizes,
    const unsigned num_xor_idx_lists, const ComboIndexSpanPair& idx_spans,
    result_t* __restrict__ xor_results) {
  const auto block_size = blockDim.x;
  const auto num_indices = idx_spans.first.size() + idx_spans.second.size();
  unsigned* xor_chunk_idx_ptr = &dynamic_shared[kXorChunkIdx];
  __shared__ bool any_xor_compat;
  if (!threadIdx.x) any_xor_compat = false;
  __syncthreads();
  for (; xor_chunk_idx * block_size < num_indices; ++xor_chunk_idx) {
    if (get_next_XOR_sources_chunk(source, xor_chunk_idx, xor_src_lists,
            xor_src_list_start_indices, xor_idx_lists,
            xor_idx_list_start_indices, xor_idx_list_sizes, num_xor_idx_lists,
            idx_spans, xor_results)) {
      any_xor_compat = true;
    }
    __syncthreads();
    if (any_xor_compat) break;
  }
  if (!threadIdx.x) *xor_chunk_idx_ptr = xor_chunk_idx + 1;
  return any_xor_compat;
}

// Test if the supplied source is:
// * XOR-compatible with any of the supplied XOR sources
// * OR-compatible with any of the supplied OR sources which are
//   variation-compatible with the compatible XOR source.
//
// In other words:
// For each XOR source that is XOR-compatible with Source
//   For each OR source that is variation-compatible with XOR source
//     If OR source is OR-compatible with Source
//       is_compat = true;
__device__ bool is_compat_loop(const SourceCompatibilityData& source,
    const SourceCompatibilityData* __restrict__ xor_src_lists,
    const index_t* __restrict__ xor_src_list_start_indices,
    const index_t* __restrict__ xor_idx_lists,
    const index_t* __restrict__ xor_idx_list_start_indices,
    const index_t* __restrict__ xor_idx_list_sizes,
    const unsigned num_xor_idx_lists, const ComboIndexSpanPair& idx_spans,
    const SourceCompatibilityData* __restrict__ or_src_lists,
    const index_t* __restrict__ or_src_list_start_indices,
    const index_t* __restrict__ or_idx_lists,
    const index_t* __restrict__ or_idx_list_start_indices,
    const index_t* __restrict__ or_idx_list_sizes,
    const unsigned num_or_idx_lists, const unsigned num_or_sources,
    const combo_index_t* __restrict__ or_variations,
    const unsigned num_or_variations) {
  const auto block_size = blockDim.x;
  unsigned* xor_chunk_idx_ptr = &dynamic_shared[kXorChunkIdx];
  unsigned* or_chunk_idx_ptr = &dynamic_shared[kOrChunkIdx];
  result_t* xor_results = (result_t*)&dynamic_shared[kNumSharedIndices];
  result_t* or_results = (result_t*)&xor_results[block_size];
  __shared__ bool any_xor_compat;
  __shared__ bool any_or_compat;
  const unsigned num_xor_indices =
      idx_spans.first.size() + idx_spans.second.size();
  if (!threadIdx.x) {
    *xor_chunk_idx_ptr = 0;
    *or_chunk_idx_ptr = 0;
    // TODO: might do better with an enum here, fewer back-to-back __syncthreads
    // calls potentially
    any_xor_compat = false;
    any_or_compat = false;
  }
  __syncthreads();
  for (;;) {
    if (!any_xor_compat) {
      if (*xor_chunk_idx_ptr * block_size >= num_xor_indices) break;
      if (get_next_compatible_XOR_sources(source, *xor_chunk_idx_ptr,
              xor_src_lists, xor_src_list_start_indices, xor_idx_lists,
              xor_idx_list_start_indices, xor_idx_list_sizes, num_xor_idx_lists,
              idx_spans, xor_results)) {
        any_xor_compat = true;
      }
    }
    __syncthreads();
    if (any_xor_compat && !num_or_sources) return true;
#if 1
    if (any_xor_compat && !any_or_compat) {
      if (*or_chunk_idx_ptr * block_size >= num_or_sources) break;
      if (get_next_compatible_OR_sources(source, *or_chunk_idx_ptr,
              *xor_chunk_idx_ptr, xor_results, xor_src_lists,
              xor_src_list_start_indices, xor_idx_lists,
              xor_idx_list_start_indices, xor_idx_list_sizes, num_xor_idx_lists,
              idx_spans, or_src_lists, or_src_list_start_indices, or_idx_lists,
              or_idx_list_start_indices, or_idx_list_sizes, num_or_idx_lists,
              num_or_sources, or_variations, num_or_variations, or_results)) {
        any_or_compat = true;
      }
    }
    __syncthreads();
    if (any_or_compat) return true;
#endif
  }
  return false;
}

// explain better:
// Find sources that are:
// * XOR compatible with any of the supplied XOR sources, and
// * OR compatible with any of the supplied OR sources, which must in turn be
// * variation-compatible with the XOR source.
//
// Find compatible XOR source -> compare with variation-compatible OR sources.
__global__ void filter_kernel(
    const SourceCompatibilityData* __restrict__ src_list,
    const unsigned num_sources,
    const SourceCompatibilityData* __restrict__ xor_src_lists,
    const index_t* __restrict__ xor_src_list_start_indices,
    const index_t* __restrict__ xor_idx_lists,
    const index_t* __restrict__ xor_idx_list_start_indices,
    const index_t* __restrict__ xor_idx_list_sizes,
    const unsigned num_xor_idx_lists,
    const device::VariationIndices* __restrict__ xor_variation_indices,
    const SourceIndex* __restrict__ src_indices,
    const index_t* __restrict__ src_list_start_indices,
    const compat_src_result_t* __restrict__ compat_src_results,
    const SourceCompatibilityData* __restrict__ or_src_lists,
    const index_t* __restrict__ or_src_list_start_indices,
    const index_t* __restrict__ or_idx_lists,
    const index_t* __restrict__ or_idx_list_start_indices,
    const index_t* __restrict__ or_idx_list_sizes,
    const unsigned num_or_idx_lists, const unsigned num_or_sources,
    const combo_index_t* __restrict__ or_variations,
    const unsigned num_or_variations, result_t* __restrict__ results,
    int stream_idx) {
  const auto block_size = blockDim.x;
  // TODO __global__ compat int, cuda_memset to zero before launching kernel
  //result_t* compat_results = (result_t*)&dynamic_shared[kNumSharedIndices];
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
    auto smallest = get_smallest_src_index_spans(source, xor_variation_indices);
    if (smallest.skip) continue;
    if (is_compat_loop(source, xor_src_lists, xor_src_list_start_indices,
            xor_idx_lists, xor_idx_list_start_indices, xor_idx_list_sizes,
            num_xor_idx_lists, smallest.spans, or_src_lists,
            or_src_list_start_indices, or_idx_lists, or_idx_list_start_indices,
            or_idx_list_sizes, num_or_idx_lists, num_or_sources, or_variations,
            num_or_variations)) {
      is_compat = true;
    }
    __syncthreads();
    if (is_compat && !threadIdx.x) {
      results[src_idx.listIndex] = 1;
      is_compat = false;
    }
  }
}

}  // anonymous namespace

void run_filter_kernel(int threads_per_block, StreamData& stream,
    const MergeFilterData& mfd, const SourceCompatibilityData* device_src_list,
    const compat_src_result_t* device_compat_src_results,
    result_t* device_results, const index_t* device_list_start_indices) {
  int num_sm;
  cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0);
  int threads_per_sm;;
  cudaDeviceGetAttribute(&threads_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor, 0);

  stream.is_running = true;
  stream.sequence_num = StreamData::next_sequence_num();

  auto block_size = threads_per_block ? threads_per_block : 768;
  auto blocks_per_sm = threads_per_sm / block_size;
  assert(blocks_per_sm * block_size == threads_per_sm);
  auto grid_size = num_sm * blocks_per_sm;  // aka blocks per grid
  // xor_chunk_idx, or_chunk_idx, xor_result_idx, xor_results, or_results
  // results could probably be moved to global
  auto shared_bytes = 
      kNumSharedIndices * sizeof(unsigned) + block_size * 2 * sizeof(result_t);
  // ensure any async alloc/copies are complete on main thread stream
  cudaError_t err = cudaStreamSynchronize(cudaStreamPerThread);
  assert_cuda_success(err, "run_filter_kernel sync");
  dim3 grid_dim(grid_size);
  dim3 block_dim(block_size);
  auto num_or_sources = util::sum_sizes(mfd.host_or.compat_idx_lists);
  //std::cerr << "num or sources: " << num_or_sources << std::endl;
  stream.xor_kernel_start.record();
  filter_kernel<<<grid_dim, block_dim, shared_bytes, stream.cuda_stream>>>(
      // Sources
      device_src_list, stream.src_indices.size(),
      // XOR sources
      mfd.device_xor.src_lists, mfd.device_xor.src_list_start_indices,
      mfd.device_xor.idx_lists, mfd.device_xor.idx_list_start_indices,
      mfd.device_xor.idx_list_sizes, mfd.host_xor.compat_idx_lists.size(),
      mfd.device_xor.variation_indices,
      // Sources again
      stream.device_src_indices, device_list_start_indices,
      // XOR sources again i think, for count > 2
      device_compat_src_results,
      // OR sources
      mfd.device_or.src_lists, mfd.device_or.src_list_start_indices,
      mfd.device_or.idx_lists, mfd.device_or.idx_list_start_indices,
      mfd.device_or.idx_list_sizes, mfd.host_or.compat_idx_lists.size(),
      num_or_sources, mfd.device_or.combo_indices,
      mfd.host_or.combo_indices.size(), device_results, stream.stream_idx);
  stream.xor_kernel_stop.record();
  if constexpr (0) {
    std::cerr << "stream " << stream.stream_idx << " XOR kernel started with " << grid_size
              << " blocks of " << block_size << " threads" << std::endl;
  }
}

void run_get_compatible_sources_kernel(
    const SourceCompatibilityData* device_src_list, unsigned num_sources,
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
  // async copies are on thread stream therefore auto synchronized
  cudaStream_t stream = cudaStreamPerThread;
  get_compatible_sources_kernel<<<grid_dim, block_dim, shared_bytes, stream>>>(
      device_src_list, num_sources, device_src_desc_pairs, num_src_desc_pairs,
      device_results);
}

}  // namespace cm
