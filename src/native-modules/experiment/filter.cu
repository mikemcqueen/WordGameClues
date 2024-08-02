// filter.cu

#include <algorithm>
#include <iostream>
#include <limits>
#include <cuda_runtime.h>
#include "filter.cuh"
#include "stream-data.h"
#include "merge-filter-data.h"
#include "util.h"

namespace cm {

namespace {

/*
template <typename T> __device__ __forceinline__ T load(const T* addr) {
  return *(const volatile T*)addr;
}

template <typename T> __device__ __forceinline__ void store(T* addr, T val) {
  *(volatile T*)addr = val;
}
*/

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

extern __shared__ uint64_t dynamic_shared[];

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

/*
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
*/

const int kXorChunkIdx = 0;
const int kOrChunkIdx = 1;
const int kXorResultsIdx = 2;
const int kNumSharedIndices = 3;

namespace tag {
struct XOR {};  // XOR;
struct OR {};   // OR;
};

template <typename T>
requires std::is_same_v<T, tag::XOR> || std::is_same_v<T, tag::OR>
__device__ bool is_source_compatible(T tag,
    const SourceCompatibilityData& source, combo_index_t flat_idx,
    const SourceCompatibilityData* __restrict__ src_lists,
    const index_t* __restrict__ src_list_start_indices,
    const index_t* __restrict__ idx_lists,
    const index_t* __restrict__ idx_list_start_indices,
    const index_t* __restrict__ idx_list_sizes, unsigned num_idx_lists) {
  for (int list_idx{int(num_idx_lists) - 1}; list_idx >= 0; --list_idx) {
    const auto src_list = &src_lists[src_list_start_indices[list_idx]];
    const auto idx_list = &idx_lists[idx_list_start_indices[list_idx]];
    const auto idx_list_size = idx_list_sizes[list_idx];
    const auto src_idx = idx_list[flat_idx % idx_list_size];
    const auto& src = src_list[src_idx];
    if constexpr (std::is_same_v<T, tag::XOR>) {
      if (!src.isXorCompatibleWith(source)) return false;
    } else if constexpr (std::is_same_v<T, tag::OR>) {
      if (!src.isOrCompatibleWith(source)) return false;
    }
    flat_idx /= idx_list_size;
  }
  return true;
}

__device__ void dump_variation(
    const UsedSources::Variations& v, const char* p = "") {
  printf("%d,%d,%d,%d,%d,%d,%d,%d,%d %s\n", (int)v[0], (int)v[1], (int)v[2], (int)v[3],
         (int)v[4], (int)v[5], (int)v[6], (int)v[7], (int)v[8], p);
}

__device__ void dump_all_sources(uint64_t flat_idx,
    const SourceCompatibilityData* __restrict__ src_lists,
    const index_t* __restrict__ src_list_start_indices,
    const index_t* __restrict__ idx_lists,
    const index_t* __restrict__ idx_list_start_indices,
    const index_t* __restrict__ idx_list_sizes, unsigned num_idx_lists) {
  UsedSources::Variations v = {-1, -1, -1, -1, -1, -1, -1, -1, -1};
  printf("all sources:\n");
  auto orig_flat_idx = flat_idx;
  for (int list_idx{int(num_idx_lists) - 1}; list_idx >= 0; --list_idx) {
    const auto src_list = &src_lists[src_list_start_indices[list_idx]];
    const auto idx_list = &idx_lists[idx_list_start_indices[list_idx]];
    const auto idx_list_size = idx_list_sizes[list_idx];
    const auto idx = flat_idx % idx_list_size;
    const auto src_idx = idx_list[idx];
    const auto& src = src_list[src_idx];
    flat_idx /= idx_list_size;
    auto b = UsedSources::merge_variations(v, src.usedSources.variations);
    src.dump("merging");
    printf("orig_flat_idx: %lu, list_idx: %d, list_size: %u"
           ", remain_flat_idx: %lu, idx: %d, src_idx: %u into:\n",
        orig_flat_idx, list_idx, idx_list_size, flat_idx, int(idx), src_idx);
    dump_variation(v, b ? "- success" : "- failed");
  }
}
__device__ unsigned variation_merge_failure = 0;

__device__ auto build_variations(uint64_t flat_idx,
    const SourceCompatibilityData* __restrict__ src_lists,
    const index_t* __restrict__ src_list_start_indices,
    const index_t* __restrict__ idx_lists,
    const index_t* __restrict__ idx_list_start_indices,
    const index_t* __restrict__ idx_list_sizes, unsigned num_idx_lists,
    bool xor_flag, bool& fail) {
  auto orig_flat_idx = flat_idx;
  //  printf("block: %d, flat: %lu, xor: %d\n", blockIdx.x, flat_idx, xor_flag ? 1 : 0);
  UsedSources::Variations v = {-1, -1, -1, -1, -1, -1, -1, -1, -1};
  for (int list_idx{int(num_idx_lists) - 1}; list_idx >= 0; --list_idx) {
    const auto src_list = &src_lists[src_list_start_indices[list_idx]];
    const auto idx_list = &idx_lists[idx_list_start_indices[list_idx]];
    const auto idx_list_size = idx_list_sizes[list_idx];
    const auto idx = flat_idx % idx_list_size;
    const auto src_idx = idx_list[idx];
    const auto& src = src_list[src_idx];
    flat_idx /= idx_list_size;
    auto b = UsedSources::merge_variations(v, src.usedSources.variations);
    if (!b) {
      fail = true;
      break;
    }
#if 0
    if (!b && !atomicCAS(&variation_merge_failure, 0, 1)) {
      fail = true;
      src.dump("failed merging");
      printf("block %d, tid %d, orig_flat_idx: %lu, list_idx: %d, list_size: %u"
             ", remain_flat_idx: %lu, idx: %d, src_idx: %u into:\n",
          blockIdx.x, threadIdx.x, orig_flat_idx, list_idx, idx_list_size,
          flat_idx, int(idx), src_idx);
      dump_variation(v);
      dump_all_sources(orig_flat_idx, src_lists, src_list_start_indices,
          idx_lists, idx_list_start_indices, idx_list_sizes, num_idx_lists);
    }
#endif
    //assert(b);
  }
  return v;
}

#define PRECOMPUTE_XOR_VARIATIONS 0

__device__ auto are_variations_compatible(
    const UsedSources::Variations& xor_variations_param,
    const combo_index_t xor_combo_idx,
    const SourceCompatibilityData* __restrict__ xor_src_lists,
    const index_t* __restrict__ xor_src_list_start_indices,
    const index_t* __restrict__ xor_idx_lists,
    const index_t* __restrict__ xor_idx_list_start_indices,
    const index_t* __restrict__ xor_idx_list_sizes, unsigned num_xor_idx_lists,
    const uint64_t or_flat_idx,
    const SourceCompatibilityData* __restrict__ or_src_lists,
    const index_t* __restrict__ or_src_list_start_indices,
    const index_t* __restrict__ or_idx_lists,
    const index_t* __restrict__ or_idx_list_start_indices,
    const index_t* __restrict__ or_idx_list_sizes,
    const unsigned num_or_idx_lists) {
  bool fail{};
#if PRECOMPUTE_XOR_VARIATIONS
  const auto& xor_variations = xor_variations_param;
#else
  auto xor_variations = build_variations(xor_combo_idx, xor_src_lists,
      xor_src_list_start_indices, xor_idx_lists, xor_idx_list_start_indices,
      xor_idx_list_sizes, num_xor_idx_lists, true, fail);
  if (fail) printf("XOR combo_idx failed: %lu\n", xor_combo_idx);
  fail = false;
#endif
  auto or_variations = build_variations(or_flat_idx, or_src_lists,
      or_src_list_start_indices, or_idx_lists, or_idx_list_start_indices,
      or_idx_list_sizes, num_or_idx_lists, false, fail);
  if (fail) return false;
  return UsedSources::are_variations_compatible(xor_variations, or_variations);
}

// Get the next block-sized chunk of OR sources and test them for variation-
// compatibililty with the XOR source specified by the supplied xor_combo_index,
// and for OR-compatibility with the supplied source.
// Return true if at least one OR source is compatible.
__device__ bool get_next_OR_sources_chunk(const SourceCompatibilityData& source,
    unsigned or_chunk_idx, const UsedSources::Variations& xor_variations,
    const combo_index_t xor_combo_idx,
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
    result_t* __restrict__ or_results) {
  or_results[threadIdx.x] = 0;
  const auto or_flat_idx = get_flat_idx(or_chunk_idx);
  if (or_flat_idx < num_or_sources) {
    //    const auto or_combo_idx =
    //        get_combo_index(or_flat_idx, or_idx_list_sizes, num_or_idx_lists);
#define VARIATIONS 0
#if VARIATIONS
    if (are_variations_compatible(xor_variations, xor_combo_idx, xor_src_lists,
            xor_src_list_start_indices, xor_idx_lists,
            xor_idx_list_start_indices, xor_idx_list_sizes, num_xor_idx_lists,
            or_flat_idx, or_src_lists, or_src_list_start_indices, or_idx_lists,
            or_idx_list_start_indices, or_idx_list_sizes, num_or_idx_lists)) {
#endif
      if (is_source_compatible(tag::OR{}, source, or_flat_idx, or_src_lists,
              or_src_list_start_indices, or_idx_lists,
              or_idx_list_start_indices, or_idx_list_sizes, num_or_idx_lists)) {
        or_results[threadIdx.x] = 1;
        return true;
      }
#if VARIATONS
    }
#endif
  }
  return false;
}

// For one XOR source
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
    result_t* __restrict__ or_results) {
  const auto block_size = blockDim.x;
  uint64_t* or_chunk_idx_ptr = &dynamic_shared[kOrChunkIdx];
  __shared__ bool any_or_compat;
  __shared__ UsedSources::Variations xor_variations;
  if (!threadIdx.x) {
    any_or_compat = false;
#if PRECOMPUTE_XOR_VARIATIONS
    // TODO: or:
    // if (threadIdx.x < xor_variations.size())
    //   v[threadIdx.x] = get_variation(xor...);
    // but even more importantly, it seems we are doing this for the same xor_idx
    // for **multiple** *blocks*?
    bool fail{};
    auto v = build_variations(xor_combo_idx, xor_src_lists,
        xor_src_list_start_indices, xor_idx_lists, xor_idx_list_start_indices,
        xor_idx_list_sizes, num_xor_idx_lists, true, fail);
    assert (!fail);
    std::copy(v.begin(), v.end(), xor_variations.begin());
#endif
  }
  __syncthreads();
  for (; or_chunk_idx * block_size < num_or_sources; ++or_chunk_idx) {
    if (get_next_OR_sources_chunk(source, or_chunk_idx, xor_variations, xor_combo_idx,
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

// For all XOR results
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
    result_t* __restrict__ or_results) {
  const auto block_size = blockDim.x;
  uint64_t* xor_results_idx_ptr = &dynamic_shared[kXorResultsIdx];
  uint64_t* or_chunk_idx_ptr = &dynamic_shared[kOrChunkIdx];
  __shared__ bool any_or_compat;
  if (!threadIdx.x) {
    if (!or_chunk_idx) {
      // First or_chunk resets xor_results_idx to the first valid results index.
      *xor_results_idx_ptr = next_xor_result_idx(0, xor_results);
    }
    any_or_compat = false;
  }
  __syncthreads();
  auto xor_results_idx = *xor_results_idx_ptr;
  while (xor_results_idx < block_size) {
    const auto xor_flat_idx = get_flat_idx(xor_chunk_idx, xor_results_idx);
    const auto xor_combo_idx = get_xor_combo_index(xor_flat_idx, idx_spans);
    if (get_next_compatible_OR_sources(source, or_chunk_idx, xor_combo_idx,
            xor_src_lists, xor_src_list_start_indices, xor_idx_lists,
            xor_idx_list_start_indices, xor_idx_list_sizes, num_xor_idx_lists,
            or_src_lists, or_src_list_start_indices, or_idx_lists,
            or_idx_list_start_indices, or_idx_list_sizes, num_or_idx_lists,
            num_or_sources, or_results)) {
      any_or_compat = true;
    }
    __syncthreads();
    if (any_or_compat) break;
    xor_results_idx = next_xor_result_idx(xor_results_idx + 1, xor_results);
  }
  if (!threadIdx.x) {
    if (xor_results_idx < block_size) {
      *xor_results_idx_ptr = xor_results_idx;
    } else {
      // If all valid xor_results have evaluated by this block, reset the
      // xor_results_idx.
      *xor_results_idx_ptr = 0;
      //*or_chunk_idx_ptr = 0; // not actually sure about this.
    }
  }
  return any_or_compat;
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
  xor_results[threadIdx.x] = 0;
  const auto xor_flat_idx = get_flat_idx(xor_chunk_idx);
  if (xor_flat_idx < num_xor_indices) {
    const auto xor_combo_idx = get_xor_combo_index(xor_flat_idx, idx_spans);
    if (is_source_compatible(tag::XOR{}, source, xor_combo_idx, xor_src_lists,
            xor_src_list_start_indices, xor_idx_lists,
            xor_idx_list_start_indices, xor_idx_list_sizes,
            num_xor_idx_lists)) {
      xor_results[threadIdx.x] = 1;
      return true;
    }
  }
  return false;
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
  uint64_t* xor_chunk_idx_ptr = &dynamic_shared[kXorChunkIdx];
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
    const unsigned num_or_idx_lists, const unsigned num_or_sources) {
  const auto block_size = blockDim.x;
  uint64_t* xor_chunk_idx_ptr = &dynamic_shared[kXorChunkIdx];
  uint64_t* or_chunk_idx_ptr = &dynamic_shared[kOrChunkIdx];
  result_t* xor_results = (result_t*)&dynamic_shared[kNumSharedIndices];
  result_t* or_results = (result_t*)&xor_results[block_size];
  __shared__ bool any_xor_compat;
  __shared__ bool any_or_compat;
  const unsigned num_xor_indices =
      idx_spans.first.size() + idx_spans.second.size();
  if (!threadIdx.x) {
    *xor_chunk_idx_ptr = 0;
    *or_chunk_idx_ptr = 0;
    any_xor_compat = false;
    any_or_compat = false;
  }
  __syncthreads();
  for (;;) {
    if (!any_xor_compat) {
      if (*xor_chunk_idx_ptr * block_size >= num_xor_indices) break; // WHAT
      if (get_next_compatible_XOR_sources(source, *xor_chunk_idx_ptr,
              xor_src_lists, xor_src_list_start_indices, xor_idx_lists,
              xor_idx_list_start_indices, xor_idx_list_sizes, num_xor_idx_lists,
              idx_spans, xor_results)) {
        any_xor_compat = true;
      }
    }
    __syncthreads();
    if (any_xor_compat && !num_or_sources) return true;
    if (any_xor_compat && !any_or_compat) {
      if (*or_chunk_idx_ptr * block_size >= num_or_sources) break; // WHAT
      if (get_next_compatible_OR_sources(source, *or_chunk_idx_ptr,
              *xor_chunk_idx_ptr, xor_results, xor_src_lists,
              xor_src_list_start_indices, xor_idx_lists,
              xor_idx_list_start_indices, xor_idx_list_sizes, num_xor_idx_lists,
              idx_spans, or_src_lists, or_src_list_start_indices, or_idx_lists,
              or_idx_list_start_indices, or_idx_list_sizes, num_or_idx_lists,
              num_or_sources, or_results)) {
        any_or_compat = true;
      }
    }
    // TODO: could move this to beginning of loop?
    __syncthreads();
    if (any_or_compat) return true;
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
    result_t* __restrict__ results, int stream_idx) {
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
            or_idx_list_sizes, num_or_idx_lists, num_or_sources)) {
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
  int threads_per_sm;
  cudaDeviceGetAttribute(
      &threads_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor, 0);

  stream.is_running = true;
  stream.sequence_num = StreamData::next_sequence_num();

  auto block_size = threads_per_block ? threads_per_block : 768;
  auto blocks_per_sm = threads_per_sm / block_size;
  assert(blocks_per_sm * block_size == threads_per_sm);
  auto grid_size = num_sm * blocks_per_sm;  // aka blocks per grid
  // xor_chunk_idx, or_chunk_idx, xor_result_idx, xor_results, or_results
  // results could probably be moved to global
  auto shared_bytes = 
      kNumSharedIndices * sizeof(uint64_t) + block_size * 2 * sizeof(result_t);
  // ensure any async alloc/copies are complete on main thread stream
  cudaError_t err = cudaStreamSynchronize(cudaStreamPerThread);
  assert_cuda_success(err, "run_filter_kernel sync");
  dim3 grid_dim(grid_size);
  dim3 block_dim(block_size);
  auto num_or_src_combos =
      util::multiply_sizes_with_overflow_check(mfd.host_or.compat_idx_lists);
  std::cerr << "num or source combinations: " << num_or_src_combos << std::endl;
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
      num_or_src_combos, device_results, stream.stream_idx);
  stream.xor_kernel_stop.record();
  if constexpr (0) {
    std::cerr << "stream " << stream.stream_idx << " XOR kernel started with "
              << grid_size << " blocks of " << block_size << " threads"
              << std::endl;
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
