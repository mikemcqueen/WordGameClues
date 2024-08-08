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

#if 1
#define RESTRICT __restrict__
#else
#define RESTRICT
#endif
  
const unsigned BIG = 2'100'000'000;
  //#define DEBUG_OR_COUNTS
//#define DEBUG_XOR_COUNTS
// #define DISABLE_OR
// #define FORCE_XOR_COMPAT
// #define FORCE_ALL_XOR
//#define MAX_SOURCES 2
// #define USE_LOCAL_XOR_COMPAT
//#define PRINTF

#if defined(DEBUG_XOR_COUNTS) || defined(DEBUG_OR_COUNTS)
__device__ unsigned count_xor_src_considered = 0;
__device__ unsigned count_xor_src_compat = 0;

__device__ unsigned count_or_src_considered = 0;
__device__ unsigned count_or_xor_compat = 0;
__device__ unsigned count_or_src_compat = 0;

__device__ unsigned count_xor_bits_compat;
__device__ unsigned count_xor_variations_compat;
#endif

void init_counts() {
  unsigned value = 0;
  cudaError_t err{};

#ifdef DEBUG_XOR_COUNTS
  err = cudaMemcpyToSymbol(count_xor_src_considered, &value, sizeof(unsigned));
  assert_cuda_success(err, "init count");

  err = cudaMemcpyToSymbol(count_xor_src_compat, &value, sizeof(unsigned));
  assert_cuda_success(err, "init count");
#endif


#ifdef USE_LOCAL_XOR_COMPAT
  err = cudaMemcpyToSymbol(count_xor_bits_compat, &value, sizeof(unsigned));
  assert_cuda_success(err, "init count");

  err = cudaMemcpyToSymbol(count_xor_variations_compat, &value,  //
      sizeof(unsigned));
  assert_cuda_success(err, "init count");
#endif


#ifdef DEBUG_OR_COUNTS
  err = cudaMemcpyToSymbol(count_or_src_considered, &value, sizeof(unsigned));
  assert_cuda_success(err, "init count");

  err = cudaMemcpyToSymbol(count_or_src_compat, &value, sizeof(unsigned));
  assert_cuda_success(err, "init count");

  err = cudaMemcpyToSymbol(count_or_xor_compat, &value, sizeof(unsigned));
  assert_cuda_success(err, "init count");
#endif
}

void display_counts() {
  cudaError_t err{};

#ifdef DEBUG_XOR_COUNTS
  unsigned considered_xor_count;
  unsigned compat_xor_count;

  err = cudaMemcpyFromSymbol(
      &considered_xor_count, count_xor_src_considered, sizeof(unsigned));
  assert_cuda_success(err, "display count");

  err = cudaMemcpyFromSymbol(
      &compat_xor_count, count_xor_src_compat, sizeof(unsigned));
  assert_cuda_success(err, "display count");
#endif

#ifdef USE_LOCAL_XOR_COMPAT
  unsigned compat_xor_bits_count;
  unsigned compat_xor_variations_count;

  err = cudaMemcpyFromSymbol(
      &compat_xor_bits_count, count_xor_bits_compat, sizeof(unsigned));
  assert_cuda_success(err, "display count");

  err = cudaMemcpyFromSymbol(
      &compat_xor_variations_count, count_xor_variations_compat, sizeof(unsigned));
  assert_cuda_success(err, "display count");
#endif

#ifdef DEBUG_OR_COUNTS
  unsigned considered_or_count;
  unsigned compat_or_xor_count;  // variations
  unsigned compat_or_count;

  err = cudaMemcpyFromSymbol(
      &considered_or_count, count_or_src_considered, sizeof(unsigned));
  assert_cuda_success(err, "display count");

  err = cudaMemcpyFromSymbol(
      &compat_or_xor_count, count_or_xor_compat, sizeof(unsigned));
  assert_cuda_success(err, "display count");

  err = cudaMemcpyFromSymbol(
      &compat_or_count, count_or_src_compat, sizeof(unsigned));
  assert_cuda_success(err, "display count");
#endif

  std::cerr
#ifdef DEBUG_XOR_COUNTS
      << " xor_considered: " << considered_xor_count
      << " xor_compat: " << compat_xor_count
#endif
#ifdef DEBUG_OR_COUNTS
      << " or_considered: " << considered_or_count
      << " or_xor_compat: " << compat_or_xor_count
      << " or_compat: " << compat_or_count
#endif
#ifdef USE_LOCAL_XOR_COMPAT
      << " xor_bits_compat: " << compat_xor_bits_count
      << " xor_variations_compat: " << compat_xor_variations_count
#endif
      << std::endl;
}

__constant__ FilterData::DeviceCommon<fat_index_t> xor_data;
__constant__ FilterData::DeviceOr or_data;

__device__ auto are_xor_compatible(const UsedSources& a, const UsedSources& b) {
  // compare bits
  if (a.getBits().intersects(b.getBits())) return false;
  //  atomicInc(&count_xor_bits_compat, BIG);
  // compare variations
  if (!UsedSources::allVariationsMatch(a.variations, b.variations))
    return false;
  // atomicInc(&count_xor_variations_compat, BIG);
  return true;
}

__device__ auto are_or_compatible(const UsedSources& a, const UsedSources& b) {
  return UsedSources::are_variations_compatible(a.variations, b.variations)
         && a.getBits().is_disjoint_or_subset(b.getBits());
  // a.getBits().intersects(b.getBits()) ||
  // a.getBits().is_subset_of(b.getBits()))
}

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
    const UsedSources::SourceDescriptorPair* RESTRICT src_desc_pairs,
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
    const SourceCompatibilityData* RESTRICT sources,
    const unsigned num_sources,
    const UsedSources::
        SourceDescriptorPair* RESTRICT incompat_src_desc_pairs,
    const unsigned num_src_desc_pairs, result_t* RESTRICT results) {
  // one block per source
  for (unsigned idx{blockIdx.x}; idx < num_sources; idx += gridDim.x) {
    const auto& source = sources[idx];
    // I don't understand why this is required, but fails synccheck and
    // produces wobbly results without.
    __syncthreads();
    if (!source_contains_any_descriptor_pair(
          source, incompat_src_desc_pairs, num_src_desc_pairs)) {
      /// TODO: this looks wrong. why thread 0 only? can i cg::block::any this?
      if (!threadIdx.x) {
        results[idx] = 1;
      }
    }
  }
}

extern __shared__ fat_index_t dynamic_shared[];

enum class Compat {
  All,
  None,
  Some
};

struct SmallestSpansResult {
  Compat compat;
  FatIndexSpanPair pair;
};

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
__device__ SmallestSpansResult get_smallest_src_index_spans(
    const SourceCompatibilityData& source) {
  using enum Compat;
  index_t fewest_indices{std::numeric_limits<index_t>::max()};
  int sentence_with_fewest{-1};
  for (int s{}; s < kNumSentences; ++s) {
    // if there are no xor_sources that contain a primary source from this
    // sentence, skip it. (it would result in num_indices == all_indices which
    // is the worst case).
    const auto& vi = xor_data.variation_indices[s];
    if (!vi.num_variations) continue;

    // if the candidate source has no primary source from this sentence, skip
    // it. (same reason as above).
    const auto variation = source.usedSources.variations[s] + 1;
    if (!variation) continue;

    // sum the xor_src indices that have no variation (index 0), with those
    // that have the same variation as the candidate source, for this sentence.
    // remember the sentence with the smallest sum.
    const auto num_indices = vi.num_indices_per_variation[0]
                             + vi.num_indices_per_variation[variation];
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
    return {All};
  }
  if (!fewest_indices) {
    // both the candidate source and all xor_sources contain a primary source
    // from sentence_with_fewest, but all xor_sources use a different variation
    // than the candidate source. we can skip all xor-compat checks since they
    // will all fail due to variation mismatch.
    return {None};
  }
  const auto variation = source.usedSources.variations[sentence_with_fewest];
  const auto& vi = xor_data.variation_indices[sentence_with_fewest];
  return {Some,  //
      std::make_pair(vi.get_index_span(0), vi.get_index_span(variation + 1))};
}

__device__ fat_index_t get_flat_idx(
    fat_index_t block_idx, unsigned thread_idx = threadIdx.x) {
  return block_idx * fat_index_t(blockDim.x) + thread_idx;
}

__device__ auto get_xor_combo_index(
    fat_index_t flat_idx, const FatIndexSpanPair& idx_spans) {
  if (flat_idx < idx_spans.first.size()) return idx_spans.first[flat_idx];
  flat_idx -= idx_spans.first.size();
  // TODO #ifdef ASSERTS
  assert(flat_idx < idx_spans.second.size());
  return idx_spans.second[flat_idx];
}

const int kXorChunkIdx = 0;
const int kOrChunkIdx = 1;
const int kXorResultsIdx = 2;
const int kNumSharedIndices = 3;

namespace tag {
struct XOR {};  // XOR;
struct OR {};   // OR;
};

template <typename IndexT, typename TagT>
requires std::is_same_v<TagT, tag::XOR> || std::is_same_v<TagT, tag::OR>
__device__ bool is_source_compatible(
    TagT tag, const SourceCompatibilityData& source, fat_index_t flat_idx) {
  FilterData::DeviceCommon<IndexT> const* data{};
  if constexpr (std::is_same_v<TagT, tag::XOR>) data = &xor_data;
  else if constexpr (std::is_same_v<TagT, tag::OR>) data = &or_data;

#ifdef DISABLE_OR
  if constexpr (std::is_same_v<TagT, tag::OR>) return false;
#endif
#ifdef FORCE_XOR_COMPAT
  else if constexpr (std::is_same_v<TagT, tag::XOR>)
    return true;
#endif

  for (int list_idx{int(data->num_idx_lists) - 1}; list_idx >= 0; --list_idx) {
    const auto& src = data->get_source(flat_idx, list_idx);
    if constexpr (std::is_same_v<TagT, tag::XOR>) {
#ifdef USE_LOCAL_XOR_COMPAT
      if (!are_xor_compatible(src.usedSources, source.usedSources)) return false;
#else
      if (!src.isXorCompatibleWith(source)) return false;
#endif
    } else if constexpr (std::is_same_v<TagT, tag::OR>) {
      //if (!src.isOrCompatibleWith(source)) return false;
      //src.usedSources.isOrCompatibleWith(source.usedSources)) return false;
      if (!are_or_compatible(src.usedSources, source.usedSources))
        return false;
    }
    flat_idx /= data->idx_list_sizes[list_idx];
  }
  return true;
}

__device__ void dump_variation(
    const UsedSources::Variations& v, const char* p = "") {
  printf("%d,%d,%d,%d,%d,%d,%d,%d,%d %s\n", (int)v[0], (int)v[1], (int)v[2], (int)v[3],
         (int)v[4], (int)v[5], (int)v[6], (int)v[7], (int)v[8], p);
}

__device__ void dump_all_sources(fat_index_t flat_idx,
    const SourceCompatibilityData* RESTRICT src_lists,
    const index_t* RESTRICT src_list_start_indices,
    const index_t* RESTRICT idx_lists,
    const index_t* RESTRICT idx_list_start_indices,
    const index_t* RESTRICT idx_list_sizes, unsigned num_idx_lists) {
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

// only used for XOR variations currently
__device__ variation_index_t get_one_variation(
    int sentence, fat_index_t flat_idx) {
  // const FilterData::DeviceXor::Base* RESTRICT data) {
  for (int list_idx{int(xor_data.num_idx_lists) - 1}; list_idx >= 0; --list_idx) {
    const auto& src = xor_data.get_source(flat_idx, list_idx);
    const auto variation = src.usedSources.getVariation(sentence);
    if (variation > -1) return variation;
    flat_idx /= xor_data.idx_list_sizes[list_idx];
  }
  return -1;
}

__device__ unsigned variation_merge_failure = 0;

// faster than UsedSources version
__device__ auto merge_variations(UsedSources::Variations& to,
    const UsedSources::Variations& from, bool force = false) {
  for (int s{0}; s < kNumSentences; ++s) {
    if (force || (to[s] == -1)) to[s] = from[s];
  }
  return true;
}

// only used for OR variations currently
__device__ auto build_variations(fat_index_t flat_idx) {
  UsedSources::Variations v;
  bool force = true;
  for (int list_idx{int(or_data.num_idx_lists) - 1}; list_idx >= 0; --list_idx) {
    const auto& src = or_data.get_source(flat_idx, list_idx);
    flat_idx /= or_data.idx_list_sizes[list_idx];
    // TODO: put UsedSources version of this in an #ifdef ASSERTS block
    merge_variations(v, src.usedSources.variations, force);
    force = false;
  }
  return v;
}

__device__ auto are_variations_compatible(
    const UsedSources::Variations& xor_variations,
    const fat_index_t or_flat_idx) {
  const auto or_variations = build_variations(or_flat_idx);
  return UsedSources::are_variations_compatible(xor_variations, or_variations);
}

// Get a block-sized chunk of OR sources and test them for variation-
// compatibililty with the XOR source specified by the supplied
// xor_combo_index, and for OR-compatibility with the supplied source.
// Return true if at least one OR source is compatible.
__device__ bool get_OR_sources_chunk(const SourceCompatibilityData& source,
    unsigned or_chunk_idx, const UsedSources::Variations& xor_variations) {
  // one thread per compat_idx
  const auto or_compat_idx = get_flat_idx(or_chunk_idx);
  if (or_compat_idx < or_data.num_compat_indices) {

    #ifdef DEBUG_OR_COUNTS
    atomicInc(&count_or_src_considered, BIG);
    #endif

    const auto or_flat_idx = or_data.compat_indices[or_compat_idx];
    if (are_variations_compatible(xor_variations, or_flat_idx)) {

      #ifdef DEBUG_OR_COUNTS
      atomicInc(&count_or_xor_compat, BIG);
      #endif

      if (is_source_compatible<index_t>(tag::OR{}, source, or_flat_idx)) {

        #ifdef DEBUG_OR_COUNTS
        atomicInc(&count_or_src_compat, BIG);
        #endif

        return true;
      }
    }
  }
  return false;
}

// For one XOR source.
// Walk through OR-sources one block-sized chunk at a time, until we find
// a chunk that contains at least one OR source that is variation-compatible
// with the XOR source indentified by the supplied xor_combo_idx, and
// OR-compatible with the supplied source.
__device__ bool get_next_compatible_OR_sources(
    const SourceCompatibilityData& source, fat_index_t xor_combo_idx) {
  const auto block_size = blockDim.x;
  fat_index_t* or_chunk_idx_ptr = &dynamic_shared[kOrChunkIdx];
  __shared__ bool any_or_compat;
  __shared__ UsedSources::Variations xor_variations;
  if (threadIdx.x < xor_variations.size()) {
    if (!threadIdx.x) any_or_compat = false;
    xor_variations[threadIdx.x] =
        get_one_variation(threadIdx.x + 1, xor_combo_idx);
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
__device__ auto next_xor_result_idx(unsigned result_idx) {
  result_t* xor_results = (result_t*)&dynamic_shared[kNumSharedIndices];
  const auto block_size = blockDim.x;
  while ((result_idx < block_size) && !xor_results[result_idx])
    result_idx++;
  return result_idx;
}

// For all XOR results
__device__ bool is_any_OR_source_compatible(
    const SourceCompatibilityData& source, unsigned xor_chunk_idx,
    const FatIndexSpanPair& xor_idx_spans) {
  result_t* xor_results = (result_t*)&dynamic_shared[kNumSharedIndices];
  __shared__ bool any_or_compat;
  if (!threadIdx.x) any_or_compat = false;
  const auto block_size = blockDim.x;
  const auto num_xor_indices =
      xor_idx_spans.first.size() + xor_idx_spans.second.size();
  auto max_results = num_xor_indices - blockDim.x * xor_chunk_idx;
  if (max_results > block_size) max_results = block_size;
  __syncthreads();
  for (unsigned xor_results_idx{}; xor_results_idx < max_results;) {
    const auto xor_flat_idx = get_flat_idx(xor_chunk_idx, xor_results_idx);
    const auto xor_combo_idx = get_xor_combo_index(xor_flat_idx, xor_idx_spans);
    if (get_next_compatible_OR_sources(source, xor_combo_idx)) {
      any_or_compat = true;
    }
    __syncthreads();

#ifdef PRINTF
    if (/*!blockIdx.x &&*/ !threadIdx.x) {
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

// Get the next block-sized chunk of XOR sources and test them for 
// compatibility with the supplied source.
// Return true if at least one XOR source is compatible.
__device__ bool get_next_XOR_sources_chunk(
    const SourceCompatibilityData& source, const unsigned xor_chunk_idx,
    const FatIndexSpanPair& xor_idx_spans) {
  result_t* xor_results = (result_t*)&dynamic_shared[kNumSharedIndices];
  const auto block_size = blockDim.x;
  const auto num_xor_indices =
      xor_idx_spans.first.size() + xor_idx_spans.second.size();
  xor_results[threadIdx.x] = 0;
  const auto xor_flat_idx = get_flat_idx(xor_chunk_idx);
  if (xor_flat_idx < num_xor_indices) {

    #ifdef DEBUG_XOR_COUNTS
    atomicInc(&count_xor_src_considered, BIG);
    #endif

    const auto xor_combo_idx = get_xor_combo_index(xor_flat_idx, xor_idx_spans);
    if (is_source_compatible<fat_index_t>(tag::XOR{}, source, xor_combo_idx)) {

#if defined(DEBUG_XOR_COUNTS)
      atomicInc(&count_xor_src_compat, BIG);
      #endif

      xor_results[threadIdx.x] = 1;
      return true;
    }
  }
  return false;
}

// Loop through block-sized chunks of XOR sources until we find one that 
// contains at least one XOR source that is XOR-compatibile with the supplied
// source, or until all XOR sources are exhausted.
// Return true if at least one XOR source is compatible.
__device__ bool get_next_compatible_XOR_sources(
    const SourceCompatibilityData& source, unsigned xor_chunk_idx,
    const FatIndexSpanPair& xor_idx_spans) {
  result_t* xor_results = (result_t*)&dynamic_shared[kNumSharedIndices];
  const auto block_size = blockDim.x;
  const auto num_xor_indices = xor_idx_spans.first.size() + xor_idx_spans.second.size();
  fat_index_t* xor_chunk_idx_ptr = &dynamic_shared[kXorChunkIdx];
  __shared__ bool any_xor_compat;
  if (!threadIdx.x) any_xor_compat = false;
  __syncthreads();
  for (; xor_chunk_idx * block_size < num_xor_indices; ++xor_chunk_idx) {
    if (get_next_XOR_sources_chunk(source, xor_chunk_idx, xor_idx_spans)) {
#ifndef FORCE_ALL_XOR
      any_xor_compat = true;
      #endif
    }
    __syncthreads();

    #ifdef PRINTF
    if (/*!blockIdx.x && */!threadIdx.x) {
      printf(" block: %u get_next_XOR xor_chunk_idx: %u, compat: %d\n", blockIdx.x,
          xor_chunk_idx, any_xor_compat ? 1 : 0);
    }
    #endif

    if (any_xor_compat) break;
  }
  if (!threadIdx.x) *xor_chunk_idx_ptr = xor_chunk_idx;
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
    const FatIndexSpanPair& xor_idx_spans) {
  const auto block_size = blockDim.x;
  fat_index_t* xor_chunk_idx_ptr = &dynamic_shared[kXorChunkIdx];
  result_t* xor_results = (result_t*)&dynamic_shared[kNumSharedIndices];
  __shared__ bool any_xor_compat;
  __shared__ bool any_or_compat;
  const auto num_xor_indices =
      xor_idx_spans.first.size() + xor_idx_spans.second.size();
  if (!threadIdx.x) {
    *xor_chunk_idx_ptr = 0;
    any_xor_compat = false;
    any_or_compat = false;
  }
  for (;;) {
    __syncthreads();
    if (*xor_chunk_idx_ptr * block_size >= num_xor_indices) return false;
    if (get_next_compatible_XOR_sources(source, *xor_chunk_idx_ptr,  //
            xor_idx_spans)) {
      any_xor_compat = true;
    }
    __syncthreads();

    if (any_xor_compat) {
      if (!or_data.num_compat_indices) return true;
#if 0
      if (is_any_OR_source_compatible(source, *xor_chunk_idx_ptr,  //
              xor_idx_spans)) {
        any_or_compat = true;
      }
    }
    __syncthreads();
    if (any_or_compat) return true;
#else
    }
#endif

    if (!threadIdx.x) {
      any_xor_compat = false;
      (*xor_chunk_idx_ptr)++;
    }
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
__global__ void filter_kernel(const SourceCompatibilityData* RESTRICT src_list,
    unsigned num_sources, const SourceIndex* RESTRICT src_indices,
    const index_t* RESTRICT src_list_start_indices,
    const result_t* RESTRICT compat_src_results, result_t* RESTRICT results,
    int stream_idx) {
  const auto block_size = blockDim.x;
  __shared__ bool is_compat;
  if (!threadIdx.x) is_compat = false;

  #if 1 || defined(PRINTF)
  if (!blockIdx.x && !threadIdx.x) {
    printf("+++kernel+++ blocks: %u\n", gridDim.x);
    if (or_data.variation_indices) {
      printf(" num_or_indices: %u\n", or_data.num_compat_indices);
    }
  }
  #endif

  // for each source (one block per source)
  #if MAX_SOURCES
  num_sources = num_sources < MAX_SOURCES ? num_sources : MAX_SOURCES;
  #endif
  for (unsigned idx{blockIdx.x}; idx < num_sources; idx += gridDim.x) {
    __syncthreads();
    const auto src_idx = src_indices[idx];
    const auto flat_idx =
        src_list_start_indices[src_idx.listIndex] + src_idx.index;
    // TODO: understand if this is right (bounds)
    if (compat_src_results && !compat_src_results[flat_idx]) continue;
    const auto& source = src_list[flat_idx];
    auto spans_result = get_smallest_src_index_spans(source);

    #ifdef PRINTF
    if (!threadIdx.x && (idx == num_sources - 1)) {
      auto num_xor_indices =
          spans_result.pair.first.size() + spans_result.pair.second.size();
      printf(" block: %u, spans.compat = %d, num_xor_indices: %lu, chunks: %u\n",
          blockIdx.x, int(spans_result.compat), num_xor_indices,
          unsigned(num_xor_indices / blockDim.x));
    }
    #endif

    if (spans_result.compat == Compat::None) continue;
    if ((spans_result.compat == Compat::All) && !or_data.num_compat_indices) {
      results[src_idx.listIndex] = 1;
      continue;
    }
    if (is_compat_loop(source, spans_result.pair)) {
      is_compat = true;
    }
    __syncthreads();
    if (is_compat && !threadIdx.x) {

      #ifdef PRINTF
      printf(" block %u, compat list_index: %u\n", blockIdx.x,  //
          src_idx.listIndex);
      #endif

      results[src_idx.listIndex] = 1;
      is_compat = false;
    }
  }
}

}  // anonymous namespace

void run_filter_kernel(int threads_per_block, StreamData& stream,
    const FilterData& mfd, const SourceCompatibilityData* device_src_list,
    const result_t* device_compat_src_results, result_t* device_results,
    const index_t* device_list_start_indices) {
  int num_sm;
  cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0);
  int threads_per_sm;
  cudaDeviceGetAttribute(
      &threads_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor, 0);

  stream.is_running = true;
  stream.sequence_num = StreamData::next_sequence_num();

  auto block_size = threads_per_block ? threads_per_block : 768;
  auto blocks_per_sm = threads_per_sm / block_size;
  // assert(blocks_per_sm * block_size == threads_per_sm); // allow 1024 tpb
  auto grid_size = num_sm * blocks_per_sm;  // aka blocks per grid
  // xor_chunk_idx, or_chunk_idx, xor_result_idx, xor_results
  // results could probably be moved to global
  auto shared_bytes = 
      kNumSharedIndices * sizeof(fat_index_t) + block_size * sizeof(result_t);
  // ensure any async alloc/copies are complete on main thread stream
  #if defined(DEBUG_XOR_COUNTS) || defined(DEBUG_OR_COUNTS)
  init_counts();
  #endif 
  cudaError_t err = cudaStreamSynchronize(cudaStreamPerThread);
  assert_cuda_success(err, "run_filter_kernel sync");
  dim3 grid_dim(grid_size);
  dim3 block_dim(block_size);
  auto num_or_src_combos =
      util::multiply_sizes_with_overflow_check(mfd.host_or.compat_idx_lists);
  if (log_level(Verbose)) std::cerr << ".";
  // "num or source combinations: " << num_or_src_combos << // std::endl;
  stream.xor_kernel_start.record();
  filter_kernel<<<grid_dim, block_dim, shared_bytes, stream.cuda_stream>>>(
      device_src_list, stream.src_indices.size(), stream.device_src_indices,
      device_list_start_indices, device_compat_src_results, device_results,
      stream.stream_idx);
  assert_cuda_success(cudaPeekAtLastError(), "filter kernel launch failed");
  stream.xor_kernel_stop.record();
  #if defined(DEBUG_XOR_COUNTS) || defined(DEBUG_OR_COUNTS)
  display_counts();
  #endif 
  if constexpr (0) {
    std::cerr << "stream " << stream.stream_idx << " XOR kernel started with "
              << grid_size << " blocks of " << block_size << " threads"
              << std::endl;
  }
}

void copy_filter_data(FilterData& mfd) {
  cudaError_t err{};

  const auto xor_bytes = sizeof(FilterData::DeviceXor::Base);
  err = cudaMemcpyToSymbol(xor_data, &mfd.device_xor, xor_bytes);
  assert_cuda_success(err, "copy xor filter data");

  const auto or_bytes = sizeof(FilterData::DeviceOr);
  err = cudaMemcpyToSymbol(or_data, &mfd.device_or, or_bytes);
  assert_cuda_success(err, "copy or filter data");
  std::cerr << "OR.num_variation_indices: "
            << mfd.device_or.num_variation_indices << std::endl;
}

void run_get_compatible_sources_kernel(
    const SourceCompatibilityData* device_src_list, unsigned num_sources,
    const UsedSources::SourceDescriptorPair* device_src_desc_pairs,
    unsigned num_src_desc_pairs, result_t* device_results) {
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
