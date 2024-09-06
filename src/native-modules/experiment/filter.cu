// filter.cu

#include <algorithm>
#include <iostream>
#include <limits>
#include <cuda_runtime.h>
#include "filter.cuh"
#include "or-filter.cuh"
#include "stream-data.h"
#include "merge-filter-data.h"
#include "util.h"

namespace cm {

__constant__ FilterData::DeviceXor xor_data;
__constant__ FilterData::DeviceOr or_data;
__constant__ FilterData::DeviceSources sources_data;

#if defined(DEBUG_XOR_COUNTS) || defined(DEBUG_OR_COUNTS)
__device__ atomic64_t count_xor_src_considered = 0;
__device__ atomic64_t count_xor_src_variation_compat = 0;
__device__ atomic64_t count_xor_check_src_compat = 0;
__device__ atomic64_t count_xor_src_compat = 0;

__device__ atomic64_t get_next_xor_compat_clocks = 0;
__device__ atomic64_t init_src_clocks = 0;
__device__ atomic64_t xor_compute_compat_uv_indices_clocks = 0;

__device__ atomic64_t is_any_or_compat_clocks = 0;
__device__ atomic64_t or_compute_compat_uv_indices_clocks = 0;
__device__ atomic64_t or_get_compat_idx_clocks = 0;
__device__ atomic64_t or_build_variation_clocks = 0;
__device__ atomic64_t or_are_variations_compat_clocks = 0;
__device__ atomic64_t or_check_src_compat_clocks = 0;
__device__ atomic64_t or_is_src_compat_clocks = 0;
__device__ atomic64_t or_incompat_xor_clocks = 0;

__device__ atomic64_t count_or_get_compat_idx = 0;
__device__ atomic64_t count_or_src_considered = 0;
__device__ atomic64_t count_or_src_variation_compat = 0;
__device__ atomic64_t count_or_check_src_compat = 0;
__device__ atomic64_t count_or_src_compat = 0;
__device__ atomic64_t count_or_xor_chunks = 0;
__device__ atomic64_t count_or_incompat_xor_chunks = 0;
__device__ atomic64_t count_or_incompat_xor_chunk_sources = 0;
#endif

namespace {

#if 1
#define RESTRICT __restrict__
#else
#define RESTRICT
#endif
  
void init_counts() {
  atomic64_t value = 0;
  cudaError_t err{};

#ifdef CLOCKS
  err = cudaMemcpyToSymbol(
      get_next_xor_compat_clocks, &value, sizeof(atomic64_t));
  assert_cuda_success(err, "init count");

  err = cudaMemcpyToSymbol(init_src_clocks, &value, sizeof(atomic64_t));
  assert_cuda_success(err, "init count");

  err = cudaMemcpyToSymbol(is_any_or_compat_clocks, &value, sizeof(atomic64_t));
  assert_cuda_success(err, "init count");

  err = cudaMemcpyToSymbol(
      or_compute_compat_uv_indices_clocks, &value, sizeof(atomic64_t));
  assert_cuda_success(err, "init count");

  err = cudaMemcpyToSymbol(or_get_compat_idx_clocks, &value, sizeof(atomic64_t));
  assert_cuda_success(err, "init count");

  err =
      cudaMemcpyToSymbol(or_build_variation_clocks, &value, sizeof(atomic64_t));
  assert_cuda_success(err, "init count");

  err = cudaMemcpyToSymbol(
      or_are_variations_compat_clocks, &value, sizeof(atomic64_t));
  assert_cuda_success(err, "init count");

  err = cudaMemcpyToSymbol(
      or_check_src_compat_clocks, &value, sizeof(atomic64_t));
  assert_cuda_success(err, "init count");

  err = cudaMemcpyToSymbol(or_is_src_compat_clocks, &value, sizeof(atomic64_t));
  assert_cuda_success(err, "init count");

  err = cudaMemcpyToSymbol(or_incompat_xor_clocks, &value, sizeof(atomic64_t));
  assert_cuda_success(err, "init count");
#endif

#ifdef DEBUG_XOR_COUNTS
  err = cudaMemcpyToSymbol(count_xor_src_considered, &value, sizeof(atomic64_t));
  assert_cuda_success(err, "init count");

  err = cudaMemcpyToSymbol(count_xor_src_variation_compat, &value,  //
      sizeof(atomic64_t));
  assert_cuda_success(err, "init count");

  err = cudaMemcpyToSymbol(count_xor_check_src_compat, &value, sizeof(atomic64_t));
  assert_cuda_success(err, "init count");

  err = cudaMemcpyToSymbol(count_xor_src_compat, &value, sizeof(atomic64_t));
  assert_cuda_success(err, "init count");
#endif

#ifdef DEBUG_OR_COUNTS
  err = cudaMemcpyToSymbol(count_or_get_compat_idx, &value, sizeof(atomic64_t));
  assert_cuda_success(err, "init count");

  err = cudaMemcpyToSymbol(count_or_src_considered, &value, sizeof(atomic64_t));
  assert_cuda_success(err, "init count");

  err = cudaMemcpyToSymbol(
      count_or_src_variation_compat, &value, sizeof(atomic64_t));
  assert_cuda_success(err, "init count");

  err = cudaMemcpyToSymbol(count_or_check_src_compat, &value, sizeof(atomic64_t));
  assert_cuda_success(err, "init count");

  err = cudaMemcpyToSymbol(count_or_src_compat, &value, sizeof(atomic64_t));
  assert_cuda_success(err, "init count");

  err = cudaMemcpyToSymbol(count_or_xor_chunks, &value,
      sizeof(atomic64_t));
  assert_cuda_success(err, "init count");

  err = cudaMemcpyToSymbol(count_or_incompat_xor_chunks, &value,
      sizeof(atomic64_t));
  assert_cuda_success(err, "init count");

  err = cudaMemcpyToSymbol(count_or_incompat_xor_chunk_sources, &value,
      sizeof(atomic64_t));
  assert_cuda_success(err, "init count");
#endif
}

void display_counts() {
  cudaError_t err{};

#ifdef CLOCKS
  atomic64_t l_get_next_xor_compat_clocks;
  atomic64_t l_init_src_clocks;
  atomic64_t l_is_any_or_compat_clocks;   
  atomic64_t l_or_compute_uv_indices_clocks;
  atomic64_t l_or_get_compat_idx_clocks;
  atomic64_t l_or_build_variation_clocks;
  atomic64_t l_or_are_variations_compat_clocks;
  atomic64_t l_or_check_src_compat_clocks;
  atomic64_t l_or_is_src_compat_clocks;
  atomic64_t l_or_incompat_xor_clocks;

  err = cudaMemcpyFromSymbol(&l_get_next_xor_compat_clocks,
      get_next_xor_compat_clocks, sizeof(atomic64_t));
  assert_cuda_success(err, "display count");

  err = cudaMemcpyFromSymbol(&l_init_src_clocks,
      init_src_clocks, sizeof(atomic64_t));
  assert_cuda_success(err, "display count");

  err = cudaMemcpyFromSymbol(&l_is_any_or_compat_clocks,
      is_any_or_compat_clocks, sizeof(atomic64_t));
  assert_cuda_success(err, "display count");

  err = cudaMemcpyFromSymbol(&l_or_compute_uv_indices_clocks,
      or_compute_compat_uv_indices_clocks, sizeof(atomic64_t));
  assert_cuda_success(err, "display count");

  err = cudaMemcpyFromSymbol(
      &l_or_get_compat_idx_clocks, or_get_compat_idx_clocks, sizeof(atomic64_t));
  assert_cuda_success(err, "display count");

  err = cudaMemcpyFromSymbol(
      &l_or_build_variation_clocks, or_build_variation_clocks, sizeof(atomic64_t));
  assert_cuda_success(err, "display count");

  err = cudaMemcpyFromSymbol(&l_or_are_variations_compat_clocks,
      or_are_variations_compat_clocks, sizeof(atomic64_t));
  assert_cuda_success(err, "display count");

  err = cudaMemcpyFromSymbol(&l_or_check_src_compat_clocks,
      or_check_src_compat_clocks, sizeof(atomic64_t));
  assert_cuda_success(err, "display count");

  err = cudaMemcpyFromSymbol(&l_or_is_src_compat_clocks,
      or_is_src_compat_clocks, sizeof(atomic64_t));
  assert_cuda_success(err, "display count");

  err = cudaMemcpyFromSymbol(&l_or_incompat_xor_clocks, or_incompat_xor_clocks,
      sizeof(atomic64_t));
  assert_cuda_success(err, "display count");
#endif
  
#ifdef DEBUG_XOR_COUNTS
  atomic64_t considered_xor_count;
  atomic64_t compat_xor_src_variation_count;
  atomic64_t compat_xor_check_src_count;
  atomic64_t compat_xor_count;

  err = cudaMemcpyFromSymbol(&considered_xor_count, count_xor_src_considered,
      sizeof(atomic64_t));
  assert_cuda_success(err, "display count");

  err = cudaMemcpyFromSymbol(&compat_xor_src_variation_count,
      count_xor_src_variation_compat, sizeof(atomic64_t));
  assert_cuda_success(err, "display count");

  err = cudaMemcpyFromSymbol(&compat_xor_check_src_count,
      count_xor_check_src_compat, sizeof(atomic64_t));
  assert_cuda_success(err, "display count");

  err = cudaMemcpyFromSymbol(&compat_xor_count, count_xor_src_compat,
      sizeof(atomic64_t));
  assert_cuda_success(err, "display count");
#endif

#ifdef DEBUG_OR_COUNTS
  atomic64_t get_or_compat_idx_count;
  atomic64_t considered_or_count;
  atomic64_t compat_or_src_variation_count;
  atomic64_t compat_or_check_src_count;
  atomic64_t compat_or_count;
  atomic64_t num_or_xor_chunks;
  atomic64_t incompat_xor_chunks;
  atomic64_t incompat_xor_chunk_sources;

  err = cudaMemcpyFromSymbol(&get_or_compat_idx_count, count_or_get_compat_idx,
      sizeof(atomic64_t));
  assert_cuda_success(err, "display count");

  err = cudaMemcpyFromSymbol(&considered_or_count, count_or_src_considered,
      sizeof(atomic64_t));
  assert_cuda_success(err, "display count");

  err = cudaMemcpyFromSymbol(&compat_or_src_variation_count,
      count_or_src_variation_compat, sizeof(atomic64_t));
  assert_cuda_success(err, "display count");

  err = cudaMemcpyFromSymbol(&compat_or_check_src_count,
      count_or_check_src_compat, sizeof(atomic64_t));
  assert_cuda_success(err, "display count");

  err = cudaMemcpyFromSymbol(&compat_or_count, count_or_src_compat,
      sizeof(atomic64_t));
  assert_cuda_success(err, "display count");

  err = cudaMemcpyFromSymbol(&num_or_xor_chunks, count_or_xor_chunks,
      sizeof(atomic64_t));
  assert_cuda_success(err, "display count");

  err = cudaMemcpyFromSymbol(&incompat_xor_chunks, count_or_incompat_xor_chunks,
      sizeof(atomic64_t));
  assert_cuda_success(err, "display count");

  err = cudaMemcpyFromSymbol(&incompat_xor_chunk_sources, count_or_incompat_xor_chunk_sources,
      sizeof(atomic64_t));
  assert_cuda_success(err, "display count");
#endif

  std::cerr
#ifdef CLOCKS
      << " is_compat_loop: " << std::endl
      << "   get_next_xor_compat_clocks: " << l_get_next_xor_compat_clocks
      << std::endl
      << "   init_src_clocks: " << l_init_src_clocks  //
      << std::endl
      << "   is_any_or_compat_clocks: " << l_is_any_or_compat_clocks  //
      << std::endl
      << " or_compute_uv_indices clocks: " << l_or_compute_uv_indices_clocks
      << std::endl
#endif

#if defined(DEBUG_OR_COUNTS) || defined(CLOCKS)
      << " get_or_compat_idx"
#ifdef DEBUG_OR_COUNTS
      << " count: " << get_or_compat_idx_count
#endif
#ifdef CLOCKS
      << " clocks: " << l_or_get_compat_idx_clocks
#endif
      << std::endl
#endif

#ifdef DEBUG_XOR_COUNTS
      << " xor_considered: " << considered_xor_count
      << " xor_compat: " << compat_xor_count << std::endl
#endif

#ifdef DEBUG_OR_COUNTS
      << " or_considered: " << considered_or_count << std::endl
#endif
#ifdef CLOCKS
      << " build_variation clocks: " << l_or_build_variation_clocks  //
      << std::endl
#endif

#if defined(DEBUG_OR_COUNTS) || defined(CLOCKS)
      << " or_src_var_compat"
#ifdef DEBUG_OR_COUNTS
      << " count: " << compat_or_src_variation_count
#endif
#ifdef CLOCKS
      << " clocks: " << l_or_are_variations_compat_clocks
#endif
      << std::endl
#endif


#if defined(DEBUG_OR_COUNTS) || defined(CLOCKS)
      << " or_check_src_compat"
#ifdef DEBUG_OR_COUNTS
      << " count: " << compat_or_check_src_count
#endif
#ifdef CLOCKS
      << " clocks: " << l_or_check_src_compat_clocks
#endif
      << std::endl
#endif


#if defined(DEBUG_OR_COUNTS) || defined(CLOCKS)
      << " or_is_compat"
#ifdef DEBUG_OR_COUNTS
      << " count: " << compat_or_count
#endif
#ifdef CLOCKS
      << " clocks: " << l_or_is_src_compat_clocks << std::endl
#endif
#endif
#if defined(DEBUG_OR_COUNTS)
      << " incompat xor chunks " << incompat_xor_chunks << " of "
      << num_or_xor_chunks << ", sources: " << incompat_xor_chunk_sources
      << " clocks: " << l_or_incompat_xor_clocks
#endif
      << std::endl;
}

/*
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

enum class Compat {
  All,
  None,
  Some
};

struct SmallestSpansResult {
  Compat compat;
  IndexSpanPair pair;
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
    // if there are no xor sources that contain a primary source from this
    // sentence, skip it. (it would result in num_indices == all_indices which
    // is the worst case).
    const auto& xor_vi = xor_data.variation_indices[s];
    if (!xor_vi.num_variations) continue;

    // if the supplied source has no primary source from this sentence, skip
    // it. (same reason as above).
    const auto src_variation = source.usedSources.variations[s] + 1;
    if (!src_variation) continue;

    // sum the xor source indices that have no variation (index 0), with those
    // that have the same variation as the supplied source, for this sentence.
    // remember the sentence with the smallest sum.

    // it's possible the variation of the supplied source is greater than the
    // number of variations for all xor sources in this sentence.
    index_t num_xor_indices_with_src_variation{};
    if (src_variation < xor_vi.num_variations) {
      num_xor_indices_with_src_variation =
          xor_vi.num_indices_per_variation[src_variation];
    }
    const auto num_xor_indices = xor_vi.num_indices_per_variation[0]
                                 + num_xor_indices_with_src_variation;
    if (num_xor_indices < fewest_indices) {
      fewest_indices = num_xor_indices;
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
  const auto src_variation =
      source.usedSources.variations[sentence_with_fewest] + 1;
  const auto& xor_vi = xor_data.variation_indices[sentence_with_fewest];
  // check again if the source variation exceeds all xor source variations
  IndexSpan src_variation_span{};
  if (src_variation < xor_vi.num_variations) {
    src_variation_span = xor_vi.get_index_span(src_variation);
  }
  return {Some, std::make_pair(xor_vi.get_index_span(0), src_variation_span)};
}

__device__ SourceIndex get_source_index(index_t idx,
    const MergeData::Device& data) {
  for (unsigned list_idx{}; list_idx < data.num_idx_lists; ++list_idx) {
    auto list_size = data.idx_list_sizes[list_idx];
    if (idx < list_size) return {list_idx, idx};
    idx -= list_size;
  }
  assert(0);
  return {};
}

__device__ auto source_bits_are_XOR_compatible(
    const UsedSources::SourceBits& a, const UsedSources::SourceBits& b) {
  using SourceBits = UsedSources::SourceBits;
  for (int i{}; i < SourceBits::wc() / 2; ++i) {
    const auto w = a.long_word(i) & b.long_word(i);
    if (w) return false;
  }
  for (int i{}; i < SourceBits::wc() - (SourceBits::wc() / 2) * 2; ++i) {
    const auto w = a.word(i) & b.word(i);
    if (w) return false;
  }
  return true;
}

// #define USE_LOCAL_XOR_COMPAT

__device__ bool is_source_XOR_compatible(const SourceCompatibilityData& source,
    fat_index_t combo_idx) {
  for (int list_idx{int(xor_data.num_idx_lists) - 1}; list_idx >= 0;
      --list_idx) {
    const auto& src = xor_data.get_source(combo_idx, list_idx);
#ifdef USE_LOCAL_XOR_COMPAT
    if (!source_bits_are_XOR_compatible(source.usedSources.bits,
            src.usedSources.bits)) {
      return false;
    }
#else
    if (!src.isXorCompatibleWith(source)) return false;
#endif
    combo_idx /= xor_data.idx_list_sizes[list_idx];
  }
  return true;
}

// This tests SourceBits-compatibility with the supplied source and each
// OR-source in each OR-arg source list, and stores the results in the
// or_data.compat_src_results flag array.
//
// Return true if the source is Bits-compatible with at least one OR-source
// from each OR-arg source list.
//
// If so, that flag array is used by check_src_compat_results() to reject
// incompatible OR-source combo-indices when generating OR-source chunks.
//
// It's currently not used for testing compatibility with XOR sources, and
// the XOR code in is_source_compatible_with() should be reviewed when it is.
//
template <typename TagT, typename T>
requires /*std::is_same_v<TagT, tag::XOR> ||*/ std::is_same_v<TagT, tag::OR>
__device__ auto compute_src_compat_results(
    const SourceCompatibilityData& source, T& data) {
  __shared__ result_t any_compat[kMaxOrArgs];
  if (threadIdx.x < kMaxOrArgs) any_compat[threadIdx.x] = false;
  __syncthreads();
  for (auto idx{threadIdx.x}; idx < data.sum_idx_list_sizes;
      idx += blockDim.x) {
    const auto src_idx = get_source_index(idx, data);
    const auto src_list_idx = data.src_list_start_indices[src_idx.listIndex];
    const auto src_list = &data.src_lists[src_list_idx];
    const auto idx_list_idx = data.idx_list_start_indices[src_idx.listIndex];
    const auto idx_list = &data.idx_lists[idx_list_idx];
    const auto& src = src_list[idx_list[src_idx.index]];
#if 0
    // TODO: overloads. Variations as struct/class. SourceBits also.
    auto compat = UsedSources::are_variations_compatible(
        src.usedSources.variations, source.usedSources.variations);
    if (compat) {
      if constexpr (std::is_same_v<TagT, tag::OR>) {
        compat = source_bits_are_OR_compatible(
            src.usedSources.getBits(), source.usedSources.getBits());
      } else if constexpr (std::is_same_v<TagT, tag::XOR>) {
        compat = source_bits_are_XOR_compatible(
            src.usedSources.getBits(), source.usedSources.getBits());
      }
    }
#else
    const auto compat = is_source_compatible_with<TagT>(source, src);
#endif
    const auto result_idx = blockIdx.x * data.sum_idx_list_sizes + idx;
    if (compat) {
      any_compat[src_idx.listIndex] = 1;
      data.src_compat_results[result_idx] = 1;
    } else {
      data.src_compat_results[result_idx] = 0;
    }
  }
  __syncthreads();
  if (!threadIdx.x) {
    // determine if source is compatible with one source from each src_list
    int num_compat_lists{};
    for (unsigned idx{}; idx < data.num_idx_lists; ++idx) {
      if (any_compat[idx]) ++num_compat_lists;
    }
    if (num_compat_lists == data.num_idx_lists) {
      // hmm tricky
      //#if defined(DEBUG_XOR_COUNTS) || defined(DEBUG_OR_COUNTS)
      //atomicAdd(&count_or_compat_results, 1);
      //#endif
      return true;
    }
  }
  return false;
}

__device__ auto init_source(const SourceCompatibilityData& source) {
  uint8_t* num_src_sentences = (uint8_t*)&dynamic_shared[kNumSrcSentences];
  uint8_t* src_sentences = &num_src_sentences[1];
  if (!threadIdx.x) {
    // initialize num_src_sentences and src_sentences for this source
    *num_src_sentences = 0;
    for (int s{}; s < kNumSentences; ++s) {
      if (source.usedSources.variations[s] > -1) {
        src_sentences[(*num_src_sentences)++] = (uint8_t)s;
      }
    }
  }
  return compute_src_compat_results<tag::OR>(source, or_data);
}

__device__ auto compute_compat_XOR_uv_indices(const Variations& src_variations) {
  const auto begin = clock64();
  index_t num_uv_indices{};
  if (compute_variations_compat_results(src_variations, xor_data,
          xor_data.variations_results_per_block)) {
    num_uv_indices = compute_compat_uv_indices(xor_data.num_unique_variations,
        xor_data.src_compat_uv_indices, xor_data.variations_results_per_block);
  }

#ifdef CLOCKS
  atomicAdd(&xor_compute_compat_uv_indices_clocks, clock64() - begin);
  #endif

  return num_uv_indices;
}

// incremental indexing of xor_data.src_compat_uv_indices
__device__ fat_index_t get_XOR_compat_idx_incremental_uv(
    const index_t chunk_idx, const index_t num_uv_indices) {
  assert(num_uv_indices > 0);
  auto desired_idx = chunk_idx * blockDim.x + threadIdx.x;
  const auto uvi_offset = blockIdx.x * xor_data.num_unique_variations;
  const UniqueVariations* uv{};
  auto uvi_idx = dynamic_shared[kXorStartUvIdx];
  auto start_idx = dynamic_shared[kXorStartSrcIdx];
  for (; uvi_idx < num_uv_indices; ++uvi_idx) {
    const auto xor_uv_idx = xor_data.src_compat_uv_indices[uvi_offset + uvi_idx];
    uv = &xor_data.unique_variations[xor_uv_idx];
    if (desired_idx < (uv->num_indices - start_idx)) break;
    desired_idx -= (uv->num_indices - start_idx);
    start_idx = 0;
  }
  __syncthreads();
  auto result = (uvi_idx < num_uv_indices)
      ? uv->first_compat_idx + start_idx + desired_idx
      : xor_data.num_compat_indices;
  // the last thread will have the highest uvi_idx/desired_idx combo
  if (threadIdx.x == blockDim.x - 1) {
    if (uvi_idx < num_uv_indices) {
      // set start_idx for next block's first thread to this block's last+1
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
    dynamic_shared[kXorStartUvIdx] = uvi_idx;
    dynamic_shared[kXorStartSrcIdx] = start_idx;
  }
  return result;
}

// Get the next block-sized chunk of XOR sources and test them for 
// compatibility with the supplied source.
// Return true if at least one XOR source is compatible.
__device__ bool get_next_XOR_sources_chunk(
    const SourceCompatibilityData& source, const index_t xor_chunk_idx,
    const IndexSpanPair& xor_idx_spans, const index_t num_uv_indices) {
  result_t* xor_results = (result_t*)&dynamic_shared[kXorResults];
  // a __sync will happen in calling function
  xor_results[threadIdx.x] = 0;
#ifdef XOR_SPANS
  const auto num_xor_indices = xor_idx_spans.first.size()
      + xor_idx_spans.second.size();
  // one thread per xor_flat_idx
  const auto xor_flat_idx = get_flat_idx(xor_chunk_idx);
  if (xor_flat_idx >= num_xor_indices) return false;

  #ifdef DEBUG_XOR_COUNTS
  atomicAdd(&count_xor_src_considered, 1);
  #endif

  const auto xor_combo_idx = get_xor_combo_index(xor_flat_idx, xor_idx_spans);
#else
  // one thread per compat_idx
  const auto xor_compat_idx = get_XOR_compat_idx_incremental_uv(xor_chunk_idx,
      num_uv_indices);
  if (xor_compat_idx >= xor_data.num_compat_indices) return false;

  #ifdef DEBUG_XOR_COUNTS
  atomicAdd(&count_xor_src_considered, 1);
  #endif

  const auto xor_combo_idx = xor_data.compat_indices[xor_compat_idx];
#endif

#if 0
  // When/if I get kNumSentences initialized early for source
  if (!are_source_bits_XOR_compatible(source, xor_combo_idx, xor_data))
    return false;
#else
  if (!is_source_XOR_compatible(source, xor_combo_idx))
    return false;
#endif

  #if defined(DEBUG_XOR_COUNTS)
  atomicAdd(&count_xor_src_compat, 1);
  #endif

  xor_results[threadIdx.x] = 1;
  return true;
}

// Loop through block-sized chunks of XOR sources until we find one that
// contains at least one XOR source that is XOR-compatibile with the supplied
// source, or until all XOR sources are exhausted.
// Return true if at least one XOR source is compatible.
__device__ bool get_next_compatible_XOR_sources_chunk(
    const SourceCompatibilityData& source, index_t xor_chunk_idx,
    const IndexSpanPair& xor_idx_spans, const index_t num_uv_indices) {
  result_t* xor_results = (result_t*)&dynamic_shared[kXorResults];
  const auto block_size = blockDim.x;
  const auto num_xor_indices =
#ifdef XOR_SPANS
      xor_idx_spans.first.size() + xor_idx_spans.second.size();
#else
      xor_data.num_compat_indices;
#endif
  __shared__ bool any_xor_compat;
  if (!threadIdx.x) any_xor_compat = false;
  __syncthreads();
  for (; xor_chunk_idx * block_size < num_xor_indices; ++xor_chunk_idx) {
#ifndef XOR_SPANS
    assert(xor_chunk_idx * block_size < num_xor_indices);
#endif
    if (get_next_XOR_sources_chunk(source, xor_chunk_idx, xor_idx_spans,
            num_uv_indices)) {
      any_xor_compat = true;
    }
    __syncthreads();
    if (any_xor_compat) break;
    if (dynamic_shared[kXorStartUvIdx] == num_uv_indices) break;
  }
  if (!threadIdx.x) dynamic_shared[kXorChunkIdx] = xor_chunk_idx;
  return any_xor_compat;
}

// Test if the supplied source is:
// * XOR-compatible with any of the supplied XOR sources
// * OR-compatible with any of the supplied OR sources, which are in turn
//   variation-compatible with the compatible XOR source.
//
// In other words:
// For each XOR source that is XOR-compatible with Source
//   For each OR source that is variation-compatible with XOR source
//     If OR source is OR-compatible with Source
//       is_compat = true
__device__ bool is_compat_loop(const SourceCompatibilityData& source,
    const IndexSpanPair& xor_idx_spans) {
  const auto block_size = blockDim.x;
  index_t* xor_chunk_idx_ptr = &dynamic_shared[kXorChunkIdx];
  __shared__ bool any_xor_compat;
  __shared__ bool any_or_compat;
  __shared__ bool src_init_done; 
  const auto num_xor_indices =
#ifdef XOR_SPANS
      xor_idx_spans.first.size() + xor_idx_spans.second.size();
#else
      xor_data.num_compat_indices;
#endif
  if (!threadIdx.x) {
    *xor_chunk_idx_ptr = 0;
    any_xor_compat = false;
    any_or_compat = false;
    src_init_done = false;
    dynamic_shared[kXorStartUvIdx] = 0;
    dynamic_shared[kXorStartSrcIdx] = 0;
  }
  auto num_uv_indices =
      compute_compat_XOR_uv_indices(source.usedSources.variations);
  if (!num_uv_indices) return false;
  for (;;) {
    __syncthreads();
    if (*xor_chunk_idx_ptr * block_size >= num_xor_indices) return false;

    auto begin = clock64();
    // TODO: passing shared variable not necessary
    if (get_next_compatible_XOR_sources_chunk(source, *xor_chunk_idx_ptr,
            xor_idx_spans, num_uv_indices)) {
      any_xor_compat = true;
    }
    __syncthreads();

    #ifdef CLOCKS
    atomicAdd(&get_next_xor_compat_clocks, clock64() - begin);
    #endif

    if (any_xor_compat) {
      if (!or_data.num_compat_indices) return true;
      begin = clock64();
      if (!src_init_done && init_source(source)) {
        src_init_done = true;
      }

      #ifdef CLOCKS
      atomicAdd(&init_src_clocks, clock64() - begin);
      #endif
    }
    // TODO: pretty sure this can be inside the above if block
    __syncthreads();

    if (any_xor_compat && src_init_done) {
      begin = clock64();
      if (is_any_OR_source_compatible(source, *xor_chunk_idx_ptr,
              xor_idx_spans)) {
        any_or_compat = true;
      }

      #ifdef CLOCKS
      atomicAdd(&is_any_or_compat_clocks, clock64() - begin);
      #endif
    }

    __syncthreads();
    if (any_or_compat) return true;
    if (dynamic_shared[kXorStartUvIdx] == num_uv_indices) break;

    if (!threadIdx.x) {
      any_xor_compat = false;
      (*xor_chunk_idx_ptr)++;
    }
  }
  return false;
}

__device__ void dump(const device::VariationIndices* dvi_array) {
  char buf[128];
  char smolbuf[16];
  uint64_t total_indices{};
  for (int s{}; s < kNumSentences; ++s) {
    const auto& dvi = dvi_array[s];
    cuda_strcpy(buf, "S");
    cuda_itoa(s + 1, smolbuf);
    cuda_strcat(buf, smolbuf);
    cuda_strcat(buf, ": variations(");
    cuda_itoa(dvi.num_variations, smolbuf);
    cuda_strcat(buf, smolbuf);
    cuda_strcat(buf, "), indices(");
    cuda_itoa(dvi.num_indices, smolbuf);
    cuda_strcat(buf, smolbuf);
    cuda_strcat(buf, ")\n");
    printf(buf);
    total_indices += dvi.num_indices;
    size_t sum{};
    for (size_t i{0}; i < dvi.num_variations; ++i) {
      cuda_strcpy(buf, "  v");
      cuda_itoa(int(i) - 1, smolbuf);
      cuda_strcat(buf, smolbuf);
      cuda_strcat(buf, ": indices(");
      cuda_itoa(dvi.num_indices_per_variation[i], smolbuf);
      cuda_strcat(buf, smolbuf);
      cuda_strcat(buf, "), offset: ");
      cuda_itoa(dvi.variation_offsets[i], smolbuf);
      cuda_strcat(buf, smolbuf);
      cuda_strcat(buf, "\n");
      printf(buf);
      sum += dvi.num_indices_per_variation[i];
    }
    cuda_strcpy(buf, "  sum(");
    cuda_itoa(sum, smolbuf);
    cuda_strcat(buf, smolbuf);
    cuda_strcat(buf, ")\n");
    printf(buf);
  }
}

__device__ void dump(const FilterData::DeviceXor& data) {
  dump(data.variation_indices);
}

// explain better:
// Find sources that are:
// * XOR compatible with any of the supplied XOR sources, and
// * OR compatible with any of the supplied OR sources, which must in turn be
// * variation-compatible with the XOR source.
//
// Find compatible XOR source -> compare with variation-compatible OR sources.
__global__ void filter_kernel(const SourceCompatibilityData* RESTRICT src_list,
    int num_sources, const SourceIndex* RESTRICT src_indices,
    const index_t* RESTRICT src_list_start_indices,
    const result_t* RESTRICT compat_src_results,
    const UniqueVariations* src_unique_variations,
    int num_src_unique_variations, result_t* RESTRICT results, int stream_idx) {
  const auto block_size = blockDim.x;
  __shared__ bool is_compat;
  if (!threadIdx.x) is_compat = false;
  //if (!threadIdx.x && !blockIdx.x) dump(xor_data);

  #if 1 || defined(PRINTF)
  if (!blockIdx.x && !threadIdx.x) {
    printf("+++kernel+++ blocks: %u\n", gridDim.x);
  }
  #endif

  // for each source (one block per source)
  #if MAX_SOURCES
  num_sources = num_sources < MAX_SOURCES ? num_sources : MAX_SOURCES;
  #endif
  for (index_t idx{blockIdx.x}; idx < num_sources; idx += gridDim.x) {
    __syncthreads();
    const auto src_idx = src_indices[idx];
    const auto flat_idx = src_idx.index;
    //        src_list_start_indices[src_idx.listIndex] + src_idx.index;
    if (compat_src_results && !compat_src_results[flat_idx]) continue;
    const auto& source = src_list[flat_idx];
    auto spans_result = get_smallest_src_index_spans(source);

    dynamic_shared[kSrcFlatIdx] = src_idx.listIndex;
#if defined(LOGGY)
    if (!threadIdx.x) {
      if (src_idx.listIndex == 4396 || src_idx.listIndex == 4397) {
        printf("begin: %u, flat_idx %u\n", src_idx.listIndex,
               flat_idx);
      }
    }
#endif

#ifdef XOR_SPANS
    if (spans_result.compat == Compat::None) continue;
    if ((spans_result.compat == Compat::All) && !or_data.num_compat_indices) {
      results[src_idx.listIndex] = 1;
      continue;
    }
#endif
    if (is_compat_loop(source, spans_result.pair)) {
      is_compat = true;
    }
    __syncthreads();

#if defined(LOGGY)
    if (!threadIdx.x) {
      if (src_idx.listIndex == 4396 || src_idx.listIndex == 4397) {
        printf(": %u, flat_idx %u, compat: %d\n", src_idx.listIndex,
            flat_idx, int(is_compat));
      }
    }
#endif

    if (is_compat && !threadIdx.x) {

      #ifdef PRINTF
      printf(" block %u, compat list_index: %u\n", blockIdx.x,
          src_idx.listIndex);
      #endif

      results[src_idx.listIndex] = 1;
      is_compat = false;
    }
  }
}

void copy_filter_data(const FilterData& mfd) {
  cudaError_t err{};

  const auto xor_bytes = sizeof(FilterData::DeviceXor);
  err = cudaMemcpyToSymbol(xor_data, &mfd.device_xor, xor_bytes);
  assert_cuda_success(err, "copy xor filter data");

  const auto or_bytes = sizeof(FilterData::DeviceOr);
  err = cudaMemcpyToSymbol(or_data, &mfd.device_or, or_bytes);
  assert_cuda_success(err, "copy or filter data");

  const auto sources_bytes = sizeof(FilterData::DeviceSources);
  err = cudaMemcpyToSymbol(sources_data, &mfd.device_sources, sources_bytes);
  assert_cuda_success(err, "copy sources filter data");
}

}  // anonymous namespace

void run_filter_kernel(int threads_per_block, StreamData& stream,
    FilterData& mfd, const SourceCompatibilityData* device_src_list,
    const result_t* device_compat_src_results,
    const UniqueVariations* device_unique_variations, int num_unique_variations,
    result_t* device_results, const index_t* device_list_start_indices) {
  // TODO: move to device_attr class
  int num_sm;
  cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0);
  int threads_per_sm;
  cudaDeviceGetAttribute(
      &threads_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor, 0);

  stream.is_running = true;
  stream.sequence_num = StreamData::next_sequence_num();

  const auto block_size = threads_per_block ? threads_per_block : 128;
  const auto blocks_per_sm = threads_per_sm / block_size;
  assert(blocks_per_sm * block_size == threads_per_sm);
  const auto grid_size = num_sm * blocks_per_sm;  // aka blocks per grid
  // results could probably be moved to global
  const auto shared_bytes =
      kSharedIndexCount * sizeof(shared_index_t)  // indices
      + kNumSentenceDataBytes * sizeof(uint8_t)   // src_sentence data
      + block_size * sizeof(result_t);            // xor_results
  //      + block_size * sizeof(index_t);        // OR index producer buffer
  const auto cuda_stream = cudaStreamPerThread;
  // TODO: move to helper function
  // v- Populate any remaining mfd.device_xx values BEFORE copy_filter_data() -v
  if (!mfd.device_xor.variations_compat_results) {
    const auto max_uv = std::max(mfd.device_or.num_unique_variations,
        mfd.device_xor.num_unique_variations);
    const auto xor_results_bytes = grid_size * max_uv * sizeof(result_t);
    cuda_malloc_async((void**)&mfd.device_xor.variations_compat_results,
        xor_results_bytes, cuda_stream, "xor.variations_compat_results");
    cuda_malloc_async((void**)&mfd.device_xor.variations_scan_results,
        xor_results_bytes, cuda_stream, "xor.variations_scan_results");
    mfd.device_xor.variations_results_per_block = max_uv;

    // NB: these have the potential to grow large as num_variations grow
    const auto or_indices_bytes =
        grid_size * mfd.device_or.num_unique_variations * sizeof(index_t);
    cuda_malloc_async((void**)&mfd.device_xor.or_compat_uv_indices,
        or_indices_bytes, cuda_stream, "xor.or_compat_uv_indices");
    const auto src_indices_bytes =
        grid_size * mfd.device_xor.num_unique_variations * sizeof(index_t);
    cuda_malloc_async((void**)&mfd.device_xor.src_compat_uv_indices,
        src_indices_bytes, cuda_stream, "xor.src_compat_uv_indices");
  }
  if (!mfd.device_or.src_compat_results) {
    const auto or_results_bytes =
        grid_size * mfd.device_or.sum_idx_list_sizes * sizeof(result_t);
    cuda_malloc_async((void**)&mfd.device_or.src_compat_results,
        or_results_bytes, cuda_stream, "or.src_compat_results");
  }
  // ^- Populate any reamining mfd.device_xx values BEFORE copy_filter_data() -^
  copy_filter_data(mfd);

  static bool log_once{true};
  if (log_once) {
    cuda_memory_dump("before filter kernel");
    log_once = false;
  }

  #if defined(DEBUG_XOR_COUNTS) || defined(DEBUG_OR_COUNTS) || defined(CLOCKS)
  init_counts();
  #endif 

  // ensure any async alloc/copies are complete on main thread stream
  cudaError_t err = cudaStreamSynchronize(cuda_stream);
  assert_cuda_success(err, "run_filter_kernel sync");
  const dim3 grid_dim(grid_size);
  const dim3 block_dim(block_size);

  #if 0
  const auto num_or_src_combos =
      util::multiply_sizes_with_overflow_check(mfd.host_or.compat_idx_lists);
  //if (log_level(Verbose)) std::cerr << ".";
  //"num or source combinations: " << num_or_src_combos << // std::endl;
  #endif

  stream.xor_kernel_start.record();
  filter_kernel<<<grid_dim, block_dim, shared_bytes, stream.cuda_stream>>>(
      device_src_list, int(stream.src_indices.size()),
      stream.device_src_indices, device_list_start_indices,
      device_compat_src_results, device_unique_variations,
      num_unique_variations, device_results, stream.stream_idx);
  assert_cuda_success(cudaPeekAtLastError(), "filter kernel launch failed");
  stream.xor_kernel_stop.record();

  #if defined(DEBUG_XOR_COUNTS) || defined(DEBUG_OR_COUNTS)
  if (log_level(ExtraVerbose)) {
    display_counts();
  }
  #endif 
  if constexpr (0) {
    std::cerr << "stream " << stream.stream_idx << " XOR kernel started with "
              << grid_size << " blocks of " << block_size << " threads"
              << std::endl;
  }
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
