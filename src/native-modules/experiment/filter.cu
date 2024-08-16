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

__constant__ FilterData::DeviceCommon<fat_index_t> xor_data;
__constant__ FilterData::DeviceOr or_data;

const unsigned BIG = 2'100'000'000;

#if defined(DEBUG_XOR_COUNTS) || defined(DEBUG_OR_COUNTS)
__device__ atomic64_t count_xor_src_considered = 0;
__device__ atomic64_t count_xor_src_compat = 0;

__device__ atomic64_t count_or_compat_results = 0;
__device__ atomic64_t count_or_src_considered = 0;
__device__ atomic64_t count_or_xor_compat = 0;
__device__ atomic64_t count_or_src_variation_compat = 0;
__device__ atomic64_t count_or_check_src_compat = 0;
__device__ atomic64_t count_or_src_compat = 0;

__device__ atomic64_t count_xor_bits_compat;
__device__ atomic64_t count_xor_variations_compat;
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

#ifdef DEBUG_XOR_COUNTS
  err = cudaMemcpyToSymbol(count_xor_src_considered, &value, sizeof(atomic64_t));
  assert_cuda_success(err, "init count");

  err = cudaMemcpyToSymbol(count_xor_src_compat, &value, sizeof(atomic64_t));
  assert_cuda_success(err, "init count");
#endif


#ifdef USE_LOCAL_XOR_COMPAT
  err = cudaMemcpyToSymbol(count_xor_bits_compat, &value, sizeof(atomic64_t));
  assert_cuda_success(err, "init count");

  err = cudaMemcpyToSymbol(count_xor_variations_compat, &value,  //
      sizeof(atomic64_t));
  assert_cuda_success(err, "init count");
#endif


#ifdef DEBUG_OR_COUNTS
  err = cudaMemcpyToSymbol(count_or_compat_results, &value, sizeof(atomic64_t));
  assert_cuda_success(err, "init count");

  err = cudaMemcpyToSymbol(count_or_src_considered, &value, sizeof(atomic64_t));
  assert_cuda_success(err, "init count");

  err = cudaMemcpyToSymbol(count_or_xor_compat, &value, sizeof(atomic64_t));
  assert_cuda_success(err, "init count");

  err = cudaMemcpyToSymbol(count_or_src_variation_compat, &value,  //
      sizeof(atomic64_t));
  assert_cuda_success(err, "init count");

  err = cudaMemcpyToSymbol(count_or_check_src_compat, &value, sizeof(atomic64_t));
  assert_cuda_success(err, "init count");

  err = cudaMemcpyToSymbol(count_or_src_compat, &value, sizeof(atomic64_t));
  assert_cuda_success(err, "init count");
#endif
}

void display_counts() {
  cudaError_t err{};

#ifdef DEBUG_XOR_COUNTS
  atomic64_t considered_xor_count;
  atomic64_t compat_xor_count;

  err = cudaMemcpyFromSymbol(
      &considered_xor_count, count_xor_src_considered, sizeof(atomic64_t));
  assert_cuda_success(err, "display count");

  err = cudaMemcpyFromSymbol(
      &compat_xor_count, count_xor_src_compat, sizeof(atomic64_t));
  assert_cuda_success(err, "display count");
#endif

#ifdef USE_LOCAL_XOR_COMPAT
  atomic64_t compat_xor_bits_count;
  atomic64_t compat_xor_variations_count;

  err = cudaMemcpyFromSymbol(
      &compat_xor_bits_count, count_xor_bits_compat, sizeof(atomic64_t));
  assert_cuda_success(err, "display count");

  err = cudaMemcpyFromSymbol(
      &compat_xor_variations_count, count_xor_variations_compat, sizeof(atomic64_t));
  assert_cuda_success(err, "display count");
#endif

#ifdef DEBUG_OR_COUNTS
  atomic64_t compat_or_results_count;
  atomic64_t considered_or_count;
  atomic64_t compat_or_xor_count;  // variations
  atomic64_t compat_or_src_variation_count;
  atomic64_t or_check_src_compat_count;
  atomic64_t compat_or_count;

  err = cudaMemcpyFromSymbol(
      &compat_or_results_count, count_or_compat_results, sizeof(atomic64_t));
  assert_cuda_success(err, "display count");

  err = cudaMemcpyFromSymbol(
      &considered_or_count, count_or_src_considered, sizeof(atomic64_t));
  assert_cuda_success(err, "display count");

  err = cudaMemcpyFromSymbol(
      &compat_or_xor_count, count_or_xor_compat, sizeof(atomic64_t));
  assert_cuda_success(err, "display count");

  err = cudaMemcpyFromSymbol(
      &compat_or_src_variation_count, count_or_src_variation_compat, sizeof(atomic64_t));
  assert_cuda_success(err, "display count");

  err = cudaMemcpyFromSymbol(
      &or_check_src_compat_count, count_or_check_src_compat, sizeof(atomic64_t));
  assert_cuda_success(err, "display count");

  err = cudaMemcpyFromSymbol(
      &compat_or_count, count_or_src_compat, sizeof(atomic64_t));
  assert_cuda_success(err, "display count");
#endif

  std::cerr
#ifdef DEBUG_XOR_COUNTS
      << " xor_considered: " << considered_xor_count
      << " xor_compat: " << compat_xor_count
#endif
#ifdef DEBUG_OR_COUNTS
      << std::endl
      << " or_compat_results: " << compat_or_results_count
      << " or_considered: " << considered_or_count  //
      << std::endl
      << " or_xor_compat: " << compat_or_xor_count
      << " or_src_variation_compat: " << compat_or_src_variation_count
      << std::endl
      << " or_check_compat: " << or_check_src_compat_count
      << " or_compat: " << compat_or_count
#endif
#ifdef USE_LOCAL_XOR_COMPAT
      << std::endl
      << " xor_bits_compat: " << compat_xor_bits_count
      << " xor_variations_compat: " << compat_xor_variations_count
#endif
      << std::endl;
}

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

template <typename IndexT, typename TagT>
requires std::is_same_v<TagT, tag::XOR> // || std::is_same_v<TagT, tag::OR>
__device__ bool is_source_compatible(
    TagT tag, const SourceCompatibilityData& source, fat_index_t flat_idx) {
  FilterData::DeviceCommon<IndexT> const* data{};
  if constexpr (std::is_same_v<TagT, tag::XOR>) data = &xor_data;
  //  else if constexpr (std::is_same_v<TagT, tag::OR>) data = &or_data;

  #ifdef DISABLE_OR
  if constexpr (std::is_same_v<TagT, tag::OR>) return false;
  #endif
  #ifdef FORCE_XOR_COMPAT
  else if constexpr (std::is_same_v<TagT, tag::XOR>) return true;
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

// Get the next block-sized chunk of XOR sources and test them for 
// compatibility with the supplied source.
// Return true if at least one XOR source is compatible.
__device__ bool get_next_XOR_sources_chunk(
    const SourceCompatibilityData& source, const unsigned xor_chunk_idx,
    const FatIndexSpanPair& xor_idx_spans) {
  result_t* xor_results = (result_t*)&dynamic_shared[kXorResults];
  const auto block_size = blockDim.x;
  const auto num_xor_indices =
      xor_idx_spans.first.size() + xor_idx_spans.second.size();
  xor_results[threadIdx.x] = 0;
  const auto xor_flat_idx = get_flat_idx(xor_chunk_idx);
  if (xor_flat_idx < num_xor_indices) {

    #ifdef DEBUG_XOR_COUNTS
    atomicAdd(&count_xor_src_considered, 1);
    #endif

    const auto xor_combo_idx = get_xor_combo_index(xor_flat_idx, xor_idx_spans);
    if (is_source_compatible<fat_index_t>(tag::XOR{}, source, xor_combo_idx)) {

      #if defined(DEBUG_XOR_COUNTS)
      atomicAdd(&count_xor_src_compat, 1);
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
  result_t* xor_results = (result_t*)&dynamic_shared[kXorResults];
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
    if (!threadIdx.x) {
      printf(" block: %u get_next_XOR xor_chunk_idx: %u, compat: %d\n", blockIdx.x,
          xor_chunk_idx, any_xor_compat ? 1 : 0);
    }
    #endif

    if (any_xor_compat) break;
  }
  if (!threadIdx.x) *xor_chunk_idx_ptr = xor_chunk_idx;
  return any_xor_compat;
}

__device__ SourceIndex get_source_index(
    unsigned idx, const MergeData::Device& data) {
  for (unsigned list_idx{}; list_idx < data.num_idx_lists; ++list_idx) {
    auto list_size = data.idx_list_sizes[list_idx];
    if (idx < list_size) return {list_idx, idx};
    idx -= list_size;
  }
  assert(0);
  return {};
}

// Determine which OR sources are compatible with the supplied source.
template <typename TagT, typename T>
requires std::is_same_v<TagT, tag::XOR> || std::is_same_v<TagT, tag::OR>
__device__ auto get_src_compat_results(
    /* TagT tag,*/ const SourceCompatibilityData& source, T& data) {
  fat_index_t* debug_idx_ptr = &dynamic_shared[kDebugIdx];
  __shared__ result_t any_compat[kMaxOrArgs];
  __shared__ bool all_compat;
  if (threadIdx.x < kMaxOrArgs) {
    any_compat[threadIdx.x] = false;
    all_compat = false;
  }
  __syncthreads();
  for (auto idx{threadIdx.x}; idx < data.sum_idx_list_sizes;
      idx += blockDim.x) {
    const auto src_idx = get_source_index(idx, data);
#if 0
    const auto idx_list = &data.idx_lists[src_idx.listIndex];
    //const auto start_idx = data.src_list_start_indices[src_idx.listIndex];
    const auto the_src_idx = idx_list[src_idx.index];
    const auto& src = data.src_lists[the_src_idx];
#else
    // TODO: i think i need to drill into idx_list first here? maybe?
    const auto start_idx = data.src_list_start_indices[src_idx.listIndex];
    const auto& src = data.src_lists[start_idx + src_idx.index];
#endif
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
    int num_compat{};
    for (unsigned idx{}; idx < data.num_idx_lists; ++idx) {
      if (any_compat[idx]) ++num_compat;
    }
    if (num_compat == data.num_idx_lists) {
      //#if defined(DEBUG_XOR_COUNTS) || defined(DEBUG_OR_COUNTS)
      //atomicAdd(&count_or_compat_results, 1);
      //#endif
      all_compat = true;
    }
  }
  __syncthreads();
  return all_compat;
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
  // required __syncthreads() will happen in get_src_compat_data()
  return get_src_compat_results<tag::XOR>(source, xor_data)
    && get_src_compat_results<tag::OR>(source, or_data);
}

/*
  if (*debug_idx_ptr) {
    source.dump("8090__10");
    for (unsigned result_idx{}, count{}, list_start_idx{}, list_idx{};
        result_idx < data.sum_idx_list_sizes; ++result_idx) {
      if (result_idx - list_start_idx == data.idx_list_sizes[list_idx]) {
        list_start_idx = result_idx;
        ++list_idx;
        printf("%u: %u  ", list_idx, count);
        count = 0;
      }
      if (data.src_compat_results[result_idx]) ++count;
    }
    printf("\n");
  }
*/

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
    const FatIndexSpanPair& xor_idx_spans) {
  const auto block_size = blockDim.x;
  fat_index_t* xor_chunk_idx_ptr = &dynamic_shared[kXorChunkIdx];
  __shared__ bool any_xor_compat;
  __shared__ bool any_or_compat;
  __shared__ bool src_init_done; 
  const auto num_xor_indices =
      xor_idx_spans.first.size() + xor_idx_spans.second.size();
  if (!threadIdx.x) {
    *xor_chunk_idx_ptr = 0;
    any_xor_compat = false;
    any_or_compat = false;
    src_init_done = false;
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
      if (!src_init_done && init_source(source)) {
        src_init_done = true;
      }
    }
    __syncthreads();

    // TODO: try: if !have continue
    if (src_init_done
        && is_any_OR_source_compatible(source, *xor_chunk_idx_ptr,  //
            xor_idx_spans)) {
      any_or_compat = true;
    }
    __syncthreads();
    if (any_or_compat) return true;

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
  fat_index_t* debug_idx_ptr = &dynamic_shared[kDebugIdx];
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

    #if 1 // DEBUG
    if (!threadIdx.x) {
      *debug_idx_ptr = (src_idx.listIndex == 8090 || src_idx.listIndex == 10);
      //if (*debug_idx_ptr) source.dump("8090or10");
    }
    #endif

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

void copy_filter_data(const FilterData& mfd) {
  cudaError_t err{};

  const auto xor_bytes = sizeof(FilterData::DeviceXor::Base);
  err = cudaMemcpyToSymbol(xor_data, &mfd.device_xor, xor_bytes);
  assert_cuda_success(err, "copy xor filter data");

  const auto or_bytes = sizeof(FilterData::DeviceOr);
  err = cudaMemcpyToSymbol(or_data, &mfd.device_or, or_bytes);
  assert_cuda_success(err, "copy or filter data");
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

  const auto block_size = threads_per_block ? threads_per_block : 768;
  const auto blocks_per_sm = threads_per_sm / block_size;
  // assert(blocks_per_sm * block_size == threads_per_sm); // allow 1024 tpb
  const auto grid_size = num_sm * blocks_per_sm;  // aka blocks per grid

  // results could probably be moved to global
  const auto shared_bytes = kSharedIndexCount * sizeof(fat_index_t)  // indices
                            + 16 * sizeof(uint8_t)  // src_sentence data
                            + block_size * sizeof(result_t);  // xor_results

  const auto cuda_stream = cudaStreamPerThread;
  // v- Populate any remaining mfd.device_xx values BEFORE copy_filter_data() -v
  const auto xor_results_bytes =
      grid_size * mfd.device_xor.sum_idx_list_sizes * sizeof(result_t);
  cuda_malloc_async((void**)&mfd.device_xor.src_compat_results,
      xor_results_bytes, cuda_stream, "src_compat_results");
  const auto or_results_bytes =
      grid_size * mfd.device_or.sum_idx_list_sizes * sizeof(result_t);
  cuda_malloc_async((void**)&mfd.device_or.src_compat_results, or_results_bytes,
      cuda_stream, "src_compat_results");
  // ^- Populate any reamining mfd.device_xx values BEFORE copy_filter_data() -^
  copy_filter_data(mfd);

  static bool log_once{true};
  if (log_once) {
    cuda_memory_dump("before filter kernel");
    log_once = false;
  }

  #if defined(DEBUG_XOR_COUNTS) || defined(DEBUG_OR_COUNTS)
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
