// filter.cu

#include <algorithm>
#include <iostream>
#include <limits>
#include <cuda_runtime.h>
#include "cuda-device.h"
#include "filter.cuh"
#include "filter-stream.h"
#include "merge-filter-data.h"
#include "or-filter.cuh"
#include "util.h"

namespace cm {

__constant__ FilterData::DeviceXor xor_data;
__constant__ FilterData::DeviceOr or_data;
__constant__ FilterStreamData::Device stream_data[kMaxStreams];
__constant__ SourceCompatibilityData* sources_data[kMaxSums];
//__constant__ FilterData::DeviceSources sources_data;

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

// Test if the supplied sources match both of the primary sources described
// by any of the supplied source descriptor pairs.
__device__ bool sources_match_any_descriptor_pair(
    const SourceCompatibilityData& source1,
    const SourceCompatibilityData& source2,
    const SourceDescriptorPair* RESTRICT src_desc_pairs,
    const unsigned num_src_desc_pairs) {
  __shared__ bool match;
  if (!threadIdx.x) match = false;
  __syncthreads();
  // one thread per src_desc_pair
  for (unsigned idx{}; idx * blockDim.x < num_src_desc_pairs; ++idx) {
    const auto pair_idx = idx * blockDim.x + threadIdx.x;
    if (pair_idx < num_src_desc_pairs) {
      const auto pair = src_desc_pairs[pair_idx];
      const auto first = source1.usedSources.has(pair.first)
          || source2.usedSources.has(pair.first);
      const auto second = source1.usedSources.has(pair.second)
          || source2.usedSources.has(pair.second);
      if (first && second) { match = true; }
    }
    __syncthreads();
    if (match) return true;
  }
  return false;
}

#if 1
__global__ void get_compatible_sources_kernel(
    const CompatSourceIndices* RESTRICT compat_src_indices,
    const unsigned num_compat_src_indices,
    const SourceDescriptorPair* RESTRICT incompat_src_desc_pairs,
    const unsigned num_src_desc_pairs, result_t* RESTRICT results) {
  // one block per source-(index)-pair
  for (unsigned idx{blockIdx.x}; idx < num_compat_src_indices;
      idx += gridDim.x) {
    const auto csi = compat_src_indices[idx];
    const auto& source1 =
        sources_data[csi.first().count()][csi.first().index()];
    const auto& source2 =
        sources_data[csi.second().count()][csi.second().index()];
    __syncthreads();
    if (!sources_match_any_descriptor_pair(source1, source2,
            incompat_src_desc_pairs, num_src_desc_pairs)) {
      if (!threadIdx.x) { results[idx] = 1; }
    }
  }
}

#else

// "filter compatible sources"
__global__ void get_compatible_sources_kernel(
    const CompatSourceIndices* RESTRICT compat_src_indices,
    const unsigned num_compat_src_indices,
    const UsedSources::SourceDescriptorPair* RESTRICT incompat_src_desc_pairs,
    const unsigned num_src_desc_pairs, result_t* RESTRICT results) {
  // one block per source-(index)-pair
  for (unsigned idx{blockIdx.x}; idx < num_compat_src_indices;
      idx += gridDim.x) {
    const auto csi = compat_src_indices[idx];
    const auto& source1 =
        sources_data[csi.first().count()][csi.first().index()];
    const auto& source2 =
        sources_data[csi.second().count()][csi.second().index()];
    // I don't understand why this is required, but fails synccheck and
    // produces wobbly results without.
    __syncthreads();
    if (!sources_match_any_descriptor_pair(source1, source2,
            incompat_src_desc_pairs, num_src_desc_pairs)) {
      if (!threadIdx.x) {
        results[idx] = 1;
      }
    }
  }
}
#endif

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

#define USE_LOCAL_XOR_COMPAT

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
// If true, the flag array is used by check_src_compat_results() to reject
// incompatible OR-source combo-indices when generating OR-source chunks.
//
// It's currently not used for testing compatibility with XOR sources, and
// the XOR code in are_sources_compatible() should be reviewed when it is.
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
    const auto& other = src_list[idx_list[src_idx.index]];
    const auto compat = are_sources_compatible<TagT>(source, other);
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
        src_sentences[(*num_src_sentences)++] = static_cast<uint8_t>(s);
      }
    }
  }
  return compute_src_compat_results<tag::OR>(source, or_data);
}

__device__ auto compute_compat_XOR_uv_indices(const Variations& src_variations) {
  const auto begin = clock64();
  index_t num_uv_indices{};
  if (compute_variations_compat_results(src_variations, xor_data,
          xor_data.src_compat_uv_indices)) {
    num_uv_indices = compute_compat_uv_indices(xor_data.num_unique_variations,
        xor_data.src_compat_uv_indices);
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
    if (uvi_idx == num_uv_indices) {
      start_idx = xor_data.num_compat_indices;
    }
    dynamic_shared[kXorStartUvIdx] = uvi_idx;
    dynamic_shared[kXorStartSrcIdx] = start_idx;
  }
  return result;
}

// Get the next block-sized chunk of XOR sources and test them for 
// compatibility with the supplied source.
// Return true if at least one XOR source is compatible.
__device__ bool get_next_XOR_sources_chunk(const SourceCompatibilityData& source,
    const index_t xor_chunk_idx, const index_t num_uv_indices) {
  result_t* xor_results = (result_t*)&dynamic_shared[kXorResults];
  // a __sync will happen in calling function
  xor_results[threadIdx.x] = 0;
  // one thread per compat_idx
  const auto xor_compat_idx = get_XOR_compat_idx_incremental_uv(xor_chunk_idx,
      num_uv_indices);
  if (xor_compat_idx >= xor_data.num_compat_indices) return false;

  #ifdef DEBUG_XOR_COUNTS
  atomicAdd(&count_xor_src_considered, 1);
  #endif

  const auto xor_combo_idx = xor_data.compat_indices[xor_compat_idx];

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
// source, or until all XOR sources are exhausted. Return true if at least one
// XOR source is compatible.
__device__ bool get_next_compatible_XOR_sources_chunk(
    const SourceCompatibilityData& source, index_t xor_chunk_idx,
    const index_t num_uv_indices) {
  result_t* xor_results = (result_t*)&dynamic_shared[kXorResults];
  const auto block_size = blockDim.x;
  const auto num_xor_indices = xor_data.num_compat_indices;
  __shared__ bool any_xor_compat;
  if (!threadIdx.x) any_xor_compat = false;
  __syncthreads();
  for (; xor_chunk_idx * block_size < num_xor_indices; ++xor_chunk_idx) {
    if (get_next_XOR_sources_chunk(source, xor_chunk_idx, num_uv_indices)) {
      any_xor_compat = true;
    }
    __syncthreads();
    if (any_xor_compat) break;
    if (dynamic_shared[kXorStartUvIdx] == num_uv_indices) break;
  }
  if (!threadIdx.x) dynamic_shared[kXorChunkIdx] = xor_chunk_idx;
  return any_xor_compat;
}

// Test if the supplied source both is:
// * XOR-compatible with any of the supplied XOR sources
// * OR-compatible with any of the supplied OR sources, which are in turn
//   variation-compatible with the compatible XOR source.
//
// In other words:
// For each XOR source that is XOR-compatible with Source
//   For each OR source that is variation-compatible with XOR source
//     If OR source is OR-compatible with Source
//       is_compat = true
__device__ bool is_compat_loop(const SourceCompatibilityData& source) {
  const auto block_size = blockDim.x;
  index_t* xor_chunk_idx_ptr = &dynamic_shared[kXorChunkIdx];
  __shared__ bool any_xor_compat;
  __shared__ bool any_or_compat;
  __shared__ bool src_init_done;
  const auto num_xor_indices = xor_data.num_compat_indices;
  if (!threadIdx.x) {
    *xor_chunk_idx_ptr = 0;
    any_xor_compat = false;
    any_or_compat = false;
    src_init_done = false;
    dynamic_shared[kXorStartUvIdx] = 0;
    dynamic_shared[kXorStartSrcIdx] = 0;
  }
  __syncthreads();
  const auto num_uv_indices =
      compute_compat_XOR_uv_indices(source.usedSources.variations);
  if (!num_uv_indices) return false;
  for (;;) {
    //    __syncthreads();
    if (*xor_chunk_idx_ptr * block_size >= num_xor_indices) return false;

    auto begin = clock64();
    // TODO: passing shared variable not necessary
    if (get_next_compatible_XOR_sources_chunk(source, *xor_chunk_idx_ptr,
            num_uv_indices)) {
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
      // TODO: passing shared variable not necessary
      if (is_any_OR_source_compatible(source, *xor_chunk_idx_ptr)) {
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
    __syncthreads();
  }
  return false;
}

__device__ SourceCompatibilityData merge_sources(
    const CompatSourceIndices compat_src_indices) {
  // NOTE: this could be faster with 9 threads for example
  // and, shared memory
  const auto csi1 = compat_src_indices.first();
  const auto csi2 = compat_src_indices.second();
  SourceCompatibilityData source{sources_data[csi1.count()][csi1.index()]};
  source.mergeInPlace(sources_data[csi2.count()][csi2.index()]);
  return source;
}

__device__ int dumped = 0;

// explain better:
// Find sources that are:
// * XOR compatible with any of the supplied XOR sources, and
// * OR compatible with any of the supplied OR sources, which must in turn be
// * variation-compatible with the corresponding/aforementioned XOR source.
//
// Find compatible XOR source -> compare with variation-compatible OR sources.
__global__ void filter_kernel(
    const CompatSourceIndices* RESTRICT compat_src_indices,
    int num_compat_src_indices, const SourceIndex* RESTRICT src_indices,
    const result_t* RESTRICT compat_src_results, result_t* RESTRICT results,
    index_t stream_idx) {
  __shared__ SourceCompatibilityData source;
  __shared__ bool is_compat;
  if (!threadIdx.x) {
    dynamic_shared[kStreamIdx] = stream_idx;
    is_compat = false;
  }

#if defined(PRINTF)
    if (!blockIdx.x && !threadIdx.x) {
      printf("+++kernel+++ blocks: %u\n", gridDim.x);
    }
#endif

#if MAX_SOURCES
    if (num_compat_src_indices >= MAX_SOURCES) {
      num_compat_src_indices = MAX_SOURCES;
    }
#endif
  // for each source (one block per source)
  for (index_t idx{blockIdx.x}; idx < num_compat_src_indices;
      idx += gridDim.x) {
    __syncthreads();
    const auto src_idx = src_indices[idx];
    if (!compat_src_results || compat_src_results[src_idx.index]) {
      if (!threadIdx.x) {
        dynamic_shared[kSrcListIdx] = src_idx.listIndex;
        dynamic_shared[kSrcIdx] = src_idx.index;
        const auto csi1 = compat_src_indices[src_idx.index].first();
        const auto csi2 = compat_src_indices[src_idx.index].second();
        // TODO: compare to source.reset() followed by 2x mergeInPlace
        source = sources_data[csi1.count()][csi1.index()];
#if 1
        if (!source.usedSources.mergeInPlace(
                sources_data[csi2.count()][csi2.index()].usedSources)) {
          printf("idx %u, num_indices %u, src_idx.listIndex %u, src_idx.index "
                 "%u, csi1 count %u index %u, csi2 count %u index %u\n",
              idx, num_compat_src_indices, src_idx.listIndex, src_idx.index,
              csi1.count(), csi1.index(), csi2.count(), csi2.index());
        }
#endif
        // sync happens inside is_compat_loop
      }
      if (is_compat_loop(source)) { is_compat = true; }
    }
    __syncthreads();
    if (!threadIdx.x && is_compat) {
      results[src_idx.listIndex] = 1;
      is_compat = false;
    }
  }
}

}  // anonymous namespace

std::pair<int, int> get_filter_kernel_grid_block_sizes() {
  // hard-code 64 due to cub::BlockScan
  const auto block_size = 64;  // threads_per_block ? threads_per_block : 64;
  const auto max_threads_per_sm = CudaDevice::get().max_threads_per_sm();
  const auto blocks_per_sm = max_threads_per_sm / block_size;
  assert(blocks_per_sm * block_size == max_threads_per_sm);
  const auto grid_size = CudaDevice::get().num_sm() * blocks_per_sm;
  return std::make_pair(grid_size, block_size);
}

void copy_filter_data_to_symbols(const FilterData& mfd, cudaStream_t stream) {
  cudaError_t err{};
  const auto xor_bytes = sizeof(FilterData::DeviceXor);
  err = cudaMemcpyToSymbolAsync(xor_data, &mfd.device_xor, xor_bytes, 0,
      cudaMemcpyHostToDevice, stream);
  assert_cuda_success(err, "copy xor filter data");

  const auto or_bytes = sizeof(FilterData::DeviceOr);
  err = cudaMemcpyToSymbolAsync(or_data, &mfd.device_or, or_bytes, 0,
      cudaMemcpyHostToDevice, stream);
  assert_cuda_success(err, "copy or filter data");

  /*
  const auto sources_bytes = sizeof(FilterData::DeviceSources);
  err = cudaMemcpyToSymbol(sources_data, &mfd.device_sources, sources_bytes);
  assert_cuda_success(err, "copy sources filter data");
  */
}

void run_filter_kernel(int /*threads_per_block*/, FilterStream& stream,
    const CompatSourceIndices* device_src_indices,
    const result_t* device_compat_src_results, result_t* device_results) {
  stream.is_running = true;
  stream.increment_sequence_num();
  const auto [grid_size, block_size] = get_filter_kernel_grid_block_sizes();
  // xor_results could probably be moved to global
  const auto shared_bytes = kSharedIndexCount
          * sizeof(shared_index_t)               // indices
      + kNumSentenceDataBytes * sizeof(uint8_t)  // src_sentence data
      + block_size * sizeof(result_t);           // xor_results

  static bool log_once{true};
  if (log_once) {
    cuda_memory_dump("before filter kernel");
    log_once = false;
  }

  #if defined(DEBUG_XOR_COUNTS) || defined(DEBUG_OR_COUNTS) || defined(CLOCKS)
  init_counts();
  #endif 

  // ensure any async alloc/copies are complete on main thread stream
  // cudaError_t err = cudaStreamSynchronize(cuda_stream);
  // assert_cuda_success(err, "run_filter_kernel sync");
  const dim3 grid_dim(grid_size);
  const dim3 block_dim(block_size);
  stream.record(stream.kernel_start);
  filter_kernel<<<grid_dim, block_dim, shared_bytes, stream.cuda_stream>>>(
      device_src_indices, int(stream.host.src_idx_list.size()),
      stream.device.src_idx_list, device_compat_src_results, device_results,
      stream.host.global_idx());
  assert_cuda_success(cudaPeekAtLastError(), "filter_kernel");
  stream.record(stream.kernel_stop);

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
    const CompatSourceIndices* device_src_indices, size_t num_src_indices,
    const SourceDescriptorPair* device_src_desc_pairs,
    size_t num_src_desc_pairs, result_t* device_results,
    cudaStream_t sync_stream, cudaStream_t stream) {
  const auto block_size = 768;  // aka threads per block
  const auto max_threads_per_sm = CudaDevice::get().max_threads_per_sm();
  const auto blocks_per_sm = max_threads_per_sm / block_size;
  assert(blocks_per_sm * block_size == max_threads_per_sm);
  const auto grid_size = CudaDevice::get().num_sm() * blocks_per_sm;
  const auto shared_bytes = 0;

  const dim3 grid_dim(grid_size);
  const dim3 block_dim(block_size);
  // ensure any async alloc/copies aee complete on sync stream
  // TODO: change to a event.synchronize. maybe pass in a new 
  // CudaEventStream class?
  auto err = cudaStreamSynchronize(sync_stream);
  assert_cuda_success(err, "get_compat_sources_kernel sync");
  get_compatible_sources_kernel<<<grid_dim, block_dim, shared_bytes, stream>>>(
      device_src_indices, unsigned(num_src_indices), device_src_desc_pairs,
      unsigned(num_src_desc_pairs), device_results);
  assert_cuda_success(cudaPeekAtLastError(), "get_compat_sources_kernel");
}

}  // namespace cm
