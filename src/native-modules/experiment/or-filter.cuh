#pragma once
#include <cassert>
#include <cub/block/block_scan.cuh>
#include <cooperative_groups.h>
#include "cuda-types.h"
#include "merge-filter-data.h"

namespace cm {

namespace cg = cooperative_groups;

#if 1
#define CLOCKS
#define DEBUG_OR_COUNTS
#define DEBUG_XOR_COUNTS
#endif
//#define MAX_SOURCES 1
//#define PRINTF

class SourceCompatibilityData;

using shared_index_t = index_t;

inline constexpr auto kSharedIndexSize = sizeof(shared_index_t);  // in bytes

inline constexpr auto kXorChunkIdx = 0;
inline constexpr auto kXorStartUvIdx = 1;
inline constexpr auto kXorStartSrcIdx = 2;
inline constexpr auto kOrStartUvIdx = 3;
inline constexpr auto kOrStartSrcIdx = 4;
inline constexpr auto kSrcFlatIdx = 5;
inline constexpr auto kDebugIdx = 6;
inline constexpr auto kSharedIndexCount = 7;

// num source sentences (uint8_t) starts at end of indices
inline constexpr auto kNumSrcSentences = kSharedIndexCount;
inline constexpr auto kNumSentenceDataBytes = 12;
// source sentence data (unit8_t) follows for 9 more bytes, round to 12 total

// xor_results (result_t) starts after of sentence data
inline constexpr auto kXorResults =
    kNumSrcSentences + (kNumSentenceDataBytes / kSharedIndexSize);

extern __shared__ index_t dynamic_shared[];

namespace tag {

struct XOR {};
struct OR {};

}  // namespace tag

extern __constant__ FilterData::DeviceXor xor_data;
extern __constant__ FilterData::DeviceOr or_data;

__device__ __forceinline__ index_t get_flat_idx(
    index_t block_idx, index_t thread_idx = threadIdx.x) {
  return block_idx * blockDim.x + thread_idx;
}

/*
// test "OR compatibility" in one pass (!intersects || is_subset_of)
// TODO: this could go into mmebitset.h i think?
__device__ __forceinline__ bool is_disjoint_or_subset(
    const UsedSources::SourceBits& a, const UsedSources::SourceBits& b) {
  using SourceBits = UsedSources::SourceBits;
  for (int i{}; i < SourceBits::wc() / 2; ++i) {
    const auto w = a.long_word(i) & b.long_word(i);
    if (w && (w != a.long_word(i))) return false;
  }
  for (int i{}; i < SourceBits::wc() - (SourceBits::wc() / 2) * 2; ++i) {
    const auto w = a.word(i) & b.word(i);
    if (w && (w != a.word(i))) return false;
  }
  return true;
}

__device__ __forceinline__ auto source_bits_are_OR_compatible(
    const UsedSources::SourceBits& a, const UsedSources::SourceBits& b) {
  return is_disjoint_or_subset(a, b);
}
*/

__device__ __forceinline__ auto are_source_bits_OR_compatible(
    const UsedSources::SourceBits& a, const UsedSources::SourceBits& b,
    int word_idx) {
  const auto w = a.word(word_idx) & b.word(word_idx);
  if (w && (w != a.word(word_idx))) return false;
  return true;
}

__device__ __forceinline__ auto are_source_bits_XOR_compatible(
    const UsedSources::SourceBits& a, const UsedSources::SourceBits& b,
    int word_idx) {
  return (a.word(word_idx) & b.word(word_idx)) == 0u;
}

template <typename TagT>
requires std::is_same_v<TagT, tag::XOR> || std::is_same_v<TagT, tag::OR>
__device__ inline auto is_source_compatible_with(
    const SourceCompatibilityData& source,
    const SourceCompatibilityData& other) {
  uint8_t* num_src_sentences = (uint8_t*)&dynamic_shared[kNumSrcSentences];
  uint8_t* src_sentences = &num_src_sentences[1];

  for (uint8_t idx{}; idx < *num_src_sentences; ++idx) {
    auto sentence = src_sentences[idx];
    // we know all source.variations[s] are > -1. no need to compare bits if
    // other.variations[s] == -1.
    if (other.usedSources.variations[sentence] > -1) {
      if constexpr (std::is_same_v<TagT, tag::OR>) {
        // NB: order of params matters here (or_src first)
        if (!are_source_bits_OR_compatible(other.usedSources.getBits(),
                source.usedSources.getBits(), sentence)) {
          return false;
        }
      } else if constexpr (std::is_same_v<TagT, tag::XOR>) {
#if 1
        if (!are_source_bits_XOR_compatible(other.usedSources.getBits(),
                source.usedSources.getBits(), sentence)) {
          return false;
        }
#else
        // NOTE: that this is a little weird. Things would probably speed
        // up if I actually used these results to test for source-bit
        // compatibility as above. But my main goal is speeding up XOR-
        // variation compatibility, so this is a bit of a hack for now.
        if (other.usedSources.variations[sentence]
            != source.usedSources.variations[sentence])
          return false;
#endif
      }
    }
  }
  return true;
}

// With all sources identified by combo_idx
template <typename TagT, typename T>
requires std::is_same_v<TagT, tag::XOR> || std::is_same_v<TagT, tag::OR>
__device__ auto is_source_compatible_with_all(
    const SourceCompatibilityData& source, fat_index_t combo_idx,
    const T& data) {
  for (int list_idx{int(data.num_idx_lists) - 1}; list_idx >= 0;
      --list_idx) {
    const auto& src = data.get_source(combo_idx, list_idx);
    if (!is_source_compatible_with<TagT>(source, src)) return false;
    combo_idx /= data.idx_list_sizes[list_idx];
  }
  return true;
}

template <typename T>
__device__ auto check_src_compat_results(fat_index_t combo_idx, const T& data) {
  const auto block_start_idx = blockIdx.x * data.sum_idx_list_sizes;
  auto list_start_idx = index_t(data.sum_idx_list_sizes);
  for (int list_idx{int(data.num_idx_lists) - 1}; list_idx >= 0; --list_idx) {
    const auto list_size = data.idx_list_sizes[list_idx];
    const auto src_idx = combo_idx % list_size;
    assert(list_start_idx >= list_size);
    list_start_idx -= list_size;
    if (!data.src_compat_results[block_start_idx + list_start_idx + src_idx])
      return false;
    combo_idx /= list_size;
  }
  return true;
}

// faster than UsedSources version
__device__ __forceinline__ void merge_variations(
    Variations& to, const Variations& from, bool force = false) {
  for (int s{}; s < kNumSentences; ++s) {
    if (force || (to[s] == -1)) to[s] = from[s];
  }
}

template <typename T>
__device__ auto build_variations(fat_index_t combo_idx, const T& data) {
  Variations v;
  bool force = true;
  for (int list_idx{int(data.num_idx_lists) - 1}; list_idx >= 0; --list_idx) {
    const auto& src = data.get_source(combo_idx, list_idx);
    combo_idx /= data.idx_list_sizes[list_idx];
    // TODO: put UsedSources version of this in an #ifdef ASSERTS block
    merge_variations(v, src.usedSources.variations, force);
    force = false;
  }
  return v;
}

__device__ inline auto compute_variations_compat_results(
    const Variations& src_variations,
    const FilterData::DeviceCommon& tgt_uv_data, index_t* compat_results) {
  __shared__ bool any_compat;
  if (!threadIdx.x) any_compat = false;
  __syncthreads();
#ifdef ONE_ARRAY
  const auto results_offset = blockIdx.x * tgt_uv_data.num_unique_variations;
  const auto results = &out_compat_results[results_offset];
#else
  const auto results_offset = blockIdx.x * xor_data.variations_results_per_block;
  const auto results = &xor_data.variations_compat_results[results_offset];
#endif
  for (auto idx{threadIdx.x}; idx < tgt_uv_data.num_unique_variations;
      idx += blockDim.x) {
    const auto compat = UsedSources::are_variations_compatible(src_variations,
        tgt_uv_data.unique_variations[idx].variations);
    results[idx] = compat ? 1 : 0;
    if (compat) any_compat = true;
  }
  __syncthreads();
  return any_compat;
}

constexpr auto kBlockSize = 64;

// compute prefix sums from compat results flag array
__device__ inline void compute_prefix_sums_in_place(index_t* results,
    const index_t num_results) {
  const auto last_result_idx = num_results - 1;
  const auto last_compat_result = results[last_result_idx];

  using BlockScan = cub::BlockScan<index_t, kBlockSize>;
  __shared__ typename BlockScan::TempStorage temp_storage;

  const auto max_idx = blockDim.x
      * ((num_results + blockDim.x - 1) / blockDim.x);
  __shared__ index_t prefix_sum;
  if (!threadIdx.x) prefix_sum = 0;
  for (index_t idx{threadIdx.x}; idx < max_idx; idx += blockDim.x) {
    const auto compat_result = idx < num_results ? results[idx] : 0;
    index_t total{};
    BlockScan(temp_storage).ExclusiveSum(compat_result, results[idx], total);
    __syncthreads();
    if (idx < num_results) {
      results[idx] += prefix_sum;
      if (idx == last_result_idx) results[idx] += last_compat_result;
    }
    if (!threadIdx.x) prefix_sum += total;
  }
}

// compact prefix sums into separate indices array
__device__ inline auto compact_indices(const index_t* scan_results,
    const index_t num_results, index_t* indices,
    const index_t last_compat_result) {
  const auto last_result_idx = num_results - 1;
  for (index_t idx{threadIdx.x}; idx < last_result_idx; idx += blockDim.x) {
    if (scan_results[idx] < scan_results[idx + 1]) {  //
      indices[scan_results[idx]] = idx;
    }
  }
  // sync for access to last *scan* result on all threads
  __syncthreads();
  const auto last_scan_result = scan_results[last_result_idx];
  if (!threadIdx.x && last_compat_result) {
    indices[last_scan_result] = last_result_idx;
  }
  // return last computed sum = total number of set flags = total num indices
  return last_scan_result;
}

inline __device__ bool shown = false;

// compact indices from prefix sums in same array
__device__ inline auto compact_indices(index_t* results,
    const index_t num_results, const index_t last_compat_result) {
  __shared__ index_t indices[kBlockSize];
  __shared__ index_t total_indices;
  __shared__ index_t num_indices_this_block;
  if (!threadIdx.x) total_indices = 0;
  const auto last_result_idx = num_results - 1;
  const auto last_scan_result = results[last_result_idx];
  // NOTE: this is almost certainly broken if blockDim.x != kBlockSize
  const auto max_idx = blockDim.x
      * ((num_results + blockDim.x - 1) / blockDim.x);
  for (index_t idx{threadIdx.x}; idx < max_idx; idx += blockDim.x) {
    if (!threadIdx.x) num_indices_this_block = 0;
    __syncthreads();  // sync total_indices, num_indices_this_block
    if (idx < last_result_idx) {
      if (results[idx] < results[idx + 1]) {  //
        if (results[idx] - total_indices >= kBlockSize) {
          printf("block %u results[%u]: %u, total: %u, diff: %u\n", blockIdx.x,
              idx, results[idx], total_indices, results[idx] - total_indices);
        } else
          indices[results[idx] - total_indices] = idx;
      }
      if (threadIdx.x == blockDim.x - 1) {
        num_indices_this_block = results[idx + 1] - total_indices;
      }
    } else if (idx == last_result_idx) {
      num_indices_this_block = results[idx] + last_compat_result
          - total_indices;
    }
    __syncthreads();
    if (threadIdx.x < num_indices_this_block) {
      results[total_indices + threadIdx.x] = indices[threadIdx.x];
      if (!blockIdx.x && !shown) {
        printf("block %u, idx %u, tid %u, total %u, num_this_block %u"
               ", results[%u] = %u\n",
            blockIdx.x, idx, threadIdx.x, total_indices, num_indices_this_block,
            total_indices + threadIdx.x, indices[threadIdx.x]);
      }
    }
    if (!threadIdx.x) total_indices += num_indices_this_block;
  }
#if 1
  __syncthreads();
  if (!blockIdx.x && !threadIdx.x) shown = true;
#endif

#if 0
  if (last_compat_result) {
    ++num_indices;
    if (!threadIdx.x) results[last_scan_result] = last_result_idx;
  }
#endif
  return last_scan_result + last_compat_result;
}

// convert compat results flag array -> prefix sums -> indices
__device__ inline auto compute_compat_uv_indices(
    const index_t num_unique_variations, index_t* in_results_out_indices) {
#ifdef ONE_ARRAY
  const auto results_offset = blockIdx.x * num_unique_variations;
  const auto results = &in_results_out_indices[results_offset];
#else
  const auto results_offset = blockIdx.x * xor_data.variations_results_per_block;
  const auto results = &xor_data.variations_compat_results[results_offset];
#endif
  const auto last_result_idx = num_unique_variations - 1;
  const auto last_compat_result = results[last_result_idx];

  compute_prefix_sums_in_place(results, num_unique_variations);
#ifdef ONE_ARRAY
  auto num_indices = compact_indices_in_place(results, num_unique_variations);
  if (last_compat_result) {
    ++num_indices;
    if (!threadIdx.x) results[num_indices] = last_result_idx;
  }
#else
  const auto indices_offset = blockIdx.x * num_unique_variations;
  const auto indices = &in_results_out_indices[indices_offset];
  const auto num_indices = compact_indices(results, num_unique_variations,
      indices, last_compat_result);
#endif
  return num_indices;
}

  /*
__device__ inline auto compute_compat_uv_indices(
    const index_t num_unique_variations, index_t* dst_uv_indices,
    const index_t num_results_per_block) {
  const auto results_offset = blockIdx.x * num_results_per_block;
  const auto results = &xor_data.variations_compat_results[results_offset];
  const auto last_result_idx = num_unique_variations - 1;
  const auto last_compat_result = results[last_result_idx];
  compute_prefix_sums_in_place(results, num_unique_variations);

  const auto uv_indices_offset = blockIdx.x * num_unique_variations;
  const auto indices = &dst_uv_indices[uv_indices_offset];
  const auto num_indices = compact_indices(results, num_unique_variations,
      indices, last_compat_result);

  return num_indices;
}
  */
  
__device__ bool is_any_OR_source_compatible(
    const SourceCompatibilityData& source, const index_t xor_chunk_idx);

}  // namespace cm
