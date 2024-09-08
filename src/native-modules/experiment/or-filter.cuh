#pragma once
#include <cassert>
#include <cub/block/block_scan.cuh>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/copy.h>
#include "cuda-types.h"
#include "merge-filter-data.h"

namespace cm {

#if 1
//#define LOGGY
//#define XOR_SPANS

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

// TODO: dst_compat_results param maybe
__device__ inline auto compute_variations_compat_results(
    const Variations& src_variations,
    const FilterData::DeviceCommon& tgt_uv_data,
    const index_t num_results_per_block) {
  __shared__ bool any_compat;
  if (!threadIdx.x) any_compat = false;
  __syncthreads();
  const auto results_offset = blockIdx.x * num_results_per_block;
  auto results = &xor_data.variations_compat_results[results_offset];
  // round up to nearest block size
  const auto max_uv_idx = blockDim.x
      * ((tgt_uv_data.num_unique_variations + blockDim.x - 1) / blockDim.x);
  for (auto idx{threadIdx.x}; idx < max_uv_idx; idx += blockDim.x) {
    if (idx < tgt_uv_data.num_unique_variations) {
      const auto compat = UsedSources::are_variations_compatible(src_variations,
          tgt_uv_data.unique_variations[idx].variations);
      results[idx] = compat ? 1 : 0;
      if (compat) any_compat = true;
    } else {
      results[idx] = 0;
    }
  }
  __syncthreads();
  return any_compat;
}

inline __device__ bool shown = false;

__device__ inline auto compute_compat_uv_indices(
    const index_t num_unique_variations, index_t* dst_uv_indices,
    const index_t num_results_per_block) {
  // convert flag array (variations_compat_results) to prefix sums
  // (variations_scan_results)
  const auto results_offset = blockIdx.x * num_results_per_block;
  const auto compat_results =
      &xor_data.variations_compat_results[results_offset];
  const auto scan_results = &xor_data.variations_scan_results[results_offset];
  const auto last_result_idx = num_unique_variations - 1;

  constexpr auto kBlockSize = 64;
  using BlockScan = cub::BlockScan<index_t, kBlockSize>;
  __shared__ typename BlockScan::TempStorage temp_storage;

  const auto max_idx = blockDim.x
      * ((num_unique_variations + blockDim.x - 1) / blockDim.x);
  __shared__ index_t prefix_sum;
  if (!threadIdx.x) prefix_sum = 0;
  for (index_t idx{threadIdx.x}; idx < max_idx; idx += blockDim.x) {
    index_t total;
    BlockScan(temp_storage)
        .ExclusiveSum(compat_results[idx], scan_results[idx], total);
    __syncthreads();
    if (idx < num_unique_variations) {
      scan_results[idx] += prefix_sum;
      if (idx == last_result_idx) scan_results[idx] += compat_results[idx];
    }
    if (!threadIdx.x) prefix_sum += total;
  }
  // generate indices from flag array + prefix sums
  const auto uv_indices_offset = blockIdx.x * num_unique_variations;
  for (index_t idx{threadIdx.x}; idx < num_unique_variations; idx += blockDim.x) {
    if (compat_results[idx]) {
      dst_uv_indices[uv_indices_offset + scan_results[idx]] = idx;
    }
  }
  // sync all threads correct value of scan_results[last_result_idx]
  // there may be other ways to achieve this, e.g. atomic store/load
  __syncthreads();
  // compute total number of set flags (which is total num indices)
  return scan_results[last_result_idx];
}

__device__ bool is_any_OR_source_compatible(
    const SourceCompatibilityData& source, const index_t xor_chunk_idx);

}  // namespace cm
