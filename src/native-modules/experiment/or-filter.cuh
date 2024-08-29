#pragma once
#include <cassert>
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

#define CLOCKS
#define DEBUG_OR_COUNTS
#define DEBUG_XOR_COUNTS
#endif
//#define MAX_SOURCES 1
// #define DISABLE_OR
// #define FORCE_XOR_COMPAT
// #define FORCE_ALL_XOR
// #define USE_LOCAL_XOR_COMPAT
// #define PRINTF

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

__device__ __forceinline__ auto get_xor_combo_index(
    index_t flat_idx, const IndexSpanPair& idx_spans) {
  index_t compat_idx{};
  if (flat_idx < idx_spans.first.size()) {
    compat_idx = idx_spans.first[flat_idx];
  } else {
    flat_idx -= idx_spans.first.size();
    // TODO #ifdef ASSERTS
    assert(flat_idx < idx_spans.second.size());
    compat_idx = idx_spans.second[flat_idx];
  }
  return xor_data.compat_indices[compat_idx];
}

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
    const Variations& xor_variations,
    const FilterData::DeviceUniqueVariations& uv_data,
    const index_t num_results_per_block) {
  __shared__ bool any_compat;
  if (!threadIdx.x) any_compat = false;
  __syncthreads();
  for (auto idx{threadIdx.x}; idx < uv_data.num_unique_variations;
      idx += blockDim.x) {
    const auto& src_variations = uv_data.unique_variations[idx].variations;
    const auto compat =
        UsedSources::are_variations_compatible(xor_variations, src_variations);
    const auto results_offset = blockIdx.x * num_results_per_block;
    xor_data.variations_compat_results[results_offset + idx] = compat ? 1 : 0;
    if (compat) any_compat = true;
  }
  __syncthreads();
  return any_compat;
}

__device__ inline auto compute_compat_uv_indices(
    const index_t num_unique_variations, index_t* dst_uv_indices,
    const index_t num_results_per_block) {
  // convert flag array (variations_compat_results) to prefix sums
  // (variations_scan_results)
  const auto results_offset = blockIdx.x * num_results_per_block;
  const auto d_flags = &xor_data.variations_compat_results[results_offset];
  auto d_scan_output = &xor_data.variations_scan_results[results_offset];
  thrust::device_ptr<const result_t> d_flags_ptr(d_flags);
  thrust::device_ptr<result_t> d_scan_output_ptr(d_scan_output);
  thrust::exclusive_scan(thrust::device, d_flags_ptr,
      d_flags_ptr + num_unique_variations, d_scan_output_ptr);

  // generate indices from flag array + prefix sums
  const auto uv_indices_offset = blockIdx.x * num_unique_variations;
  auto d_indices = &dst_uv_indices[uv_indices_offset];
  thrust::device_ptr<index_t> d_indices_ptr(d_indices);
  thrust::counting_iterator<int> count_begin(0);
  thrust::for_each(thrust::device, count_begin,
      count_begin + num_unique_variations,
      [d_flags, d_scan_output, d_indices] __device__(const index_t idx) {
        if (!threadIdx.x && d_flags[idx]) {
          d_indices[d_scan_output[idx]] = idx;
        }
      });
  // compute total number of set flags (which is total num indices)
  const auto last_idx = num_unique_variations - 1;
  return d_scan_output[last_idx] + d_flags[last_idx];
}

__device__ bool is_any_OR_source_compatible(
    const SourceCompatibilityData& source, index_t xor_chunk_idx,
    const IndexSpanPair& xor_idx_spans);

}  // namespace cm
