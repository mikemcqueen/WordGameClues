#pragma once
#include <cassert>
#include "cuda-types.h"

namespace cm {

#if 1
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

inline constexpr auto kSharedIndexCount = 2;
inline constexpr auto kSharedIndexSize = 8;  // in bytes

inline constexpr auto kXorChunkIdx = 0;
inline constexpr auto kDebugIdx = 1;

// num source sentences (uint8_t) starts at end of indices
inline constexpr auto kNumSrcSentences = 2;
// source sentence data (unit8_t) follows for 9 more bytes

// xor_results (result_t) starts after of sentence data, rounded to 8 bytes
inline constexpr auto kXorResults = 4;  // 2 + 16/2

extern __shared__ fat_index_t dynamic_shared[];

namespace tag {

struct XOR {};
struct OR {};

}  // namespace tag

__device__ __forceinline__ auto get_xor_combo_index(
    fat_index_t flat_idx, const FatIndexSpanPair& idx_spans) {
  if (flat_idx < idx_spans.first.size()) return idx_spans.first[flat_idx];
  flat_idx -= idx_spans.first.size();
  // TODO #ifdef ASSERTS
  assert(flat_idx < idx_spans.second.size());
  return idx_spans.second[flat_idx];
}

__device__ __forceinline__ fat_index_t get_flat_idx(
    fat_index_t block_idx, unsigned thread_idx = threadIdx.x) {
  return block_idx * fat_index_t(blockDim.x) + thread_idx;
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

__device__ __forceinline__ auto source_bits_are_XOR_compatible(
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
  return (a.word(word_idx) & b.word(word_idx)) == 0;
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
#if 0
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
  index_t list_start_idx{data.sum_idx_list_sizes};
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
__device__ __forceinline__ void merge_variations(UsedSources::Variations& to,
    const UsedSources::Variations& from, bool force = false) {
  for (int s{}; s < kNumSentences; ++s) {
    if (force || (to[s] == -1)) to[s] = from[s];
  }
}

template <typename T>
__device__ auto build_variations(fat_index_t combo_idx, const T& data) {
  UsedSources::Variations v;
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

__device__ bool is_any_OR_source_compatible(
    const SourceCompatibilityData& source, unsigned xor_chunk_idx,
    const FatIndexSpanPair& xor_idx_spans);

}  // namespace cm
