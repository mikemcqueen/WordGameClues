#pragma once
#include <cassert>
#include "cuda-types.h"

namespace cm {

#define DEBUG_OR_COUNTS
#define DEBUG_XOR_COUNTS
// #define DISABLE_OR
// #define FORCE_XOR_COMPAT
// #define FORCE_ALL_XOR
// #define MAX_SOURCES 2
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

struct XOR {};  // XOR;
struct OR {};   // OR;

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
        if (!are_source_bits_XOR_compatible(other.usedSources.getBits(),
                source.usedSources.getBits(), sentence)) {
          return false;
        }
      }
    }
  }
  return true;
}

/*
__device__ auto is_source_XOR_compatible(const SourceCompatibilityData&
source, const SourceCompatibilityData& or_src) { uint8_t* num_src_sentences
= (uint8_t*)&dynamic_shared[kNumSrcSentences]; uint8_t* src_sentences =
&num_src_sentences[1];

  for (uint8_t idx{}; idx < *num_src_sentences; ++idx) {
    auto sentence = src_sentences[idx];
    // we know all source.variations[s] are > -1. no need to compare bits if
    // or_src.variations[s] == -1.
    if (or_src.usedSources.variations[sentence] > -1) {
      // NB: order of bits params matters here
      if (!source_bits_are_XOR_compatible(or_src.usedSources.getBits(),
              source.usedSources.getBits(), sentence)) {
        return false;
      }
    }
  }
  return true;
}
*/

__device__ bool is_any_OR_source_compatible(
    const SourceCompatibilityData& source, unsigned xor_chunk_idx,
    const FatIndexSpanPair& xor_idx_spans);

}  // namespace cm
