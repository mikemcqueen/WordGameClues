#pragma once
#include <cassert>
#include "cuda-types.h"

namespace cm {

#define DEBUG_OR_COUNTS
// #define DEBUG_XOR_COUNTS
// #define DISABLE_OR
// #define FORCE_XOR_COMPAT
// #define FORCE_ALL_XOR
// #define MAX_SOURCES 2
// #define USE_LOCAL_XOR_COMPAT
// #define PRINTF

class SourceCompatibilityData;

inline const int kXorChunkIdx = 0;
inline const int kOrChunkIdx = 1;
inline const int kXorResultsIdx = 2;
inline const int kNumSharedIndices = 3;

__device__ __forceinline__ /*inline*/ auto get_xor_combo_index(
    fat_index_t flat_idx, const FatIndexSpanPair& idx_spans) {
  if (flat_idx < idx_spans.first.size()) return idx_spans.first[flat_idx];
  flat_idx -= idx_spans.first.size();
  // TODO #ifdef ASSERTS
  assert(flat_idx < idx_spans.second.size());
  return idx_spans.second[flat_idx];
}

__device__ inline fat_index_t get_flat_idx(
    fat_index_t block_idx, unsigned thread_idx = threadIdx.x) {
  return block_idx * fat_index_t(blockDim.x) + thread_idx;
}

__device__ bool is_any_OR_source_compatible(
    const SourceCompatibilityData& source, unsigned xor_chunk_idx,
    const FatIndexSpanPair& xor_idx_spans);

}  // namespace cm
