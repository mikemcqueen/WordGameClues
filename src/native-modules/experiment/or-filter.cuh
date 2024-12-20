#pragma once
#include <cassert>
#include <cub/block/block_scan.cuh>
#include "combo-maker.h"
#include "cuda-types.h"
#include "filter-stream.h"
#include "merge-filter-data.h"

namespace cm {

//#define ONE_ARRAY

#if 0
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
inline constexpr auto kSrcListIdx = 5;  // src_idx.listIndex
inline constexpr auto kSrcIdx = 6;      // src_idx.index
inline constexpr auto kSwarmIdx = 7;
inline constexpr auto kStreamIdx = 8;
inline constexpr auto kDebugIdx = 9;
inline constexpr auto kSharedIndexCount = 16;  // for SourceCompatData alignment

// "The Source" that is being tested by current block in filter_kernel
inline constexpr auto kTheSourceIdx = kSharedIndexCount;

// xor_results (result_t) starts after source
inline constexpr auto kXorResultsIdx = kTheSourceIdx
    + (sizeof(SourceCompatibilityData) / kSharedIndexSize);

namespace tag {

struct XOR {};
struct OR {};

}  // namespace tag

extern __shared__ index_t dynamic_shared[];

extern __constant__ FilterData::DeviceXor xor_data;
extern __constant__ FilterData::DeviceOr or_data;
extern __constant__ FilterSwarmData::Device swarm_data_[kMaxSwarms];
extern __constant__ FilterStreamData::Device stream_data_[kMaxStreams];
// also declared extern in filter.cuh, required in filter-support.cpp
// extern __constant__ SourceCompatibilityData* sources_data[32];

__device__ __forceinline__ auto& source() {
  return reinterpret_cast<SourceCompatibilityData&>(
      dynamic_shared[kTheSourceIdx]);
}

__device__ __forceinline__ auto& swarm_data() {
  return swarm_data_[dynamic_shared[kSwarmIdx]];
}

__device__ __forceinline__ auto& stream_data() {
  return stream_data_[dynamic_shared[kStreamIdx]];
}

__device__ __forceinline__ index_t get_flat_idx(index_t block_idx,
    index_t thread_idx = threadIdx.x) {
  return block_idx * blockDim.x + thread_idx;
}

#if 0
__device__ __forceinline__ auto are_source_bits_OR_compatible(
    const UsedSources::SourceBits& other_bits, int word_idx) {
  const auto& src_bits = source().usedSources.getBits();
  const auto other_word = other_bits.word(word_idx);
  const auto w = src_bits.word(word_idx) & other_word;
  if (w && (w != other_word)) return false;
  return true;
}

__device__ __forceinline__ auto are_source_bits_XOR_compatible(
    const UsedSources::SourceBits& a, const UsedSources::SourceBits& b,
    int word_idx) {
  return (a.word(word_idx) & b.word(word_idx)) == 0u;
}

// each thread calls this with a different "other" source
template <typename TagT>
requires /*std::is_same_v<TagT, tag::XOR> ||*/ std::is_same_v<TagT, tag::OR>
__device__ inline auto are_source_bits_compatible_with(
    const SourceCompatibilityData& other) {
  const uint8_t* num_sentences = (uint8_t*)&dynamic_shared[kNumSentencesIdx];
  const uint8_t* sentences = &num_sentences[1];

  // TODO: i could just increment 'sentences++' here instead of indexing
  for (uint8_t idx{}; idx < *num_sentences; ++idx) {
    const auto sentence = sentences[idx];
    // we know all source.variations[s] == NoVariation. no need to compare bits
    // if other.variations[s] == NoVariation.
    if (other.usedSources.variations[sentence] != NoVariation) {
      if constexpr (std::is_same_v<TagT, tag::OR>) {
        // NB: order of params matters here (or_src first)
        if (!are_source_bits_OR_compatible(other.usedSources.getBits(),
                sentence)) {
          return false;
        }
      } else if constexpr (std::is_same_v<TagT, tag::XOR>) {
        if (!are_source_bits_XOR_compatible(other.usedSources.getBits(),
                source().usedSources.getBits(), sentence)) {
          return false;
        }
      }
    }
  }
  return true;
}
#else

template <typename T>
__device__ __forceinline__ auto are_words_OR_compatible(const T a, const T b) {
  const auto w = a & b;
  if (w && (w != a)) return false;
  return true;
}

inline __device__ auto are_source_bits_OR_compatible_with(
    const UsedSources::SourceBits& other_bits) {
  const auto& bits = source().usedSources.getBits();
  if (!are_words_OR_compatible(other_bits.quad_word(0), bits.quad_word(0)))
    return false;
  if (!are_words_OR_compatible(other_bits.quad_word(1), bits.quad_word(1)))
    return false;
  if (!are_words_OR_compatible(other_bits.word(8), bits.word(8))) return false;
  return true;
}

// each thread calls this with a different "other" source
template <typename TagT>
requires /*std::is_same_v<TagT, tag::XOR> ||*/ std::is_same_v<TagT, tag::OR>
__device__ __forceinline__ auto are_source_bits_compatible_with(
    const SourceCompatibilityData& other) {
  if constexpr (std::is_same_v<TagT, tag::OR>) {
    if (!are_source_bits_OR_compatible_with(other.usedSources.getBits()))
      return false;
  } else if constexpr (std::is_same_v<TagT, tag::XOR>) {
    return false;
  }
  return true;
}
#endif
  
// With all sources identified by combo_idx
template <typename TagT, typename T>
requires /*std::is_same_v<TagT, tag::XOR> ||*/ std::is_same_v<TagT, tag::OR>
__device__ auto are_source_bits_compatible_with_all(fat_index_t combo_idx,
    const T& data) {
  for (int list_idx{int(data.num_idx_lists) - 1}; list_idx >= 0;
      --list_idx) {
    const auto& other = data.get_source(combo_idx, list_idx);
    if (!are_source_bits_compatible_with<TagT>(other)) return false;
    combo_idx /= data.idx_list_sizes[list_idx];
  }
  return true;
}

__device__ __forceinline__ auto are_source_bits_OR_compatible_with_all(
    fat_index_t combo_idx) {
  return are_source_bits_compatible_with_all<tag::OR>(combo_idx, or_data);
}

template <typename T>
__device__ auto check_src_compat_results(fat_index_t combo_idx, const T& data) {
  const auto block_start_idx = blockIdx.x * data.num_src_compat_results;
  auto list_start_idx = data.num_src_compat_results;
  for (int list_idx{int(data.num_idx_lists) - 1}; list_idx >= 0; --list_idx) {
    const auto list_size = data.idx_list_sizes[list_idx];
    const auto src_idx = combo_idx % list_size;
    assert(list_start_idx >= list_size);
    list_start_idx -= list_size;
    if (!stream_data().or_src_bits_compat_results[block_start_idx
            + list_start_idx + src_idx])
      return false;
    combo_idx /= list_size;
  }
  return true;
}
  
// faster than UsedSources version
__device__ __forceinline__ void merge_variations_unchecked(
    Variations& to, const Variations& from, bool force = false) {
  for (int s{}; s < kNumSentences; ++s) {
    if (force || (to[s] == NoVariation)) to[s] = from[s];
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
    merge_variations_unchecked(v, src.usedSources.variations, force);
    force = false;
  }
  return v;
}

/* debugging */
__device__ inline void dump_array(const index_t* results,
    const index_t num_results) {
  for (index_t idx{}; idx < num_results; ++idx) {
    printf("block %u, results[%u] = %u\n", blockIdx.x, idx, results[idx]);
  }
}

__device__ inline auto compute_variations_compat_results(
    const Variations& src_variations,
    const FilterData::DeviceCommon& tgt_uv_data, index_t* compat_results) {
  __shared__ bool any_compat;
  if (!threadIdx.x) any_compat = false;
  __syncthreads();
  const auto num_results = tgt_uv_data.num_unique_variations;
#ifdef ONE_ARRAY
  const auto results_offset = blockIdx.x * num_results;
  const auto results = &compat_results[results_offset];
#else
  const auto results_offset = blockIdx.x
      * stream_data().num_variations_results_per_block;
  const auto results = &stream_data().variations_compat_results[results_offset];
#endif
  for (auto idx{threadIdx.x}; idx < num_results; idx += blockDim.x) {
    const auto compat = UsedSources::are_variations_compatible(src_variations,
        tgt_uv_data.unique_variations[idx].variations);
    results[idx] = compat ? 1u : 0u;
    if (compat) any_compat = true;
  }
  __syncthreads();
  return any_compat;
}

constexpr unsigned kBlockSize = 64u;

// compute prefix sums from compat results flag array
__device__ inline auto compute_prefix_sums_in_place(index_t* results,
    const index_t num_results) {
  const auto last_result_idx = num_results - 1;
  const auto last_compat_result = results[last_result_idx];

  using BlockScan = cub::BlockScan<index_t, kBlockSize>;
  __shared__ typename BlockScan::TempStorage temp_storage;

  // or: num_results + blockDim.x - (num_results % blockDim.x)
  // or: copy block_size_bits to constant memory and use shift left/right
  const auto max_idx = blockDim.x
      * ((num_results + blockDim.x - 1) / blockDim.x);
  __shared__ index_t cumulative_total;
  __shared__ index_t last_total;
  if (!threadIdx.x) cumulative_total = 0;
  __syncthreads();
  for (index_t idx{threadIdx.x}; idx < max_idx; idx += blockDim.x) {
    const auto compat_result = idx < num_results ? results[idx] : 0u;
    index_t scan_result;
    index_t total;
    BlockScan(temp_storage).ExclusiveSum(compat_result, scan_result, total);
    if (!threadIdx.x) {
      if (idx > 0) cumulative_total += last_total;
      last_total = total;
    }
    __syncthreads();
    if (idx < num_results) {
      results[idx] = cumulative_total + scan_result;
    }
  }
  // ensure global results array updates are visible to all threads
  // volatile reads would also achieve this, presumably.
  __syncthreads();
  return cumulative_total + last_total;
}

// compact prefix sums into separate indices array
__device__ inline void compact_sums_to_indices(const index_t* prefix_sums,
    const index_t num_prefix_sums, index_t aggregate_sum, index_t* indices) {
  // determine whether a prefix sum value represents what was originally a "1"
  // compat result by comparing it to the next prefix sum value.
  const auto last_idx = num_prefix_sums - 1;
  for (index_t idx{threadIdx.x}; idx < last_idx; idx += blockDim.x) {
    if (prefix_sums[idx] < prefix_sums[idx + 1]) {
      indices[prefix_sums[idx]] = idx;
    }
  }
  // we can't compare the last prefix sum value to the next... because it's the
  // last! but, if it's (exactly 1) less than the aggregate total sum, we know
  // it too was originally a "1" compat result.
  if (!threadIdx.x) {
    const auto last_prefix_sum = prefix_sums[last_idx];
    if (last_prefix_sum < aggregate_sum) {
      assert(last_prefix_sum + 1 == aggregate_sum);
      indices[last_prefix_sum] = last_idx;
    }
  }
}

// compact indices from prefix sums in-place (broken probably, not in use)
__device__ inline auto compact_indices_in_place(index_t* results,
    const index_t num_results, const index_t last_compat_result) {
  __shared__ index_t indices[kBlockSize];
  __shared__ index_t total_indices;
  __shared__ index_t last_idx;
  __shared__ index_t first_idx;
  if (!threadIdx.x) {
    last_idx = 0;
    first_idx = 0;
    total_indices = 0;
  }
  const auto last_result_idx = num_results - 1;
  const auto last_scan_result = results[last_result_idx];
  // NOTE: this is almost certainly broken if blockDim.x != kBlockSize
  assert(blockDim.x == kBlockSize);  // TODO: move this to run_kernel()
  const auto max_idx = blockDim.x
      * ((num_results + blockDim.x - 1) / blockDim.x);
  for (index_t idx{threadIdx.x}; idx < max_idx; idx += blockDim.x) {
    __syncthreads();
    // copy up to one block of indices to shared memory for this iteration
    if (idx < last_result_idx) {
      if (results[idx] < results[idx + 1]) {
#if 1
        if ((results[idx] < first_idx)
            || (results[idx] - first_idx >= blockDim.x)) {
          printf("idx %u, blockDim.x %u, num_results %u, max %u, results[idx] "
                 "%u, first_idx %u, diff %u\n",
              idx, blockDim.x, num_results, max_idx, results[idx], first_idx,
              results[idx] - first_idx);
        }
#endif
        indices[results[idx] - first_idx] = idx;
      }
      if (threadIdx.x == blockDim.x - 1) { //
        last_idx = results[idx + 1];
      }
    } else if (idx == last_result_idx) {
      indices[results[idx] - first_idx] = idx;
      last_idx = results[idx];
    }
    __syncthreads();
    const auto num_indices = last_idx - total_indices;
    // copy from shared memory back to results array
    if (threadIdx.x < num_indices) {
      results[total_indices + threadIdx.x] = indices[threadIdx.x];
    }
    if (!threadIdx.x) {
      total_indices += num_indices;
      // set first index for next iter to last index this iter
      first_idx = last_idx;
    }
  }

#if 0
  __syncthreads();
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
  const auto results_offset = blockIdx.x
      * stream_data().num_variations_results_per_block;
  const auto results = &stream_data().variations_compat_results[results_offset];
#endif

  // should get a "unused local" with ONE_ARRAY defined. good. fix it.
  const auto aggregate_sum = compute_prefix_sums_in_place(results,
      num_unique_variations);

#ifdef ONE_ARRAY
  // TODO: don't need to pass in as parameter; can be computed in function
  // also TODO: i think using last_compat_result is wrong. the element at
  // that index is actually a sum, not a "flag" (0 or 1). we're treating
  // it as a flag in compute_indices_in_place(). need to do some kind of
  // computation using the total sum (return value of compute_prefix_sums
  // _in_place) similar to how compat_indices() is doing it.
  const auto last_compat_result = results[num_unique_variations - 1];
  // unused variable
  auto num_indices = compact_indices_in_place(results, num_unique_variations,
      last_compat_result);
#else
  const auto indices_offset = blockIdx.x * num_unique_variations;
  const auto indices = &in_results_out_indices[indices_offset];
  compact_sums_to_indices(results, num_unique_variations, aggregate_sum,
      indices);
#endif
  return aggregate_sum;
}

__device__ bool is_any_OR_source_compatible();

}  // namespace cm
