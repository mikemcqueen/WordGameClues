#pragma once
#include <cassert>
//#include <cooperative_groups.h>
#include <cub/block/block_scan.cuh>
#include <cuda/atomic>
#include "cuda-types.h"
#include "merge-filter-data.h"

namespace cm {

//namespace cg = cooperative_groups;

#define ONE_ARRAY

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
inline constexpr auto kSrcListIdx = 5;
inline constexpr auto kSrcIdx = 6;
inline constexpr auto kDebugIdx = 7;
inline constexpr auto kSharedIndexCount = 8;

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

/* debugging */
__device__ inline void copy_array(const index_t* src_results,
    const index_t num_results, index_t* dst_results,
    const index_t dst_num_items_per_block) {
  const auto dst_offset = blockIdx.x * dst_num_items_per_block;
  const auto dst = &dst_results[dst_offset];
  for (auto idx{threadIdx.x}; idx < num_results; idx += blockDim.x) {
    dst[idx] = src_results[idx];
  }
}

__device__ inline void dump_array(const index_t* results,
    const index_t num_results) {
  for (index_t idx{}; idx < num_results; ++idx) {
    printf("block %u, results[%u] = %u\n", blockIdx.x, idx, results[idx]);
  }
}

__device__ inline void dump_array(const index_t* results,
    const index_t num_results,
    const index_t num_items_per_block) {
  const auto offset = blockIdx.x * num_items_per_block;
  return dump_array(&results[offset], num_results);
}

__device__ inline bool record_done = false;

__device__ inline auto record(bool enabled = true) {
  if (!enabled) record_done = true;
  if (record_done) return false;

  bool record = false;
  switch (dynamic_shared[kSrcListIdx]) {
#if 0
  case 1419:
    if (dynamic_shared[kSrcIdx] == 49337) record = true;
    break;
  case 1420:
    if (dynamic_shared[kSrcIdx] == 48892) record = true;
    break;
  case 2883:
    if (dynamic_shared[kSrcIdx] == 62243) record = true;
    break;
  case 1839:
    if (dynamic_shared[kSrcIdx] == 53636) record = true;
    break;
  case 2828:
    if (dynamic_shared[kSrcIdx] == 63986) record = true;
    break;
  case 2943:
    if (dynamic_shared[kSrcIdx] == 58378) record = true;
    break;
#endif

#if 0
  case 11149:
    if (dynamic_shared[kSrcIdx] == 3958) record = true;
    break;
  case 11144:;
    if (dynamic_shared[kSrcIdx] == 4210) record = true;
    break;
  case 7385:;
    if (dynamic_shared[kSrcIdx] == 4239) record = true;
    break;
  case 6533:;
    if (dynamic_shared[kSrcIdx] == 27937) record = true;
    break;
  case 5585:;:
    if (dynamic_shared[kSrcIdx] == 41251) record = true;
    break;
#endif

  case 2943:
    if (dynamic_shared[kSrcIdx] == 58378) record = true;
    break;
#if 0
  case 1839:
    if (dynamic_shared[kSrcIdx] == 53636) record = true;
#endif
    break;

  default:
    break;
  }
  return record;
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
  const auto results_offset = blockIdx.x * xor_data.variations_results_per_block;
  const auto results = &xor_data.variations_compat_results[results_offset];
#endif
#if 1
  if (!threadIdx.x && record()) { printf("num_results: %u\n", num_results); }
#endif
  for (auto idx{threadIdx.x}; idx < num_results; idx += blockDim.x) {
    const auto compat = UsedSources::are_variations_compatible(src_variations,
        tgt_uv_data.unique_variations[idx].variations);
    results[idx] = compat ? 1u : 0u;
#if 0
    if (record()) {
      printf("idx %u results[0]: %u\n", idx, results[0]);
    }
#endif
    if (compat) any_compat = true;
  }
  __syncthreads();
#if 1
  if (!threadIdx.x && record()) {
#if 1
    if (blockIdx.x > 0) {
      printf("15 elems prior/overlapping flag array %u\n", dynamic_shared[kSrcListIdx]);
      dump_array(&compat_results[(blockIdx.x - 1) * num_results + 350], 15);
    }
#endif
    printf("flag array %u\n", dynamic_shared[kSrcListIdx]);
    dump_array(results, num_results);
  }
#endif
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

  // or: num_results + blockDim.x - (num_results % blockDim.x)
  // or: copy block_size_bits to constant memory and use shift left/right
  const auto max_idx = blockDim.x
      * ((num_results + blockDim.x - 1) / blockDim.x);
  __shared__ index_t prefix_sum;
  __shared__ index_t last_total;
  if (!threadIdx.x) prefix_sum = 0;
  for (index_t idx{threadIdx.x}; idx < max_idx; idx += blockDim.x) {
    const auto compat_result = idx < num_results ? results[idx] : 0;
    index_t scan_result;
    index_t total;
    BlockScan(temp_storage).ExclusiveSum(compat_result, scan_result, total);
    if (!threadIdx.x) {
      if (idx > 0) prefix_sum += last_total;
      last_total = total;
    }
    __syncthreads();
    if (idx < num_results) {
      results[idx] = prefix_sum + scan_result;
      //if (idx == last_result_idx) results[idx] += last_compat_result;
    }
  }
#if 0
  // if num_indices is an even  multiple of block_size, fixup the last result
  if ((threadIdx.x == blockDim.x - 1) && (num_indices == max_idx)) {
    results[idx] += last_compat_result;
  }
#endif

#if 1
  __syncthreads();
  if (!threadIdx.x) {
    if (record()) {
      printf("prefix sums %u\n", dynamic_shared[kSrcListIdx]);
      dump_array(results, num_results);
    }
  }
#endif
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

__device__ inline int shown = 0;

// compact indices from prefix sums in-place
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
  const auto max_idx = blockDim.x
      * ((num_results + blockDim.x - 1) / blockDim.x);
  for (index_t idx{threadIdx.x}; idx < max_idx; idx += blockDim.x) {
    __syncthreads();

#if 1
    if (!threadIdx.x) {
      if (record()) {
        printf("indices round %u: %u, idx %u, first %u, total %u \n",
            idx / blockDim.x, dynamic_shared[kSrcListIdx], idx, first_idx,
            total_indices);
        dump_array(results, num_results);
      }
    }
#endif

    // copy up to one block of indices to shared memory for this iteration
    if (idx < last_result_idx) {
      if (results[idx] < results[idx + 1]) {
        if (results[idx] - first_idx >= kBlockSize) {

          if (1 || record()) {
            printf("list %u idx %u OOR: results[%u](%u) - %u = %u\n",
                dynamic_shared[kSrcListIdx], dynamic_shared[kSrcIdx], idx,
                results[idx], first_idx, results[idx] - first_idx);
          }

        } else {
          indices[results[idx] - first_idx] = idx;
        }
      }
      if (threadIdx.x == blockDim.x - 1) { //
        last_idx = results[idx + 1];

        if (last_idx > num_results) {
          printf("ONE: list %u, last_idx %u\n", dynamic_shared[kSrcListIdx],
              last_idx);
        }

      }
    } else if (idx == last_result_idx) {
      indices[results[idx] - first_idx] = idx;
      last_idx = results[idx];
      if (last_compat_result) { // (num_results even multiple of block_size)
        //last_idx++;
        //if (idx == last_result_idx) results[idx] += last_compat_result;
      }

      if (last_idx > num_results) {
        printf("TWO results[%u]: %u, first_idx %u, last_idx %u, "
               "last_compat_result %u\n",
            idx, results[idx], first_idx, last_idx, last_compat_result);
        dump_array(results + first_idx, last_idx - first_idx);
      }

    }
    __syncthreads();
    const auto num_indices = last_idx - total_indices;
    // copy from shared memory back to results array
    if (threadIdx.x < num_indices) {
      if (total_indices + threadIdx.x >= num_results) {

#if 0
        printf("block %u OOR tid %u total_idx %u sum %u last %u num_idx %u\n",
            blockIdx.x, threadIdx.x, total_indices, threadIdx.x + total_indices,
            last_idx, num_indices);
#endif

      } else {
        results[total_indices + threadIdx.x] = indices[threadIdx.x];
      }
    }
    if (!threadIdx.x) {
      total_indices += num_indices;
      // set first index for next iter to last index this iter
      first_idx = last_idx;
    }
  }

#if 1
  __syncthreads();
  if (!threadIdx.x && record()) {
    printf("list %u count %u\n", dynamic_shared[kSrcListIdx],
        last_scan_result + last_compat_result);
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
  auto num_indices = compact_indices_in_place(results, num_unique_variations,
      last_compat_result);
#else
  const auto indices_offset = blockIdx.x * num_unique_variations;
  const auto indices = &in_results_out_indices[indices_offset];
  const auto num_indices = compact_indices(results, num_unique_variations,
      indices, last_compat_result);
#endif

  if (!threadIdx.x && record()) {
    printf("DONE - %u\n", dynamic_shared[kSrcListIdx]);
    record(false);
  }

  return num_indices;
}

__device__ bool is_any_OR_source_compatible(
    const SourceCompatibilityData& source, const index_t xor_chunk_idx);

}  // namespace cm
