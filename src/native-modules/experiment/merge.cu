#include <cassert>
#include <chrono>
#include <iostream>
#include <cuda_runtime.h>
#include "combo-maker.h"
#include "merge.cuh"
#include "peco.h"

namespace cm {

namespace {

constexpr auto kMaxMatrices = 10u;

// Given two arrays of sources, test xor compatibility of all combinations
// of source pairs consisting of one source from each list.
//
// Indices are used because only a subset (as specified by indices arrays)
// of each source array is actually considered.
//
// compat_results can be thought of as a matrix, with with num_src1_indices
// rows and num_src2_indices columns.
//
__global__ void list_pair_compat_kernel(const SourceCompatibilityData* sources1,
    const SourceCompatibilityData* sources2, const index_t* src1_indices,
    unsigned num_src1_indices, const index_t* src2_indices,
    unsigned num_src2_indices, result_t* compat_results, MergeType merge_type,
    bool flag) {
  // for each source1 (one block per row)
  for (unsigned idx1{blockIdx.x}; idx1 < num_src1_indices; idx1 += gridDim.x) {
    const auto src1_idx = src1_indices[idx1];
    const auto& src1 = sources1[src1_idx];
    // for each source2 (one thread per column)
    for (unsigned idx2{threadIdx.x}; idx2 < num_src2_indices;
         idx2 += blockDim.x) {
      const auto src2_idx = src2_indices[idx2];
      const auto& src2 = sources2[src2_idx];
      bool compat{};
      if (merge_type == MergeType::XOR) {
        compat = src1.isXorCompatibleWith(src2);
      } else {  // MergeType::OR
        compat = src1.hasCompatibleVariationsWith(src2);
      }
      // aka: row * num_cols_per_row + col
      const auto result_idx = idx1 * num_src2_indices + idx2;
      compat_results[result_idx] = compat ? 1 : 0;
    }
  }
}
  ;
// Given N compatibility matrices, representing the results of comparing every
// pair of sources arrays (via list_pair_compat_kernel), find all N-tuples of
// of compatible results, representing combinations of compatible sources to
// be merged.
//
// The logic in this function is dependent on the order that pairs are fed to
// list_pair_compat_kernel via for_each_list_pair(), and the order of,,, ??
// both of which are intentional and important, and ensure that the mapping of
// a "flat" (linear) index to a particular combination of index lists is in a
// specific order. There is logic outside of this function that depends on this
// order.
//
// Ex: for the 2 index lists: [[0, 1], [0, 1, 2]], the flat-index order
// of their combinations is:
//
// 0:[0, 0], 1:[0, 1], 2:[0, 2], 3:[1,0], 4:[1, 1], 5:[1,2]
//
// Ex: list-pair order for 4 matrices stored in compat_matrics:
//
// [2,3],[1,3],[0,3],[1,2],[0,2],[0,1]
//
__global__ void get_compat_combos_kernel(uint64_t first_idx,
    uint64_t num_indices, const result_t* compat_matrices,
    const index_t* compat_matrix_start_indices, const index_t* idx_list_sizes,
    unsigned num_idx_lists, result_t* results, bool flag) {
  const auto grid_size = gridDim.x * blockDim.x;
  const auto thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (auto idx{thread_idx}; idx < num_indices; idx += grid_size) {
    bool compat{true};
    auto col_flat_idx = first_idx + idx;
    for (int col_list_idx{int(num_idx_lists) - 1}, matrix_idx{};
        compat && (col_list_idx > 0); --col_list_idx) {
      const auto col_list_size = idx_list_sizes[col_list_idx];
      const auto col = col_flat_idx % col_list_size;
      col_flat_idx /= col_list_size;
      auto row_flat_idx = col_flat_idx;
      for (auto row_list_idx{col_list_idx - 1}; row_list_idx >= 0;
          --row_list_idx, ++matrix_idx) {
        const auto row_list_size = idx_list_sizes[row_list_idx];
        const auto row = row_flat_idx % row_list_size;
        const auto offset = row * col_list_size + col;
        const auto matrix_start_idx = compat_matrix_start_indices[matrix_idx];
        if (!compat_matrices[matrix_start_idx + offset]) {
          compat = false;
          break;
        }
        row_flat_idx /= row_list_size;
      }
    }
    results[idx] = compat ? 1 : 0;
  }
}

/*
struct ComboIndex {
  index_t row_idx;    // row index
  index_t elem1_idx;  // first element of the combination
  index_t elem2_idx;  // second element of the combination
};

__device__ void get_combo_index(unsigned idx, unsigned row_size,
  unsigned combos_per_row, ComboIndex& result) {
  // Calculate sublist idx
  result.row_idx = idx / combos_per_row;
  // Calculate the remainder to determine the combination within the sublist
  unsigned r = idx % combos_per_row;
  // Find the combination (elem1_idx, elem2_idx) based on r
  result.elem1_idx = 0;
  while (r >= (row_size - result.elem1_idx - 1)) {
    r -= (row_size - result.elem1_idx - 1);
    result.elem1_idx++;
  }
  result.elem2_idx = result.elem1_idx + r + 1;
}
*/

}  // anonymous namespace

int run_list_pair_compat_kernel(const SourceCompatibilityData* device_sources1,
    const SourceCompatibilityData* device_sources2,
    const index_t* device_indices1, unsigned num_device_indices1,
    const index_t* device_indices2, unsigned num_device_indices2,
    result_t* device_compat_results, MergeType merge_type, cudaStream_t stream,
    bool flag) {
  int num_sm;
  cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0);
  int threads_per_sm;
  cudaDeviceGetAttribute(&threads_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor, 0);
  auto block_size = 256; // should be variable probably
  auto blocks_per_sm = threads_per_sm / block_size;
  assert(blocks_per_sm * block_size == threads_per_sm);
  auto grid_size = num_sm * blocks_per_sm;  // aka blocks per grid
  auto shared_bytes = 0;

  dim3 grid_dim(grid_size);
  dim3 block_dim(block_size);
  list_pair_compat_kernel<<<grid_dim, block_dim, shared_bytes, stream>>>(
      device_sources1, device_sources2, device_indices1, num_device_indices1,
      device_indices2, num_device_indices2, device_compat_results, merge_type,
      flag);
  return 0;
}

int run_get_compat_combos_kernel(uint64_t first_idx, uint64_t num_indices,
    const result_t* device_compat_matrices,
    const index_t* device_compat_matrix_start_indices,
    const index_t* device_idx_list_sizes, unsigned num_idx_lists,
    result_t* device_results, cudaStream_t stream, bool flag) {
  assert((num_idx_lists <= kMaxMatrices)
         && "max compat matrix count exceeded (easy fix)");
  int num_sm;
  cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0);
  int threads_per_sm;
  cudaDeviceGetAttribute(&threads_per_sm, cudaDevAttrMaxThreadsPerMultiProcessor, 0);
  auto block_size = 256;
  auto blocks_per_sm = threads_per_sm / block_size;
  assert(blocks_per_sm * block_size == threads_per_sm);
  auto grid_size = num_sm * blocks_per_sm;  // aka blocks per grid
  auto shared_bytes = 0;

  dim3 grid_dim(grid_size);
  dim3 block_dim(block_size);
  get_compat_combos_kernel<<<grid_dim, block_dim, shared_bytes, stream>>>(
      first_idx, num_indices, device_compat_matrices,
      device_compat_matrix_start_indices, device_idx_list_sizes, num_idx_lists,
      device_results, flag);
  return 0;
}

}  // namespace cm
