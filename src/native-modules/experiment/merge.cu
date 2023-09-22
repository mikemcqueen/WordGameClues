#include <chrono>
#include <iostream>
#include <cuda_runtime.h>
#include "filter-types.h"
#include "merge.h"
#include "peco.h"
#include "combo-maker.h"

namespace {

using namespace cm;

constexpr auto kMaxMatrices = 10u;

// Given two arrays of sources, test xor compatibility of all combinations
// of two sources (pairs), containing one source from each list.
//
// Indices are used because only a subset (as specified by indicee arrays)
// of each source array is actually considered.
//
// compat_results can be thought of as a matrix, with withh num_src1_indices
// rows and num_src2_indices columns.
//
__global__ void list_pair_compat_kernel(
  const SourceCompatibilityData* sources1,
  const SourceCompatibilityData* sources2,
  const index_t* src1_indices, unsigned num_src1_indices,
  const index_t* src2_indices, unsigned num_src2_indices,
  result_t* compat_results) {
  // for each source1 (one block per row)
  for (unsigned idx1{blockIdx.x}; idx1 < num_src1_indices; idx1 += gridDim.x) {
    const auto src1_idx = src1_indices[idx1];
    const auto& src1 = sources1[src1_idx];

    // for each source2 (one thread per column)
    for (unsigned idx2{threadIdx.x}; idx2 < num_src2_indices;
         idx2 += blockDim.x) {
      const auto src2_idx = src2_indices[idx2];
      const auto& src2 = sources2[src2_idx];
      const auto result_idx = idx1 * num_src2_indices + idx2;
      compat_results[result_idx] = src1.isXorCompatibleWith(src2) ? 1 : 0;
    }
  }
}

// Given N compatibility matrices, representing the results of comparing every
// pair of sources arrays (via list_pair_compat_kernel), find all N-tuples of
// of compatible results, representing combinations of compatible sources to
// be merged.
//
__global__ void get_compat_combos_kernel(uint64_t first_combo,
  uint64_t num_combos, const result_t* compat_matrices,
  const index_t* compat_matrix_start_indices, unsigned num_compat_matrices,
  const index_t* idx_list_sizes, result_t* results) {
  //
  const unsigned threads_per_grid = gridDim.x * blockDim.x;
  const unsigned thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (uint64_t idx{thread_idx}; idx < num_combos; idx += threads_per_grid) {
    index_t row_indices[kMaxMatrices];
    auto combo_idx{first_combo + idx};
    for (int i{(int)num_compat_matrices - 1}; i >= 0; --i) {
      const auto idx_list_size = idx_list_sizes[i];
      row_indices[i] = combo_idx % idx_list_size;
      combo_idx /= idx_list_size;
    }
    bool compat = true;
    for (size_t i{}, n{}; compat && (i < num_compat_matrices - 1); ++i) {
      for (size_t j{i + 1}; j < num_compat_matrices; ++j, ++n) {
        auto offset = row_indices[i] * idx_list_sizes[j] + row_indices[j];
        if (!compat_matrices[compat_matrix_start_indices[n] + offset]) {
          compat = false;
          break;
        }
      }
    }
    results[idx] = compat ? 1 : 0;
  }
}

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

}  // namespace

namespace cm {

int run_list_pair_compat_kernel(const SourceCompatibilityData* device_sources1,
  const SourceCompatibilityData* device_sources2,
  const index_t* device_indices1, unsigned num_device_indices1,
  const index_t* device_indices2, unsigned num_device_indices2,
  result_t* device_compat_results) {
  //
  int num_sm;
  cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0);
  auto threads_per_sm = 2048;
  auto block_size = 256;
  auto blocks_per_sm = threads_per_sm / block_size;
  auto grid_size = num_sm * blocks_per_sm;  // aka blocks per grid
  auto shared_bytes = 0;

  dim3 grid_dim(grid_size);
  dim3 block_dim(block_size);
  cudaStream_t stream = cudaStreamPerThread;
  cudaStreamSynchronize(cudaStreamPerThread);
  list_pair_compat_kernel<<<grid_dim, block_dim, shared_bytes, stream>>>(
    device_sources1, device_sources2, device_indices1, num_device_indices1,
    device_indices2, num_device_indices2, device_compat_results);
  return 0;
}

int run_get_compat_combos_kernel(uint64_t first_combo, uint64_t num_combos,
  const result_t* device_compat_matrices,
  const index_t* device_compat_matrix_start_indices,
  unsigned num_compat_matrices, const index_t* device_idx_list_sizes,
  result_t* device_results) {
  //
  assert((num_compat_matrices <= kMaxMatrices)
         && "max compat matrix count exceeded (easy fix)");
  int num_sm;
  cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0);
  auto threads_per_sm = 2048;
  auto block_size = 256;
  auto blocks_per_sm = threads_per_sm / block_size;
  auto grid_size = num_sm * blocks_per_sm;  // aka blocks per grid
  auto shared_bytes = 0;

  dim3 grid_dim(grid_size);
  dim3 block_dim(block_size);
  cudaStream_t stream = cudaStreamPerThread;
  cudaStreamSynchronize(cudaStreamPerThread);
  get_compat_combos_kernel<<<grid_dim, block_dim, shared_bytes, stream>>>(
    first_combo, num_combos, device_compat_matrices,
    device_compat_matrix_start_indices, num_compat_matrices,
    device_idx_list_sizes, device_results);
  return 0;
}

}  // namespace cm
