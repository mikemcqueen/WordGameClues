#include <chrono>
#include <iostream>
#include <cuda_runtime.h>
#include "filter-types.h" // index_t, StreamSwarm, should generalize header
#include "merge.h"
#include "peco.h"
#include "combo-maker.h"

namespace {

using namespace cm;

constexpr auto kMaxMatrices = 10u;

__global__ void list_pair_compat_kernel(
  const SourceCompatibilityData* sources1,
  const SourceCompatibilityData* sources2,
  const index_t* src1_indices, unsigned num_src1_indices,
  const index_t* src2_indices, unsigned num_src2_indices,
  result_t* compat_results) {
  // for each source (one block per source)
  for (unsigned idx1{blockIdx.x}; idx1 < num_src1_indices; idx1 += gridDim.x) {
    const auto src1_idx = src1_indices[idx1];
    const auto& src1 = sources1[src1_idx];

    for (unsigned idx2{threadIdx.x}; idx2 < num_src2_indices;
         idx2 += blockDim.x) {
      const auto src2_idx = src2_indices[idx2];
      const auto& src2 = sources2[src2_idx];
      const auto result_idx = idx1 * num_src2_indices + idx2;
      compat_results[result_idx] = src1.isXorCompatibleWith(src2) ? 1 : 0;
    }
  }
}

__global__ void get_compat_combos_kernel(unsigned first_combo, unsigned max_combos,
  const result_t* compat_matrices,
  const index_t* compat_matrix_start_indices,
  MatrixDim* compat_matrix_dims, unsigned num_compat_matrices,
  result_t* results) {
  //
  const unsigned threads_per_grid = gridDim.x * blockDim.x;
  const unsigned thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (unsigned idx{thread_idx}; idx < max_combos; idx += threads_per_grid) {
    index_t offset_indices[kMaxMatrices];
    auto tmp_idx{idx};
    for (int m{(int)num_compat_matrices - 1}; m >= 0; --m) {
      auto matrix_size = compat_matrix_dims[m].rows * compat_matrix_dims[m].columns;
      offset_indices[m] = tmp_idx % matrix_size;
      tmp_idx /= matrix_size;
    }
    bool compatible = true;
    for (unsigned m{}; m < num_compat_matrices; ++m) {
      auto result =
        compat_matrices[compat_matrix_start_indices[m] + offset_indices[m]];
      //if (!compat_matrices[compat_matrix_start_indices[m] + offset_indices[m]]) {
      if (!result) {
        compatible = false;
        break;
      }
      assert(result == 1);
    }
    results[idx] = compatible ? 1 : 0;
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

__global__ void merge_kernel(const SourceCompatibilityData* sources,
  const index_t* list_start_indices, const index_t* flat_indices,
  unsigned row_size, unsigned num_rows, merge_result_t* results) {
  //
  const unsigned threads_per_grid = gridDim.x * blockDim.x;
  const unsigned combos_per_row = row_size * (row_size - 1) / 2;  // nC2 formula
  const unsigned num_combos = num_rows * combos_per_row;
  const unsigned thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned result_idx = thread_idx / combos_per_row;
  if (!(thread_idx % combos_per_row)) {
    results[result_idx] = 0;
  }
  __syncthreads();

  // hack
  //  int i{};

  for (unsigned idx{}; idx < num_combos; idx += threads_per_grid) {
    ComboIndex combo_index;
    get_combo_index(idx, row_size, combos_per_row, combo_index);
    const index_t* row = &flat_indices[combo_index.row_idx * row_size];
    const auto& src1 = sources[list_start_indices[combo_index.elem1_idx]
                               + row[combo_index.elem1_idx]];
    const auto& src2 = sources[list_start_indices[combo_index.elem2_idx]
                               + row[combo_index.elem2_idx]];
    if (src1.isXorCompatibleWith(src2)) {
      // i++;
      atomicAdd(&results[result_idx], 1);
    }
  }
  // TODO: parallel reduce all finished rows within this block
}

}  // namespace

namespace cm {

int run_merge_kernel(cudaStream_t stream, int threads_per_block,
  const SourceCompatibilityData* device_sources,
  const index_t* device_list_start_indices, const index_t* device_flat_indices,
  unsigned row_size, unsigned num_rows, merge_result_t* device_results) {
  //
  int num_sm;
  cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0);
  auto threads_per_sm = 2048;
  auto block_size = threads_per_block ? threads_per_block : 1024;
  auto blocks_per_sm = threads_per_sm / block_size;
  // assert(blocks_per_sm * block_size == threads_per_sm);
  auto grid_size = num_sm * blocks_per_sm;  // aka blocks per grid
  auto shared_bytes = 0;

  //  stream.is_running = true;
  //  stream.sequence_num = StreamData::next_sequence_num();
  //  stream.start_time = std::chrono::high_resolution_clock::now();
  dim3 grid_dim(grid_size);
  dim3 block_dim(block_size);
  cudaStreamSynchronize(cudaStreamPerThread);
  merge_kernel<<<grid_dim, block_dim, shared_bytes, stream>>>(device_sources,
    device_list_start_indices, device_flat_indices, row_size, num_rows,
    device_results);

#if 1 || defined(STREAM_LOG)
  std::cerr << "stream " << 0
            << " started with " << grid_size << " blocks"
            << " of " << block_size << " threads"
            << std::endl;
#endif
  return 0;
}

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

int run_get_compat_combos_kernel(unsigned first_combo, unsigned max_combos,
  const result_t* device_compat_matrices,
  const index_t* device_compat_matrix_start_indices,
  MatrixDim* device_compat_matrix_dims, unsigned num_compat_matrices,
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
    first_combo, max_combos, device_compat_matrices,
    device_compat_matrix_start_indices, device_compat_matrix_dims,
    num_compat_matrices, device_results);
  return 0;
}

}  // namespace cm
