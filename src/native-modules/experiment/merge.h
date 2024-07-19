#ifndef INCLUDE_MERGE_H
#define INCLUDE_MERGE_H

#pragma once
#include <vector>
//#include <cuda_runtime.h>
//#include "peco.h"
#include "combo-maker.h"
//#include "cuda-types.h"
#include "merge-filter-data.h"

namespace cm {

  //using merge_result_t = int;

  /*
// merge.cu

int run_list_pair_compat_kernel(const SourceCompatibilityData* device_sources1,
    const SourceCompatibilityData* device_sources2,
    const index_t* device_indices1, unsigned num_indices1,
    const index_t* device_indices2, unsigned num_indices2,
    result_t* device_compat_results);

int run_get_compat_combos_kernel(uint64_t first_combo, uint64_t num_combos,
    const result_t* device_compat_matrices, unsigned num_compat_matrices,
    const index_t* device_compat_matrix_start_indices,
    const index_t* device_idx_list_sizes, result_t* device_results);
  */

  /*
auto cuda_get_compat_xor_src_indices(const std::vector<SourceList>& src_lists,
    const SourceCompatibilityData* device_src_lists,
    const std::vector<IndexList>& idx_lists, const index_t* device_idx_lists,
    const index_t* device_idx_list_sizes) -> std::vector<uint64_t>;

auto get_compatible_indices(
    const std::vector<SourceList>& src_lists) -> std::vector<IndexList>;

auto xor_merge_sources(const std::vector<SourceList>& src_lists,
    const std::vector<IndexList>& idx_lists,
    const std::vector<uint64_t>& combo_indices) -> XorSourceList;
  */

// merge-support.cpp

auto get_merge_data(const std::vector<SourceList>& src_lists,
    MergeData::Host& host, MergeData::Device& device, bool merge_only) -> bool;

auto merge_xor_compatible_src_lists(
    const std::vector<SourceList>& src_lists) -> SourceList;

/*
SourceCompatibilityData* alloc_copy_src_lists(
  const std::vector<SourceList>& src_lists, size_t* num_bytes = nullptr);

index_t* alloc_copy_idx_lists(
  const std::vector<IndexList>& idx_lists, size_t* num_bytes = nullptr);

index_t* alloc_copy_list_sizes(
  const std::vector<index_t>& list_sizes, size_t* num_bytes = nullptr);
*/

}  // namespace cm

#endif  // INCLUDE_MERGE_H
