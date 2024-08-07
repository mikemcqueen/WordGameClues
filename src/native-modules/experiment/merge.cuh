#pragma once
#include "cuda-types.h"
#include "merge-type.h"

namespace cm {

struct SourceCompatibilityData;

int run_list_pair_compat_kernel(const SourceCompatibilityData* device_sources1,
    const SourceCompatibilityData* device_sources2,
    const index_t* device_indices1, unsigned num_indices1,
    const index_t* device_indices2, unsigned num_indices2,
    result_t* device_compat_results, MergeType merge_type, cudaStream_t stream,
    bool flag = false);

int run_get_compat_combos_kernel(uint64_t first_idx, uint64_t num_indices,
    const result_t* device_compat_matrices,
    const index_t* device_compat_matrix_start_indices,
    const index_t* device_idx_list_sizes, unsigned num_idx_lists,
    result_t* device_results, cudaStream_t stream, bool flag = false);

}  // namespace cm
