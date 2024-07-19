#pragma once
#include "combo-maker.h"
#include "cuda-types.h"

namespace cm {

int run_list_pair_compat_kernel(const SourceCompatibilityData* device_sources1,
    const SourceCompatibilityData* device_sources2,
    const index_t* device_indices1, unsigned num_indices1,
    const index_t* device_indices2, unsigned num_indices2,
    result_t* device_compat_results);

int run_get_compat_combos_kernel(uint64_t first_combo, uint64_t num_combos,
    const result_t* device_compat_matrices, unsigned num_compat_matrices,
    const index_t* device_compat_matrix_start_indices,
    const index_t* device_idx_list_sizes, result_t* device_results);

}
