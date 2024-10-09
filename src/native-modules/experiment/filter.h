#ifndef INCLUDE_FILTER_H
#define INCLUDE_FILTER_H

#pragma once
#include <optional>
#include <cuda_runtime.h> // cudaStream_t
#include "cuda-types.h"
#include "filter-types.h"
#include "merge-filter-data.h"

namespace cm {

auto filter_candidates_cuda(FilterData& mfd, const FilterParams& params)  //
    -> std::optional<CompatSourceIndicesSet>;

filter_result_t get_filter_result();

void set_incompatible_sources(FilterData& mfd,
    const CompatSourceIndicesSet& incompat_src_indices, cudaStream_t stream);

void alloc_copy_filter_indices(FilterData& mfd, cudaStream_t stream);

}  // namespace cm

#endif  // INCLUDE_FILTER_H
