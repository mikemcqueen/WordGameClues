#ifndef INCLUDE_FILTER_H
#define INCLUDE_FILTER_H

#pragma once
#include <optional>
//#include <cuda_runtime.h> // cudaStream_t
#include "cuda-types.h"
#include "filter-types.h"
#include "merge-filter-data.h"

namespace cm {

void filter_candidates_cuda(FilterData& mfd, const FilterParams& params);

filter_result_t get_filter_result();

void set_incompatible_sources(FilterData& mfd,
    const CompatSourceIndicesSet& incompat_src_indices, cudaStream_t stream);

void filter_init(FilterData& mfd);
void filter_cleanup();

}  // namespace cm

#endif  // INCLUDE_FILTER_H
