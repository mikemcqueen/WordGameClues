// filter.cuh

#pragma once
#include "combo-maker.h"
#include "cuda-types.h"

namespace cm {

struct FilterData;
struct StreamData;

void run_get_compatible_sources_kernel(
    const SourceCompatibilityData* device_src_list, unsigned num_sources,
    const UsedSources::SourceDescriptorPair* device_incompatible_src_desc_pairs,
    unsigned num_src_desc_pairs, result_t* device_results);

void copy_filter_data(FilterData& mfd);

void run_filter_kernel(int threads_per_block, StreamData& stream,
    const FilterData& mfd, const SourceCompatibilityData* device_src_list,
    const result_t* device_compat_src_results,
    result_t* device_results, const index_t* device_list_start_indices);

}  // namespace cm
