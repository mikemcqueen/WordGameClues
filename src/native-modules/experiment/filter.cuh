// filter.cuh

#pragma once
#include "combo-maker.h"
#include "cuda-types.h"

namespace cm {

struct MergeFilterData;
struct StreamData;

void run_get_compatible_sources_kernel(
    const SourceCompatibilityData* device_sources, unsigned num_sources,
    const UsedSources::SourceDescriptorPair* device_incompatible_src_desc_pairs,
    unsigned num_src_desc_pairs, compat_src_result_t* device_results);

void run_xor_kernel(StreamData& stream, int threads_per_block,
    const MergeFilterData& mfd, const SourceCompatibilityData* device_sources,
    const compat_src_result_t* device_compat_src_results,
    result_t* device_results, const index_t* device_list_start_indices);

void show_or_arg_counts(unsigned num_or_args);

}  // namespace cm
