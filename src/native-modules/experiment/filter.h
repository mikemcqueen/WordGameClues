#ifndef INCLUDE_FILTER_H
#define INCLUDE_FILTER_H

#pragma once
#include <utility>
#include <vector>
#include "filter-types.h"
#include "merge-filter-data.h"

namespace cm {

struct FilterArgs {
  int threads_per_block{};
  int streams{};
  int stride{};
  int iters{};
  bool synchronous{};
};

// functions

//auto cuda_markAllXorCompatibleOrSources(const MergeFilterData& mfd)
//    -> std::vector<result_t>;

//unsigned move_marked_or_sources(device::OrSourceData* device_or_src_list,
//    const std::vector<result_t>& mark_results);

//void run_mark_or_sources_kernel(
//    const MergeFilterData& mfd, result_t* device_results);

/*
[[nodiscard]] SourceCompatibilityData* cuda_allocCopyXorSources(
  const XorSourceList& xorSourceList);
*/

// filter-support

auto filter_candidates_cuda(const MergeFilterData& mfd, int sum,
    int threads_per_block, int num_streams, int stride, int iters,
    bool synchronous) -> std::optional<SourceCompatibilitySet>;

filter_result_t get_filter_result();

[[nodiscard]] UsedSources::SourceDescriptorPair*
cuda_alloc_copy_source_descriptor_pairs(
    const std::vector<UsedSources::SourceDescriptorPair>& src_desc_pairs);

[[nodiscard]] SourceCompatibilityData* cuda_alloc_copy_sources(
    const SourceCompatibilitySet& sources);

[[nodiscard]] std::pair<device::OrSourceData*, unsigned>
cuda_allocCopyOrSources(const OrArgList& orArgList);

[[nodiscard]] uint64_t* cuda_alloc_copy_compat_indices(
    const std::vector<uint64_t>& compat_indices);

[[nodiscard]] auto cuda_allocCopySentenceVariationIndices(
    const SentenceVariationIndices& sentenceVariationIndices, cudaStream_t stream)
    -> device::VariationIndices*;

}  // namespace cm

#endif  // INCLUDE_FILTER_H
