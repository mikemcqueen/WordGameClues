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

auto filter_candidates_cuda(const MergeFilterData& mfd, int sum,
    int threads_per_block, int num_streams, int stride, int iters,
    bool synchronous) -> std::optional<SourceCompatibilitySet>;

filter_result_t get_filter_result();

[[nodiscard]] UsedSources::SourceDescriptorPair*
cuda_alloc_copy_source_descriptor_pairs(
    const std::vector<UsedSources::SourceDescriptorPair>& src_desc_pairs,
    cudaStream_t stream);

[[nodiscard]] std::pair<device::OrSourceData*, unsigned>
cuda_allocCopyOrSources(const OrArgList& orArgList, cudaStream_t stream);

[[nodiscard]] auto cuda_allocCopySentenceVariationIndices(
    const SentenceVariationIndices& sentenceVariationIndices,
    cudaStream_t stream) -> device::VariationIndices*;

}  // namespace cm

#endif  // INCLUDE_FILTER_H
