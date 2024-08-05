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

auto filter_candidates_cuda(const FilterData& mfd, int sum,
    int threads_per_block, int num_streams, int stride, int iters,
    bool synchronous) -> std::optional<SourceCompatibilitySet>;

filter_result_t get_filter_result();

[[nodiscard]] auto cuda_alloc_copy_source_descriptor_pairs(
    const std::vector<UsedSources::SourceDescriptorPair>& src_desc_pairs,
    cudaStream_t stream) -> UsedSources::SourceDescriptorPair*;

[[nodiscard]] auto cuda_allocCopySentenceVariationIndices(
    const SentenceVariationIndices& sentenceVariationIndices,
    cudaStream_t stream) -> device::FatVariationIndices*;

[[nodiscard]] auto cuda_alloc_copy_variation_indices(
    const VariationIndices& host_variation_indices,
    cudaStream_t stream) -> device::VariationIndices*;

void alloc_copy_filter_data(FilterData& mfd,
    const UsedSources::VariationsList& or_variations_list, cudaStream_t stream);

template <typename T>
void alloc_copy_start_indices(MergeData::Host& host,
    FilterData::DeviceCommon<T>& device, cudaStream_t stream) {
  auto src_list_start_indices = make_start_indices(host.src_lists);
  device.src_list_start_indices = cuda_alloc_copy_start_indices(
      src_list_start_indices, stream);  // "src_list_start_indices"
  auto idx_list_start_indices = make_start_indices(host.compat_idx_lists);
  device.idx_list_start_indices = cuda_alloc_copy_start_indices(
      idx_list_start_indices, stream);  // "idx_list_start_indices"
}

}  // namespace cm

#endif  // INCLUDE_FILTER_H
