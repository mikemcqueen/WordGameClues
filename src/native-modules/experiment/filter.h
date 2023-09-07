#ifndef INCLUDE_FILTER_H
#define INCLUDE_FILTER_H

#include <utility>
#include <vector>
#include "filter-types.h"

namespace cm {

void filterCandidatesCuda(
  int sum, int threads_per_block, int num_streams, int stride, int iters);

void run_xor_kernel(StreamData& stream, int threads_per_block,
  const SourceCompatibilityData* device_sources, result_t* device_results,
  const index_t* device_list_start_indices);

filter_result_t get_filter_result();

[[nodiscard]] SourceCompatibilityData* cuda_allocCopyXorSources(
  const XorSourceList& xorSourceList);

[[nodiscard]] std::pair<device::OrSourceData*, unsigned>
cuda_allocCopyOrSources(const OrArgList& orArgList);

[[nodiscard]] auto cuda_allocCopySentenceVariationIndices(
  const SentenceVariationIndices& sentenceVariationIndices)
  -> device::VariationIndices*;

[[nodiscard]] index_t* cuda_allocCopyXorSourceIndices(
  const std::vector<index_t> xorSourceIndices);

}  // namespace cm

#endif  // INCLUDE_FILTER_H
