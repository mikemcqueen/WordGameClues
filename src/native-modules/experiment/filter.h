#ifndef INCLUDE_FILTER_H
#define INCLUDE_FILTER_H

#include <utility>
#include <vector>
#include "filter-types.h"
#include "merge-filter-data.h"

namespace cm {

// functions

void filterCandidatesCuda(int sum, int threads_per_block, int num_streams,
  int stride, int iters, bool synchronous);

void run_xor_kernel(StreamData& stream, int threads_per_block,
  const SourceCompatibilityData* device_sources, result_t* device_results,
  const index_t* device_list_start_indices);

filter_result_t get_filter_result();

[[nodiscard]] SourceCompatibilityData* cuda_allocCopyXorSources(
  const XorSourceList& xorSourceList);

[[nodiscard]] std::pair<device::OrSourceData*, unsigned>
cuda_allocCopyOrSources(const OrArgList& orArgList);

[[nodiscard]] uint64_t* cuda_alloc_copy_compat_indices(
  const std::vector<uint64_t>& compat_indices);

[[nodiscard]] auto cuda_allocCopySentenceVariationIndices(
  const SentenceVariationIndices& sentenceVariationIndices)
  -> device::VariationIndices*;

}  // namespace cm

#endif  // INCLUDE_FILTER_H
