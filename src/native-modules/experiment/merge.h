#ifndef INCLUDE_MERGE_H
#define INCLUDE_MERGE_H

#include <vector>
#include <cuda_runtime.h>
#include "peco.h"
#include "combo-maker.h"
#include "filter-types.h"

namespace cm {

using merge_result_t = int;

int getNumEmptySublists(const std::vector<SourceList>& sourceLists);

bool filterAllXorIncompatibleIndices(Peco::IndexListVector& indexLists,
  const std::vector<SourceList>& sourceLists);

int list_size(const Peco::IndexList& indexList);

int64_t vec_product(const std::vector<size_t>& v);

std::string vec_to_string(const std::vector<size_t>& v);

auto cuda_mergeCompatibleXorSourceCombinations(
  const std::vector<SourceList>& sourceLists) -> XorSourceList;

int run_list_pair_compat_kernel(const SourceCompatibilityData* device_sources1,
  const SourceCompatibilityData* device_sources2,
  const index_t* device_indices1, unsigned num_device_indices1,
  const index_t* device_indices2, unsigned num_device_indices2,
  result_t* device_compat_results);

int run_merge_kernel(cudaStream_t stream, int threads_per_block,
  const SourceCompatibilityData* device_sources,
  const index_t* device_list_start_indices, const index_t* device_flat_indices,
  unsigned row_size, unsigned num_rows, merge_result_t* device_results);

}  // namespace cm

#endif  // INCLUDE_MERGE_H
