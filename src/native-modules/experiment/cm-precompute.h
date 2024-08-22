#ifndef INCLUDE_CM_PRECOMPUTE_H
#define INCLUDE_CM_PRECOMPUTE_H

#pragma once
#include <vector>
#include "combo-maker.h"
#include "cuda-types.h"
#include "filter-types.h"

namespace cm {

// functions

auto build_src_lists(
    const std::vector<NCDataList>& nc_data_lists) -> std::vector<SourceList>;

auto buildSentenceVariationIndices(const std::vector<SourceList>& xor_src_lists,
    const std::vector<IndexList>& compat_idx_lists,
    const FatIndexList& compat_indices)
    -> SentenceXorVariationIndices;

auto build_OR_variation_indices(
    const UsedSources::VariationsList& variations_list,
    const FatIndexList& compat_indices)
  -> SentenceOrVariationIndices;

}  // namespace cm

#endif // INCLUDE_CM_PRECOMPUTE_H
