#ifndef INCLUDE_CM_PRECOMPUTE_H
#define INCLUDE_CM_PRECOMPUTE_H

#pragma once
#include <vector>
#include "combo-maker.h"
#include "cuda-types.h"

namespace cm {

// functions

auto build_src_lists(
    const std::vector<NCDataList>& nc_data_lists) -> std::vector<SourceList>;

auto build_or_arg_list(const std::vector<SourceList>& or_src_lists) -> OrArgList;

/*
void markAllXorCompatibleOrSources(OrArgList& or_arg_list,
  const std::vector<SourceList>& xor_src_lists,
  const std::vector<IndexList>& compat_idx_lists,
  const ComboIndexList& compat_indices);
*/

auto buildSentenceVariationIndices(const std::vector<SourceList>& xor_src_lists,
  const std::vector<IndexList>& compat_idx_lists,
  const ComboIndexList& compat_indices) -> SentenceVariationIndices;
}

#endif // INCLUDE_CM_PRECOMPUTE_H
