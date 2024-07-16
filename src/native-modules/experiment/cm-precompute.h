#ifndef INCLUDE_CM_PRECOMPUTE_H
#define INCLUDE_CM_PRECOMPUTE_H

#include <unordered_map>
#include <unordered_set>
#include <string>
#include <vector>
#include "combo-maker.h"

namespace cm {

// functions

auto buildSourceListsForUseNcData(const std::vector<NCDataList>& useNcDataLists)
  -> std::vector<SourceList>;

auto buildOrArgList(std::vector<SourceList>&& or_src_lists) -> OrArgList;

#if 0
void markAllXorCompatibleOrSources(OrArgList& or_arg_list,
  const std::vector<SourceList>& xor_src_lists,
  const std::vector<IndexList>& compat_idx_lists,
  const ComboIndexList& compat_indices);
#endif

auto buildSentenceVariationIndices(const std::vector<SourceList>& xor_src_lists,
  const std::vector<IndexList>& compat_idx_lists,
  const ComboIndexList& compat_indices) -> SentenceVariationIndices;
}

#endif // INCLUDE_CM_PRECOMPUTE_H
