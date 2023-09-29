#ifndef INCLUDE_CM_PRECOMPUTE_H
#define INCLUDE_CM_PRECOMPUTE_H

#include <unordered_map>
#include <unordered_set>
#include <string>
#include <vector>
#include "combo-maker.h"

namespace cm {

// functions

auto buildSourceListsForUseNcData(const std::vector<NCDataList>& useNcDataLists
  /*,const SourceListMap& sourceListMap*/) -> std::vector<SourceList>;

auto buildSentenceVariationIndices(const std::vector<SourceList>& xor_src_lists,
  const std::vector<IndexList>& compat_idx_lists,
  const std::vector<uint64_t>& compat_indices) -> SentenceVariationIndices;

}

#endif // INCLUDE_CM_PRECOMPUTE_H
