#ifndef INCLUDE_MERGE_FILTER_DATA_H
#define INCLUDE_MERGE_FILTER_DATA_H

#include "combo-maker.h"
#include "cuda-types.h"

namespace cm {

struct MergeFilterData {
  /* merge + filter */

  std::vector<SourceList> xor_src_lists;
  SourceListMap sourceListMap;

  SourceCompatibilityData* device_src_lists{};
  index_t* device_idx_lists{};

  /* filter */
  
  index_t* device_src_list_start_indices{};
  index_t* device_idx_list_start_indices{};
  index_t* device_idx_list_sizes{};

  std::vector<IndexList> compat_idx_lists;
  ComboIndexList combo_indices;
  combo_index_t* device_combo_indices{};

  device::VariationIndices* device_variation_indices{};
  unsigned num_nariation_indices{};

#if 0
  // TEMPORARY>>
  XorSourceList xorSourceList;
  SourceCompatibilityData* device_xorSources{};
  index_t* device_legacy_xor_src_indices{};
  std::vector<int> xorSourceIndices;
  // <<TEMPORARY
#endif  

  // OrArgList orArgList;
  unsigned num_or_args{};
  device::OrSourceData* device_or_sources{}; 
  unsigned num_or_sources{};
  //  SentenceVariationIndices sentenceVariationIndices;
};

inline MergeFilterData MFD;

}

#endif  // INCLUDE_MERGE_FILTER_DATA_H
