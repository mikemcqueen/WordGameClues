#ifndef INCLUDE_MERGE_FILTER_DATA_H
#define INCLUDE_MERGE_FILTER_DATA_H

#include "combo-maker.h"
#include "cuda-types.h"

namespace cm {

namespace device {  // on-device data structures

struct OrSourceData {
  SourceCompatibilityData source;
  unsigned or_arg_idx;
};

/*
struct OrArgData {
  OrSourceData* or_sources;
  unsigned num_or_sources;
};
*/

struct VariationIndices {
  combo_index_t* device_data;  // one chunk of allocated data; other pointers
                               //  below point inside this chunk.
  combo_index_t* combo_indices;
  index_t* num_combo_indices;  // per variation
  index_t* variation_offsets;  // offsets into combo_indices
  index_t num_variations;

  constexpr ComboIndexSpan get_index_span(int variation) const {
    return {&combo_indices[variation_offsets[variation]],
      num_combo_indices[variation]};
  }
};

}  // namespace device

struct MergeFilterData {
  /* merge + filter */

  std::vector<SourceList> xor_src_lists;

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
  //  SentenceVariationIndices sentenceVariationIndices;

  OrArgList or_arg_list;
  unsigned num_or_args{};
  device::OrSourceData* device_or_sources{}; 
  unsigned num_or_sources{};
};

inline MergeFilterData MFD;

}

#endif  // INCLUDE_MERGE_FILTER_DATA_H
