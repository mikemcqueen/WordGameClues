#ifndef INCLUDE_MERGE_FILTER_DATA_H
#define INCLUDE_MERGE_FILTER_DATA_H

#include "combo-maker.h"
#include "cuda-types.h"

namespace cm {

namespace device {  // on-device data structures

struct OrSourceData {
  SourceCompatibilityData src;
  unsigned or_arg_idx;
};

struct VariationIndices {
  combo_index_t* device_data;  // one chunk of allocated data; other pointers
                               // below point inside this chunk.
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
  struct Host {
    // merge-only
    SourceList merged_xor_src_list;

    // merge + filter
    std::vector<SourceList> xor_src_lists;

    // filter
    std::vector<UsedSources::SourceDescriptorPair> incompatible_src_desc_pairs;

    std::vector<IndexList> compat_idx_lists;
    ComboIndexList combo_indices;

    OrArgList or_arg_list;
  } host;

  struct Device {
    // merge + filter
    SourceCompatibilityData* src_lists{};
    index_t* idx_lists{};
    index_t* idx_list_sizes{};

    // filter
    UsedSources::SourceDescriptorPair* incompatible_src_desc_pairs{};
    unsigned num_incompatible_sources{};

    index_t* src_list_start_indices{};
    index_t* idx_list_start_indices{};

    device::VariationIndices* variation_indices{};
    unsigned num_variation_indices{};

    device::OrSourceData* or_src_list{};
    unsigned num_or_sources{};
  } device;
};  // struct MergeFilterData

}  // namespace cm

#endif  // INCLUDE_MERGE_FILTER_DATA_H
