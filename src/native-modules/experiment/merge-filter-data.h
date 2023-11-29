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
};

#if 0
  //
  // mergeCompatible():
  //

  // makes this 
  std::vector<SourceList> xor_src_lists;
  // makes these in a really convoluted way, as a consequence of making
  // compat_idx_lists/combo_indices. really should separate these out
  SourceCompatibilityData* device_src_lists{};
  index_t* device_idx_lists{};
  index_t* device_idx_list_sizes{};
  // makes these as a consequence of making g_merged_xor_src_list, and
  // saves them here for filter
  std::vector<IndexList> compat_idx_lists;
  ComboIndexList combo_indices;

  //
  // filterPreparation()
  // should probably call it make_filter_indices
  //

  // makes these. 
  index_t* device_src_list_start_indices{};
  index_t* device_idx_list_start_indices{};

  // makes these via cuda_allocCopySentenceVariationIndices, which has to
  // sync device because it's doing async copy of local host-side data. should
  // store host-side data here to eliminate sync.
  // also names a local host-side array with "device_" prefix which is wrong.
  device::VariationIndices* device_variation_indices{};
  unsigned num_nariation_indices{};

  // unused?
  //  combo_index_t* device_combo_indices{};

  //
  // set_or_arg
  // should probably call it make_or_arg_data
  // 

  // makes these
  OrArgList or_arg_list;
  unsigned num_or_args{};
  device::OrSourceData* device_or_sources{}; 
  unsigned num_or_sources{};
#endif

}  // namespace cm

#endif  // INCLUDE_MERGE_FILTER_DATA_H
