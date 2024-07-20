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

struct MergeData {
  struct Host {
    std::vector<IndexList> compat_idx_lists;
    ComboIndexList combo_indices;
  } host;

  struct Device {
    void cuda_free() {
      cm::cuda_free(src_lists);
      cm::cuda_free(idx_lists);
      cm::cuda_free(idx_list_sizes);
    }

    SourceCompatibilityData* src_lists{};
    index_t* idx_lists{};
    index_t* idx_list_sizes{};
  } device;
};

struct MergeFilterData {
  struct Host : MergeData::Host {
    // merge-only
    // currently used by showComponents (-t) and conistency check v1.
    // consistency check v1 can be removed, and showComponents can be
    // updated to do everything on c++ side, obviating the need for this.
    SourceList merged_xor_src_list;

    /*
    std::vector<IndexList> compat_idx_lists;
    ComboIndexList combo_indices;
    */

    // filter
    std::vector<SourceList> xor_src_lists;

    std::vector<UsedSources::SourceDescriptorPair> incompatible_src_desc_pairs;

    OrArgList or_arg_list;
  } host;

  struct Device : MergeData::Device {
    /*
    // merge + filter
    SourceCompatibilityData* src_lists{};
    index_t* idx_lists{};
    index_t* idx_list_sizes{};
    */

    void cuda_free() {
      MergeData::Device::cuda_free();
      cm::cuda_free(incompatible_src_desc_pairs);
      cm::cuda_free(src_list_start_indices);
      cm::cuda_free(idx_list_start_indices);
      cm::cuda_free(or_src_list);
    }

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
