#ifndef INCLUDE_MERGE_FILTER_DATA_H
#define INCLUDE_MERGE_FILTER_DATA_H

#pragma once
#include "combo-maker.h"
#include "cuda-types.h"
#include "merge-type.h"

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
    Host() = default;
    Host(const Host&) = delete; // disable copy
    Host(Host&&) = delete; // disable move

    std::vector<IndexList> compat_idx_lists;
    ComboIndexList combo_indices;
  } host;

  struct Device {
  private:
    void reset_pointers() {
      src_lists = nullptr;
      idx_lists = nullptr;
      idx_list_sizes = nullptr;
    }

  public:
    Device() = default;
    ~Device() { cuda_free(); }
    Device(const Device&) = delete; // disable copy
    Device(Device&&) = delete; // disable mvoe

    void cuda_free() {
      cm::cuda_free(src_lists);
      cm::cuda_free(idx_lists);
      cm::cuda_free(idx_list_sizes);
      reset_pointers();
    }

    SourceCompatibilityData* src_lists{};
    index_t* idx_lists{};
    index_t* idx_list_sizes{};
  } device;
};

struct MergeFilterData {
  // XOR kernel
  struct HostXor : MergeData::Host {
    // merge-only
    // currently used by showComponents (-t) and conistency check v1.
    // consistency check v1 can be removed, and showComponents can be
    // updated to do everything on c++ side, obviating the need for this.
    SourceList merged_xor_src_list;

    std::vector<UsedSources::SourceDescriptorPair> incompat_src_desc_pairs;
    std::vector<SourceList> src_lists;
  } host_xor;

  struct DeviceXor : MergeData::Device {
  private:
    void reset_pointers() {
      incompat_src_desc_pairs = nullptr;
      src_list_start_indices = nullptr;
      idx_list_start_indices = nullptr;
    }

  public:
    void cuda_free() {
      MergeData::Device::cuda_free();
      cm::cuda_free(incompat_src_desc_pairs);
      cm::cuda_free(src_list_start_indices);
      cm::cuda_free(idx_list_start_indices);
      reset_pointers();
    }

    UsedSources::SourceDescriptorPair* incompat_src_desc_pairs{};

    index_t* src_list_start_indices{};
    index_t* idx_list_start_indices{};

    device::VariationIndices* variation_indices{};
    unsigned num_variation_indices{};
  } device_xor;

  // OR kernel
  struct HostOr : MergeData::Host {
    OrArgList arg_list;
  } host_or;

  struct DeviceOr : MergeData::Device {
  private:
    void reset_pointers() {
      src_list = nullptr;
    }

  public:
    void cuda_free() {
      MergeData::Device::cuda_free();
      cm::cuda_free(src_list);
      reset_pointers();
    }

    device::OrSourceData* src_list{};
    unsigned num_sources{};
  } device_or;

};  // struct MergeFilterData

}  // namespace cm

#endif  // INCLUDE_MERGE_FILTER_DATA_H
