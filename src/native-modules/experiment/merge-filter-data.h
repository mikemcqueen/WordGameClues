#ifndef INCLUDE_MERGE_FILTER_DATA_H
#define INCLUDE_MERGE_FILTER_DATA_H

#pragma once
#include "combo-maker.h"
#include "cuda-types.h"
#include "merge-type.h"

namespace cm {

struct MergeData {
  struct Host {
    Host() = default;
    Host(const Host&) = delete; // disable copy
    Host(Host&&) = delete; // disable move

    std::vector<SourceList> src_lists;
    std::vector<IndexList> compat_idx_lists;
    FatIndexList compat_indices;
  } host;

  struct Device {
  protected:
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
    unsigned num_idx_lists{};
  } device;
};

struct MergeFilterData {
  //
  // Common
  //
  struct DeviceCommon : MergeData::Device {
  protected:
    void reset_pointers() {
      MergeData::Device::reset_pointers();
      src_list_start_indices = nullptr;
      idx_list_start_indices = nullptr;
      variation_indices = nullptr;
    }

  public:
    void cuda_free() {
      MergeData::Device::cuda_free();
      cm::cuda_free(src_list_start_indices);
      cm::cuda_free(idx_list_start_indices);
      cm::cuda_free(variation_indices);
      reset_pointers();
    }

    constexpr const auto& get_source(
        fat_index_t flat_idx, index_t list_idx) const {
      const auto src_list = &src_lists[src_list_start_indices[list_idx]];
      const auto idx_list = &idx_lists[idx_list_start_indices[list_idx]];
      const auto idx_list_size = idx_list_sizes[list_idx];
      const auto src_idx = idx_list[flat_idx % idx_list_size];
      return src_list[src_idx];
    }

    index_t* src_list_start_indices{};
    index_t* idx_list_start_indices{};
    device::VariationIndices* variation_indices{};
    unsigned num_variation_indices{};
  };

  DeviceCommon* device_xor_data{};
  DeviceCommon* device_or_data{};

  //
  // XOR
  //
  struct HostXor : MergeData::Host {
    // merge-only
    // currently used by showComponents (-t) and conistency check v1.
    // consistency check v1 can be removed, and showComponents can be
    // updated to do everything on c++ side, obviating the need for this.
    SourceList merged_xor_src_list;
    std::vector<UsedSources::SourceDescriptorPair> incompat_src_desc_pairs;
  } host_xor;

  struct DeviceXor : MergeFilterData::DeviceCommon {
  private:
    void reset_pointers() {
      DeviceCommon::reset_pointers();
      incompat_src_desc_pairs = nullptr;
    }

  public:
    void cuda_free() {
      DeviceCommon::cuda_free();
      cm::cuda_free(incompat_src_desc_pairs);
      reset_pointers();
    }

    UsedSources::SourceDescriptorPair* incompat_src_desc_pairs{};
  } device_xor;

  //
  // OR
  //
  struct HostOr : MergeData::Host {
  } host_or;

  struct DeviceOr : DeviceCommon {
  private:
    void reset_pointers() {
      DeviceCommon::reset_pointers();
    }

  public:
    void cuda_free() {
      DeviceCommon::cuda_free();
      reset_pointers();
    }
  } device_or;

};  // struct MergeFilterData

}  // namespace cm

#endif  // INCLUDE_MERGE_FILTER_DATA_H
