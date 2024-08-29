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
    void cuda_free() {
      cm::cuda_free(src_lists);
      cm::cuda_free(idx_lists);
      cm::cuda_free(idx_list_sizes);
      reset_pointers();
    }

    SourceCompatibilityData* src_lists;
    index_t* idx_lists;
    index_t* idx_list_sizes;
    int num_idx_lists;
    int sum_idx_list_sizes;
  } device;
};

struct FilterData {
  //
  // Common
  //
  struct HostCommon : MergeData::Host {
    std::vector<UniqueVariations> unique_variations;
  };

  struct DeviceUniqueVariations {
  protected:
    void reset_pointers() {
      unique_variations = nullptr;
    }

  public:
    void cuda_free() {
      cm::cuda_free(unique_variations);
      reset_pointers();
    }

    UniqueVariations* unique_variations;
    int num_unique_variations;
  };

  struct DeviceCommon : MergeData::Device, DeviceUniqueVariations {
  protected:
    void reset_pointers() {
      MergeData::Device::reset_pointers();
      DeviceUniqueVariations::reset_pointers();
      src_list_start_indices = nullptr;
      idx_list_start_indices = nullptr;
      compat_indices = nullptr;
    }

  public:
    void cuda_free() {
      MergeData::Device::cuda_free();
      DeviceUniqueVariations::cuda_free();
      cm::cuda_free(src_list_start_indices);
      cm::cuda_free(idx_list_start_indices);
      cm::cuda_free(compat_indices);
      reset_pointers();
    }

    constexpr const auto& get_source(
        fat_index_t combo_idx, index_t list_idx) const {
      const auto src_list = &src_lists[src_list_start_indices[list_idx]];
      const auto idx_list = &idx_lists[idx_list_start_indices[list_idx]];
      const auto idx_list_size = idx_list_sizes[list_idx];
      const auto src_idx = idx_list[combo_idx % idx_list_size];
      return src_list[src_idx];
    }

    index_t* src_list_start_indices;
    index_t* idx_list_start_indices;
    fat_index_t* compat_indices;
    int num_compat_indices;
  };

  //
  // XOR
  //
  struct HostXor : HostCommon {
    // merge-only
    // currently used by showComponents (-t) and conistency check v1.
    // consistency check v1 can be removed, and showComponents can be
    // updated to do everything on c++ side, obviating the need for this.
    SourceList merged_xor_src_list;
    std::vector<UsedSources::SourceDescriptorPair> incompat_src_desc_pairs;
  } host_xor;

  struct DeviceXor : DeviceCommon {
    using Base = DeviceCommon;
  private:
    void reset_pointers() {
      Base::reset_pointers();
      incompat_src_desc_pairs = nullptr;
      variation_indices = nullptr;
      variations_compat_results = nullptr;
      variations_scan_results = nullptr;
      src_compat_uv_indices = nullptr;
      or_compat_uv_indices = nullptr;
    }

  public:
    void cuda_free() {
      Base::cuda_free();
      cm::cuda_free(incompat_src_desc_pairs);
      cm::cuda_free(variation_indices);
      cm::cuda_free(variations_compat_results);
      cm::cuda_free(variations_scan_results);
      cm::cuda_free(src_compat_uv_indices);
      cm::cuda_free(or_compat_uv_indices);
      reset_pointers();
    }

    UsedSources::SourceDescriptorPair* incompat_src_desc_pairs;
    device::VariationIndices* variation_indices;
    // flag array (0/1) of compatible entries with source variations and
    // or_data.unique_variations (dual use, allocated for greater of two)
    result_t* variations_compat_results;
    // exclusive_scan results, also dual use
    result_t* variations_scan_results;
    // list of xor.unique_variations indices compatible with current source
    index_t* src_compat_uv_indices;
    // list of or.unique_variations indices compatible with current xor source
    index_t* or_compat_uv_indices;
    index_t variations_results_per_block;
  } device_xor;

  //
  // OR
  //
  struct HostOr : HostCommon {
  } host_or;

  struct DeviceOr : DeviceCommon {
    using Base = DeviceCommon;
  public:
    void reset_pointers() {
      Base::reset_pointers();
      src_compat_results = nullptr;
    }

    void cuda_free() {
      Base::cuda_free();
      cm::cuda_free(src_compat_results);
      reset_pointers();
    }

    result_t* src_compat_results;
  } device_or;

  struct DeviceSources : DeviceUniqueVariations {
  protected:
    void reset_pointers() {
      DeviceUniqueVariations::reset_pointers();
    }

  public:
    void cuda_free() {
      DeviceUniqueVariations::cuda_free();
      reset_pointers();
    }
  } device_sources;

  DeviceXor* device_xor_data{};
  DeviceOr* device_or_data{};
  DeviceSources* device_sources_data{};
};  // struct FilterData

}  // namespace cm

#endif  // INCLUDE_MERGE_FILTER_DATA_H
