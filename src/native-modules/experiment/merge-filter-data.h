#ifndef INCLUDE_MERGE_FILTER_DATA_H
#define INCLUDE_MERGE_FILTER_DATA_H

#pragma once

#include "combo-maker.h"
#include "cuda-types.h"
#include "source-index.h"

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

    constexpr SourceIndex get_source_index(index_t flat_idx) {
      for (index_t list_idx{}; list_idx < num_idx_lists; ++list_idx) {
        const auto list_size = idx_list_sizes[list_idx];
        if (flat_idx < list_size) return {list_idx, flat_idx};
        flat_idx -= list_size;
      }
      assert(0);
      return {};
    }

    SourceCompatibilityData* src_lists;
    index_t* idx_lists;
    index_t* idx_list_sizes;
    index_t num_idx_lists;
    index_t num_src_compat_results; // sum of all idx_list_sizes
  } device;
};

struct FilterData {
  //
  // Common
  //
  struct HostCommon : MergeData::Host {
    std::vector<UniqueVariations> unique_variations;
  };

  struct DeviceCommon : MergeData::Device {
  protected:
    void reset_pointers() {
      MergeData::Device::reset_pointers();
      src_list_start_indices = nullptr;
      idx_list_start_indices = nullptr;
      compat_indices = nullptr;
      unique_variations = nullptr;
    }

  public:
    void cuda_free() {
      MergeData::Device::cuda_free();
      cm::cuda_free(src_list_start_indices);
      cm::cuda_free(idx_list_start_indices);
      cm::cuda_free(compat_indices);
      cm::cuda_free(unique_variations);
      reset_pointers();
    }

    constexpr const auto& get_source(const fat_index_t combo_idx,
        const index_t list_idx) const {
      const auto src_list = &src_lists[src_list_start_indices[list_idx]];
      const auto idx_list = &idx_lists[idx_list_start_indices[list_idx]];
      const auto idx_list_size = idx_list_sizes[list_idx];
      const auto src_idx = idx_list[combo_idx % idx_list_size];
      return src_list[src_idx];
    }

    constexpr const auto& get_source(const SourceIndex src_idx) {
      const auto src_list_idx = src_list_start_indices[src_idx.listIndex];
      const auto src_list = &src_lists[src_list_idx];
      const auto idx_list_idx = idx_list_start_indices[src_idx.listIndex];
      const auto idx_list = &idx_lists[idx_list_idx];
      return src_list[idx_list[src_idx.index]];
    }

    index_t* src_list_start_indices;
    index_t* idx_list_start_indices;
    fat_index_t* compat_indices;
    UniqueVariations* unique_variations;
    int num_compat_indices;
    int num_unique_variations;
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
    std::vector<SourceDescriptorPair> incompat_src_desc_pairs;
  } host_xor;

  struct DeviceXor : DeviceCommon {
    using Base = DeviceCommon;
  private:
    void reset_pointers() {
      Base::reset_pointers();
      incompat_src_desc_pairs = nullptr;
    }

  public:
    void cuda_free() {
      Base::cuda_free();
      cm::cuda_free(incompat_src_desc_pairs);
      reset_pointers();
    }

    SourceDescriptorPair* incompat_src_desc_pairs;
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
    }

    void cuda_free() {
      Base::cuda_free();
      reset_pointers();
    }
  } device_or;


  DeviceXor* device_xor_data{};
  DeviceOr* device_or_data{};
};  // struct FilterData

}  // namespace cm

#endif  // INCLUDE_MERGE_FILTER_DATA_H
