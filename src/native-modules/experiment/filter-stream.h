#pragma once

#include <vector>
#include <cuda_runtime.h> // cudaStream_t
#include "merge-filter-data.h"
#include "source-desc.h"
#include "source-index.h"
#include "stream-data.h"

namespace cm {

struct FilterStream;

struct FilterStreamData {
  struct Device;

  struct Host : HasGlobalDeviceData {
    std::vector<SourceIndex> src_idx_list{};

  private:
    friend struct Device;
    bool device_initialized{false};
  };

  struct Device {
    void init(FilterStream& stream, FilterData& mfd);

    // list of ...TODO...
    SourceIndex* src_idx_list;

    // Buffers

    // xor_data.unique_variations indices compatible with current source
    index_t* xor_src_compat_uv_indices;

    // or_data.unique_variations indices compatible with current xor source
    index_t* or_xor_compat_uv_indices;

#if 1
    // serves as both flag-array of variation compatibility test, and result
    // in-place exclusive scan.
    // necessary because i thought compact_indices_in_place was broken so I
    // temporarily reverted to compact them into this.
    index_t* variations_compat_results;
#endif

    // flag-array of SourceBits-compatibility tests between current source and
    // all or-sources
    result_t* or_src_bits_compat_results;

    // TODO:
    // result_t* results;

    // length of src_idx_list
    index_t num_src_idx;

  private:
    void alloc_buffers(FilterData& fd, cudaStream_t stream);
    void cuda_copy_to_symbol(index_t idx, cudaStream_t stream);
  };
};  // struct FilterStreamData

using FilterStreamBase =
    StreamData<FilterStreamData::Host, FilterStreamData::Device>;

struct FilterSwarmData {
  struct Host : HasGlobalDeviceData {
    std::vector<SourceDescriptorPair> incompat_src_desc_pairs;
  };

  struct Device {
    // list of incompatible primary source descriptor pairs from sum==2
    SourceDescriptorPair* incompat_src_desc_pairs;
  };
};  // struct FilterSwarmData

class IndexStates;

struct FilterStream : public FilterStreamBase {
  using FilterStreamBase::FilterStreamBase;

  void init(FilterData& mfd) { device.init(*this, mfd); }

  int num_ready(const IndexStates& indexStates) const;

  int num_done(const IndexStates& indexStates) const;

  int num_compatible(const IndexStates& indexStates) const;

  bool fill_source_indices(IndexStates& idx_states, index_t max_indices);

  bool fill_source_indices(IndexStates& idx_states) {
    return fill_source_indices(idx_states, stride);
  }

  void alloc_copy_source_indices();

  auto hasWorkRemaining() const { return !host.src_idx_list.empty(); }

  void dump() const;
};  // struct FilterStream

}  // namespace cm
