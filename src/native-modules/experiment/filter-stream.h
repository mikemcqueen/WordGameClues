#pragma once

#include <vector>
#include <cuda_runtime.h> // cudaStream_t
#include "merge-filter-data.h"
#include "source-desc.h"
#include "source-index.h"
#include "stream-swarm.h"

namespace cm {

#define VARIATIONS_RESULTS

struct FilterStream;

struct FilterStreamData {
  struct Device;

  struct Host : HasStreamDeviceData {
    std::vector<SourceIndex> src_idx_list{};

  private:
    friend struct Device;
    bool device_initialized{false};
  };

  struct Device {
    void init() {
      src_idx_list = nullptr;
      num_src_idx = 0;
    }
    void init(FilterStream& stream, FilterData& mfd);
    void alloc_copy_source_index_list(const Host& host, cudaStream_t stream);

    // list of ...TODO...
    SourceIndex* src_idx_list;

    // xor_data.unique_variations indices compatible with current source
    index_t* xor_src_compat_uv_indices;

    // or_data.unique_variations indices compatible with current xor source
    index_t* or_xor_compat_uv_indices;

#ifdef VARIATIONS_RESULTS
    // serves as both flag-array of variation compatibility test, and results
    // of in-place exclusive scan.
    // necessary because i thought compact_indices_in_place was broken so I
    // temporarily reverted to compact them from this.
    index_t* variations_compat_results;
    index_t num_variations_results_per_block;
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
    void copy_to_symbol(index_t idx, cudaStream_t stream);
    };
};  // struct FilterStreamData

using FilterStreamBase =
    StreamData<FilterStreamData::Host, FilterStreamData::Device>;

struct FilterSwarmData {
  struct Host : HasSwarmDeviceData {
  };

  struct Device {
    void init() { reset(); }

    void reset() {
      compat_src_indices = nullptr;
      compat_src_results = nullptr;
    }

    void cuda_free(cudaStream_t stream) {
      if (compat_src_indices) cuda_free_async(compat_src_indices, stream);
      if (compat_src_results) cuda_free_async(compat_src_results, stream);
      reset();
    }

    void update(index_t swarm_idx,
        CompatSourceIndices* device_compat_src_indices,
        result_t* device_compat_src_results, cudaStream_t stream) {
      cuda_free(stream);
      compat_src_indices = device_compat_src_indices;
      compat_src_results = device_compat_src_results;
      copy_to_symbol(swarm_idx, stream);
    }

    CompatSourceIndices* compat_src_indices;
    result_t* compat_src_results;

    private:
      void copy_to_symbol(index_t idx, cudaStream_t stream);
  };
};  // struct FilterSwarmData

class IndexStates;

struct FilterStream : public FilterStreamBase {
  using FilterStreamBase::FilterStreamBase;

  void init(FilterData& mfd) { device.init(*this, mfd); }

  int num_ready(const IndexStates& indexStates) const;

  int num_done(const IndexStates& indexStates) const;

  int num_compatible(const IndexStates& indexStates) const;

  bool fill_source_index_list(IndexStates& idx_states, index_t max_indices);

  bool fill_source_index_list(IndexStates& idx_states) {
    return fill_source_index_list(idx_states, stride);
  }

  void alloc_copy_source_index_list();

  auto hasWorkRemaining() const { return !host.src_idx_list.empty(); }

  void dump() const;
};  // struct FilterStream

using FilterSwarm =
    StreamSwarm<FilterStream, FilterSwarmData::Host, FilterSwarmData::Device>;

}  // namespace cm
