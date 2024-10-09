// filter.cuh

#pragma once
#include "combo-maker.h"
#include "cuda-types.h"

namespace cm {

constexpr unsigned kMaxOrArgs = 20;

struct FilterData;
struct StreamData;

extern __constant__ SourceCompatibilityData* sources_data[32];

void run_get_compatible_sources_kernel(
    const CompatSourceIndices* device_src_indices, unsigned num_src_indices,
    const UsedSources::SourceDescriptorPair* device_incompat_src_desc_pairs,
    unsigned num_src_desc_pairs, result_t* device_results);

void run_filter_kernel(int threads_per_block, StreamData& stream,
    FilterData& mfd, const CompatSourceIndices* device_src_indices,
    const result_t* device_compat_src_results, result_t* device_results);

__device__ __host__ inline void dump_compat_src_indices(
    const CompatSourceIndices* compat_src_indices, int num_csi,
    const char* header) {
  printf("%s:\n", header);
  for (int idx{}; idx < num_csi; ++idx) {
    const auto csi1 = compat_src_indices[idx].first();
    const auto csi2 = compat_src_indices[idx].second();
    printf(" %2d: first (%u, %u) second (%u, %u)\n", idx, csi1.count(), csi1.index(),
        csi2.count(), csi2.index());
  }
}

}  // namespace cm

