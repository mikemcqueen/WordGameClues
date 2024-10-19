// filter.cuh

#pragma once
#include "combo-maker.h"
#include "cuda-types.h"

namespace cm {

constexpr unsigned kMaxOrArgs = 20;
constexpr unsigned kMaxSums = 32;

struct FilterData;
struct StreamData;

extern __constant__ SourceCompatibilityData* sources_data[kMaxSums];

void run_get_compatible_sources_kernel(
    const CompatSourceIndices* device_src_indices, size_t num_src_indices,
    const UsedSources::SourceDescriptorPair* device_incompat_src_desc_pairs,
    size_t num_src_desc_pairs, result_t* device_resultsy,
    cudaStream_t sync_stream, cudaStream_t stream);

void run_filter_kernel(int threads_per_block, StreamData& stream,
    FilterData& mfd, const CompatSourceIndices* device_src_indices,
    const result_t* device_compat_src_results, result_t* device_results);

#if 0
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
#endif

}  // namespace cm

