#include <algorithm>
#include <exception>
#include <cuda_runtime.h>
//#include <helper_cuda.h>
#include "candidates.h"
#include "combo-maker.h"

namespace cm {

PreComputedData PCD;

__device__ auto isSourceORCompatibleWithAnyOrSource(
  const SourceCompatibilityData& compatData, const OrSourceList& orSourceList)
{
  auto compatible = false;
  for (const auto& orSource : orSourceList) {
    // skip any sources that were already determined to be XOR incompatible
    // or AND compatible with --xor sources.
    if (!orSource.xorCompatible || orSource.andCompatible) continue;
    compatible = compatData.isOrCompatibleWith(orSource.source);
    if (compatible) break;
  }
  return compatible;
};

__device__ auto isSourceCompatibleWithEveryOrArg(
  const SourceCompatibilityData& compatData, const OrArgDataList& orArgDataList)
{
  auto compatible = true; // if no --or sources specified, compatible == true
  for (const auto& orArgData : orArgDataList) {
    // TODO: skip calls to here if container.compatible = true  which may have
    // been determined in Precompute phase @ markAllANDCompatibleOrSources()
    // and skip the XOR check as well in this case.
    compatible = isSourceORCompatibleWithAnyOrSource(compatData,
     orArgData.orSourceList);
    if (!compatible) break;
  }
  return compatible;
}
 
// "temporarily" re-added this original version without indices, as part of 
// prototyping cuda integration
__device__ auto isSourceXORCompatibleWithAnyXorSource(
  const SourceCompatibilityData& compatData, const XorSourceList& xorSourceList)
{
  bool compatible = true; // empty list == compatible
  for (const auto& xorSource: xorSourceList) {
    compatible = compatData.isXorCompatibleWith(xorSource);
    if (compatible) break;
  }
  return compatible;
};

//__device__
auto isSourceXORCompatibleWithAnyXorSource(
  const SourceCompatibilityData& compatData, const XorSourceList& xorSourceList,
  const std::vector<int>& indices)
{
  bool compatible = true; // empty list == compatible
#if PERF
  isany_perf.range_calls++;
#endif
  for (auto index : indices) {
    const auto& xorSource = xorSourceList[index];
#if PERF
    isany_perf.comps++;
#endif
    compatible = compatData.isXorCompatibleWith(xorSource);
    if (compatible) break;
  }
#if PERF
  if (compatible) isany_perf.compat++;
#endif
  return compatible;
};

//__device__
auto isSourceXORCompatibleWithAnyXorSource(
  const SourceCompatibilityData& compatData, const XorSourceList& xorSourceList,
  const std::array<VariationIndicesMap, kNumSentences>& variationIndicesMaps)
{
#if PERF
  isany_perf.calls++;
#endif
  for (auto s = 0; s < kNumSentences; ++s) {
    auto variation = compatData.usedSources.variations[s];
    const auto& map = variationIndicesMaps[s];
    if ((variation < 0) || (map.size() == 1)) continue;
#if PERF
    isany_perf.ss_attempt++;
#endif
    if (auto it = map.find(variation); it != map.end()) {
      if (isSourceXORCompatibleWithAnyXorSource(compatData, xorSourceList,
        it->second))
      {
        return true;
      }
    }
    if (auto it = map.find(-1); it != map.end()) {
      if (isSourceXORCompatibleWithAnyXorSource(compatData, xorSourceList,
        it->second))
      {
        return true;
      }
    }
#if PERF
    isany_perf.ss_fail++;
#endif
    return false;
  }
#if PERF
  isany_perf.full++;
#endif
  return isSourceXORCompatibleWithAnyXorSource(compatData, xorSourceList,
    variationIndicesMaps[1].at(-1)); // hack: we know "sentence" 2 doesn't exist
}
  
__device__ bool isAnySourceCompatibleWithUseSources(
  const SourceCompatibilityList& sourceCompatList)
{
  if (sourceCompatList.empty()) return true;
  auto compatible = false;
  for (const auto& compatData : sourceCompatList) {
    compatible = isSourceXORCompatibleWithAnyXorSource(compatData,
      PCD.xorSourceList); // , PCD.variationIndicesMaps);
    // if there were --xor sources specified, and none are compatible with the
    // current source, no further compatibility checking is necessary; continue
    // to next source.
    if (!compatible) continue;
    compatible = isSourceCompatibleWithEveryOrArg(compatData, PCD.orArgDataList);
    if (compatible) break;
  }
  return compatible;
};

__global__ void kernel(const SourceCompatibilityList *compatLists, size_t count) {
  auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  isAnySourceCompatibleWithUseSources(compatLists[index]);
}

void filterCandidatesCuda(int sum) {
  std::cerr << "filerCandidatesCuda" << std::endl;
  const auto& sourceCompatLists = allSumsCandidateData[sum - 2].sourceCompatLists;
  kernel<<<40, 32>>>(&sourceCompatLists[0], sourceCompatLists.size());
}

} // namespace cm
