#include <algorithm>
#include <format>
#include <iostream>
#include "combo-maker.h"

namespace cm {

PreComputedData PCD;

auto isSourceORCompatibleWithAnyOrSource(const SourceCompatibilityData& compatData,
  const OrSourceList& orSourceList)
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

auto isSourceCompatibleWithEveryOrArg(const SourceCompatibilityData& compatData,
  const OrArgDataList& orArgDataList)
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
 
auto isSourceXORCompatibleWithAnyXorSource(
  const SourceCompatibilityData& compatData, const XorSourceList& xorSourceList,
  const std::vector<int>& indices)
{
  bool compatible = true; // empty list == compatible
  isany_perf.range_calls++;
  for (auto index : indices) {
    const auto& xorSource = xorSourceList[index];
    isany_perf.comps++;
    compatible = compatData.isXorCompatibleWith(xorSource);
    if (compatible) break;
  }
  if (compatible) isany_perf.compat++;
  return compatible;
};

auto isSourceXORCompatibleWithAnyXorSource(
  const SourceCompatibilityData& compatData, const XorSourceList& xorSourceList,
  const std::array<VariationIndicesMap, kNumSentences>& variationIndicesMaps)
{
  isany_perf.calls++;
  for (auto s = 0; s < kNumSentences; ++s) {
    auto variation = compatData.usedSources.variations[s];
    const auto& map = variationIndicesMaps[s];
    if ((variation < 0) || (map.size() == 1)) continue;
    isany_perf.ss_attempt++;
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
    isany_perf.ss_fail++;
    return false;
  }
  isany_perf.full++;
  return isSourceXORCompatibleWithAnyXorSource(compatData, xorSourceList,
    variationIndicesMaps[1].at(-1)); // hack: we know this is the full index list
}
  
bool isAnySourceCompatibleWithUseSources(
  const SourceCompatibilityList& sourceCompatList)
{
  if (sourceCompatList.empty()) return true;
  auto compatible = false;
  for (const auto& compatData : sourceCompatList) {
    compatible = isSourceXORCompatibleWithAnyXorSource(compatData,
      PCD.xorSourceList, PCD.variationIndicesMaps);
    // if there were --xor sources specified, and none are compatible with the
    // current source, no further compatibility checking is necessary; continue
    // to next source.
    if (!compatible) continue;
    compatible = isSourceCompatibleWithEveryOrArg(compatData, PCD.orArgDataList);
    if (compatible) break;
  }
  return compatible;
};

} // namespace cm
