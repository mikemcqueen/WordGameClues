#include <algorithm>
//#include <format>
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
 
auto buildCompatibleIndices(const SourceCompatibilityData& compatData,
  const std::array<VariationIndicesMap, kNumSentences>& variationIndicesMaps)
{
  std::vector<int> indices{};
  for (auto s = 0; s < kNumSentences; ++s) {
    auto variation = compatData.usedSources.variations[s];
    if (variation < 0) continue;
    const auto& map = variationIndicesMaps[s];
    if (map.empty()) continue;
    auto it = map.find(variation);
    variation = (it != map.end()) ? variation : -1;
    const auto& index_set = (it != map.end()) ? it->second : map.at(-1);
    if (!indices.size()) {
      indices.insert(indices.cend(), index_set.cbegin(), index_set.cend());
      if (variation != -1) {
        indices.insert(indices.cend(), map.at(-1).cbegin(), map.at(-1).cend());
      }
    } else if (variation > -1) {
      std::vector<int> intersection{};
      std::set_intersection(indices.cbegin(), indices.cend(),
        index_set.cbegin(), index_set.cend(), std::back_inserter(intersection));
      //if (variation == -1) {
        indices = std::move(intersection);
      /*      
      } else {
        indices.clear();
        // TODO: this is slow for sentences with no variations (legacy sentences).
        // we're doing an unnecessary intersect of everything. should figure out
        // some way to mark a map as "only -1".  hey how about size() == 1?
        std::set_intersection(intersection.cbegin(), intersection.cend(),
          map.at(-1).cbegin(), map.at(-1).cend(), std::back_inserter(indices));
      }
      */
      if (!indices.size()) break;
    }
  }
  return indices;
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
    //auto x_compatibleIndices = buildCompatibleIndices(compatData,
    //  PCD.variationIndicesMaps);
    //std::vector<int> compatibleIndices{}; 

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
