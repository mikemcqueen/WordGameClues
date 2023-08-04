#include <algorithm>
//#include <format>
#include <iostream>
#include "combo-maker.h"

namespace cm {

  PreComputedData PCD;

namespace {

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
  int* compat_index)
{
  bool compatible = true; // empty list == compatible
  for (size_t i{}; i < xorSourceList.size(); ++i) {
    const auto& xorSource = xorSourceList.at(i);
    compatible = compatData.isXorCompatibleWith(xorSource);
    if (compatible) {
      if (compat_index) *compat_index = i;
      break;
    }
  }
  return compatible;
};

auto isSourceXORCompatibleWithAnyXorSource(
  const SourceCompatibilityData& compatData, const XorSourceList& xorSourceList,
  const std::vector<int>& indices, int* compat_index)
{
  bool compatible = true; // empty list == compatible
  isany_perf.range_calls++;
  for (auto index : indices) {
    const auto& xorSource = xorSourceList[index];
    isany_perf.comps++;
    compatible = compatData.isXorCompatibleWith(xorSource);
    if (compatible) {
      global_compat_indices.insert(index);
      if (compat_index) *compat_index = index;
      break;
    }
  }
  if (compatible) isany_perf.compat++;
  return compatible;
};

auto isSourceXORCompatibleWithAnyXorSource(
  const SourceCompatibilityData& compatData, const XorSourceList& xorSourceList,
  const std::array<VariationIndicesMap, kNumSentences>& variationIndicesMaps,
  int* compat_index)
{
  isany_perf.calls++;
  for (auto s = 0; s < kNumSentences; ++s) {
    auto variation = compatData.usedSources.variations[s];
    const auto& map = variationIndicesMaps[s];
    if ((variation < 0) || (map.size() == 1)) continue;
    isany_perf.ss_attempt++;
    if (auto it = map.find(variation); it != map.end()) {
      if (isSourceXORCompatibleWithAnyXorSource(compatData, xorSourceList,
        it->second, compat_index))
      {
        return true;
      }
    }
    if (auto it = map.find(-1); it != map.end()) {
      if (isSourceXORCompatibleWithAnyXorSource(compatData, xorSourceList,
        it->second, compat_index))
      {
        return true;
      }
    }
    isany_perf.ss_fail++;
    return false;
  }
  isany_perf.full++;
  return isSourceXORCompatibleWithAnyXorSource(compatData, xorSourceList,
    variationIndicesMaps[1].at(-1), compat_index); // hack: we know this is the full index list
}

} // anon namespace
  
void check(const SourceCompatibilityList& sources, int list_index, int index)
{
  if (global_isany_call_counter != list_index) return;

  char buf[32];
  snprintf(buf, sizeof(buf), "%d:%d", list_index, index);
  auto& source = sources.at(index);
  source.dump(buf);
  int compat_index{ -1 };
  auto compat = isSourceXORCompatibleWithAnyXorSource(source,
     PCD.xorSourceList, &compat_index);
  std::cerr << "compat: " << compat << " (" << compat_index << ")"
            << std::endl;
}
  
bool isAnySourceCompatibleWithUseSources(
  const SourceCompatibilityList& sources)
{
  assert(!sources.empty());// return true;

  //check(sources, 2657, 0);
  //dump_xor(2408);

  auto compatible = false;

  for (size_t src_index{}; src_index < sources.size(); ++src_index) {
    const auto& src = sources.at(src_index);
    //auto x_compatibleIndices = buildCompatibleIndices(src,
    //  PCD.variationIndicesMaps);
    //std::vector<int> compatibleIndices{}; 
    int compat_index{-1};
    compatible = isSourceXORCompatibleWithAnyXorSource(src,
      PCD.xorSourceList, PCD.variationIndicesMaps, &compat_index);
    #if 0
    if (compatible) {
      std::cout << global_isany_call_counter << ":" << src_index
                << std::endl;
    }
    #endif
    // if there were --xor sources specified, and none are compatible with the
    // current source, no further compatibility checking is necessary; continue
    // to next source.
    if (!compatible) continue;
    compatible = isSourceCompatibleWithEveryOrArg(src,
      PCD.orArgDataList);
    if (compatible) break;
  }
  ++global_isany_call_counter;
  return compatible;
};

} // namespace cm
