#include <iostream>
#include "combo-maker.h"

namespace cm {

PreComputedData PCD;

//
//
auto isSourceORCompatibleWithAnyOrSource(const SourceBits& sourceBits,
  const OrSourceList& orSourceList)
{
    auto compatible = false;
    for (const auto& orSource : orSourceList) {
      // skip any sources that were already determined to be XOR incompatible or AND compatible
      // with command-line supplied --xor sources.
      if (!orSource.xorCompatible || orSource.andCompatible) continue;
      auto andBits = (sourceBits & orSource.source.sourceBits);
      // OR == XOR || AND
      compatible = andBits.none() || (andBits == orSource.source.sourceBits);
      if (compatible) break;
    }
    return compatible;
};

//
//
auto isSourceCompatibleWithEveryOrArg(const SourceBits& sourceBits,
  const OrArgDataList& orArgDataList)
{
  auto compatible = true; // if no --or sources specified, compatible == true
  for (const auto& orArgData : orArgDataList) {
    // TODO: skip calls to here if container.compatible = true  which may have
    // been determined in Precompute phase @ markAllANDCompatibleOrSources()
    // and skip the XOR check as well in this case.
    compatible = isSourceORCompatibleWithAnyOrSource(sourceBits,
      orArgData.orSourceList);
    if (!compatible) break;
  }
  return compatible;
}
 
auto isSourceXORCompatibleWithAnyXorSource(const SourceCompatibilityData& compatData,
  const XorSourceList& xorSourceList)
{
  bool compatible = xorSourceList.empty(); // empty list == compatible
  for (const auto& xorSource : xorSourceList) {
    compatible = compatData.isCompatibleWith(xorSource);
    if (compatible) break;
  }
  return compatible;
};

bool isAnySourceCompatibleWithUseSources(const SourceCompatibilityList& sourceCompatList) {
  if (sourceCompatList.empty()) return true;
  auto compatible = false;
  for (const auto& compatData : sourceCompatList) {
    compatible = isSourceXORCompatibleWithAnyXorSource(compatData, PCD.xorSourceList);
    // if there were --xor sources specified, and none are compatible with the
    // current source, no further compatibility checking is necessary; continue
    // to next source.
    if (!compatible) continue;

    // TODO
    compatible = isSourceCompatibleWithEveryOrArg(compatData.sourceBits, PCD.orArgDataList);
    if (compatible) break;
  }
  return compatible;
};

} // namespace cm
