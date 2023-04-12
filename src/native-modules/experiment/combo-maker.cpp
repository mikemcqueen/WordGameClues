#include <iostream>
#include "combo-maker.h"

namespace cm {

PreComputedData PCD;

#if 0
struct OrSource : SourceData {
  bool xorCompatible = false;
  bool andCompatible = false;
};

// One OrArgData contains all of the data for a single --or argument.
//
struct OrArgData {
  std::vector<OrSource> orSourceList;
  bool compatible = false;
};
using OrArgDataList = std::vector<OrArgData>;
#endif

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
      auto andBits = (sourceBits & orSource.source.primarySrcBits);
      // OR == XOR || AND
      compatible = andBits.none() || (andBits == orSource.source.primarySrcBits);
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

auto isSourceXORCompatibleWithAnyXorSource(const SourceBits& sourceBits,
  const XorSourceList& xorSourceList, bool flag)
{
  using namespace std;
  bool compatible = xorSourceList.empty(); // empty list == compatible
  for (const auto& xorSource : xorSourceList) {
    compatible = (sourceBits & xorSource.primarySrcBits).none();
    if (compatible) break;
  }
  return compatible;
};

bool isAnySourceCompatibleWithUseSources(const SourceBitsList& sourceBitsList, bool flag) {
  if (sourceBitsList.empty()) return true;
  auto compatible = false;
  for (const auto& sourceBits : sourceBitsList) {
    compatible = isSourceXORCompatibleWithAnyXorSource(sourceBits, PCD.xorSourceList, flag);
    // if there were --xor sources specified, and none are compatible with the
    // current source, no further compatibility checking is necessary; continue
    // to next source.
    if (!compatible) continue;

    compatible = isSourceCompatibleWithEveryOrArg(sourceBits, PCD.orArgDataList);
    if (compatible) break;
  }
  return compatible;
};

} // namespace cm
