#include <iostream>
#include "combo-maker.h"

namespace cm {

PreComputedData PCD;

auto isSourceCompatibleWithEveryOrSource(const SourceData& source,
  const OrSourceList& orSourceList)
{
  return true;
}

#if 0
//
//
let isSourceXORCompatibleWithAnyXorSource = (source: SourceData, xorSourceList: XorSource[]): boolean => {
    let compatible = listIsEmpty(xorSourceList); // empty list == compatible
    for (let xorSource of xorSourceList) {
        compatible = !CountBits.intersects(source.primarySrcBits, xorSource.primarySrcBits);
        if (compatible) break;
    }
    return compatible;
};
#endif

auto isSourceXORCompatibleWithAnyXorSource(const SourceData& source,
  const XorSourceList& xorSourceList)
{
#if 0
  using namespace std;
  auto it = std::find_if(source.ncList.begin(), source.ncList.end(),
    [](const NameCount& nc){ return nc.name == "volleyball"; });
  auto vb = (it != source.ncList.end());
#endif

  bool compatible = xorSourceList.empty(); // empty list == compatible
  for (const auto& xorSource : xorSourceList) {
    compatible = (source.primarySrcBits & xorSource.primarySrcBits).none();
#if 0
    if (compatible) {
      if (vb) {
	cout << source.primarySrcBits.to_string() << endl
	     << xorSource.primarySrcBits.to_string() << endl
	     << "---" << endl;
      }
    }
#endif
    if (compatible) break;
  }
  return compatible;
};

#if 0
//
//
let isAnySourceCompatibleWithUseSources = (sourceList: SourceList, pcd: PreComputedData): boolean => {
    // TODO: this is why --xor is required with --or. OK for now. Fix later.
    if (listIsEmpty(pcd.useSourceLists.xor)) return true;

    let compatible = false;
    for (let source of sourceList) {
        compatible = isSourceXORCompatibleWithAnyXorSource(source, pcd.useSourceLists.xor);
        // if there were --xor sources specified, and none are compatible with the
        // current source, no further compatibility checking is necessary; continue
        // to next source.
        if (!compatible) continue;

        compatible = isSourceCompatibleWithEveryOrSource(source, pcd.useSourceLists.or);
        if (compatible) break;
    }
    return compatible;
};
#endif

bool isAnySourceCompatibleWithUseSources(const SourceList& sourceList) {
  if (sourceList.empty()) return true;
  auto compatible = false;
  for (const auto& source : sourceList) {
    compatible = isSourceXORCompatibleWithAnyXorSource(source, PCD.xorSourceList);
    // if there were --xor sources specified, and none are compatible with the
    // current source, no further compatibility checking is necessary; continue
    // to next source.
    if (!compatible) continue;

    compatible = isSourceCompatibleWithEveryOrSource(source, PCD.orSourceList);
    if (compatible) break;
  }
  return compatible;
};

} // namespace cm
