#include <iostream>
#include "combo-maker.h"

namespace cm {

PreComputedData PCD;

auto isSourceCompatibleWithEveryOrSource(const SourceBits& sourceBits,
  const OrSourceList& orSourceList)
{
  return true;
}

auto isSourceXORCompatibleWithAnyXorSource(const SourceBits& sourceBits,
  const XorSourceList& xorSourceList, bool flag)
{
  using namespace std;
  bool compatible = xorSourceList.empty(); // empty list == compatible
  for (const auto& xorSource : xorSourceList) {
    compatible = (sourceBits & xorSource.primarySrcBits).none();
    if (flag) {
      cout << sourceBits.to_string() << endl
	   << xorSource.primarySrcBits.to_string() << endl
	   << "+++++ " << std::boolalpha << compatible << endl;
    }
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

    compatible = isSourceCompatibleWithEveryOrSource(sourceBits, PCD.orSourceList);
    if (compatible) break;
  }
  return compatible;
};

} // namespace cm
