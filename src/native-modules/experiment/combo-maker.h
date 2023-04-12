#ifndef include_combo_maker_h
#define include_combo_maker_h

#include <bitset>
#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <napi.h>

namespace cm {

constexpr auto kMaxPrimarySources = 111;
using SourceBits = std::bitset<kMaxPrimarySources>;
using SourceBitsList = std::vector<SourceBits>;

struct NameCount {
  std::string name;
  int count;

  std::string toString() const {
    char buf[128] = { 0 };
    sprintf(buf, "%s:%d", name.c_str(), count);
    return buf;
  }

  static std::string listToString(const std::vector<NameCount>& list) {
    char buf[1280] = { 0 };
    for (auto it = list.cbegin(); it != list.cend(); ++it) {
      strcat(buf, it->toString().c_str());
      if ((it + 1) != list.cend()) {
	strcat(buf, ",");
      }
    }
    return buf;
  }

  static std::string listToString(const std::vector<const NameCount*>& list) {
    char buf[1280] = { 0 };
    for (auto it = list.cbegin(); it != list.cend(); ++it) {
      strcat(buf, (*it)->toString().c_str());
      if ((it + 1) != list.cend()) {
	strcat(buf, ",");
      }
    }
    return buf;
  }

  static auto listToCountSet(const std::vector<NameCount>& list) {
    std::unordered_set<int> count_set;
    for (const auto& nc : list) {
      count_set.insert(nc.count);
    }
    return count_set;
  }

  static auto listToSourceBits(const std::vector<NameCount>& list) {
    SourceBits bits{};
    for (const auto& nc : list) {
      bits.set(nc.count);
    }
    return bits;
  }

};

using NameCountList = std::vector<NameCount>;

struct NCData {
  NameCountList ncList;
};

using NCDataList = std::vector<NCData>;

struct SourceBase {
  std::vector<NameCount> primaryNameSrcList;
  SourceBits primarySrcBits;
  std::vector<NameCount> ncList;
};

struct SourceData : SourceBase {
  std::vector<std::string> sourceNcCsvList;
  // synonymCounts
};

using SourceList = std::vector<SourceData>;
using SourceListMap = std::unordered_map<std::string, SourceList>;
using SourceCRef = std::reference_wrapper<const SourceData>;
using SourceCRefList = std::vector<SourceCRef>;

using XorSource = SourceBase;
using XorSourceList = std::vector<XorSource>;

struct OrSourceData {
  SourceData source;
  bool xorCompatible = false;
  bool andCompatible = false;
};
using OrSourceList = std::vector<OrSourceData>;

// One OrArgData contains all of the data for a single --or argument.
//
struct OrArgData {
  OrSourceList orSourceList;
  bool compatible = false;
};
using OrArgDataList = std::vector<OrArgData>;

struct PreComputedData {
  XorSourceList xorSourceList;
  OrArgDataList orArgDataList;
  SourceListMap sourceListMap;
};

struct MergedSources {
  SourceBits primarySrcBits;
  SourceCRefList sourceCRefList;
};

using MergedSourcesList = std::vector<MergedSources>;

using StringList = std::vector<std::string>;

// functions
 
std::vector<SourceCRefList> buildSourceListsForUseNcData(
  const std::vector<NCDataList>& useNcDataLists,
  const SourceListMap& sourceListMap);

MergedSourcesList mergeAllCompatibleSources(const NameCountList& ncList,
  const SourceListMap& sourceListMap);

XorSourceList mergeCompatibleXorSourceCombinations(
  const std::vector<SourceCRefList>& sourceLists);

bool isAnySourceCompatibleWithUseSources(const SourceBitsList& sourceBitsList, bool flag);

} // namespace cm

#endif // include_combo_maker_h
