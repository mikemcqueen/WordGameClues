#ifndef include_combo_maker_h
#define include_combo_maker_h

#include <bitset>
#include <napi.h>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cm {

constexpr auto kMaxPrimarySources = 111;

using SourceBits = std::bitset<kMaxPrimarySources>;
using SourceBitsList = std::vector<SourceBits>;
using UsedSources = std::array<std::set<uint32_t>, 10>;
using UsedSourcesList = std::vector<UsedSources>;

struct SourceCompatibilityData {
  SourceBits sourceBits;
  UsedSources usedSources;

  SourceCompatibilityData() = default;
  // copy assign allowed for now for precompute.mergeAllCompatibleXorSources
  SourceCompatibilityData(const SourceCompatibilityData&) = default;
  SourceCompatibilityData& operator=(const SourceCompatibilityData&) = default;
  SourceCompatibilityData(SourceCompatibilityData&&) = default;
  SourceCompatibilityData& operator=(SourceCompatibilityData&&) = default;

  // copy components
  SourceCompatibilityData(const SourceBits& sourceBits,
      const UsedSources& usedSources):
    sourceBits(sourceBits),
    usedSources(usedSources)
  {}

  // move components
  SourceCompatibilityData(SourceBits&& sourceBits,
      UsedSources&& usedSources):
    sourceBits(std::move(sourceBits)),
    usedSources(std::move(usedSources))
  {}

  static auto isCandidate(int src) { return src >= 1'000'000; }
  static auto getCandidateSentence(int src) { return src / 1'000'000; }
  static auto getCandidateSource(int src) { return src % 1'000'000; }
  static auto getVariation(int src) { return (src % 1'000'000) / 100; }

  static auto areUsedSourcesCompatible(const UsedSources& usedSources, const UsedSources& other) {
    for (auto i = 1u; i < usedSources.size(); ++i) {
      if (usedSources[i].empty() || other[i].empty()) continue;
      if (getVariation(*usedSources[i].cbegin()) != 
	  getVariation(*other[i].cbegin()))
      {
	return false;
      }
      for (auto it = usedSources[i].cbegin(); it != usedSources[i].cend(); ++it) {
	if (other[i].find(*it) != other[i].end()) {
	  return false;
	}
      }
    }
    return true;
  }

  auto isCompatibleWith(const SourceCompatibilityData& other) const {
    if ((sourceBits & other.sourceBits).any()) {
      return false;
    }
    return areUsedSourcesCompatible(usedSources, other.usedSources);
  }

  auto addUsedSource(int src) {
    return addUsedSource(usedSources, src);
  }

  void mergeWith() {
  }

  static auto addUsedSource(UsedSources& usedSources, int src) -> bool {
    auto nothrow = false;
    if (!isCandidate(src)) throw "poop";
    auto sentence = getCandidateSentence(src);
    auto source = getCandidateSource(src);
    auto& set = usedSources[sentence];
    // defensive incompatible variation index check
    if (!set.empty()) {
      if (getVariation(*set.begin()) != getVariation(source)) {
	if (nothrow) return false;
	throw "oopsie1"; // new Error(`oopsie ${anyElem}, ${source}`);
      }
      if (set.find(source) != set.end()) {
	if (nothrow) return false;
	throw "oopsie2"; // new Error(`poopsie ${source}, [${[...set]}]`);
      }
    }
    set.insert(source);
    return true;
  }
};

struct NameCount;
using NameCountList = std::vector<NameCount>;

using SourceCompatibilityList = std::vector<SourceCompatibilityData>;

struct SourceData : SourceCompatibilityData {
  NameCountList primaryNameSrcList;
  NameCountList ncList;

  SourceData() = default;
  SourceData(NameCountList&& primaryNameSrcList, SourceBits&& sourceBits,
      UsedSources&& usedSources, NameCountList&& ncList) :
    SourceCompatibilityData(std::move(sourceBits), std::move(usedSources)),
    primaryNameSrcList(std::move(primaryNameSrcList)),
    ncList(std::move(ncList))
  {}

  // copy assign allowed for now for precompute.mergeAllCompatibleXorSources
  SourceData(const SourceData&) = default;
  SourceData& operator=(const SourceData&) = default;
  SourceData(SourceData&&) = default;
  SourceData& operator=(SourceData&&) = default;
};

/*
struct SourceData : SourceBase {
  //std::vector<std::string> sourceNcCsvList; // TODO: I don't think this is even used anymore
  // synonymCounts

  SourceData() = default;
  SourceData(NameCountList&& primaryNameSrcList, SourceBits&& primarySrcBits,
      UsedSources&& usedSources, NameCountList&& ncList): //, std::vector<std::string>&& sourceNcCsvList) :
    SourceBase(std::move(primaryNameSrcList), std::move(primarySrcBits),
      std::move(usedSources), std::move(ncList))
      //,sourceNcCsvList(std::move(sourceNcCsvList))
  {}

  SourceData(const SourceData&) = delete;
  SourceData& operator=(const SourceData&) = delete;
  SourceData(SourceData&&) = default;
  SourceData& operator=(SourceData&&) = default;
};
*/

using SourceList = std::vector<SourceData>;
using SourceListMap = std::unordered_map<std::string, SourceList>;
using SourceCRef = std::reference_wrapper<const SourceData>;
using SourceCRefList = std::vector<SourceCRef>;

using XorSource = SourceData;
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

struct MergedSources : SourceCompatibilityData {
  SourceCRefList sourceCRefList;

  MergedSources() = default;
  MergedSources(const MergedSources&) = default; // allow, dangerous?
  MergedSources& operator=(const MergedSources&) = delete;
  MergedSources(MergedSources&&) = default;
  MergedSources& operator=(MergedSources&&) = default;

  // copy from SourceData
  MergedSources(const SourceData& source) :
      SourceCompatibilityData(source.sourceBits, source.usedSources),
      sourceCRefList(SourceCRefList{SourceCRef{source}})
  {}
};

using MergedSourcesList = std::vector<MergedSources>;

using StringList = std::vector<std::string>;

struct NameCount {
  std::string name;
  int count;

  NameCount(std::string&& name, int count) : name(std::move(name)), count(count) {}
  NameCount() = default;
  NameCount(const NameCount&) = default;
  NameCount& operator=(const NameCount&) = default;
  NameCount(NameCount&&) = default;
  NameCount& operator=(NameCount&&) = default;

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
      if (nc.count < 1'000'000) {
	bits.set(nc.count);
      }
    }
    return bits;
  }

  static auto listToUsedSources(const std::vector<NameCount>& list) {
    UsedSources usedSources{};
    for (const auto& nc : list) {
      if (nc.count >= 1'000'000) {
	SourceCompatibilityData::addUsedSource(usedSources, nc.count);
      }
    }
    return usedSources;
  }

  static auto listMerge(const std::vector<NameCount>& list1,
    const std::vector<NameCount>& list2)
  {
    auto result = list1; // copy (ok)
    result.insert(result.end(), list2.begin(), list2.end()); // copy (ok)
    return result;
  }
};

struct NCData {
  NameCountList ncList;
};

using NCDataList = std::vector<NCData>;

// functions
 
void debugSourceList(const SourceList& sourceList, std::string_view sv);

auto buildSourceListsForUseNcData(const std::vector<NCDataList>& useNcDataLists,
  const SourceListMap& sourceListMap) -> std::vector<SourceList>;

XorSourceList mergeCompatibleXorSourceCombinations(
  const std::vector<SourceList>& sourceLists);

bool isAnySourceCompatibleWithUseSources(const SourceCompatibilityList& sourceCompatList);

void mergeUsedSourcesInPlace(UsedSources& to, const UsedSources& from);

} // namespace cm

#endif // include_combo_maker_h
