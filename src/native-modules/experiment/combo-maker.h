#ifndef include_combo_maker_h
#define include_combo_maker_h

#include <vector>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <napi.h>

namespace cm {

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

  static auto listToCountSet(const std::vector<NameCount>& list) {
    std::unordered_set<int> count_set;
    for (const auto& nc : list) {
      count_set.insert(nc.count);
    }
    return count_set;
  }

};

using NameCountList = std::vector<NameCount>;

struct NCData {
  NameCountList ncList;
};

using NCDataList = std::vector<NCData>;

struct SourceData {
  std::vector<NameCount> primaryNameSrcList;
  std::vector<NameCount> ncList;
  std::vector<std::string> sourceNcCsvList;
  // synonymCounts
};

using SourceList = std::vector<SourceData>;
using SourceListMap = std::unordered_map<std::string, SourceList>;

struct XorSource {
  std::vector<NameCount> primaryNameSrcList;
  std::vector<NameCount> ncList;
  std::unordered_set<int> primarySrcSet;
};

using XorSourceList = std::vector<XorSource>;

using StringList = std::vector<std::string>;

//
 
std::vector<SourceList> buildSourceListsForUseNcData(
  const std::vector<NCDataList>& useNcDataLists, const SourceListMap& sourceListMap);

XorSourceList mergeCompatibleXorSourceCombinations(const std::vector<SourceList>& sourceLists);


#if 0
template <typename T>
typename std::vector<T>::iterator append(std::vector<T>& dst, const std::vector<T>& src)
{
    typename std::vector<T>::iterator result;
    if (dst.empty()) {
        dst = src;
        result = std::begin(dst);
    } else {
        result = dst.insert(std::end(dst), std::cbegin(src), std::cend(src));
    }
    return result;
}

template <typename T>
typename std::vector<T>::const_iterator append(std::vector<T>& dst, std::vector<T>&& src)
{
    typename std::vector<T>::const_iterator result;
    if (dst.empty()) {
        dst = std::move(src);
        result = std::cbegin(dst);
    } else {
        result = dst.insert(std::end(dst),
                             std::make_move_iterator(std::begin(src)),
                             std::make_move_iterator(std::end(src)));
    }
    src.clear();
    src.shrink_to_fit();
    return result;
}
#endif

} // namespace cm

#endif // include_combo_maker_h
