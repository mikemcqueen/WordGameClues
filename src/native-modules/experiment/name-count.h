#pragma once

#include <functional>
#include <string>
#include <unordered_set>
#include <vector>
#include "source-compat.h"

namespace cm {

struct NameCount;
using NameCountList = std::vector<NameCount>;
using NameCountCRef = std::reference_wrapper<const NameCount>;
using NameCountCRefList = std::vector<NameCountCRef>;

struct NameCount {
  NameCount(const std::string& name, int count) : name(name), count(count) {}
  /*
  NameCount() = default;
  NameCount(const NameCount&) = default;
  NameCount(NameCount&&) = default;
  NameCount& operator=(const NameCount&) = default;
  NameCount& operator=(NameCount&&) = default;
  */

  std::string toString() const;

  static std::string makeString(const std::string& name, int count);
  static std::string makeString(const NameCount& nc1, const NameCount& nc2);
  static std::string makeString(const std::string& name1,
      const std::string& name2);

  static std::vector<std::string> listToNameList(const NameCountList& list);

  static void listSort(NameCountList& list);
  static void listSort(NameCountCRefList& cref_list);

  static std::string listToNameCsv(const NameCountCRefList& cref_list);
  static std::string listToString(const NameCountCRefList& cref_list);
  static std::string listToString(const std::vector<std::string>& list);
  static std::string listToString(const NameCountList& list);
  static std::string listToString(const std::vector<const NameCount*>& list);

  static auto listToCountSet(const NameCountList& list) {
    std::unordered_set<int> count_set;
    for (const auto& nc : list) {
      count_set.insert(nc.count);
    }
    return count_set;
  }

  static auto listToCountList(const NameCountList& list) {
    std::vector<int> result;
    result.reserve(list.size());
    for (const auto& nc : list) {
      result.push_back(nc.count);
    }
    return result;
  }

  static UsedSources listToUsedSources(const NameCountList& list);

  static auto listMerge(
    const NameCountList& list1, const NameCountList& list2) {
    //
    auto result = list1;                                     // copy (ok)
    result.insert(result.end(), list2.begin(), list2.end()); // copy (ok)
    return result;
  }

  static auto listContains(
    const NameCountList& list, const std::string& name, int count) {
    //
    for (const auto& nc : list) {
      if ((nc.name == name) && (nc.count == count)) {
        return true;
      }
    }
    return false;
  }

  std::string name;
  int count{};
};

struct NCData {
  NameCountList ncList;
};
using NCDataList = std::vector<NCData>;

}  // namespace cm
