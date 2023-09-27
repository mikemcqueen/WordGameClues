#include <cassert>
#include <string>
#include <unordered_set>
#include <vector>
#include "clue-manager.h"
#include "util.h"

namespace clue_manager {

using namespace cm;

namespace {

// types
struct KnownSourceMapValue {
  SourceList src_list;
};
using KnownSourceMap = std::unordered_map<std::string, KnownSourceMapValue>;

// globals
std::vector<NcResultMap> ncResultMaps;
NameSourcesMap primaryClueNameSourcesMap;
std::vector<std::unordered_set<std::string>> compoundClueNameSets;
std::vector<KnownSourceMap> knownSourceMaps;

// functions
NcResultData& getNcResult(const NameCount& nc) {
  if ((int)ncResultMaps.size() <= nc.count) {
    ncResultMaps.resize(nc.count + 1);
  }
  auto& map = ncResultMaps.at(nc.count);
  // TODO: specialize std::hash for NameCount?
  const auto nc_str = nc.toString();
  auto it = map.find(nc_str);
  if (it == map.end()) {
    map.insert(std::make_pair(nc_str, NcResultData{}));
    it = map.find(nc_str);
  }
  return it->second;
}

const auto& get_compound_clue_names(int count) {
  return compoundClueNameSets.at(count - 2);
}

KnownSourceMap& get_known_source_map(int count, bool force_create = false) {
  const auto idx = count - 2;
  if (force_create && ((int)knownSourceMaps.size() == idx)) {
    knownSourceMaps.emplace_back(KnownSourceMap{});
  }
  return knownSourceMaps.at(idx);
}

auto& get_known_source_map_entry(
  int count, const std::string& key) {
  auto& map = get_known_source_map(count);
  auto it = map.find(key);
  assert(it != map.end());
  return it->second;
}

}  // namespace

auto get_known_nc_source(const cm::NameCount& nc, int idx) -> SourceData {
  return getNcResult(nc).src_list.at(idx);
}

auto getNumNcResults(const NameCount& nc) -> int {
  return getNcResult(nc).src_list.size();
}

void appendNcResults(const NameCount& nc, SourceList& src_list) {
  auto& nc_result = getNcResult(nc);
  for (auto& src: src_list) {
    if (nc_result.src_compat_set.find(src) == nc_result.src_compat_set.end()) {
      // add to set *before* moving to list
      nc_result.src_compat_set.insert(src);
      nc_result.src_list.emplace_back(std::move(src));
    }
  }
}

int appendNcResultsFromSourceMap(
  const cm::NameCount& nc, const std::string& src_key) {
  //
  auto& nc_result = getNcResult(nc);
  const auto& src_map_entry = get_known_source_map_entry(nc.count, src_key);
  nc_result.src_list.insert(nc_result.src_list.end(),
    src_map_entry.src_list.begin(), src_map_entry.src_list.end());
  return std::distance(
    src_map_entry.src_list.begin(), src_map_entry.src_list.end());
}

auto buildNameSourcesMap(std::vector<std::string>& names,
  std::vector<IndexList>& src_lists) -> NameSourcesMap {
  NameSourcesMap nameSourcesMap;
  for (size_t i{}; i < names.size(); ++i) {
    nameSourcesMap.emplace(
      std::make_pair(std::move(names.at(i)), std::move(src_lists.at(i))));
  }
  return nameSourcesMap;
}

void setPrimaryClueNameSourcesMap(NameSourcesMap&& nameSourcesMap) {
  assert(primaryClueNameSourcesMap.empty());
  primaryClueNameSourcesMap = std::move(nameSourcesMap);
}

const IndexList& getSourcesForPrimaryClueName(const std::string& name) {
  const auto& map = primaryClueNameSourcesMap;
  auto it = map.find(name);
  assert(it != map.end());
  return it->second;
}

void setCompoundClueNames(int count, std::vector<std::string>& name_list) {
  auto index = count - 2;
  assert((int)compoundClueNameSets.size() == index);
  std::unordered_set<std::string> name_set;
  for (auto& name: name_list) {
    name_set.emplace(std::move(name));
  }
  compoundClueNameSets.emplace_back(std::move(name_set));
}

bool is_known_source_map_entry(int count, const std::string& key) {
  const auto idx = count - 2;
  if ((int)knownSourceMaps.size() > idx) {
    return get_known_source_map(count).contains(key);
  }
  return false;
}

void init_known_source_map_entry(
  int count, const std::vector<std::string>& name_list, SourceList&& src_list) {
  //
  // True is arbitrary here. I *could* support replacing an existing src_list,
  // but i'm unaware of any situation that requires it, and I want things to
  // blow up when it happens.
  auto& map = get_known_source_map(count, true);
  const auto key = util::join(name_list, ",");
  auto [_, success] = map.insert(std::make_pair(key, std::move(src_list)));
  assert(success);
}

bool is_known_name_count(const std::string& name, int count) {
  return (count == 1) ? primaryClueNameSourcesMap.contains(name)
                      : get_compound_clue_names(count).contains(name);
}

}  // namespace clue_manager
