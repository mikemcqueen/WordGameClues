#include <cassert>
#include <string>
#include <unordered_set>
#include <vector>
#include "clue-manager.h"

namespace clue_manager {

using namespace cm;

namespace {

NameSourcesMap primaryClueNameSourcesMap;
std::vector<std::unordered_set<std::string>> compoundClueNameSets;

const auto& get_compound_clue_names(int count) {
  return compoundClueNameSets.at(count - 2);
}

}  // namespace

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

bool is_known_name_count(const std::string& name, int count) {
  return (count == 1) ? primaryClueNameSourcesMap.contains(name)
                      : get_compound_clue_names(count).contains(name);
}

}  // namespace clue_manager
