#ifndef INCLUDE_CLUE_MANAGER_H
#define INCLUDE_CLUE_MANAGER_H

#include <string>
#include <unordered_map>
#include <vector>
#include "cuda-types.h"

namespace clue_manager {

using NameSourcesMap = std::unordered_map<std::string, cm::IndexList>;

auto buildNameSourcesMap(std::vector<std::string>& names,
  std::vector<cm::IndexList>& src_lists) -> NameSourcesMap;

void setPrimaryClueNameSourcesMap(NameSourcesMap&& nameSourcesMap);

void setCompoundClueNames(int count, std::vector<std::string>& name_list);

const cm::IndexList& getSourcesForPrimaryClueName(const std::string& name);

bool is_known_name_count(const std::string& name, int count);

}  // namespace clue_manager

#endif // INCLUDE_CLUE_MANAGER_H
