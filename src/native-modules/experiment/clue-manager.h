#ifndef INCLUDE_CLUE_MANAGER_H
#define INCLUDE_CLUE_MANAGER_H

#include <string>
#include <unordered_map>
#include <vector>
#include "cuda-types.h"

namespace clue_manager {

using NameSourcesMap = std::unordered_map<std::string, cm::IndexList>;

void buildNameSourcesMap(
  std::vector<std::string>& names, std::vector<cm::IndexList>& src_lists);

const cm::IndexList& getSourcesForPrimaryClueName(const std::string& name);

}  // namespace clue_manager

#endif // INCLUDE_CLUE_MANAGER_H
