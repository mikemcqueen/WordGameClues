#include <cassert>
#include "clue-manager.h"

namespace clue_manager {

using namespace cm;

namespace {
NameSourcesMap nameSourcesMap;
}

void buildNameSourcesMap(
  std::vector<std::string>& names, std::vector<IndexList>& src_lists) {
  assert(nameSourcesMap.empty());
  for (size_t i{}; i < names.size(); ++i) {
    nameSourcesMap.emplace(
      std::make_pair(std::move(names.at(i)), std::move(src_lists.at(i))));
  }
}

const IndexList& getSourcesForPrimaryClueName(const std::string& name) {
  auto it = nameSourcesMap.find(name);
  assert(it != nameSourcesMap.end());
  return it->second;
}

}  // namespace clue_manager
