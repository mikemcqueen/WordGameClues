// candidates.cpp

#include <chrono>
#include "combo-maker.h"
#include "candidates.h"

namespace cm {

auto addCandidate(int sum, const std::string& combo, int index) -> int {
  IndexComboListMap& indexCombosMap =
    allSumsCandidateData.find(sum)->second.indexComboListMap;
  auto it = indexCombosMap.find(index);
  assert(it != indexCombosMap.end());
  it->second.insert(combo);
  return index;
}

auto addCandidate(int sum, std::string&& combo,
  cm::SourceCompatibilityList&& compatList) -> int {
  //
  int index{};
  std::set<std::string> combos{};
  combos.emplace(std::move(combo));

  // TODO this could be simplified. add sumData if new sum, then common code.
  if (auto it = allSumsCandidateData.find(sum);
      it == allSumsCandidateData.end()) {
    SourceCompatibilityLists sourceCompatLists{};
    sourceCompatLists.emplace_back(std::move(compatList));

    IndexComboListMap indexComboListMap;
    auto [ignore, success] =
      indexComboListMap.insert(std::make_pair(index, std::move(combos)));
    assert(success);

    OneSumCandidateData oneSumData{
      std::move(sourceCompatLists), std::move(indexComboListMap)};
    allSumsCandidateData.emplace(std::make_pair(sum, std::move(oneSumData)));
  } else {
    auto& oneSumData = allSumsCandidateData.find(sum)->second;
    auto& sourceCompatLists = oneSumData.sourceCompatLists;
    sourceCompatLists.emplace_back(std::move(compatList));
    index = sourceCompatLists.size() - 1;

    auto& indexComboListMap = oneSumData.indexComboListMap;
    auto [ignore, success] =
      indexComboListMap.insert(std::make_pair(index, std::move(combos)));
    assert(success);
  }
  return index;
}

} // namespace cm

