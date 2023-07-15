#ifndef INCLUDE_CANDIDATES_H
#define INCLUDE_CANDIDATES_H

#include <unordered_map>
#include <vector>
#include "combo-maker.h"

namespace cm {
  using SourceCompatibilityLists = std::vector<SourceCompatibilityList>;
  // TODO: map is dumb. use vector. ComboLists
  using IndexComboListMap = std::unordered_map<int, std::vector<std::string>>;
  
  struct OneSumCandidateData {
    SourceCompatibilityLists sourceCompatLists;
    IndexComboListMap indexComboListMap;
  };

  //
  
  inline std::vector<OneSumCandidateData> allSumsCandidateData{};
  
  //

  auto addCandidate(int sum, std::string&& combo, int index) -> int;
  auto addCandidate(int sum, std::string&& combo,
    cm::SourceCompatibilityList&& compatList) -> int;

  void filterCandidates(int sum);
}

#endif // INCLUDE_CANDIDATES_H
