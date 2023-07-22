#ifndef INCLUDE_CANDIDATES_H
#define INCLUDE_CANDIDATES_H

#include <unordered_map>
#include <vector>
#include "combo-maker.h"

namespace cm {
  using SourceCompatibilityLists = std::vector<SourceCompatibilityList>;

  // TODO: map is dumb. use vector. ComboLists
  using IndexComboListMap = std::unordered_map<int, std::set<std::string>>;
  
  struct OneSumCandidateData {
    SourceCompatibilityLists sourceCompatLists;
    IndexComboListMap indexComboListMap;
  };

  //

  auto addCandidate(int sum, const std::string& combo, int index) -> int;
  auto addCandidate(int sum, std::string&& combo,
    cm::SourceCompatibilityList&& compatList) -> int;

  [[nodiscard]] XorSource* cuda_allocCopyXorSources(
    const XorSourceList& xorSourceList, const std::vector<int> sortedIndices);

  [[nodiscard]]
  auto cuda_allocCopySentenceVariationIndices(
    const SentenceVariationIndices& sentenceVariationIndices)
    -> device::VariationIndices*;

  void filterCandidates(int sum);

  //
  
  inline std::unordered_map<int, OneSumCandidateData> allSumsCandidateData{};
} // namespace cm

#endif // INCLUDE_CANDIDATES_H
