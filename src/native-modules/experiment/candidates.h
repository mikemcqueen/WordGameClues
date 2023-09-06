#ifndef INCLUDE_CANDIDATES_H
#define INCLUDE_CANDIDATES_H

#include <future>
#include <unordered_map>
#include <utility>
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

  auto addCandidate(int sum, const std::string& combo, int index) -> int;
  auto addCandidate(int sum, std::string&& combo,
    cm::SourceCompatibilityList&& compatList) -> int;

  void filterCandidates(
    int sum, int threads_per_block, int streams, int stride, int iters);
  void filterCandidatesCuda(
    int sum, int threads_per_block, int streams, int stride, int iters);

  using filter_result_t = std::unordered_set<std::string>;
  filter_result_t get_filter_result();

  inline std::unordered_map<int, OneSumCandidateData> allSumsCandidateData{};
} // namespace cm

#endif // INCLUDE_CANDIDATES_H
