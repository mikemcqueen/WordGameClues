#ifndef INCLUDE_CANDIDATES_H
#define INCLUDE_CANDIDATES_H

#include <functional>
#include <unordered_map>
#include <set>
#include <string>
#include <vector>
#include "combo-maker.h"

namespace cm {

// types

using SourceCompatibilityLists = std::vector<SourceCompatibilityList>;

struct CandidateData {
  std::reference_wrapper<const SourceCompatibilityList> src_list_cref;
  std::set<std::string> combos;  // TODO: why is this a set vs. unordered_set?
};

using CandidateList = std::vector<CandidateData>;

// functions

inline auto count_candidates(const CandidateList& candidates) {
  size_t num{};
  for (const auto& candidate : candidates) {
    num += candidate.src_list_cref.get().size();
  }
  return num;
}

void consider_candidate(const NameCountList& ncList, int sum);

void filterCandidates(
  int sum, int threads_per_block, int streams, int stride, int iters);

// globals

inline std::unordered_map<int, CandidateList> allSumsCandidateData{};

}  // namespace cm

#endif // INCLUDE_CANDIDATES_H
