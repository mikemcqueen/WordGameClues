#ifndef INCLUDE_CANDIDATES_H
#define INCLUDE_CANDIDATES_H

#include <functional>
#include <unordered_map>
#include <set>
#include <string>
#include <utility>
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
using CandidateMap = std::unordered_map<int, CandidateList>;

// functions

void consider_candidate(const NameCountList& ncList);

void clear_candidates(int sum);

int count_candidates(const CandidateList& candidates);

// globals

inline CandidateMap allSumsCandidateData{};

}  // namespace cm

#endif // INCLUDE_CANDIDATES_H
