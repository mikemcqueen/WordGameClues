#ifndef INCLUDE_CANDIDATES_H
#define INCLUDE_CANDIDATES_H

#include <functional>
#include <unordered_map>
//#include <utility>
#include <set>
#include <string>
#include <vector>
#include "combo-maker.h"

namespace cm {

// types

using SourceCompatibilityLists = std::vector<SourceCompatibilityList>;
// TODO: map is dumb. use vector. ComboLists
//using IndexComboListMap = std::unordered_map<int, std::set<std::string>>;

struct CandidateData {
  std::reference_wrapper<const SourceCompatibilityList> src_list_cref;
  std::set<std::string> combos;  // TODO: why is this a set vs. unordered_set?
};

using CandidateList = std::vector<CandidateData>;

/*
struct OneSumCandidateData {
  SourceCompatibilityLists sourceCompatLists;
  IndexComboListMap indexComboListMap;
};
*/
  
// functions

void consider_candidate(const NameCountList& ncList, int sum);

int add_candidate(int sum, const std::string&& combo, int index);
int add_candidate(
  int sum, std::string&& combo, SourceCompatibilityList&& src_list);

void filterCandidates(
  int sum, int threads_per_block, int streams, int stride, int iters);

// globals

inline std::unordered_map<int, CandidateList> allSumsCandidateData{};

}  // namespace cm

#endif // INCLUDE_CANDIDATES_H
