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

inline auto count_candidates(const CandidateList& candidates) {
  size_t num{};
  for (const auto& candidate : candidates) {
    num += candidate.src_list_cref.get().size();
  }
  return num;
}

struct SizeIndex {
  index_t size;
  index_t index;
};
using SizeIndexList = std::vector<SizeIndex>;

inline std::pair<SizeIndexList, index_t>
make_size_idx_list(const CandidateList& candidates) {
  SizeIndexList size_idx_list;
  index_t num_sources{};
  for (index_t idx{}; const auto& candidate : candidates) {
    index_t size = candidate.src_list_cref.get().size();
    // TODO weird. this should be more straightforward?
    size_idx_list.emplace_back(SizeIndex{size, idx++});
    num_sources += size;
  }
  return {size_idx_list, num_sources};
}

void consider_candidate(const NameCountList& ncList, int sum);

// globals

inline CandidateMap allSumsCandidateData{};

}  // namespace cm

#endif // INCLUDE_CANDIDATES_H
