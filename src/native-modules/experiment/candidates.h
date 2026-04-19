#pragma once

#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include "cuda-types.h"
#include "name-count.h"

namespace cm {

// aliases/types

using FatIndexListCRef = std::reference_wrapper<const FatIndexList>;

struct CandidateCounts {
  size_t num_considers;
  size_t num_incompat;
  size_t num_candidates;
  size_t num_sources;
};

struct CandidateData {
  CompatSourceIndicesListCRef compat_src_indices_cref;
  std::unordered_set<std::string> combos;
};
using CandidateList = std::vector<CandidateData>;

// functions

// I'll get it right eventually.
/*
void consider_candidate(const NameCountList& ncList);
void consider_candidate(const NameCountCRefList& nc_cref_list);
void consider_candidate(int sum, const NameCountCRefList& nc_cref_list);
*/
int consider_candidate(int sum, const NameCountCRefList& nc_cref_list,
    const IndexList& unique_name_indices);

void clear_candidates(int sum);

auto get_candidates(int sum) -> const CandidateList&;

auto get_num_candidate_sources(const CandidateList& candidates) -> size_t;

void save_current_candidate_counts(int sum);

auto get_candidate_counts(int sum) -> CandidateCounts;

}  // namespace cm
