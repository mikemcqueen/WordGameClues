// candidates.cpp

#include <cassert>
#include <chrono>
#include <optional>
#include <unordered_map>
#include "combo-maker.h"
#include "candidates.h"

namespace {

using namespace cm;

// types

struct MergeData {
  SourceCompatibilityData merged_src;
  SourceCRefList src_cref_list;
};

struct CandidateRepoValue {
  SourceCompatibilityList merged_src_list;
  std::optional<int> opt_idx;
};

// globals

std::unordered_map<std::string, CandidateRepoValue> candidate_repo;

// functions

auto make_merge_list(const SourceCompatibilityList& merged_src_list,
  const SourceList& src_list) {
  //
  std::vector<MergeData> merge_list;
  for (const auto& merged_src : merged_src_list) {
    SourceCRefList compat_src_list;
    for (const auto& src : src_list) {
      if (merged_src.isXorCompatibleWith(src)) {
        compat_src_list.emplace_back(std::cref(src));
      }
    }
    if (!compat_src_list.empty()) {
      merge_list.push_back(MergeData{merged_src, std::move(compat_src_list)});
      //merge_list.emplace_back(merged_src, std::move(compat_src_list));
    }
  }
  return merge_list;
}

auto make_merged_src_list(const std::vector<MergeData>& merge_list) {
  SourceCompatibilityList merged_src_list;
  for (const auto& merge_data : merge_list) {
    for (const auto src_cref : merge_data.src_cref_list) {
      merged_src_list.emplace_back(
        merge_data.merged_src.copyMerge(src_cref.get()));
    }
  }
  return merged_src_list;
}

SourceCompatibilityList merge_all_compatible_sources(
  const NameCountList& ncList, const SourceListMap& src_list_map) {
  //
  assert(ncList.size() <= 2 && "ncList.size() > 2");
  SourceCompatibilityList merged_src_list;
  for (const auto& nc : ncList) {
    const auto& src_list = src_list_map.find(nc.toString())->second;
    if (merged_src_list.empty()) {
      merged_src_list.assign(src_list.begin(), src_list.end()); // deep copy of elements
      continue;
    }
    // TODO: optimization opportunity. Walk through loop building merge_lists
    // first. If any are empty, bail. After all succeed, do actual merges.
    const auto merge_list = make_merge_list(merged_src_list, src_list);
    if (merge_list.empty()) {
      return {};
    }
    merged_src_list = make_merged_src_list(merge_list);
  }
  return merged_src_list;
}

auto add_candidate(int sum, const std::string&& combo, int index) -> int {
  auto& candidate_list = allSumsCandidateData.find(sum)->second;
  candidate_list.at(index).combos.emplace(std::move(combo));
  return index;
}

int add_candidate(int sum, std::string&& combo,
  std::reference_wrapper<const SourceCompatibilityList> src_list_cref) {
  //
  if (auto it = allSumsCandidateData.find(sum);
      it == allSumsCandidateData.end()) {
    allSumsCandidateData.emplace(std::make_pair(sum, CandidateList{}));
  }
  std::set<std::string> combos{};
  combos.emplace(std::move(combo));
  // TODO: this move probably doesn't work without a specific constructor
  CandidateData candidate_data{src_list_cref, std::move(combos)};
  auto& candidate_list = allSumsCandidateData.find(sum)->second;
  candidate_list.emplace_back(std::move(candidate_data));
  return candidate_list.size() - 1;
}

}  // namespace

namespace cm {

void consider_candidate(const NameCountList& ncList, int sum) {
  auto key = NameCount::listToString(ncList);
  if (candidate_repo.find(key) == candidate_repo.end()) {
    CandidateRepoValue repo_value;
    repo_value.merged_src_list =
      std::move(merge_all_compatible_sources(ncList, PCD.sourceListMap));
    candidate_repo.emplace(std::make_pair(key, std::move(repo_value)));
  }
  auto& repo_value = candidate_repo.find(key)->second;
  if (repo_value.merged_src_list.empty()) {
    return;
  }
  auto combo = NameCount::listToString(NameCount::listToNameList(ncList));
  if (repo_value.opt_idx.has_value()) {
    add_candidate(sum, std::move(combo), repo_value.opt_idx.value());
  } else {
    repo_value.opt_idx = add_candidate(
      sum, std::move(combo), std::cref(repo_value.merged_src_list));
  }
}

} // namespace cm

