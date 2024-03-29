// candidates.cpp

#include <cassert>
#include <chrono>
#include <numeric>
#include <optional>
#include <unordered_map>
#include "candidates.h"
#include "clue-manager.h"
#include "combo-maker.h"
#include "merge-filter-data.h"

namespace {

using namespace cm;

// types

struct MergeData {
  SourceCompatibilityData merged_src;
  SourceCompatibilityCRefList src_cref_list;
};

struct CandidateRepoValue {
  SourceCompatibilityList merged_src_list;
  std::optional<int> opt_idx;
};

using CandidateRepo = std::unordered_map<std::string, CandidateRepoValue>;

// functions

auto& get_candidate_repo(int sum) {
  constexpr const int kMaxRepos = 20;
  static std::vector<CandidateRepo> candidate_repos(kMaxRepos);
  return candidate_repos.at(sum - 2);
}

auto get_src_compat_list(const NameCount& nc) {
  SourceCompatibilityList src_list;
  clue_manager::for_each_nc_source(
    nc, [&src_list](const SourceCompatibilityData& src) {
      src_list.emplace_back(src);
    });  // comment
  return src_list;
}

auto get_src_compat_cref_list(const NameCount& nc) {
  SourceCompatibilityCRefList src_cref_list;
  clue_manager::for_each_nc_source(
    nc, [&src_cref_list](const SourceCompatibilityData& src) {
      src_cref_list.emplace_back(std::cref(src));
    });
  return src_cref_list;
}

auto make_merge_list(const SourceCompatibilityList& merged_src_list,
  const SourceCompatibilityCRefList& src_cref_list) {
  //
  std::vector<MergeData> merge_list;
  for (const auto& merged_src : merged_src_list) {
    SourceCompatibilityCRefList compat_src_cref_list;
    for (const auto& src_cref : src_cref_list) {
      if (merged_src.isXorCompatibleWith(src_cref.get())) {
        compat_src_cref_list.emplace_back(src_cref);
      }
    }
    if (!compat_src_cref_list.empty()) {
      //merge_list.push_back(MergeData{merged_src, std::move(compat_src_cref_list)});
      merge_list.emplace_back(merged_src, std::move(compat_src_cref_list));
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
  const NameCountList& ncList) {
  //
  assert(ncList.size() <= 2 && "ncList.size() > 2");
  SourceCompatibilityList merged_src_list;
  for (const auto& nc : ncList) {
    if (merged_src_list.empty()) {
      merged_src_list = std::move(get_src_compat_list(nc));
      continue;
    }
    // TODO: optimization opportunity. Walk through loop building merge_lists
    // first. If any are empty, bail. After all succeed, do actual merges.
    const auto& src_cref_list = get_src_compat_cref_list(nc);
    const auto merge_list = make_merge_list(merged_src_list, src_cref_list);
    if (merge_list.empty()) {
      return {};
    }
    merged_src_list = make_merged_src_list(merge_list);
  }
  return merged_src_list;
}

auto add_candidate(int sum, const std::string&& combo, int index) {
  auto& candidate_list = allSumsCandidateData.find(sum)->second;
  candidate_list.at(index).combos.emplace(std::move(combo));
  return index;
}

int add_candidate(int sum, std::string&& combo,
  std::reference_wrapper<const SourceCompatibilityList> src_list_cref) {
  //
  if (!allSumsCandidateData.contains(sum)) {
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

void consider_candidate(const NameCountList& nc_list) {
  auto sum = std::accumulate(nc_list.begin(), nc_list.end(), 0,
    [](int total, const NameCount& nc) { return total + nc.count; });
  auto& candidate_repo = get_candidate_repo(sum);
  auto key = NameCount::listToString(nc_list);
  if (!candidate_repo.contains(key)) {
    CandidateRepoValue repo_value;
    repo_value.merged_src_list =
      std::move(merge_all_compatible_sources(nc_list));
    candidate_repo.emplace(std::make_pair(key, std::move(repo_value)));
  }
  auto& repo_value = candidate_repo.find(key)->second;
  if (repo_value.merged_src_list.empty()) {
    return;
  }
  auto combo = NameCount::listToString(NameCount::listToNameList(nc_list));
  if (repo_value.opt_idx.has_value()) {
    add_candidate(sum, std::move(combo), repo_value.opt_idx.value());
  } else {
    repo_value.opt_idx = add_candidate(
      sum, std::move(combo), std::cref(repo_value.merged_src_list));
  }
}

void clear_candidates(int sum) {
  get_candidate_repo(sum).clear();
}

int count_candidates(const CandidateList& candidates) {
  size_t num{};
  for (const auto& candidate : candidates) {
    num += candidate.src_list_cref.get().size();
  }
  return num;
}

}  // namespace cm
