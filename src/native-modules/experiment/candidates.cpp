// candidates.cpp

#include <algorithm>
#include <cassert>
#include <chrono>
#include <numeric>
#include <optional>
#include <semaphore>
#include <unordered_map>
#include "candidates.h"
#include "clue-manager.h"
#include "combo-maker.h"
#include "merge-filter-data.h"

// debugging
#include "filter.cuh"

namespace cm {

namespace {

// types

/*
// TODO: no longer used
struct MergeData {
  SourceCompatibilityData merged_src;
  SourceCompatibilityCRefList src_cref_list;
};

// TODO: no longer used
struct MergeData2 {
  SourceCompatibilityDataCRef src_cref;
  SourceCompatibilityCRefList src_cref_list;
};
*/

struct CandidateRepoValue {
  CompatSourceIndicesList compat_src_indices;
  // TODO: no longer used
  //SourceCompatibilityList merged_src_list;
  // TODO: opt_idx no longer used really
  std::optional<int> opt_idx;
};
using CandidateRepo = std::unordered_map<std::string, CandidateRepoValue>;

class Lock {
  Lock() = delete;

public:
  Lock(std::binary_semaphore& semaphore) : semaphore_(semaphore) {
    semaphore_.acquire();
  }
  ~Lock() noexcept {
    semaphore_.release();
  }
private:
  std::binary_semaphore& semaphore_;
};

// globals

constexpr const int kMaxRepos = 20;
std::vector<CandidateRepo> candidate_repos_(kMaxRepos);
std::binary_semaphore candidate_repos_semaphore_{1};

std::unordered_map<int, CandidateList> candidate_map_;
std::binary_semaphore candidate_map_semaphore_{1};

std::unordered_map<int, CandidateCounts> candidate_counts_;
std::binary_semaphore candidate_counts_semaphore_{1};

// functions

auto& get_candidate_repo(int sum) {
  assert(sum >=2 && "sum < 2");
  Lock lk(candidate_repos_semaphore_);
  return candidate_repos_.at(sum - 2);
}

/*
auto get_src_compat_list(const NameCount& nc) {
  SourceCompatibilityList src_list;
  clue_manager::for_each_nc_source(nc,
      [&src_list](const SourceCompatibilityData& src, index_t) {
        src_list.push_back(src);
      });
  return src_list;
}

// TODO: clue_manager exports this function already
auto make_source_cref_list(const NameCount& nc) {
  SourceCompatibilityCRefList src_cref_list;
  clue_manager::for_each_nc_source(nc,
      [&src_cref_list](const SourceCompatibilityData& src, index_t) {
        src_cref_list.push_back(std::cref(src));
      });
  return src_cref_list;
}

auto make_merge_list(const SourceCompatibilityList& merged_src_list,
    const SourceCompatibilityCRefList& src_cref_list) {
  std::vector<MergeData> merge_list;
  for (const auto& merged_src : merged_src_list) {
    SourceCompatibilityCRefList compat_src_cref_list;
    for (const auto& src_cref : src_cref_list) {
      if (merged_src.isXorCompatibleWith(src_cref.get())) {
        compat_src_cref_list.push_back(src_cref);
      }
    }
    if (!compat_src_cref_list.empty()) {
      merge_list.emplace_back(merged_src, std::move(compat_src_cref_list));
    }
  }
  return merge_list;
}

auto make_merge_list(const SourceCompatibilityCRefList& src_cref_list1,
    const SourceCompatibilityCRefList& src_cref_list2) {
  std::vector<MergeData2> merge_list;
  for (const auto src_cref1 : src_cref_list1) {
    SourceCompatibilityCRefList compat_src_cref_list;
    for (const auto src_cref2 : src_cref_list2) {
      if (src_cref1.get().isXorCompatibleWith(src_cref2.get())) {
        compat_src_cref_list.push_back(src_cref2);
      }
    }
    if (!compat_src_cref_list.empty()) {
      merge_list.emplace_back(src_cref1, std::move(compat_src_cref_list));
    }
  }
  return merge_list;
}

auto make_merged_src_list(const std::vector<MergeData>& merge_list) {
  SourceCompatibilityList merged_src_list;
  for (const auto& merge_data : merge_list) {
    for (const auto src_cref : merge_data.src_cref_list) {
      merged_src_list.push_back(merge_data.merged_src.copyMerge(src_cref.get()));
    }
  }
  return merged_src_list;
}

auto make_merged_src_list(const std::vector<MergeData2>& merge_list) {
  SourceCompatibilityList merged_src_list;
  for (const auto& merge_data : merge_list) {
    for (const auto src_cref : merge_data.src_cref_list) {
      merged_src_list.push_back(
          merge_data.src_cref.get().copyMerge(src_cref.get()));
    }
  }
  return merged_src_list;
}

SourceCompatibilityList merge_all_compatible_sources(
    const NameCountList& nc_list) {
  assert(nc_list.size() <= 2 && "nc_list.size() > 2");
  SourceCompatibilityList merged_src_list;
  for (const auto& nc : nc_list) {
    if (merged_src_list.empty()) {
      merged_src_list = std::move(get_src_compat_list(nc));
      continue;
    }
    // TODO: optimization opportunity. Walk through loop building merge_lists
    // first. If any are empty, bail. After all succeed, do actual merges.
    const auto& src_cref_list = make_source_cref_list(nc);
    const auto merge_list = make_merge_list(merged_src_list, src_cref_list);
    if (merge_list.empty()) {
      return {};
    }
    merged_src_list = std::move(make_merged_src_list(merge_list));
  }
  return merged_src_list;
}

// optimized for 2 sources, using more crefs
// ultimately making this a kernel is the path forward due to isXorCompat calls
SourceCompatibilityList merge_all_compatible_sources(
    const NameCountCRefList& nc_cref_list) {
  assert(nc_cref_list.size() == 2 && "nc_cref_list.size() != 2");
  const auto src_cref_list1 = make_source_cref_list(nc_cref_list.at(0).get());
  const auto src_cref_list2 = make_source_cref_list(nc_cref_list.at(1).get());
  const auto merge_list = make_merge_list(src_cref_list1, src_cref_list2);
  if (merge_list.empty()) return {};
  return make_merged_src_list(merge_list);
}

auto make_merge_list(const SourceCompatibilityCRefList& src_cref_list1,
    const SourceCompatibilityCRefList& src_cref_list2) {
  std::vector<MergeData2> merge_list;
  for (const auto src_cref1 : src_cref_list1) {
    SourceCompatibilityCRefList compat_src_cref_list;
    for (const auto src_cref2 : src_cref_list2) {
      if (src_cref1.get().isXorCompatibleWith(src_cref2.get())) {
        compat_src_cref_list.push_back(src_cref2);
      }
    }
    if (!compat_src_cref_list.empty()) {
      merge_list.emplace_back(src_cref1, std::move(compat_src_cref_list));
    }
  }
  return merge_list;
}
*/

// optimized for 2 sources, using more crefs
//
// ultimately making this a kernel is the path forward due to isXorCompat calls
// ^- Still true? -^
auto make_compat_source_indices(const NameCountCRefList& nc_cref_list,
    const IndexList& unique_name_indices) {
  assert(nc_cref_list.size() == 2 && "nc_cref_list.size() != 2");
  const auto& first_nc = nc_cref_list.at(0).get();
  const auto& second_nc = nc_cref_list.at(1).get();
  // effectively "sort" NCs (and unique_name_indices).
  const auto nc1_idx = first_nc.name < second_nc.name ? 0 : 1;
  const auto nc2_idx = 1 - nc1_idx;
  const auto& nc1 = nc_cref_list.at(nc1_idx).get();
  const auto& nc2 = nc_cref_list.at(nc2_idx).get();

  const auto start_idx1 = clue_manager::get_unique_clue_starting_source_index(
      nc1.count, unique_name_indices.at(nc1_idx));
  const auto start_idx2 = clue_manager::get_unique_clue_starting_source_index(
      nc2.count, unique_name_indices.at(nc2_idx));
  CompatSourceIndicesList compat_src_indices;

  // TODO: this is ridiculously inefficient because we repeatedly call
  //       get_known_sources_map_entries() within for_each_nc_source().
  // debugging
  //  static int n{};

  clue_manager::for_each_nc_source(nc1,
      [&](const SourceCompatibilityData& src1, index_t idx1) {
        CompatSourceIndex csi1{nc1.count, start_idx1 + idx1};

#if 0
        if (n < 10) {
          std::cerr << "count1: " << csi1.count()
                    << ", name_idx: " << unique_name_indices.at(0)
                    << ", num_sources: "
                    << clue_manager::get_num_unique_clue_sources(nc1.count,
                           unique_name_indices.at(nc1_idx))
                    << ", src_index: " << csi1.index()
                    << ", start_idx: " << start_idx1
                    << ", idx: " << idx1
                    << std::endl;
        }
#endif

        clue_manager::for_each_nc_source(nc2,
            [&](const SourceCompatibilityData& src2, index_t idx2) {
              if (src1.isXorCompatibleWith(src2)) {
                CompatSourceIndex csi2{nc2.count, start_idx2 + idx2};

#if 0
                if (n < 10) {
                  std::cerr
                      << "count2: " << csi2.count()
                      << ", name_idx: " << unique_name_indices.at(1)
                      << ", num_sources: "
                      << clue_manager::get_num_unique_clue_sources(nc2.count,
                             unique_name_indices.at(nc2_idx))
                      << ", src_index: " << csi2.index()
                      << ", start_idx: " << start_idx2 << ", idx: " << idx2
                      << std::endl;
                }
#endif

#if 0
                CompatSourceIndices indices{csi1, csi2};
                if (n++ < 10) {
                  printf("adding: first (%u, %u) second (%u, %u) from idx1 %u "
                         "idx2 %u\n",
                      csi1.count(), csi1.index(), csi2.count(), csi2.index(),
                      idx1, idx2);
                  dump_compat_src_indices(&indices, 1, "result");
                }
                compat_src_indices.push_back(indices);
#else
                compat_src_indices.emplace_back(csi1, csi2);
#endif
              }
            });
      });
  return compat_src_indices;
}

/*
auto add_candidate(int sum, std::string&& combo, int index) {
  // need to lock/find only because get_candidates() returns const reference.
  // if this were in a class this could be satisfied with a private non-const
  // get_candidates() accessor.
  Lock lk(candidate_map_semaphore_);
  auto& candidates = candidate_map_.find(sum)->second;
  candidates.at(index).combos.insert(std::move(combo));
  return index;
}

auto add_candidate(int sum, std::string&& combo,
    SourceCompatibilityListCRef src_list_cref) {
  std::unordered_set<std::string> combos{};
  combos.insert(std::move(combo));
  Lock lk(candidate_map_semaphore_);
  if (!candidate_map_.contains(sum)) {
    auto [_, success] = candidate_map_.emplace(sum, CandidateList{});
    assert(success);
  }
  // can't use get_candidates() here because it returns a const reference.
  auto& candidates = candidate_map_.find(sum)->second;
  candidates.emplace_back(std::move(src_list_cref), std::move(combos));
  return int(candidates.size()) - 1;
}
*/

auto add_candidate(int sum, std::string&& combo,
    CompatSourceIndicesListCRef compat_src_indices_cref) {
  std::unordered_set<std::string> combos{};
  combos.insert(std::move(combo));
  Lock lk(candidate_map_semaphore_);
  if (!candidate_map_.contains(sum)) {
    auto [_, success] = candidate_map_.emplace(sum, CandidateList{});
    assert(success);
  }
  // can't use get_candidates() here because it returns a const reference.
  auto& candidates = candidate_map_.find(sum)->second;
  candidates.emplace_back(compat_src_indices_cref, std::move(combos));
  return int(candidates.size()) - 1;
}

auto get_candidate_counts_ref(int sum) -> CandidateCounts& {
  Lock lk(candidate_counts_semaphore_);
  if (!candidate_counts_.contains(sum)) {
    candidate_counts_.emplace(sum, CandidateCounts{0, 0, 0, 0});
  }
  return candidate_counts_.find(sum)->second;
}

}  // anonymous namespace

/*
void consider_candidate(const NameCountList& nc_list) {
  auto sum = std::accumulate(nc_list.begin(), nc_list.end(), 0,
    [](int total, const NameCount& nc) { return total + nc.count; });
  auto& candidate_repo = get_candidate_repo(sum);
  // no lock necessary to modify repo map for a particular sum
  // TODO: auto& repo_value = repo_get_or_add(nc_list)
  // that way we can std::move(key), and only one lookup
  auto key = NameCount::listToString(nc_list);
  if (!candidate_repo.contains(key)) {
    CandidateRepoValue repo_value;
    repo_value.merged_src_list =
        std::move(merge_all_compatible_sources(nc_list));
    candidate_repo.emplace(key, std::move(repo_value));
  }
  auto& cc = get_candidate_counts_ref(sum);
  cc.num_considers++;
  auto& repo_value = candidate_repo.find(key)->second;
  if (repo_value.merged_src_list.empty()) {
    cc.num_incompat++;
    return;
  }
  auto combo = NameCount::listToString(NameCount::listToNameList(nc_list));
  if (repo_value.opt_idx.has_value()) {
    std::cerr << "sum: " << sum << ", adding '" << combo << "' to idx "
              << repo_value.opt_idx.value() << std::endl;
    add_candidate(sum, std::move(combo), repo_value.opt_idx.value());
  } else {
    repo_value.opt_idx = add_candidate(sum, std::move(combo),
        std::cref(repo_value.merged_src_list));
  }
}

// TODO: should I be sorting this nc_list?
// is it ever possible that I'll be called with one:1,two:2 followed by
// two:2,one:1?
//

void consider_candidate(const NameCountCRefList& nc_cref_list) {
  auto sum = std::accumulate(nc_cref_list.begin(), nc_cref_list.end(), 0,
      [](int total, const NameCountCRef& nc_cref) {
        return total + nc_cref.get().count;
      });
  auto& candidate_repo = get_candidate_repo(sum);
  // no lock necessary to modify repo map for a particular sum
  // TODO: auto& repo_value = repo_get_or_add(nc_list)
  // that way we can std::move(key), and only one lookup
  auto key = NameCount::listToString(nc_cref_list);
  if (!candidate_repo.contains(key)) {
    CandidateRepoValue repo_value;
    repo_value.merged_src_list =
        std::move(merge_all_compatible_sources(nc_cref_list));
    candidate_repo.emplace(key, std::move(repo_value));
  }
  auto& cc = get_candidate_counts_ref(sum);
  cc.num_considers++;
  auto& repo_value = candidate_repo.find(key)->second;
  if (repo_value.merged_src_list.empty()) {
    cc.num_incompat++;
    return;
  }
  auto combo = NameCount::listToNameCsv(nc_cref_list);
  if (repo_value.opt_idx.has_value()) {
    add_candidate(sum, std::move(combo), repo_value.opt_idx.value());
  } else {
    repo_value.opt_idx = add_candidate(sum, std::move(combo),
        std::cref(repo_value.merged_src_list));
  }
}
*/

void consider_candidate(int sum, const NameCountCRefList& nc_cref_list,
    const IndexList& unique_name_indices) {
  // no lock necessary to modify repo map for a particular sum
  auto& candidate_repo = get_candidate_repo(sum);
  auto key = NameCount::listToString(nc_cref_list);
  if (candidate_repo.contains(key)) return;
  CandidateRepoValue repo_value;
  repo_value.compat_src_indices =
      std::move(make_compat_source_indices(nc_cref_list, unique_name_indices));

#if 0
  // debugging for duplicate sources-index-list count with different 
    const auto it = candidate_map_.find(sum);
    if (!repo_value.compat_src_indices.empty()) {
      assert(it != candidate_map_.end());
      const auto& candidates = it->second;
      const auto combo = NameCount::listToNameCsv(nc_cref_list);
      bool has{false};
      for (const auto& candidate : candidates) {
        if (candidate.combos.empty()) {
          std::cerr << "empty combos\n";
        } else if (candidate.combos.contains(combo)) {
          std::cerr << "already has combo " << combo << std::endl;
          has = true;
          break;
        }
      }
      if (!has) { std::cerr << "no has combo!!" << std::endl; }
      // std::terminate();
    }
#endif

  candidate_repo.emplace(key, std::move(repo_value));
  auto& cc = get_candidate_counts_ref(sum);
  cc.num_considers++;
  auto& rv = candidate_repo.find(key)->second;
  if (rv.compat_src_indices.empty()) {
    cc.num_incompat++;
  } else {
    add_candidate(sum, NameCount::listToNameCsv(nc_cref_list),
        std::cref(rv.compat_src_indices));
  }
}

void clear_candidates(int sum) {
  // no locks strictly necessary here. while the candidate and repo maps are
  // potentially accessed from and modified by two separate threads for a
  // particular sum (main thread, and filter_task thread), their access is
  // deterministically interleaved. by the time the filter task starts, the
  // main thread is done with any modifications (via consider_candidate).
  {
    // lock/find because get_candidates() returns const reference. a private
    // non-const get_candidates() accessor in a class would also work.
    Lock lk(candidate_map_semaphore_);
    candidate_map_.find(sum)->second.clear();
  }
  get_candidate_repo(sum).clear();
}

auto get_num_candidate_sources(const CandidateList& candidates) -> size_t {
  return std::accumulate(candidates.begin(), candidates.end(), 0u,
      [](size_t total, const CandidateData& candidate) {
        return total + candidate.compat_src_indices_cref.get().size();
      });
}

auto get_candidates(int sum) -> const CandidateList& {
  Lock lk(candidate_map_semaphore_);
  const auto it = candidate_map_.find(sum);
  assert((it != candidate_map_.end()) && "no candidates for sum");
  return it->second;
}

void save_current_candidate_counts(int sum) {
  const auto& candidates = get_candidates(sum);
  auto& cc = get_candidate_counts_ref(sum);
  cc.num_candidates = candidates.size();
  cc.num_sources = get_num_candidate_sources(candidates);
}

auto get_candidate_counts(int sum) -> CandidateCounts {
  Lock lk(candidate_counts_semaphore_);
  const auto it = candidate_counts_.find(sum);
  assert((it != candidate_counts_.end()) && "no saved candidate counts for sum");
  return it->second;
}

}  // namespace cm
