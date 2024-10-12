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

struct CandidateRepoValue {
  CompatSourceIndicesList compat_src_indices;
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

int get_num_candidate_sources(int sum) {
  Lock lk(candidate_map_semaphore_);
  const auto it = candidate_map_.find(sum);
  return (it == candidate_map_.end()) ? 0 : int(get_num_candidate_sources(it->second));
}

// optimized for 2 sources, using more crefs
//
// ultimately making this a kernel is the path forward due to isXorCompat calls
// ^- Still true? -^
auto make_compat_source_indices(const NameCountCRefList& nc_cref_list,
    const IndexList& unique_name_indices, const std::pair<int, int> idx_pair) {
  assert(nc_cref_list.size() == 2 && "nc_cref_list.size() != 2");
  const auto& nc1 = nc_cref_list.at(idx_pair.first).get();
  const auto& nc2 = nc_cref_list.at(idx_pair.second).get();
  const auto start_idx1 = clue_manager::get_unique_clue_starting_source_index(
      nc1.count, unique_name_indices.at(idx_pair.first));
  const auto start_idx2 = clue_manager::get_unique_clue_starting_source_index(
      nc2.count, unique_name_indices.at(idx_pair.second));
  CompatSourceIndicesList compat_src_indices;

  // TODO: this is ridiculously inefficient because we repeatedly call
  //       get_known_sources_map_entries() within for_each_nc_source().
  clue_manager::for_each_nc_source(nc1,
      [&](const SourceCompatibilityData& src1, index_t idx1) {
        CompatSourceIndex csi1{nc1.count, start_idx1 + idx1};
        clue_manager::for_each_nc_source(nc2,
            [&](const SourceCompatibilityData& src2, index_t idx2) {
              if (src1.isXorCompatibleWith(src2)) {
                CompatSourceIndex csi2{nc2.count, start_idx2 + idx2};
                compat_src_indices.emplace_back(csi1, csi2);
              }
            });
      });
  return compat_src_indices;
}

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

std::pair<int, int> get_sorted_index_pair(
    const NameCountCRefList& nc_cref_list) {
  const auto& first_nc = nc_cref_list.at(0).get();
  const auto& second_nc = nc_cref_list.at(1).get();
  // effectively "sort" NCs
  const auto first_idx = first_nc.name < second_nc.name ? 0 : 1;
  const auto second_idx = 1 - first_idx;
  return std::make_pair(first_idx, second_idx);
}

void consider_candidate(int sum, const NameCountCRefList& nc_cref_list,
    const IndexList& unique_name_indices) {
  auto idx_pair = get_sorted_index_pair(nc_cref_list);
  const auto& nc1 = nc_cref_list.at(idx_pair.first).get();
  const auto& nc2 = nc_cref_list.at(idx_pair.second).get();
  // no lock necessary to modify repo map for a particular sum
  auto& candidate_repo = get_candidate_repo(sum);
  auto key = NameCount::makeString(nc1, nc2);
  if (candidate_repo.contains(key)) return;
  CandidateRepoValue repo_value;
  repo_value.compat_src_indices = std::move(
      make_compat_source_indices(nc_cref_list, unique_name_indices, idx_pair));
  candidate_repo.emplace(key, std::move(repo_value));
  auto& cc = get_candidate_counts_ref(sum);
  cc.num_considers++;
  auto& rv = candidate_repo.find(key)->second;
  if (rv.compat_src_indices.empty()) {
    cc.num_incompat++;
  } else {
    add_candidate(sum, NameCount::makeString(nc1.name, nc2.name),
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
