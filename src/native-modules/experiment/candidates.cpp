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
#include "known-sources.h"
#include "merge-filter-data.h"
#include "cm-hash.h"

// debugging
#include "filter.cuh"
#include <atomic>
#include <array>

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

std::atomic<long> make_indices_duration_ns_(0);

// optimized for 2 sources, using more crefs
//
// ultimately making this a kernel is the path forward due to isXorCompat calls
// Still true? Probably, eventually, when it becomes too slow.
//
// So. One problem here is duplicate sources. There are a lot of them.
// Eliminating those would probably cut the number of sources in half, which
// would cut the time by 3/4ths. Because would go from ~ 20K^2 tests to
// 10K^2 tests - 400K to 100K.
//
// maybe instead of calling make_compat_src_indices for each nc_cref_list,
// which is really an nc_cref_pair, we could just build 2 nc_cref_lists
// from .at(0) and .at(1).
//
// consider_candidate would just dump an empty compat_src_indices list into
// the repo map. after we're done with all consider_candidate calls, we
// execute a "bulk" make_compat_src_indices.
//
// what would this look like? Well the main goal is to eliminate duplicates
// at least at first. so we walk both nc_lists and add all ...
//
// ok there are a couple options here that I need to think through.
//
// 1) maps from SourceCompatibilityDataCRef -> IndexList of unique indices.
//    in fact, i think i could build these incrementally during each call to
//    consider_candidate. then the "compute_all_indices()" function just
//    walks they keys in each map testing isXorCompat, and if true, adds
//    every index combination.
//    - Too slow. doing dynamic allocation, specifically creating the
//      std::vector<int> for each source is too much. there's 100M sources
//      combined. I got it down to ~900ms total by emplacing a std::array<int,
//      10> but unfortunately some sources are shared ~2000 times so that's not
//      enough space to hold all the indices. and 100M * 2000*4 is too much
//      host memory.
//
// 2) my original idea, was a map of SourceCompatibilityDataCRef -> bool,
//    initialized to false. in the outer loop, once a source is used, set
//    it to true. any future same sources in outer-loop will skip.
//    inner loop is a problem though. need to reset every entry to false at
//    the end of each inner loop. So #1 looks better.
//
// maybe there's another idea here.
//
// 3) is there some way I can leverage unique variations? like, i get it,
//    there are 35M sources across all sources for all nc_pairs in c15,
//    but what if there were only 60K unique variations, and somehow each
//    UV pointed to a list/set of sources for that variation?
//

struct Indices {
  //  std::array<index_t, 10> array;
  IndexList indices;
  index_t max{};
};

// using Indices = std::array<index_t, 10>;

std::unordered_map<index_t, Indices> source_indices_map_;
//std::unordered_map<SourceCompatibilityDataCRef, Indices> source_indices_map_;

void add_map_entry(const SourceCompatibilityData& src) {
  Indices idx_list;
  static index_t index = 0;
  source_indices_map_.emplace(index++, std::move(idx_list));
  //source_indices_map_.emplace(std::cref(src), std::move(idx_list));
}

index_t max_ = 0;

void add_compat_source_index(const SourceCompatibilityData& src, index_t idx) {
  static index_t index = 0;
  auto it = source_indices_map_.find(index++);
  //auto it = source_indices_map_.find(std::cref(src));
  assert(it != source_indices_map_.end());
  //  it->second.array.at(0) = idx;
  if (++it->second.max > max_) {
    max_ = it->second.max;
  }
}

auto make_compat_source_indices(const NameCountCRefList& nc_cref_list,
    const IndexList& unique_name_indices, const std::pair<int, int> idx_pair) {
  assert(nc_cref_list.size() == 2 && "nc_cref_list.size() != 2");
  const auto& nc1 = nc_cref_list.at(idx_pair.first).get();
  const auto start_idx1 = clue_manager::get_unique_clue_starting_source_index(
      nc1.count, unique_name_indices.at(idx_pair.first));
  const auto& nc2 = nc_cref_list.at(idx_pair.second).get();
  const auto start_idx2 = clue_manager::get_unique_clue_starting_source_index(
      nc2.count, unique_name_indices.at(idx_pair.second));

  CompatSourceIndicesList src_indices;
#if 0
  KnownSources::for_each_nc_source_compat_data(nc1,
      [](const SourceCompatibilityData& src1, index_t idx1) {
        add_map_entry(src1);
      });
  KnownSources::for_each_nc_source_compat_data(nc2,
      [](const SourceCompatibilityData& src2, index_t idx2) {
        add_map_entry(src2);
      });

  int n{};
  auto t = util::Timer::start_timer();
  KnownSources::for_each_nc_source_compat_data(nc1,
      [&n, start_idx1](const SourceCompatibilityData& src1, index_t idx1) {
        add_compat_source_index(src1, start_idx1 + idx1);
        n++;
      });
  KnownSources::for_each_nc_source_compat_data(nc2,
      [&n, start_idx2](const SourceCompatibilityData& src2, index_t idx2) {
        add_compat_source_index(src2, start_idx2 + idx2);
        n++;
      });
#else
  auto t = util::Timer::start_timer();
  KnownSources::for_each_nc_source_compat_data(nc1,
      [count = nc1.count, start_idx1, &nc2, start_idx2, &src_indices]  //
      (const SourceCompatibilityData& src1, index_t idx1) {
        CompatSourceIndex csi1{count, start_idx1 + idx1};
        KnownSources::for_each_nc_source_compat_data(nc2,
            [count = nc2.count, start_idx2, csi1, &src1, &src_indices]  //
            (const SourceCompatibilityData& src2, index_t idx2) {
              if (src1.isXorCompatibleWith(src2)) {
                CompatSourceIndex csi2{count, start_idx2 + idx2};
                src_indices.emplace_back(csi1, csi2);
              }
            });
      });
#endif
  t.stop();
  auto ns = t.nanoseconds();
  make_indices_duration_ns_.fetch_add(ns);
  return src_indices;
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

void start_considering() {
  source_indices_map_.clear();
  max_ = 0;
}

void finish_considering() {
  // CompatSourceIndex csi1{nc1.count, start_idx1 + idx1};
  /*
  if (src1.isXorCompatibleWith(src2)) {
    CompatSourceIndex csi2{nc2.count, start_idx2 + idx2};
    compat_src_indices.emplace_back(csi1, csi2);
  }
  */
  //std::cerr << " max: " << max_ << std::endl;
}

int consider_candidate(int sum, const NameCountCRefList& nc_cref_list,
    const IndexList& unique_name_indices) {
  auto idx_pair = get_sorted_index_pair(nc_cref_list);
  const auto& nc1 = nc_cref_list.at(idx_pair.first).get();
  const auto& nc2 = nc_cref_list.at(idx_pair.second).get();
  // no lock necessary to modify repo map for a particular sum
  auto& candidate_repo = get_candidate_repo(sum);
  auto key = NameCount::makeString(nc1, nc2);
  if (candidate_repo.contains(key)) return 0;
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
    // TODO: util::append
    add_candidate(sum, NameCount::makeString(nc1.name, nc2.name),
        std::cref(rv.compat_src_indices));
  }
  return 0;
}

long get_make_indices_duration() {
  return long((double)make_indices_duration_ns_.load() / 1e6);
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
