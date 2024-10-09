// combo-maker.cpp

#include <cassert>
#include <optional>
#include "candidates.h"
#include "clue-manager.h"
#include "combo-maker.h"
#include "cm-hash.h"
#include "peco.h"
#include "util.h"

namespace cm {

namespace {

auto next_index(const std::vector<int>& count_list,
    IndexList& unique_name_indices) {
  assert(!unique_name_indices.empty());
  auto idx = int(unique_name_indices.size()) - 1;
  ++unique_name_indices.at(idx);
  // while last index is maxed: reset to zero, increment next-to-last index
  while (unique_name_indices.at(idx)
      == index_t(clue_manager::get_num_unique_clue_names(count_list.at(idx)))) {
    unique_name_indices.at(idx) = 0;
    if (--idx < 0) return false;
    ++unique_name_indices.at(idx);
  }
  return true;
}

std::optional<NameCountCRefList> next(const std::vector<int>& count_list,
    IndexList& unique_name_indices) {
  for (;;) {
    if (!next_index(count_list, unique_name_indices)) {
      return std::nullopt;
    }
    NameCountCRefList nc_cref_list;
    for (size_t idx{}; idx < count_list.size(); ++idx) {
      const auto count = count_list.at(idx);
      const auto& nc = clue_manager::get_unique_clue_nc(count,
          unique_name_indices.at(idx));
      if (!nc_cref_list.size()) {
        nc_cref_list.push_back(std::cref(nc));
      } else {
        // no duplicate names allowed
        if (nc_cref_list.at(0).get().name != nc.name) {
          nc_cref_list.push_back(std::cref(nc));
          return std::make_optional(std::move(nc_cref_list));
        }
      }
    }
  }
  assert(0);
  return std::nullopt;
}

std::optional<NameCountCRefList> first(const std::vector<int>& count_list,
    IndexList& unique_name_indices) {
  unique_name_indices.clear();
  for (const auto count : count_list) {
    if (!clue_manager::get_num_unique_clue_names(count)) return std::nullopt;
    unique_name_indices.push_back(0);
  }
  unique_name_indices.back() = -1;
  return next(count_list, unique_name_indices);
}

/*
auto make_unique_name_prefix_sums(int sum) {
  IndexList prefix_sums = {0};
  for (int i{1}; i < clue_manager::get_num_unique_clue_names(sum) - 1; ++i) {
    const auto& name = clue_manager::get_unique_clue_name(sum, i);
    size_t num_sources{};
    // TODO: expensive call
    for (const auto entry_cref :
        clue_manager::get_known_source_map_entries(name, sum)) {
      num_sources += entry_cref.get().src_list.size();
    }
    prefix_sums.push_back(int(prefix_sums.back() + num_sources));
  }
  return prefix_sums;
}
*/

/*
// There is a small space-optimization opportunity here. Up to half of the
// sources per-sum are duplicates. We could eliminate duplicates and preserve
// indices by adding the sources to a map<SourceCRef, index> while we build a
// src_list here. That'd save maybe 8MB (150000 * 56) for c2,14. Going to skip
// this for now in the interest of just getting indexed sources working in the
// first place.

using SourceCRefIndexMap = std::unordered_map<SourceCompatibilityDataCRef, int>;

void test_unique_clue_sources(int sum) {
  SourceCompatibilitySet src_set;
  //  SourceCRefIndexMap src_index_map;
  size_t num_sources{};
  for (int i{0}; i < clue_manager::get_num_unique_clue_names(sum); ++i) {
    const auto& name = clue_manager::get_unique_clue_name(sum, i);
    for (const auto entry_cref : clue_manager::get_known_source_map_entries(name, sum)) {
      const auto& src_list = entry_cref.get().src_list;
      num_sources += src_list.size();
      for (const auto& src : src_list) {
        src_set.insert(src);
      }
    }
  }
  std::cerr << "TEST " << sum << ": sources(" << num_sources << ") unique("
            << src_set.size() << ")\n";
}
*/

}  // anonymous namespace

// Given a sum, such as 4, and a max # of numbers to combine, such as 4,
// generate an array of addend arrays ("count lists"), for each 2 <= N <= max,
// that add up to that sum, such as [ [1, 3], [2, 2], [1, 1, 2], [1, 1, 1, 1] ]
// in reality though much of this code only works for max == 2.
void compute_combos_for_sum(int sum, int max) {
  assert(max == 2 && "max != 2");
  //test_unique_clue_sources(sum);
  //  auto prefix_sums = make_unique_name_prefix_sums(sum);
  auto addends = Peco::make_addends(sum, max);
  int num_candidates{};
  IndexList unique_name_indices;
  auto t = util::Timer::start_timer();
  for (const auto& count_list: addends) {
    for (auto result = first(count_list, unique_name_indices);
        result.has_value(); result = next(count_list, unique_name_indices)) {
      ++num_candidates;
      // this call is a bit weird, a legacy of migrating from list of sources to
      // list of source-indices. "result.value()" (nc_cref_list) arguably isn't
      // the right data type anymore. indices all the way down.
      consider_candidate(sum, result.value(), unique_name_indices);
      // old: consider_candidate(result.value());
    }
  }
  t.stop();
  if (log_level(Normal)) {
    std::cerr << "sum(" << sum << ") consider(C++) - count_lists("
              << addends.size() << ")"
              << " candidates(" << num_candidates << ") - " << t.count()
              << "ms\n";
  }
  //  Native.filterCandidatesForSum(sum, args.tpb, args.streams, args.stride,
  //      args.iters, args.synchronous);
}

/*
export const makeCombosForSum = (sum: number, args: any,
  synchronous: boolean = false): void =>
{
    if (_.isUndefined(args.maxResults)) {
        args.maxResults = 50000;
        // TODO: whereever this is actually enforced:
        // console.error(`Enforcing max results: ${args.maxResults}`);
    }
    args.synchronous = synchronous;
    // TODO: Fix this abomination
    args.sum = sum;
    let max = args.max;
    args.max = Math.min(args.max, args.sum);
    getCombosForUseNcLists(sum, max, args);
    args.max = max;
};

...

// run 2-clue sources synchronously to seed "incompatible sources"
// which makes subsequent sums faster.
makeCombosForSum(2, args, true);
if (first == = 2) ++first;
for (let sum = first; sum <= last; ++sum) {
  // TODO: return # of combos filtered due to note name match
  makeCombosForSum(sum, args);
}
*/

}  // namespace cm
