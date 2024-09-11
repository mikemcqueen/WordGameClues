// combo-maker.cpp

#include <cassert>
#include <optional>
#include "candidates.h"
#include "clue-manager.h"
#include "combo-maker.h"
#include "peco.h"
#include "util.h"

namespace cm {

namespace {

auto next_index(const std::vector<int>& count_list,
    std::vector<int>& clue_indices) {
  assert(!clue_indices.empty());
  auto idx = int(clue_indices.size()) - 1;
  ++clue_indices.at(idx);
  // while last index is maxed: reset to zero, increment next-to-last index
  while (clue_indices.at(idx)
         == clue_manager::get_num_unique_clue_names(count_list.at(idx))) {
    clue_indices.at(idx) = 0;
    if (--idx < 0) return false;
    ++clue_indices.at(idx);
  }
  return true;
}

std::optional<NameCountCRefList> next(const std::vector<int>& count_list,
    std::vector<int>& clue_indices) {
  for (;;) {
    if (!next_index(count_list, clue_indices)) {
      return std::nullopt;
    }
    NameCountCRefList nc_cref_list;
    auto good{true};
    for (size_t idx{}; idx < count_list.size(); ++idx) {
      const auto count = count_list.at(idx);
      const auto& nc = clue_manager::get_unique_clue_nc(count,
          clue_indices.at(idx));
      if (nc_cref_list.size()) {
        // because we are only comparing to ncList[0].name
        assert((nc_cref_list.size() < 2ul) && "logic broken");
        // no duplicate names allowed
        if (nc_cref_list.front().get().name == nc.name) {
          good = false;
          break;
        }
      }
      nc_cref_list.push_back(std::cref(nc));
    }
    if (good) {
      NameCount::listSort(nc_cref_list);
      return {std::move(nc_cref_list)};
    }
  }
  assert(0);
  return std::nullopt;
}

std::optional<NameCountCRefList> first(const std::vector<int>& count_list,
    std::vector<int>& clue_indices) {
  clue_indices.clear();
  for (const auto count : count_list) {
    if (!clue_manager::get_num_unique_clue_names(count)) return std::nullopt;
    clue_indices.push_back(0);
  }
  clue_indices.back() = -1;
  return next(count_list, clue_indices);
}

}  // anonymous namespace

void compute_combos_for_sum(int sum, int max) {
  // Given a sum, such as 4, and a max # of numbers to combine, such as 4,
  // generate an array of addend arrays ("count lists"), for each 2 <= N <= max,
  // that add up to that sum, such as [ [1, 3], [2, 2], [1, 1, 2], [1, 1, 1, 1] ]
  // in reality though much of this code only works for max == 2.
  auto addends = Peco::make_addends(sum, max);
  int num_candidates{};
  std::vector<int> clue_indices;
  auto t = util::Timer::start_timer();
  for (const auto& count_list: addends) {
    assert(count_list.size() == 2 && "count_list.size() != 2");
    auto result = first(count_list, clue_indices);
    if (!result.has_value()) continue;
    do {
      ++num_candidates;
      consider_candidate(result.value());
      result = next(count_list, clue_indices);
    } while (result.has_value());
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
