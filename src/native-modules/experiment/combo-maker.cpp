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

/*

const nextIndex = (countList: number[], clueIndices: number[]): boolean => {
    // increment last index
    let index = clueIndices.length - 1;
    clueIndices[index] += 1;
    // while last index is maxed: reset to zero, increment next-to-last index, etc.
    while (clueIndices[index] === ClueManager.getUniqueClueNameCount(countList[index])) {
        clueIndices[index] = 0;
        if (--index < 0) {
            return false;
        }
        clueIndices[index] += 1;
    }
    return true;
};

export interface FirstNextResult {
    done: boolean;
    ncList?: NameCount.List;
}
*/

auto next_index(const std::vector<int>& count_list,
    std::vector<int>& clue_indices) {
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

/*
export const next = (countList: number[], clueIndices: number[]): FirstNextResult => {
    for (;;) {
        if (!nextIndex(countList, clueIndices)) {
            return { done: true };
        }
        let ncList: NameCount.List = [];    // e.g. [ { name: "pollock", count: 2 }, { name: "jackson", count: 4 } ]
        if (countList.every((count, index) => {
            //if (skip(count, clueIndices[index])) return false;
            let name = ClueManager.getUniqueClueName(count, clueIndices[index]);
            if (ncList.length) {
                // because we are only comparing to ncList[0].name
                Assert((ncList.length < 2) && "logic broken");
                // no duplicate names allowed
                if (ncList[0].name === name) return false;
            }
            // TODO: ncList.push({ name, count });
            ncList.push(NameCount.makeNew(name, count));
            return true; // every.continue;
        })) {
            NameCount.sortList(ncList);
            return { done: false, ncList };
        }
    }
};
*/

std::optional<NameCountList> next(const std::vector<int>& count_list,
    std::vector<int>& clue_indices) {
  for (;;) {
    if (!next_index(count_list, clue_indices)) {
      return {};
    }
    NameCountList nc_list;
    auto every = true;
    for (size_t idx{}; idx < count_list.size(); ++idx) {
      const auto count = count_list.at(idx);
      const auto& name = clue_manager::get_unique_clue_name(count,
          clue_indices.at(idx));
      if (nc_list.size()) {
        // because we are only comparing to ncList[0].name
        assert((nc_list.size() < 2ul) && "logic broken");
        // no duplicate names allowed
        if (nc_list.front().name == name) {
          every = false;
          break;
        }
      }
      nc_list.emplace_back(name, count);
    }
    if (every) {
      NameCount::listSort(nc_list);
      return {std::move(nc_list)};
    }
  }
  assert(0);
  return {};
}

/*
export const first = (countList: number[], clueIndices: number[]): FirstNextResult => {
    // TODO: _.fill?
    for (let index = 0; index < countList.length; ++index) {
        if (ClueManager.getUniqueClueNameCount(countList[index]) === 0) {
            return { done: true };
        }
        clueIndices[index] = 0;
    }
    clueIndices[clueIndices.length - 1] = -1;
    return next(countList, clueIndices);
};
*/

std::optional<NameCountList> first(const std::vector<int>& count_list,
    std::vector<int>& clue_indices) {
  clue_indices.clear();
  for (const auto count : count_list) {
    if (!clue_manager::get_num_unique_clue_names(count)) {
      return std::nullopt;
    }
    clue_indices.push_back(0);
  }
  clue_indices.back() = -1;
  return next(count_list, clue_indices);
}

}  // anonymous namespace

/*
//
// args:
//   synonymMinMax
//
const getCombosForUseNcLists = (sum: number, max: number, args: any): void => {
    let comboCount = 0;
    let totalVariations = 0;
    
    const MILLY = 1000000n;
    const start = process.hrtime.bigint();

    // Given a sum, such as 4, and a max # of numbers to combine, such as 4, generate
    // an array of addend arrays ("count lists"), for each 2 <= N <= max, that add up
    // to that sum, such as [ [1, 3], [2, 2], [1, 1, 2], [1, 1, 1, 1] ]
    let countListArray: number[][] = Peco.makeNew({ sum, max }).getCombinations(); 

    let candidateCount = 0;
    // for each countList
    countListArray.forEach((countList: number[]) => {
        comboCount += 1;

        let clueIndices: number[] = [];
        let result = first(countList, clueIndices);
        if (result.done) return; // continue; 

        let numVariations = 1;

        // this is effectively Peco.getCombinations().forEach()
        let firstIter = true;
        while (!result.done) {
            if (!firstIter) {
                result = next(countList, clueIndices);
                if (result.done) break;
                numVariations += 1;
            } else {
                firstIter = false;
            }
            Native.considerCandidate(result.ncList!);
        }
        totalVariations += numVariations;
    });

    let duration = (process.hrtime.bigint() - start) / MILLY;
    Debug(`sum(${sum}) combos(${comboCount}) variations(${totalVariations})` +
        ` -${duration}ms`);

    // enhancing visibility of JS duration coz it's starting to matter
    if (1 || args.verbose) {
        console.error(`sum(${sum}) consider(JS) - combos(${comboCount})` +
            ` variations(${totalVariations}) - ${duration}ms `);
    }
    Native.filterCandidatesForSum(sum, args.tpb, args.streams, args.stride,
        args.iters, args.synchronous);
};
*/

void compute_combos_for_sum(int sum, int max) {
  auto t = util::Timer::start_timer();
  // Given a sum, such as 4, and a max # of numbers to combine, such as 4,
  // generate an array of addend arrays ("count lists"), for each 2 <= N <= max,
  // that add up to that sum, such as [ [1, 3], [2, 2], [1, 1, 2], [1, 1, 1, 1] ]
  // in reality though much of this code only works for max == 2.
  auto addends = Peco::make_addends(sum, max);
  int total_candidates{};
  std::vector<int> clue_indices;
  for (const auto& count_list: addends) {
    auto result = first(count_list, clue_indices);
    if (!result.has_value()) continue;
    int num_candidates{};
    do {
      ++num_candidates;
      consider_candidate(result.value());
      result = next(count_list, clue_indices);
    } while (result.has_value());
    total_candidates += num_candidates;
  }
  t.stop();

  // enhancing visibility of JS duration coz it's starting to matter
  if (log_level(Normal)) {
    std::cerr << "sum(" << sum << ") consider(C++) - count_lists("
              << addends.size() << ")"
              << " candidates(" << total_candidates << ") - " << t.count()
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
