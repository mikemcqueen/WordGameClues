// validator.cpp

#include <array>
#include <iostream>
#include <vector>
#include "clue-manager.h"
#include "combo-maker.h"
#include "peco.h"
#include "util.h"
#include "validator.h"

namespace cm::validator {

namespace {

auto validateSourceNamesAtCounts(const std::string& clue_name,
    const std::vector<std::string>& src_names, std::vector<int> count_list,
    NameCountList& nc_list) -> SourceList {
  // NOTE: this could also just be a nested loop. depends how slow it is.
  // build lists of src_crefs for NCs 0 and 1 that have no name conflicts
  // temporary: ignore name conflicts; eventual: for_each_nc_entry
  SourceCRefList src_cref_list0;
  clue_manager::for_each_nc_source(src_names.at(0), count_list.at(0),
      [&list = src_cref_list0](const SourceData& src) {  //
        list.emplace_back(std::cref(src));
      });
  SourceCRefList src_cref_list1;
  clue_manager::for_each_nc_source(src_names.at(1), count_list.at(1),
      [&list = src_cref_list1](const SourceData& src) {  //
        list.emplace_back(std::cref(src));
      });
  // NOTE: in order to support more than 2 sources here, we'd probably have
  // to bite the bullet and merge sources in lists 0,1 then then merge the
  // resulting list with sources from list 2, and so on.
  SourceList src_list;
  for (const auto src_cref0 : src_cref_list0) {
    const auto& src0 = src_cref0.get();
    for (const auto src_cref1 : src_cref_list1) {
      const auto& src1 = src_cref1.get();
      if (src0.isXorCompatibleWith(src1)) {
        // TODO: SourceData.copyMerge(const SourceData&)
        // copy
        SourceData merged_src{src0};
        // merge
        merged_src.mergeInPlace(src1);
        std::ranges::copy(src1.primaryNameSrcList,
            std::back_inserter(merged_src.primaryNameSrcList));
        std::ranges::copy(src1.ncList, std::back_inserter(merged_src.ncList));
        src_list.emplace_back(std::move(merged_src));
      }
    }
  }
  return src_list;
}

}  // anonymous namespace

auto validateSources(const std::string& clue_name,
    const std::vector<std::string>& src_names, int sum,
    bool validate_all) -> SourceList {
  SourceList result;
  if (src_names.size() != 2) return result;
  auto pred = [&clue_name](const std::string& src_name) {  //
    return src_name == clue_name;
  };
  // TODO: test returning same variable at multiple locations
  if (std::ranges::find_if(src_names, pred) != src_names.end()) {
    return result;
  }
  auto addends = Peco::make_addends(sum, src_names.size());
  for (auto& count_list : addends) {
    do {
      if (!clue_manager::are_known_name_counts(src_names, count_list)) continue;
      NameCountList nc_list;
      auto src_list = validateSourceNamesAtCounts(
          clue_name, src_names, count_list, nc_list);
      std::ranges::move(src_list, std::back_inserter(result));
      if (!validate_all && !result.empty()) return result;
    } while (std::ranges::next_permutation(count_list).found);
  }
  return result;
};

void show_validator_durations() {}

}  // namespace cm::validator
