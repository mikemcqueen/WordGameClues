// validator.cpp

#include <array>
#include <iostream>
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
  auto src_cref_list0 =
      clue_manager::make_src_cref_list(src_names.at(0), count_list.at(0));
  auto src_cref_list1 =
      clue_manager::make_src_cref_list(src_names.at(1), count_list.at(1));
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
        // copy src0
        SourceData merged_src{src0};
        // merging sources with common names is fine. allow duplicates.
        merged_src.merge_nc_names(src1.nc_names, true);
        // not fine: when an NC name is the same as one of its source names
        if (!merged_src.merge_nc_name(clue_name)) {
          if (log_level(ExtraVerbose)) {
            std::cerr << "failed to merge " << clue_name << ":"
                      << util::sum(count_list) << " to nc_names ["
                      << util::join(merged_src.nc_names, ",") << "]"
                      << std::endl;
            std::cerr << " after merging " << util::join(src1.nc_names, ",")
                      << " of " << NameCount::listToString(src1.ncList)
                      << " with " << util::join(src0.nc_names, ",") << " of "
                      << NameCount::listToString(src0.ncList) << std::endl;
          }
          continue;
        }
        // merge compatibility data
        merged_src.mergeInPlace(src1);
        // copy src1 primaryNameSrcList on top
        std::ranges::copy(src1.primaryNameSrcList,
            std::back_inserter(merged_src.primaryNameSrcList));
        // replace ncList with supplied names/counts
        merged_src.ncList.clear();
        merged_src.ncList.emplace_back(clue_name, util::sum(count_list));
        src_list.push_back(std::move(merged_src));
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
  // don't love this here. would prefer all of this logic were in one place.
  // I suppose this is an "optimization" though.
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
      if (!validate_all && !src_list.empty()) return result;
    } while (std::ranges::next_permutation(count_list).found);
  }
  return result;
};

void show_validator_durations() {}

}  // namespace cm::validator
