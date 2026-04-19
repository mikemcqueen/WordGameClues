// validator.cpp

#include <array>
#include <iostream>
#include "clue-manager.h"
#include "known-sources.h"
#include "peco.h"
#include "util.h"
#include "validator.h"

namespace cm::validator {

namespace {

auto validate_source_names_at_counts(const std::string& clue_name,
    const std::vector<std::string>& src_names, std::vector<int> count_list,
    NameCountList& nc_list) -> SourceComboList {
  auto src_crefs0 = KnownSources::make_src_compat_cref_list(src_names.at(0),
      count_list.at(0));
  auto src_crefs1 = KnownSources::make_src_compat_cref_list(src_names.at(1),
      count_list.at(1));
  // NOTE: in order to support more than 2 sources here, we'd probably have
  // to bite the bullet and merge sources in lists 0,1 then then merge the
  // resulting list with sources from list 2, and so on.
  SourceComboList combo_list;
  index_t idx0{};
  for (const auto src_cref0 : src_crefs0) {
    const auto& src0 = src_cref0.get();
    index_t idx1{};
    for (const auto src_cref1 : src_crefs1) {
      const auto& src1 = src_cref1.get();
      if (src0.isXorCompatibleWith(src1)) {
#if 0
        // Check nc_names compatibility without storing them
        // Build temporary nc_names set for validation
        std::set<std::string> temp_nc_names = src0.nc_names;
        for (const auto& name : src1.nc_names) {
          temp_nc_names.insert(name);  // duplicates OK
        }
        // not fine: when an NC name is the same as one of its source names
        if (!temp_nc_names.insert(clue_name).second) {
          if (log_level(ExtraVerbose)) {
            std::cerr << "failed to merge " << clue_name << ":"
                      << util::sum(count_list) << " to nc_names ["
                      << util::join(temp_nc_names, ",") << "]"
                      << std::endl;
          }
          ++idx1;
          continue;
        }
#endif
        // Create compact SourceCombo with parent references
        SourceCompatibilityData merged_compat{src0.usedSources};
        merged_compat.mergeInPlace(src1);

        SourceParentList parents{
            {src_names.at(0), count_list.at(0), idx0},
            {src_names.at(1), count_list.at(1), idx1}  //
        };
        combo_list.emplace_back(std::move(merged_compat), std::move(parents),
            std::string(clue_name), util::sum(count_list));
      }
      ++idx1;
    }
    ++idx0;
  }
  return combo_list;
}

}  // anonymous namespace

auto validate_sources(const std::string& clue_name,
    const std::vector<std::string>& src_names, int sum,
    bool validate_all) -> SourceComboList {
  SourceComboList result;
  if (src_names.size() != 2) return result;
  auto pred = [&clue_name](const std::string& src_name) {  
    return src_name == clue_name;
  };
  // TODO: test returning same variable at multiple locations
  // don't love this here. would prefer all of this logic were in one place.
  // I suppose this is an "optimization" though.
  if (std::ranges::find_if(src_names, pred) != src_names.end()) {
    return result;
  }
  auto addends = Peco::make_addends(sum, int(src_names.size()), sum);
  for (auto& count_list : addends) {
    do {
      if (!clue_manager::are_known_name_counts(src_names, count_list)) continue;
      NameCountList nc_list;
      auto combo_list = validate_source_names_at_counts(
          clue_name, src_names, count_list, nc_list);
      std::ranges::move(combo_list, std::back_inserter(result));
      if (!validate_all && !combo_list.empty()) return result;
    } while (std::ranges::next_permutation(count_list).found);
  }
  return result;
}

auto is_xor_compatible(const std::vector<std::string>& src_names,
    const std::vector<int>& count_list) -> bool {
  if (src_names.size() != 2) return false;
  if (!clue_manager::are_known_name_counts(src_names, count_list)) return false;
  auto src_crefs0 = KnownSources::make_src_compat_cref_list(
      src_names.at(0), count_list.at(0));
  auto src_crefs1 = KnownSources::make_src_compat_cref_list(
      src_names.at(1), count_list.at(1));
  for (const auto src_cref0 : src_crefs0) {
    for (const auto src_cref1 : src_crefs1) {
      if (src_cref0.get().isXorCompatibleWith(src_cref1.get())) return true;
    }
  }
  return false;
}

void show_validator_durations() {}

}  // namespace cm::validator
