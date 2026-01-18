#include <cassert>
#include <functional>
#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "cm-precompute.h"
#include "known-sources.h"
#include "log.h"
#include "util.h"
#include "cm-hash.h"

namespace cm {

namespace {

// Create a SourceCombo wrapper for a primary SourceData
// The combo has a single parent ref pointing to the primary
SourceCombo make_combo_from_primary(const SourceData& src, const std::string& name,
    index_t idx) {
  SourceCombo combo;
  combo.usedSources = src.usedSources;
  combo.parents = {{name, 1, idx}};  // count is always 1 for primaries
  combo.nc = src.ncList.at(0);
  return combo;
}

// Merge two SourceCombos into a new SourceCombo
// Only merges compat data and combines parents (no data list merging)
SourceCombo mergeSourceCombos(const SourceCombo& combo1, const SourceCombo& combo2) {
  SourceCombo result;
  result.usedSources = combo1.usedSources.copyMerge(combo2.usedSources);
  // Combine parents from both combos
  result.parents = combo1.parents;
  result.parents.insert(result.parents.end(),
      combo2.parents.begin(), combo2.parents.end());
  // The merged combo's NC combines the counts
  // (though this isn't used for filtering, just for tracking)
  // TODO: what
  result.nc = NameCount("merged", combo1.nc.count + combo2.nc.count);
  return result;
}

// Get SourceComboList for an NC
// For count == 1, wraps primaries in SourceCombo
// For count > 1, returns the stored SourceComboList directly
auto get_combo_list_for_nc(const NameCount& nc) -> SourceComboList {
  SourceComboList result;
  if (nc.count == 1) {
    // Primary sources: wrap each SourceData in a SourceCombo
    index_t idx{};
    KnownSources::for_each_primary_source(nc.name,
        [&result, &nc, &idx](const SourceData& src, index_t) {
          result.push_back(make_combo_from_primary(src, nc.name, idx));
          ++idx;
        });
  } else {
    // Compound sources: already stored as SourceComboList
    KnownSources::for_each_combo_source(nc.name, nc.count,
        [&result](const SourceCombo& combo, index_t) {
          result.push_back(combo);
        });
  }
  return result;
}

auto mergeCompatibleComboLists(
    const SourceComboList& combo_list1, const SourceComboList& combo_list2) {
  SourceComboList result{};
  for (const auto& combo1 : combo_list1) {
    for (const auto& combo2 : combo_list2) {
      if (combo1.isXorCompatibleWith(combo2)) {
        result.push_back(mergeSourceCombos(combo1, combo2));
      }
    }
  }
  return result;
}

// NOTE: for ncList.size() <= 2
auto mergeAllCompatibleSources(const NameCountList& ncList) -> SourceComboList {
  assert(ncList.size() <= 2 && "ncList.length > 2");
  const auto logging = false;

  auto combo_list = get_combo_list_for_nc(ncList[0]);
  if (logging) {
    std::cerr << "nc[0]: " << ncList[0].toString() << " (" << combo_list.size()
              << ")" << std::endl;
  }

  for (auto i = 1u; i < ncList.size(); ++i) {
    auto combo_list2 = get_combo_list_for_nc(ncList[i]);
    if (logging) {
      std::cerr << " nc[" << i << "]: " << ncList[i].toString() << " ("
                << combo_list2.size() << ")" << std::endl;
    }
    combo_list = mergeCompatibleComboLists(combo_list, combo_list2);
    if (combo_list.empty()) break;
  }
  return combo_list;
}

}  // namespace

auto build_src_lists(const std::vector<NCDataList>& nc_data_lists)
    -> std::vector<SourceComboList> {
  using StringSet = std::unordered_set<std::string>;
  using HashMap = std::unordered_map<SourceCompatibilityData, StringSet>;

  srand(-1);
  size_t total_sources{};
  int hash_hits = 0;
  const auto size = nc_data_lists[0].size();
  std::vector<HashMap> hashList(size);
  std::vector<SourceComboList> comboLists(size);

  for (const auto& nc_data_list : nc_data_lists) {
    assert(nc_data_list.size() == size);
    for (size_t i{}; i < nc_data_list.size(); ++i) {
      auto combo_list = mergeAllCompatibleSources(nc_data_list[i].ncList);
      total_sources += combo_list.size();
      for (auto& combo : combo_list) {
        // De-duplicate based on SourceCompatibilityData
        const SourceCompatibilityData& key = combo;
        if (hashList[i].find(key) != hashList[i].end()) {
          hash_hits++;
          continue;
        }
        hashList[i][key] = StringSet{};
        comboLists[i].push_back(std::move(combo));
      }
    }
  }

  if (log_level(Verbose)) {
    std::cerr << " total sources: " << total_sources
              << ", hash_hits: " << hash_hits
              << ", comboLists: " << comboLists.size()
              << ", combos: " << util::sum_sizes(comboLists) << std::endl;
  }
  return comboLists;
}

}  // namespace cm
