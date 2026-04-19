#include <cassert>
#include <functional>
#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "cm-precompute.h"
#include "deferred-source.h"
#include "known-sources.h"
#include "log.h"
#include "util.h"
#include "cm-hash.h"

namespace cm {

namespace {

// Get DeferredSourceDataList for an NC.
// For count == 1, wraps primaries in DeferredSourceData.
// For count > 1, returns the stored DeferredSourceDataList directly.
auto get_combo_list_for_nc(const NameCount& nc) -> DeferredSourceDataList {
  DeferredSourceDataList result;
  if (nc.count == 1) {
    // Primary sources: wrap each SourceData in DeferredSourceData.
    int idx{};
    KnownSources::for_each_primary_source(nc.name,
        [&result, &nc, &idx](const SourceData& src, index_t) {
          result.push_back(deferred_source::from_primary(src, nc.name, idx));
          ++idx;
        });
  } else {
    // Compound sources: already stored as DeferredSourceDataList.
    KnownSources::for_each_combo_source(nc.name, nc.count,
        [&result](const DeferredSourceData& combo, index_t) {
          result.push_back(combo);
        });
  }
  return result;
}

auto mergeCompatibleComboLists(
    const DeferredSourceDataList& combo_list1,
    const DeferredSourceDataList& combo_list2) {
  DeferredSourceDataList result{};
  for (const auto& combo1 : combo_list1) {
    for (const auto& combo2 : combo_list2) {
      if (combo1.isXorCompatibleWith(combo2)) {
        result.push_back(deferred_source::combine(combo1, combo2));
      }
    }
  }
  return result;
}

// NOTE: for ncList.size() <= 2
auto mergeAllCompatibleSources(const NameCountList& ncList)
    -> DeferredSourceDataList {
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
    -> std::vector<DeferredSourceDataList> {
  using StringSet = std::unordered_set<std::string>;
  using HashMap = std::unordered_map<SourceCompatibilityData, StringSet>;

  srand(-1);
  size_t total_sources{};
  int hash_hits = 0;
  const auto size = nc_data_lists[0].size();
  std::vector<HashMap> hashList(size);
  std::vector<DeferredSourceDataList> comboLists(size);

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
