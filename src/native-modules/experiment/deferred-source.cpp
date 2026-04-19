#include <cassert>
#include <iostream>
#include "deferred-source.h"
#include "known-sources.h"

namespace cm::deferred_source {

namespace {

void collect_primary_name_src_list(const DeferredSourceData& combo,
    NameCountList& result) {
  for (const auto& known_nci : combo.known_nci_list) {
    if (known_nci.nc.count <= 0) {
      std::cerr << "ERROR: collect_primary_name_src_list: invalid count: "
                << known_nci.nc.count << ", name: " << known_nci.nc.name
                << ", idx: " << known_nci.index << std::endl;
      assert(false && "invalid known_nci count");
    }
    if (known_nci.nc.count == 1) {
      const auto& primary = KnownSources::get_primary_source(known_nci);
      result.insert(result.end(), primary.primaryNameSrcList.begin(),
          primary.primaryNameSrcList.end());
    } else {
      const auto& parent_combo = KnownSources::get_deferred_source(known_nci);
      collect_primary_name_src_list(parent_combo, result);
    }
  }
}

void collect_nc_names(const DeferredSourceData& combo,
    std::set<std::string>& result) {
  for (const auto& known_nci : combo.known_nci_list) {
    if (known_nci.nc.count == 1) {
      const auto& primary = KnownSources::get_primary_source(known_nci);
      result.insert(primary.nc_names.begin(), primary.nc_names.end());
    } else {
      const auto& parent_combo = KnownSources::get_deferred_source(known_nci);
      collect_nc_names(parent_combo, result);
    }
  }
  result.insert(combo.nc.name);
}

}  // namespace

auto from_primary(const SourceData& src, const std::string& name, index_t idx)
    -> DeferredSourceData {
  assert(!src.ncList.empty());
  DeferredSourceData deferred_source;
  deferred_source.usedSources = src.usedSources;
  deferred_source.known_nci_list = {{{name, 1}, idx}};
  deferred_source.nc = src.ncList.at(0);
  return deferred_source;
}

auto combine(const DeferredSourceData& first, const DeferredSourceData& second)
    -> DeferredSourceData {
  DeferredSourceData result;
  result.usedSources = first.usedSources.copyMerge(second.usedSources);
  result.known_nci_list = first.known_nci_list;
  result.known_nci_list.insert(result.known_nci_list.end(),
      second.known_nci_list.begin(), second.known_nci_list.end());
  result.nc = NameCount("merged", first.nc.count + second.nc.count);
  return result;
}

auto materialize(const DeferredSourceData& combo, MaterializeMode mode)
    -> SourceData {
  NameCountList nc_list;
  nc_list.emplace_back(combo.nc.name, combo.nc.count);

  if (mode == MaterializeMode::NcListOnly) {
    return SourceData(combo.usedSources, NameCountList{}, std::move(nc_list),
        std::set<std::string>{});
  }

  NameCountList primary_name_src_list;
  collect_primary_name_src_list(combo, primary_name_src_list);

  std::set<std::string> nc_names;
  collect_nc_names(combo, nc_names);

  return SourceData(combo.usedSources, std::move(primary_name_src_list),
      std::move(nc_list), std::move(nc_names));
}

auto materialize_selected(const std::vector<index_t>& combo_indices,
    const std::vector<DeferredSourceDataList>& combo_lists, MaterializeMode mode)
    -> SourceData {
  NameCountList primary_name_src_list;
  NameCountList nc_list;
  UsedSources used_sources;

  for (size_t i{}; i < combo_indices.size(); ++i) {
    const auto& combo = combo_lists.at(i).at(combo_indices.at(i));
    used_sources.mergeInPlace(combo.usedSources);

    if (mode == MaterializeMode::NcListOnly) {
      nc_list.emplace_back(combo.nc.name, combo.nc.count);
      continue;
    }

    const auto src = materialize(combo, mode);
    primary_name_src_list.insert(primary_name_src_list.end(),
        src.primaryNameSrcList.begin(), src.primaryNameSrcList.end());
    nc_list.insert(nc_list.end(), src.ncList.begin(), src.ncList.end());
  }

  assert(!nc_list.empty() && "empty ncList");
  if (mode == MaterializeMode::Full) {
    assert(!primary_name_src_list.empty() && "empty primaryNameSrcList");
  }
  return SourceData(std::move(primary_name_src_list), std::move(nc_list),
      std::move(used_sources));
}

}  // namespace cm::deferred_source
