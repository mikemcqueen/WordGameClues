#pragma once

#include <cstdint>
#include <functional>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "name-count.h"

namespace cm {

struct SourceData;
using SourceList = std::vector<SourceData>;
using SourceCRef = std::reference_wrapper<const SourceData>;
using SourceCRefList = std::vector<SourceCRef>;

// Identifies a parent source for reconstruction of merged sources
struct SourceParent {
  std::string name;
  int count;
  uint32_t idx;  // index within the source list for (name, count)
};

using SourceParentList = std::vector<SourceParent>;

// Compact form - used during precomputation and compatibility checking
// Stores only compatibility data + parent references for later reconstruction
struct SourceCombo : SourceCompatibilityData {
  SourceCombo() = default;
  SourceCombo(SourceCompatibilityData&& compat, SourceParentList&& parent_list,
      std::string&& clue_name, int clue_count)
      : SourceCompatibilityData(std::move(compat)),
        parents(std::move(parent_list)),
        nc(std::move(clue_name), clue_count) {}

  SourceCombo(const SourceCombo&) = default;
  SourceCombo& operator=(const SourceCombo&) = default;
  SourceCombo(SourceCombo&&) = default;
  SourceCombo& operator=(SourceCombo&&) = default;

  SourceParentList parents;  // lineage for reconstruction
  NameCount nc{"", 0};       // the clue name:count for this combo
};
using SourceComboList = std::vector<SourceCombo>;

// Full form - ALWAYS has populated lists, used for final output
struct SourceData : SourceCompatibilityData {
  SourceData() = default;
  SourceData(NameCountList&& primaryNameSrcList, NameCountList&& ncList,
      UsedSources&& usedSources)
      : SourceCompatibilityData(std::move(usedSources)),
        primaryNameSrcList(std::move(primaryNameSrcList)),
        ncList(std::move(ncList)) {}

  // constructor for list.emplace
  // used_sources is const-ref because it doesn't benefit from move
  SourceData(const UsedSources& used_sources,
      NameCountList&& primary_name_src_list, NameCountList&& nc_list,
      std::set<std::string>&& nc_names)
      : SourceCompatibilityData(used_sources),
        primaryNameSrcList(std::move(primary_name_src_list)),
        ncList(std::move(nc_list)), nc_names(std::move(nc_names)) {}

  // copy assign allowed for now for precompute.mergeAllCompatibleXorSources
  SourceData(const SourceData&) = default;
  SourceData& operator=(const SourceData&) = default;
  SourceData(SourceData&&) = default;
  SourceData& operator=(SourceData&&) = default;

  static void dumpList(const SourceList& src_list);
  static void dumpList(const SourceCRefList& src_cref_list);

  NameCountList primaryNameSrcList;
  NameCountList ncList;
  std::set<std::string> nc_names;
  // NOTE: SourceData is ALWAYS fully populated. Use SourceCombo for compact storage.
};

using SourceListCRef = std::reference_wrapper<const SourceList>;
using SourceListCRefList = std::vector<SourceListCRef>;
using SourceListMap = std::unordered_map<std::string, SourceList>;

using XorSource = SourceData;
using XorSourceList = std::vector<XorSource>;

struct MergedSources : SourceCompatibilityData {
  MergedSources() = default;
  MergedSources(const MergedSources&) = default;  // allow, dangerous?
  MergedSources& operator=(const MergedSources&) = delete;
  MergedSources(MergedSources&&) = default;
  MergedSources& operator=(MergedSources&&) = default;

  // copy from SourceData
  MergedSources(const SourceData& source)
      : SourceCompatibilityData(source.usedSources),
        sourceCRefList(SourceCRefList{SourceCRef{source}}) {}

  SourceCRefList sourceCRefList;
};

using MergedSourcesList = std::vector<MergedSources>;
using StringList = std::vector<std::string>;

// functions

inline constexpr void assert_valid(const SourceList& src_list) {
  for (const auto& src : src_list) {
    src.assert_valid();
  }
}

}  // namespace cm
