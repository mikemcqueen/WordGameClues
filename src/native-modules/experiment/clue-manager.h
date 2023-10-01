#ifndef INCLUDE_CLUE_MANAGER_H
#define INCLUDE_CLUE_MANAGER_H

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>
#include "cuda-types.h"
// not thrilled about this. it's really a header organization issue though
#include "combo-maker.h"

namespace clue_manager {

// types
struct KnownSourceMapValue {
  cm::SourceList src_list;
};
using KnownSourceMapValueCRef = std::reference_wrapper<const KnownSourceMapValue>;

struct UniqueSources {
  cm::SourceList src_list;
  std::unordered_set<cm::SourceCompatibilityData> src_compat_set;
};
using NcSourcesMap = std::unordered_map<std::string, UniqueSources>;

using PrimaryNameSrcIndicesMap = std::unordered_map<std::string, cm::IndexList>;
// name -> src_csv_list 
using NameSourcesMap = std::unordered_map<std::string, std::vector<std::string>>;

// functions

// ncSourcesMaps

auto get_nc_src_list(const cm::NameCount& nc) -> const cm::SourceList&;

auto get_num_nc_sources(const cm::NameCount& nc) -> int;

int append_nc_sources_from_known_source(
  const cm::NameCount& nc, const std::string& known_src_csv);

// NOTE: this doesn't properly set nc_list. it could.
auto make_src_list_for_nc(const cm::NameCount& nc) -> cm::SourceList;

// NOTE: this doesn't properly set nc_list. it can't, without a proxy.
auto make_src_cref_list_for_nc(const cm::NameCount& nc) -> cm::SourceCRefList;

// nameSourcesMaps

auto buildPrimaryNameSrcIndicesMap(std::vector<std::string>& names,
  std::vector<cm::IndexList>& src_lists) -> PrimaryNameSrcIndicesMap;

void setPrimaryNameSrcIndicesMap(PrimaryNameSrcIndicesMap&& src_indices_map);

void setNameSourcesMap(int count, NameSourcesMap&& name_sources_map);

// knownSourceMaps

bool is_known_source_map_entry(int count, const std::string& src_csv);

void init_known_source_map_entry(int count,
  const std::vector<std::string>& name_list, cm::SourceList&& src_list);

const cm::IndexList& getPrimaryClueSrcIndices(const std::string& name);

bool is_known_name_count(const std::string& name, int count);  // known_nc

auto get_known_source_map_entries(const cm::NameCount& nc)
  -> std::vector<KnownSourceMapValueCRef>;

// uniqueClueNames

int get_num_unique_clue_names(int count);

const std::string& get_unique_clue_name(int count, int idx);

inline void for_each_nc_source(const cm::NameCount& nc, const auto& fn) {
  for (const auto& entry_cref : get_known_source_map_entries(nc)) {
    const auto& src_list = entry_cref.get().src_list;
    std::for_each(src_list.begin(), src_list.end(), fn);
  }
}

}  // namespace clue_manager

#endif // INCLUDE_CLUE_MANAGER_H
