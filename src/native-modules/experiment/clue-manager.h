#ifndef INCLUDE_CLUE_MANAGER_H
#define INCLUDE_CLUE_MANAGER_H

#include <string>
#include <unordered_map>
#include <vector>
#include "cuda-types.h"
// not thrilled about this. it's really a header organization issue though
#include "combo-maker.h"

namespace clue_manager {

// types

struct NcResultData {
  cm::SourceList src_list;
  std::unordered_set<cm::SourceCompatibilityData> src_compat_set;
};
using NcResultMap = std::unordered_map<std::string, NcResultData>;

using NameSourcesMap = std::unordered_map<std::string, cm::IndexList>;

// functions

// ncResultMaps
auto get_known_nc_source(const cm::NameCount& nc, int idx) -> cm::SourceData;

auto getNumNcResults(const cm::NameCount& nc) -> int;

void appendNcResults(const cm::NameCount& nc, cm::SourceList& src_list);

int appendNcResultsFromSourceMap(
  const cm::NameCount& nc, const std::string& src_key);

// nameSourcesMaps
auto buildNameSourcesMap(std::vector<std::string>& names,
  std::vector<cm::IndexList>& src_lists) -> NameSourcesMap;

void setPrimaryClueNameSourcesMap(NameSourcesMap&& nameSourcesMap);

void setCompoundClueNames(int count, std::vector<std::string>& name_list);

bool is_known_source_map_entry(int count, const std::string& key);

void init_known_source_map_entry(int count,
  const std::vector<std::string>& name_list, cm::SourceList&& src_list);

const cm::IndexList& getSourcesForPrimaryClueName(const std::string& name);

bool is_known_name_count(const std::string& name, int count);

}  // namespace clue_manager

#endif // INCLUDE_CLUE_MANAGER_H
