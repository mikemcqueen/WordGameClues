#ifndef INCLUDE_CLUE_MANAGER_H
#define INCLUDE_CLUE_MANAGER_H

#include <functional>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "cuda-types.h"
// not thrilled about this. it's really a header organization issue though
#include "combo-maker.h"

namespace cm::clue_manager {

// types

struct KnownSourceMapValue {
  SourceList src_list;
  std::set<std::string> clue_names;
};
using KnownSourceMapValueCRef = std::reference_wrapper<const KnownSourceMapValue>;

// primary name -> IndexList??
using PrimaryNameSrcIndicesMap = std::unordered_map<std::string, IndexList>;
// name -> source_csv list 
using NameSourcesMap = std::unordered_map<std::string, std::vector<std::string>>;

// functions

// nameSourcesMaps

void set_name_sources_map(int count, NameSourcesMap&& name_sources_map);

bool is_known_name_count(const std::string& name, int count);  // known_nc

bool are_known_name_counts(const std::vector<std::string>& name_list,
    const std::vector<int>& count_list);

const std::vector<std::string>& get_nc_sources(const NameCount& nc);

// knownSourceMaps

bool has_known_source_map(int count);

bool is_known_source_map_entry(int count, const std::string& sources_csv);

// TODO: const ref version of this as an overload
auto get_known_source_map_entry(int count, const std::string& sources_csv)
  -> KnownSourceMapValue&;

void init_known_source_map_entry(
    int count, const std::string source_csv, SourceList&& src_list);

const IndexList& getPrimaryClueSrcIndices(const std::string& name);

auto get_known_source_map_entries(const std::string& name, int count)  //
    -> std::vector<KnownSourceMapValueCRef>;

auto get_known_source_map_entries(const NameCount& nc)  //
    -> std::vector<KnownSourceMapValueCRef>;

bool add_compound_clue(const NameCount& nc, const std::string& sources_csv);

// NOTE: this doesn't properly set nc_list. it could.
auto make_src_list(const NameCount& nc) -> SourceList;

// NOTE: this doesn't properly set nc_list. it can't, without a proxy.
auto make_src_cref_list(const std::string& name, int count) -> SourceCRefList;

inline auto make_src_cref_list(const NameCount& nc) {
  return make_src_cref_list(nc.name, nc.count);
}

// uniqueClueNames

int get_num_unique_clue_names(int count);

const std::string& get_unique_clue_name(int count, int idx);

// misc

/*
inline void for_each_source_map_entry(
    const std::string& name, int count, const auto& fn) {
  for (const auto& entry_cref : get_known_source_map_entries(nc)) {
    fn(entry_cref.get());
  }
}
*/

void for_each_nc_source(const std::string& name, int count, const auto& fn) {
  for (const auto& entry_cref : get_known_source_map_entries(name, count)) {
    const auto& src_list = entry_cref.get().src_list;
    std::ranges::for_each(src_list, fn);
  }
}

inline void for_each_nc_source(const NameCount& nc, const auto& fn) {
  for_each_nc_source(nc.name, nc.count, fn);
}

void init_primary_clues(
    std::vector<std::string>&& names, std::vector<IndexList>&& idx_lists);

void dump_memory(std::string_view header = "clue-manager memory:");

}  // namespace cm::clue_manager

#endif // INCLUDE_CLUE_MANAGER_H
