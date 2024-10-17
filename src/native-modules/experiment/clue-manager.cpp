#include <cassert>
#include <charconv>
#include <chrono>
#include <cmath>
#include <string>
#include <unordered_set>
#include <vector>
#include "clue-manager.h"
#include "util.h"
#include "log.h"

namespace cm::clue_manager {

namespace {

//
// types/aliases
//

using PrimaryNameSrcIndicesMap = std::unordered_map<std::string, IndexList>;
// TODO: CompoundSrcSourceListMap
using KnownSourceMap = std::unordered_map<std::string, KnownSourceMapValue>;
// using StringCRefList = std::vector<std::reference_wrapper<const std::string>>;

//
// globals
//

// map primary clue_name -> idx_list 
PrimaryNameSrcIndicesMap primaryNameSrcIndicesMap_;

// map clue_name -> source_csv_list for each count (including primary)
std::vector<NameSourcesMap> nameSourcesMaps;

// map source_csv -> { src_list, clue_names_set } for each count
std::vector<KnownSourceMap> knownSourceMaps;

// populated on demand from primaryNameSrcIndicesMap/nameSrcMaps
std::vector<NameCountList> uniqueClueNames_;

SourceCompatibilityList uniquePrimaryClueSrcList_;

// nameSourcesMap

const auto& get_name_sources_map(int count) {
  return nameSourcesMaps.at(count - 1);
}

auto build_primary_name_sources_map(
    const PrimaryNameSrcIndicesMap& name_src_indices_map) {
  NameSourcesMap name_sources_map;
  for (auto& [name, idx_list] : name_src_indices_map) {
    std::vector<std::string> sources;
    for (auto idx : idx_list) {
      sources.push_back(std::move(std::to_string(idx)));
    }
    auto [_, success] = name_sources_map.emplace(name, std::move(sources));
    assert(success);
  }
  return name_sources_map;
}

// knownSourceMap


auto& get_known_source_map(int count, bool force_create = false) {
  // allow force-creates exactly in-sequence only, or throw an exception
  const auto idx = count - 1;
  if (force_create && ((int)knownSourceMaps.size() == idx)) {
    knownSourceMaps.push_back(KnownSourceMap{});
  }
  return knownSourceMaps.at(idx);
}

// Initialize entries of knownSourceMaps_[0] and populate the src_list fields.
void init_primary_known_source_map(
    const PrimaryNameSrcIndicesMap& name_src_indices_map,
    const NameSourcesMap& name_sources_map) {
  using namespace std::chrono;
  util::LogDuration<microseconds> ld(" init primary known_src_map");
  for (const auto& [name, sources] : name_sources_map) {
    for (size_t idx{}; idx < sources.size(); ++idx) {
      const auto name_src = name + ":" + sources.at(idx);
      // TODO: questionable behavior. I think we need to add all varitions now,
      // at leat if the names are different.
      if (is_known_source_map_entry(1, name_src)) {
        // or abort?
        std::cerr << "skip adding '" << name << "' as known primary source " << name_src
                  << std::endl;
        continue;
      }
      int int_src = name_src_indices_map.find(name)->second.at(idx);
      NameCountList primary_name_src_list;
      primary_name_src_list.emplace_back(name, int_src);
      NameCountList nc_list;
      nc_list.emplace_back(name, 1);
      std::set<std::string> nc_names;
      nc_names.insert(name);
      UsedSources used_sources;
      used_sources.addSource(int_src);
      SourceList src_list;
      src_list.emplace_back(used_sources, std::move(primary_name_src_list),
          std::move(nc_list), std::move(nc_names));
      // TODO: std::move(name_src), param as r-value reference
      init_known_source_map_entry(1, name_src, std::move(src_list));
    }
  }
}

// uniqueClueNames

void populate_unique_clue_nc_list(NameCountList& nc_list, int count) {
  for (const auto& [name, _] : get_name_sources_map(count)) {
    nc_list.emplace_back(name, count);
  }
}

const auto& get_unique_clue_nc_list(int count) {
  const auto idx{count - 1};
  if ((int)uniqueClueNames_.size() <= idx) { uniqueClueNames_.resize(idx + 1); }
  auto& nc_list = uniqueClueNames_.at(idx);
  if (nc_list.empty()) {
    populate_unique_clue_nc_list(nc_list, count);
    assert(!nc_list.empty());
  }
  return nc_list;
}

void ensure_primary_unique_clue_source_list() {
  if (uniquePrimaryClueSrcList_.empty()) {
    uniquePrimaryClueSrcList_ = std::move(make_unique_clue_source_list(1));
  }
}

// misc

auto src_list_size(const SourceList& src_list) {
  size_t s{};
  for (const auto& src : src_list) {
    s += sizeof(SourceCompatibilityData);
    for (const auto& nc : src.primaryNameSrcList) {
      s += nc.name.size() + sizeof(int);
    }
    for (const auto& nc : src.ncList) {
      s += nc.name.size() + sizeof(int);
    }
  }
  return s;
}

}  // namespace

//
// primaryNameSrcIndicesMap_
//

auto buildPrimaryNameSrcIndicesMap(std::vector<std::string>&& names,
    std::vector<IndexList>&& idx_lists) -> PrimaryNameSrcIndicesMap {
  PrimaryNameSrcIndicesMap src_indices_map;
  for (size_t i{}; i < names.size(); ++i) {
    auto [_, success] = src_indices_map.emplace(
        std::move(names.at(i)), std::move(idx_lists.at(i)));
    assert(success);
  }
  return src_indices_map;
}

const IndexList& getPrimaryClueSrcIndices(const std::string& name) {
  const auto& map = primaryNameSrcIndicesMap_;
  auto it = map.find(name);
  assert(it != map.end());
  return it->second;
}

//
// nameSourcesMaps
//

void init_known_source_map_entry(
    int count, const std::string source, SourceList&& src_list) {
  // True is arbitrary here. I *could* support replacing an existing src_list,
  // but i'm unaware of any situation that requires it, and as a result I want
  // things to blow up when it is attempted, currently.
  auto& map = get_known_source_map(count, true);
  auto [_, success] = map.emplace(std::move(source), std::move(src_list));
  if (!success) {
    std::cerr << "failed adding " << source << ":" << count
              << " to known_source_map" << std::endl;
  }
  assert(success);
}

void set_name_sources_map(int count, NameSourcesMap&& name_sources_map) {
  auto idx = count - 1;
  // allow sets exactly in-sequence only, or throw an exception
  assert((int)nameSourcesMaps.size() == idx);
  nameSourcesMaps.push_back(std::move(name_sources_map));
}

bool is_known_name_count(const std::string& name, int count) {
  return get_name_sources_map(count).contains(name);
}

bool are_known_name_counts(const std::vector<std::string>& name_list,
    const std::vector<int>& count_list) {
  // TODO: std::zip opportunity
  for (size_t i{}; i < name_list.size(); ++i) {
    if (!is_known_name_count(name_list.at(i), count_list.at(i))) {
      return false;
    }
  }
  return true;
}

const std::vector<std::string>& get_nc_sources(const NameCount& nc) {
  return get_name_sources_map(nc.count).at(nc.name);
}

  /*
const std::vector<std::string>& get_nc_sources(const NameCount& nc) {
  return get_name_sources_map(nc.count).at(nc.name);
}
  */

//
// knownSourceMaps
//

bool has_known_source_map(int count) {
  assert(count > 0);
  return (int)knownSourceMaps.size() >= count;
}

bool is_known_source_map_entry(int count, const std::string& source) {
  const auto idx = count - 1;
  if ((int)knownSourceMaps.size() > idx) {
    return get_known_source_map(count).contains(source);
  }
  return false;
}

auto get_known_source_map_entry(int count, const std::string& source)
    -> KnownSourceMapValue& {
  auto& map = get_known_source_map(count);
  auto it = map.find(source);
  assert(it != map.end());
  return it->second;
}

auto append(
    const std::string& s1, const std::string& s2, const std::string& s3) {
  return s1 + s2 + s3;
}

auto get_known_source_map_entries(const std::string& name,
    int count) -> std::vector<KnownSourceMapValueCRef> {
  std::vector<KnownSourceMapValueCRef> cref_entries;
  const auto& name_sources_map = get_name_sources_map(count);
  for (const auto& source : name_sources_map.at(name)) {
    const auto& key = count > 1 ? source : append(name, ":", source);
    cref_entries.push_back(
        std::cref(get_known_source_map_entry(count, key)));
  }
  return cref_entries;
}

auto get_known_source_map_entries(const NameCount& nc)
    -> std::vector<KnownSourceMapValueCRef> {
  return get_known_source_map_entries(nc.name, nc.count);
}

bool add_compound_clue(
    const cm::NameCount& nc, const std::string& sources_csv) {
  get_known_source_map_entry(nc.count, sources_csv).clue_names.emplace(nc.name);
  return true;
}

//
// uniqueClueNames
//

int get_num_unique_clue_names(int count) {
  return int(get_unique_clue_nc_list(count).size());
}

const NameCount& get_unique_clue_nc(int count, int idx) {
  return get_unique_clue_nc_list(count).at(idx);
}

const std::string& get_unique_clue_name(int count, int idx) {
  return get_unique_clue_nc(count, idx).name;
}

SourceCompatibilityList make_unique_clue_source_list(int count) {
  SourceCompatibilityList unique_src_list;
  const auto& name_sources_map = get_name_sources_map(count);
  for (int idx{}; idx < get_num_unique_clue_names(count); ++idx) {
    const auto& name = get_unique_clue_name(count, idx);
    for (const auto& source : name_sources_map.at(name)) {
      const auto& key = count > 1 ? source : append(name, ":", source);
      const auto& src_list = get_known_source_map_entry(count, key).src_list;
      unique_src_list.insert(unique_src_list.end(), src_list.begin(), src_list.end());
    }
  }
  if (log_level(ExtraVerbose)) {
    std::cerr << " " << count << ": make_unique_clue_sources("
              << unique_src_list.size() << "), names("
              << get_num_unique_clue_names(count) << ")\n";
  }
  return unique_src_list;
}

const SourceCompatibilityList& get_primary_unique_clue_source_list() {
  ensure_primary_unique_clue_source_list();
  return uniquePrimaryClueSrcList_;
}

const SourceCompatibilityData& get_unique_clue_source(int count, int idx) {
  assert(count == 1);
  ensure_primary_unique_clue_source_list();
  return uniquePrimaryClueSrcList_.at(idx);
}

int get_num_unique_clue_sources(int count, const std::string& name) {
  int num_sources{};
  const auto& name_sources_map = get_name_sources_map(count);
  for (const auto& source : name_sources_map.at(name)) {
    const auto& key = count > 1 ? source : append(name, ":", source);
    const auto& src_list = get_known_source_map_entry(count, key).src_list;
    num_sources += int(src_list.size());
  }
  return num_sources;
}

// TODO: this is dog slow.
int get_unique_clue_starting_source_index(int count, int unique_name_idx) {
  int start_idx{};
  for (int idx{}; idx < unique_name_idx; ++idx) {
    const auto& name = get_unique_clue_name(count, idx);
    start_idx += get_num_unique_clue_sources(count, name);
  }
  return start_idx;
}

//
// misc
//

/*
inline void for_each_source_map_entry(
    const std::string& name, int count, const auto& fn) {
  for (const auto& entry_cref : get_known_source_map_entries(nc)) {
    fn(entry_cref.get());
  }
}
*/

auto make_src_list(const NameCount& nc) -> cm::SourceList {
  SourceList src_list;
  for_each_nc_source(nc, [&src_list](const SourceData& src, index_t) {  //
    src_list.push_back(src);
  });
  return src_list;
}

auto make_src_cref_list(const std::string& name,
    int count) -> cm::SourceCRefList {
  SourceCRefList src_cref_list;
  for_each_nc_source(name, count,
      [&src_cref_list](const SourceData& src, index_t) {
        src_cref_list.push_back(std::cref(src));
      });
  return src_cref_list;
}

// JavaScript's primary name-sources map is Sentence.NameSourcesMap, which is
// a map of [string: Set<number]. On the C++ side that is unwrapped as a list
// of names, and a list of IndexLists, which are passed to us here.
//
// This function builds and initializes three globals from these two lists.
// The first two are simple map representations of the data, with either
// integer or string primary sources:
//  * primaryNameSrcIndicesMap_, which is a unordered_map<string, IndexList>
//  * nameSourcesMap_[0], which is a unordered_map<string, vector<string>>
// The third requires we build and populate actual SourceLists:
//  * knownSourcesMap_[0], which is a unordered_map<string,
//  KnownSourceMapValue>
//
void init_primary_clues(std::vector<std::string>&& names,
    std::vector<IndexList>&& idx_lists) {
  // FIRST, build and set primaryNameSrcIndicesMap_. The passed-in parameters
  // names and idx_lists are consumed here.
  assert(primaryNameSrcIndicesMap_.empty());
  primaryNameSrcIndicesMap_ = buildPrimaryNameSrcIndicesMap(std::move(names),
      std::move(idx_lists));

  // SECOND, build and set nameSourcesMap_[0]
  set_name_sources_map(1,
      build_primary_name_sources_map(primaryNameSrcIndicesMap_));

  // THIRD, populate knownSourcesMap_[1] from the above two maps.
  init_primary_known_source_map(primaryNameSrcIndicesMap_,
      get_name_sources_map(1));
}

void dump_memory(std::string_view header /* = "clue_manager memory:" */) {
  // std::unordered_map<std::string, IndexList> primaryNameSrcIndicesMap;
  size_t src_indices_map_size{};
  for (const auto& p : primaryNameSrcIndicesMap_) {
    size_t s{p.first.size()};
    s += p.second.size() * sizeof(IndexList::value_type);
    src_indices_map_size += s;
  }

  // using NameSourcesMap = std::unordered_map<std::string,
  // std::vector<std::string>>; std::vector<NameSourcesMap> nameSourcesMaps;
  size_t name_sources_maps_size{};
  for (const auto& m : nameSourcesMaps) {
    for (const auto& p : m) {
      size_t s{p.first.size()};
      for (const auto& str : p.second) {
        s += str.size();
      }
      name_sources_maps_size += s;
    }
  }

  // struct KnownSourceMapValue {
  //   SourceList src_list;
  //   std::set<std::string> clue_names;
  // };
  // using KnownSourceMap = std::unordered_map<std::string,
  // KnownSourceMapValue>; std::vector<KnownSourceMap> knownSourceMaps;
  size_t known_source_maps_size{};
  size_t num_known_source_maps_sources{};
  for (const auto& m : knownSourceMaps) {
    for (const auto& p : m) {
      size_t s{p.first.size()};
      s += src_list_size(p.second.src_list);
      num_known_source_maps_sources += p.second.src_list.size();
      for (const auto& str : p.second.clue_names) {
        s += str.size();
      }
      known_source_maps_size += s;
    }
  }

  // not counted yet
  // std::vector<StringCRefList> uniqueClueNames;
  std::cerr << header << std::endl
            << " src_indices_map_size:   "
            << util::pretty_bytes(src_indices_map_size) << std::endl
            << " name_sources_maps_size: "
            << util::pretty_bytes(name_sources_maps_size) << std::endl
            << " known_source_maps_size: "
            << util::pretty_bytes(known_source_maps_size)  //
            << ", sources(" << num_known_source_maps_sources << ")" << std::endl
            << " total:                  "
            << util::pretty_bytes(src_indices_map_size + name_sources_maps_size
                   + known_source_maps_size)
            << std::endl;
}

}  // namespace cm::clue_manager
