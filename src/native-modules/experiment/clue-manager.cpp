#include <cassert>
#include <charconv>
#include <chrono>
#include <string>
#include <unordered_set>
#include <vector>
#include "clue-manager.h"
#include "util.h"

namespace clue_manager {

using namespace cm;

namespace {

//
// types
//

// TODO: CompoundSrcSourceListMap
using KnownSourceMap = std::unordered_map<std::string, KnownSourceMapValue>;
using StringCRefList = std::vector<std::reference_wrapper<const std::string>>;

//
// globals
//

// map nc -> src_list for each count
std::vector<NcSourcesMap> ncSourcesMaps;

// map primary clue_name -> idx_list 
PrimaryNameSrcIndicesMap primaryNameSrcIndicesMap;

// map clue_name -> src_csv_list for each count (including primary)
std::vector<NameSourcesMap> nameSourcesMaps;

// map src_csv -> src_list for each count
std::vector<KnownSourceMap> knownSourceMaps;

// populated on demand from primaryNameSrcIndicesMap/nameSrcMaps
std::vector<StringCRefList> uniqueClueNames;

//
// functions
//

auto& get_nc_sources_map(int count) {
  const auto idx = count - 1;
  if ((int)ncSourcesMaps.size() <= idx) {
    ncSourcesMaps.resize(idx + 1);
  }
  return ncSourcesMaps.at(idx);
}

//////////

auto& get_known_source_map(int count, bool force_create = false) {
  // allow force-creates exactly in-sequence only, or throw an exception
  const auto idx = count - 1;
  if (force_create && ((int)knownSourceMaps.size() == idx)) {
    knownSourceMaps.emplace_back(KnownSourceMap{});
  }
  return knownSourceMaps.at(idx);
}

void init_known_source_map_entry(
  int count, const std::string& src, SourceList&& src_list) {
  // True is arbitrary here. I *could* support replacing an existing src_list,
  // but i'm unaware of any situation that requires it, and as a result I want
  // things to blow up when it is attempted, currently.
  auto& map = get_known_source_map(count, true);
  auto [_, success] = map.emplace(src, std::move(src_list));
  assert(success);
}

auto& get_known_source_map_entry(int count, const std::string& nc_str) {
  auto& map = get_known_source_map(count);
  auto it = map.find(nc_str);
  assert(it != map.end());
  return it->second;
}

///////////

const auto& get_name_sources_map(int count) {
  return nameSourcesMaps.at(count - 1);
}

auto build_primary_name_sources_map(
  const PrimaryNameSrcIndicesMap& name_src_indices_map) {
  //
  NameSourcesMap name_sources_map;
  // TODO: [name, idx_list]
  for (const auto& kv_pair : name_src_indices_map) {
    const auto& idx_list = kv_pair.second;
    std::vector<std::string> sources;
    for (auto idx: idx_list) {
      sources.emplace_back(std::to_string(idx));
    }
    name_sources_map.emplace(kv_pair.first, std::move(sources));
  }
  return name_sources_map;
}

//////////

void populate_unique_clue_names(StringCRefList& name_cref_list, int count) {
  for (const auto& [name, _] : get_name_sources_map(count)) {
    name_cref_list.emplace_back(std::cref(name));
  }
}

const auto& get_unique_clue_names(int count) {
  const auto idx{count - 1};
  if ((int)uniqueClueNames.size() <= idx) {
    uniqueClueNames.resize(idx + 1);
  }
  auto& names = uniqueClueNames.at(idx);
  if (names.empty()) {
    populate_unique_clue_names(names, count);
    assert(!names.empty());
  }
  return names;
}

}  // namespace

//
// ncSourcesMaps
//

// non-const return. would like it to be (in some cases). so a class type
// container would have some benefit here.
auto get_nc_src_list(const NameCount& nc) -> cm::SourceList& {
  // TODO: specialize std::hash for NameCount?
  auto& map{get_nc_sources_map(nc.count)};
  const auto nc_str{nc.toString()};
  if (!map.contains(nc_str)) {
    auto [_, success] = map.emplace(nc_str, UniqueSources{});
    assert(success);
  }
  return map.find(nc_str)->second.src_list;
}

/*
auto get_nc_src_list(const cm::NameCount& nc) -> const cm::SourceList& {
  return get_nc_sources(nc).src_list;
}
*/
  
auto get_num_nc_sources(const NameCount& nc) -> int {
  return get_nc_src_list(nc).size();
}

int append_nc_sources_from_known_source(
  const cm::NameCount& nc, const std::string& known_src_csv) {
  //
  auto& src_list = get_nc_src_list(nc);
  const auto& src_map_entry = get_known_source_map_entry(nc.count, known_src_csv);
  src_list.insert(src_list.end(), src_map_entry.src_list.begin(),
    src_map_entry.src_list.end());
  return std::distance(
    src_map_entry.src_list.begin(), src_map_entry.src_list.end());
}

auto make_src_list_for_nc(const NameCount& nc) -> cm::SourceList {
  SourceList src_list;
  for_each_nc_source(nc, [&src_list](const SourceData& src) {
    src_list.emplace_back(src);
  });  // cl-format
  return src_list;
}

auto make_src_cref_list_for_nc(const NameCount& nc) -> cm::SourceCRefList {
  SourceCRefList src_cref_list;
  for_each_nc_source(nc, [&src_cref_list](const SourceData& src) {
    src_cref_list.emplace_back(std::cref(src));
  });
  return src_cref_list;
}

//
// nameSourcesMaps
//

auto buildPrimaryNameSrcIndicesMap(std::vector<std::string>& names,
  std::vector<IndexList>& idx_lists) -> PrimaryNameSrcIndicesMap {
  //
  PrimaryNameSrcIndicesMap src_indices_map;
  for (size_t i{}; i < names.size(); ++i) {
    src_indices_map.emplace(std::move(names.at(i)), std::move(idx_lists.at(i)));
  }
  return src_indices_map;
}

void setNameSourcesMap(int count, NameSourcesMap&& name_sources_map) {
  auto idx = count - 1;
  // allow sets exactly in-sequence only, or throw an exception
  assert((int)nameSourcesMaps.size() == idx);
  nameSourcesMaps.emplace_back(std::move(name_sources_map));
}

bool is_known_name_count(const std::string& name, int count) {
  return get_name_sources_map(count).contains(name);
}

const std::vector<std::string>& get_nc_sources(const NameCount& nc) {
  return get_name_sources_map(nc.count).at(nc.name);
}

void setPrimaryNameSrcIndicesMap(PrimaryNameSrcIndicesMap&& src_indices_map) {
  assert(primaryNameSrcIndicesMap.empty());
  using namespace std::chrono;
  auto ksm0 = high_resolution_clock::now();
  auto name_sources_map = build_primary_name_sources_map(src_indices_map);  
  for (const auto& [name, sources] : name_sources_map) {
    SourceList src_list;  // TODO = build_primary_src_list(name, sources);
    for (const auto& str_src : sources) {
      if (is_known_source_map_entry(1, str_src))
        continue;

      NameCountList name_src_list;
      int int_src;
      auto [_, err] = std::from_chars(
        str_src.data(), str_src.data() + str_src.size(), int_src);
      assert(err == std::errc{});
      name_src_list.emplace_back(name, int_src);
      NameCountList nc_list;
      nc_list.emplace_back(name, 1);
      UsedSources used_sources;
      used_sources.addSource(int_src);
      SourceData src_data(
        std::move(name_src_list), std::move(nc_list), std::move(used_sources));
      src_list.emplace_back(std::move(src_data));
      init_known_source_map_entry(1, str_src, std::move(src_list));
    }
  }
  auto ksm1 = high_resolution_clock::now();
  auto ksm_dur = duration_cast<milliseconds>(ksm1 - ksm0).count();
  std::cerr << " init primary known_src_map - " << ksm_dur << "ms" << std::endl;

  setNameSourcesMap(1, std::move(name_sources_map));
  primaryNameSrcIndicesMap = std::move(src_indices_map);
}

const IndexList& getPrimaryClueSrcIndices(const std::string& name) {
  const auto& map = primaryNameSrcIndicesMap;
  auto it = map.find(name);
  assert(it != map.end());
  return it->second;
}

//
// knownSourceMaps
//

void init_known_source_map_entry(
  int count, const std::vector<std::string>& name_list, SourceList&& src_list) {
  //
  init_known_source_map_entry(count, util::join(name_list, ","), std::move(src_list));
}

bool is_known_source_map_entry(int count, const std::string& src_csv) {
  const auto idx = count - 1;
  if ((int)knownSourceMaps.size() > idx) {
    return get_known_source_map(count).contains(src_csv);
  }
  return false;
}

auto get_known_source_map_entries(const NameCount& nc)
  -> std::vector<KnownSourceMapValueCRef> {
  //
  std::vector<KnownSourceMapValueCRef> cref_entries;
  const auto& name_sources_map = get_name_sources_map(nc.count);
  for (const auto& src_csv : name_sources_map.at(nc.name)) {
    cref_entries.emplace_back(
      std::cref(get_known_source_map_entry(nc.count, src_csv)));
  }
  return cref_entries;
}

//
// uniqueClueNames
//

int get_num_unique_clue_names(int count) {
  return get_unique_clue_names(count).size();
}

const std::string& get_unique_clue_name(int count, int idx) {
  return get_unique_clue_names(count).at(idx).get();
}

}  // namespace clue_manager
