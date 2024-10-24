#pragma once

#include <functional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>
#include "clue-manager.h"
#include "combo-maker.h"
#include "cuda-types.h"
#include "util.h"

#if 1
#include <iostream>
#include <unordered_set>
#endif

namespace cm {

inline std::vector<std::string_view> split(std::string_view str, char delim = ':') {
    std::vector<std::string_view> tokens;
    size_t start = 0;
    size_t end = str.find(delim);
    
    while (end != std::string_view::npos) {
        tokens.push_back(str.substr(start, end - start));
        start = end + 1;
        end = str.find(delim, start);
    }
    
    tokens.push_back(str.substr(start));
    return tokens;
}

//
// KnownSources
//
class KnownSources {
public:
  struct Entry {
    SourceList src_list; // TODO: remove
    SourceCRefList src_cref_list;
    std::set<std::string> clue_names;
  };

  using EntryCRef = std::reference_wrapper<const Entry>;

  static KnownSources& get() {
    static KnownSources known_sources;
    return known_sources;
  }

#if 0
  static void test_keys() {
    std::unordered_set<std::string_view> sources;
    const auto& map = get().get_map(1);
    for (const auto& [key, entry] : map) {
      assert(entry.src_list.size() == 1);
      const auto tokens = split(key);
      std::cerr
          << tokens.at(0) << ":" << tokens.at(1) << " - "
          << entry.src_list.at(0).usedSources.get_source_descriptor().toString()
          << std::endl;
      sources.insert(tokens.at(0));
    }
    std::cerr << "primary KnownSources keys: " << map.size()
              << ", unique: " << sources.size() << std::endl;
  }
#endif

#if 1
  static void for_each_nc_source_compat_data(const NameCount& nc,
      const auto& fn) {
    index_t idx{};
    const auto& name_sources_map = clue_manager::get_name_sources_map(nc.count);
    for (const auto& source : name_sources_map.at(nc.name)) {
      // const auto& key = count > 1 ? source : util::append(name, ":", source);
      if (nc.count > 1) {
        const auto& src_list = get().get_entry(nc.count, source).src_list;
        for (const auto& src : src_list) {
          fn(static_cast<const SourceCompatibilityData&>(src), idx++);
        }
      } else {
        const auto it = get().get_primary_map().find(source);
        assert(it != get().get_primary_map().end());
        fn(it->second, idx++);
      }
    }
  }
#endif

  static void for_each_nc_source(const std::string& name, int count,
      const auto& fn) {
    index_t idx{};
    for_each_entry(name, count, [&fn, &idx](const SourceList& src_list) {
      for (const auto& src : src_list) {
        fn(src, idx++);
      }
    });
  }

  static void for_each_nc_source(const NameCount& nc, const auto& fn) {
    for_each_nc_source(nc.name, nc.count, fn);
  }

  static auto make_src_list(const NameCount& nc) -> SourceList;

  static auto make_src_cref_list(const std::string& name, int count)
      -> SourceCRefList;

  static auto make_src_cref_list(const NameCount& nc) -> SourceCRefList {
    return make_src_cref_list(nc.name, nc.count);
  }

  // TODO: i think this can be eliminated; check src/tools/todo
  static bool add_compound_clue(const NameCount& nc,
      const std::string& sources_csv);

  bool has_entries_for(int count) const;

  bool has_entries_for(int count, const std::string& source) const;

  void init_entry(int count, const std::string source, SourceList&& src_list);
  void init_primary_entry(const std::string& name, const std::string& source,
      SourceList&& src_list);

  // one entry per source_csv
  auto get_entry(int count, const std::string& source_csv) -> Entry&;
  auto get_entry(int count, const std::string& source_csv) const
      -> const Entry&;

  // (potentially) multiple entries per NC, because a given NC may have
  // multiple valid source combinations
  auto get_entries(const std::string& name, int count) const
      -> std::vector<EntryCRef>;
  auto get_entries(const NameCount& nc) const -> std::vector<EntryCRef> {
    return get_entries(nc.name, nc.count);
  }

private:
  using Map = std::unordered_map<std::string, Entry>;
  using PrimaryMap = std::unordered_map<std::string, SourceCompatibilityData>;

  auto get_map(int count, bool force_create = false) -> Map&;

  auto get_map(int count) const -> const Map& { return maps_.at(count - 1); }

  auto get_primary_map() const -> const PrimaryMap& { return primary_map_; }

  static void for_each_entry(const std::string& name, int count,
      const auto& fn) {
    const auto& name_sources_map = clue_manager::get_name_sources_map(count);
    for (const auto& source : name_sources_map.at(name)) {
      const auto& key = count > 1 ? source : util::append(name, ":", source);
      fn(get().get_entry(count, key).src_list);
    }
  }

  // map source_csv -> { src_cref_list, clue_names_set } for each count
  std::vector<Map> maps_;
  // map primary source (packed-index as string) -> SourceCompatibilityData
  PrimaryMap primary_map_;
  // ordered set of unique sources
  std::set<SourceData> src_set_;
};

}  // namespace cm
