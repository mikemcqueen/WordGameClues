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
  // Entry for primary sources (count == 1) - always fully populated
  struct Entry {
    SourceList src_list;
    std::set<std::string> clue_names;
  };

  // Entry for compound sources (count > 1) - compact storage
  struct ComboEntry {
    SourceComboList src_combo_list;
    std::set<std::string> clue_names;
  };

  using EntryCRef = std::reference_wrapper<const Entry>;

  // Map type aliases for clear storage boundaries
  using PrimaryCompatMap = std::unordered_map<std::string, SourceCompatibilityData>;
  using PrimaryEntryMap = std::unordered_map<std::string, Entry>;
  using ComboEntryMap = std::unordered_map<std::string, ComboEntry>;

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

  // Iterate over SourceCompatibilityData for an NC (works with both storage types)
  static void for_each_nc_source_compat(const std::string& name, int count,
      const auto& fn) {
    index_t idx{};
    const auto& name_sources_map = clue_manager::get_name_sources_map(count);
    for (const auto& source : name_sources_map.at(name)) {
      if (count > 1) {
        // Compound source: get source compat data from ComboEntryMap
        const auto& combo_entry = get().get_combo_entry(count, source);
        for (const auto& combo : combo_entry.src_combo_list) {
          fn(static_cast<const SourceCompatibilityData&>(combo), idx++);
        }
      } else {
        // Primary source: use PrimaryCompatMap
        const auto it = get().primary_compat_map_.find(source);
        assert(it != get().primary_compat_map_.end());
        fn(it->second, idx++);
      }
    }
  }

  static void for_each_nc_source_compat(const NameCount& nc,
      const auto& fn) {
    for_each_nc_source_compat(nc.name, nc.count, fn);
  }

  // Iterate over SourceData for primary sources (count == 1 only)
  static void for_each_primary_source(const std::string& name,
      const auto& fn) {
    index_t idx{};
    const auto& name_sources_map = clue_manager::get_name_sources_map(1);
    for (const auto& source : name_sources_map.at(name)) {
      const auto& key = util::append(name, ":", source);
      const auto& src_list = get().get_primary_entry(key).src_list;
      for (const auto& src : src_list) {
        fn(src, idx++);
      }
    }
  }

  // Iterate over SourceCombo for compound sources (count > 1)
  static void for_each_combo_source(const std::string& name, int count,
      const auto& fn) {
    assert(count > 1);
    index_t idx{};
    const auto& name_sources_map = clue_manager::get_name_sources_map(count);
    for (const auto& source : name_sources_map.at(name)) {
      const auto& combo_entry = get().get_combo_entry(count, source);
      for (const auto& combo : combo_entry.src_combo_list) {
        fn(combo, idx++);
      }
    }
  }

  static void for_each_nc_source(const std::string& name, int count,
      const auto& fn) {
    assert(count == 1 && "for_each_nc_source only supports primary sources");
    for_each_primary_source(name, fn);
  }

  static void for_each_nc_source(const NameCount& nc, const auto& fn) {
    for_each_nc_source(nc.name, nc.count, fn);
  }

  // Make a SourceList from primary sources (count == 1 only)
  static auto make_src_list(const NameCount& nc) -> SourceList;

  static auto make_src_cref_list(const std::string& name, int count)
      -> SourceCRefList;

  static auto make_src_cref_list(const NameCount& nc) -> SourceCRefList {
    return make_src_cref_list(nc.name, nc.count);
  }

  static auto make_src_compat_cref_list(const std::string& name, int count)
      -> SourceCompatCRefList;

  // THE RECONSTRUCTION POINT: Convert compact SourceCombo to full SourceData
  static SourceData reconstruct(const SourceCombo& combo);

  // Minimal reconstruction: only populates ncList from combo.nc
  // Use when only ncList is needed (e.g., merge_only path)
  static SourceData reconstruct_nclist(const SourceCombo& combo);

  // TODO: i think this can be eliminated; check src/tools/todo
  static bool add_compound_clue(const NameCount& nc,
      const std::string& sources_csv);

  bool has_entries_for(int count) const;
  bool has_entries_for(int count, const std::string& source) const;

  // Primary entry methods (count == 1)
  void init_primary_entry(const std::string& name, const std::string& source,
      SourceList&& src_list);
  auto get_primary_entry(const std::string& key) -> Entry&;
  auto get_primary_entry(const std::string& key) const -> const Entry&;

  // Combo entry methods (count > 1)
  void init_combo_entry(int count, const std::string& source,
      SourceComboList&& src_combo_list);
  auto get_combo_entry(int count, const std::string& source_csv) -> ComboEntry&;
  auto get_combo_entry(int count, const std::string& source_csv) const
      -> const ComboEntry&;

  // Legacy compatibility: get_entry delegates to appropriate type
  auto get_entry(int count, const std::string& source_csv) -> Entry&;
  auto get_entry(int count, const std::string& source_csv) const
      -> const Entry&;

  // Helpers that work with both Entry (count==1) and ComboEntry (count>1)
  const std::set<std::string>& get_entry_clue_names(int count,
      const std::string& source_csv) const;
  size_t get_num_clue_sources(int count, const std::string& key) const;

  void dump_memory() const;

private:
  auto get_combo_map(int count, bool force_create = false) -> ComboEntryMap&;
  auto get_combo_map(int count) const -> const ComboEntryMap& {
    return combo_entry_maps_.at(count - 2);  // count 2 = index 0
  }

  // Storage for primary sources (count == 1)
  // "name:source" -> Entry with full SourceData
  PrimaryEntryMap primary_entry_map_;
  // source -> SourceCompatibilityData for GPU lookups
  PrimaryCompatMap primary_compat_map_;

  // Storage for compound sources (count > 1)
  // Indexed by count-2 (count 2 = index 0)
  // source_csv -> ComboEntry with SourceComboList
  std::vector<ComboEntryMap> combo_entry_maps_;
};

}  // namespace cm
