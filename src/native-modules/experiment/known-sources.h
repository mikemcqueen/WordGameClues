#pragma once

#include <functional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>
#include "combo-maker.h"

namespace cm {

//
// KnownSources
//
class KnownSources {
  //TODO: try: 
  // static KnownSources known_sources_;

public:
  struct Entry {
    SourceList src_list; // TODO: remove
    SourceCRefList src_cref_list;
    std::set<std::string> clue_names;
  };

  using EntryCRef = std::reference_wrapper<const Entry>;

  static KnownSources& get();
  
  bool has_entries_for(int count) const;

  bool has_entries_for(int count, const std::string& source) const;

  void init_entry(int count, const std::string source, SourceList&& src_list);

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

  // TODO: i think this can be eliminated; check src/tools/todo
  bool add_compound_clue(const NameCount& nc,
      const std::string& sources_csv);

  auto make_src_list(const NameCount& nc) -> SourceList;

  auto make_src_cref_list(const std::string& name, int count)
      -> SourceCRefList;

  auto make_src_cref_list(const NameCount& nc) -> SourceCRefList {
    return make_src_cref_list(nc.name, nc.count);
  }

  inline void for_each_nc_source(const std::string& name, int count,
      const auto& fn) {
    for (index_t idx{}; const auto entry_cref : get_entries(name, count)) {
      for (const auto& src : entry_cref.get().src_list) {
        fn(src, idx++);
      }
    }
  }

  inline void for_each_nc_source(const NameCount& nc, const auto& fn) {
    for_each_nc_source(nc.name, nc.count, fn);
  }

private:
  using Map = std::unordered_map<std::string, Entry>;

  auto& get_map(int count, bool force_create = false);

  const auto& get_map(int count) const { return maps_.at(count - 1); }

  // map source_csv -> { src_cref_list, clue_names_set } for each count
  std::vector<Map> maps_;
  // ordered set of unique sources
  std::set<SourceData> src_set_;
};

}  // namespace cm
