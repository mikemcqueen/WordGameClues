#include <cassert>
#include <iostream>
#include "clue-manager.h"
#include "known-sources.h"
#include "util.h"

//
// KnownSources
//

namespace cm {

namespace {

  //KnownSources known_sources_;

}

///* static */ KnownSources& KnownSources::get() { return known_sources_; }

/*static*/ auto KnownSources::make_src_list(const NameCount& nc) -> SourceList {
  SourceList src_list;
  for_each_nc_source(nc, [&src_list](const SourceData& src, index_t) { //
    src_list.push_back(src);
  });
  return src_list;
}

/*static*/ auto KnownSources::make_src_cref_list(const std::string& name,
    int count) -> SourceCRefList {
  SourceCRefList src_cref_list;
  for_each_nc_source(name, count,
      [&src_cref_list](const SourceData& src, index_t) {
        src_cref_list.push_back(std::cref(src));
      });
  return src_cref_list;
}

// NB: NOT threadsafe (but shouldn't matter)
// TODO: i think this can be eliminated; check src/tools/todo
/*static*/ bool KnownSources::add_compound_clue(const NameCount& nc,
    const std::string& sources_csv) {
  get().get_entry(nc.count, sources_csv).clue_names.insert(nc.name);
  return true;
}

auto KnownSources::get_map(int count, bool force_create /* = false */) -> Map& {
  // allow force-creates exactly in-sequence only, or throw an exception
  const auto idx = count - 1;
  if (force_create && (int(maps_.size()) == idx)) {
    maps_.push_back(Map{});
  }
  return maps_.at(idx);
}

bool KnownSources::has_entries_for(int count) const {
  assert(count > 0);
  return int(maps_.size()) >= count;
}

bool KnownSources::has_entries_for(int count, const std::string& source) const {
  assert(count > 0);
  const auto idx = count - 1;
  if (int(maps_.size()) > idx) {
    return get_map(count).contains(source);
  }
  return false;
}

void KnownSources::init_entry(int count, const std::string source,
    SourceList&& src_list) {
  auto& map = get_map(count, true);  // true = create map if it doesn't exist
  auto [_, success] = map.emplace(std::move(source), std::move(src_list));
  if (!success) {
    std::cerr << "failed adding " << source << ":" << count
              << " to known_source_map" << std::endl;
  }
  assert(success);
}

void KnownSources::init_primary_entry(const std::string& name,
    const std::string& source, SourceList&& src_list) {
  assert(src_list.size() == 1);
  // copy before move in init_entry call
  SourceCompatibilityData src_compat{src_list.at(0).usedSources};
  init_entry(1, util::append(name, ":", source), std::move(src_list));
  // this is expected to fail a lot due to duplicates
  primary_map_.emplace(source, std::move(src_compat));
}

// one entry per source_csv
auto KnownSources::get_entry(int count, const std::string& source_csv)
    -> Entry& {
  auto& map = get_map(count);
  auto it = map.find(source_csv);
  assert(it != map.end());
  return it->second;
}

// one entry per source_csv
auto KnownSources::get_entry(int count, const std::string& source_csv) const
    -> const Entry& {
  const auto& map = get_map(count);
  auto it = map.find(source_csv);
  assert(it != map.end());
  return it->second;
}

// (potentially) multiple sources per name, because a name may have multiple
// valid source combinations
auto KnownSources::get_entries(const std::string& name, int count) const
    -> std::vector<EntryCRef> {
  std::vector<EntryCRef> cref_entries;
  const auto& name_sources_map = clue_manager::get_name_sources_map(count);
  for (const auto& source : name_sources_map.at(name)) {
    const auto& key = count > 1 ? source : util::append(name, ":", source);
    cref_entries.push_back(std::cref(get_entry(count, key)));
  }
  return cref_entries;
}

}  // namespace cm
