#include <cassert>
#include <iostream>
#include "clue-manager.h"
#include "known-sources.h"
#include "util.h"

//
// KnownSources
//

namespace cm {

/*static*/ const SourceData& KnownSources::get_primary_source(
    const NameCountIndex& nci) {
  assert(nci.nc.count == 1 && "get_primary_source: only count == 1 supported");
  const SourceData* result = nullptr;
  KnownSources::for_each_primary_source(nci.nc.name,
      [&](const SourceData& src, index_t idx) {
        if (idx == nci.index) {
          result = &src;
        }
      });
  if (!result) {
    std::cerr << "ERROR: get_primary_source: source not found, "
              << nci.nc.name << ":" << nci.nc.count << " idx " << nci.index
              << std::endl;
  }
  assert(result != nullptr);
  return *result;
}

/*static*/ const DeferredSourceData& KnownSources::get_deferred_source(
    const NameCountIndex& nci) {
  if (nci.nc.count <= 1) {
    std::cerr << "ERROR: get_deferred_source: invalid count, "
              << nci.nc.count << ", name: " << nci.nc.name << ", idx: "
              << nci.index << std::endl;
  }
  assert(nci.nc.count > 1 && "get_deferred_source: only count >= 2 supported");
  const DeferredSourceData* result = nullptr;
  KnownSources::for_each_combo_source(nci.nc.name, nci.nc.count,
      [&](const DeferredSourceData& combo, index_t idx) {
        if (idx == nci.index) {
          result = &combo;
        }
      });
  if (!result) {
    std::cerr << "ERROR: get_deferred_source: source not found, "
              << nci.nc.name << ":" << nci.nc.count << " idx " << nci.index
              << std::endl;
  }
  assert(result != nullptr);
  return *result;
}

// Make a SourceList from primary sources (count == 1 only)
/*static*/ auto KnownSources::make_src_list(const NameCount& nc) -> SourceList {
  assert(nc.count == 1 && "make_src_list only supports primary sources");
  SourceList src_list;
  for_each_primary_source(nc.name, [&src_list](const SourceData& src, index_t) {
    src_list.push_back(src);
  });
  return src_list;
}

// Make a SourceCRefList from primary sources (count == 1 only)
/*static*/ auto KnownSources::make_src_cref_list(const std::string& name,
    int count) -> SourceCRefList {
  std::cerr << "make_src_cref_list " << name << ":" << count << "\n";

  assert(count == 1 && "make_src_cref_list only supports primary sources");
  SourceCRefList src_cref_list;
  for_each_primary_source(name, [&src_cref_list](const SourceData& src, index_t) {
    src_cref_list.push_back(std::cref(src));
  });
  return src_cref_list;
}

// Make a SourceCRefList from primary sources (count == 1 only)
/*static*/ auto KnownSources::make_src_compat_cref_list(const std::string& name,
    int count) -> SourceCompatCRefList {
  SourceCompatCRefList compat_crefs;
  for_each_nc_source_compat(name, count,
      [&compat_crefs](const SourceCompatibilityData& src, index_t) {
        compat_crefs.push_back(std::cref(src));
      });
  return compat_crefs;
}

// NB: NOT threadsafe (but shouldn't matter)
// TODO: i think this can be eliminated; check src/tools/todo
/*static*/ bool KnownSources::add_compound_clue(const NameCount& nc,
    const std::string& sources_csv) {
  assert(nc.count > 1 && "add_compound_clue: count must be > 1");
  get().get_compound_entry(nc.count, sources_csv).clue_names.insert(nc.name);
  return true;
}

auto KnownSources::get_combo_map(int count, bool force_create /* = false */)
    -> CompoundEntryMap& {
  // Compound sources start at count 2, so index = count - 2
  const auto idx = count - 2;
  assert(idx >= 0 && "get_combo_map: count must be >= 2");
  if (force_create && (int(compound_entry_maps_.size()) == idx)) {
    compound_entry_maps_.push_back(CompoundEntryMap{});
  }
  return compound_entry_maps_.at(idx);
}

bool KnownSources::has_entries_for(int count) const {
  assert(count > 0);
  if (count == 1) {
    return !primary_entry_map_.empty();
  }
  const auto idx = count - 2;
  return int(compound_entry_maps_.size()) > idx;
}

bool KnownSources::has_entries_for(int count, const std::string& source) const {
  assert(count > 0);
  if (count == 1) {
    return primary_entry_map_.contains(source);
  }
  const auto idx = count - 2;
  if (int(compound_entry_maps_.size()) > idx) {
    return compound_entry_maps_.at(idx).contains(source);
  }
  return false;
}

void KnownSources::init_primary_entry(const std::string& name,
    const std::string& source, SourceList&& src_list) {
  assert(src_list.size() == 1);
  // Copy compat data before move
  SourceCompatibilityData src_compat{src_list.at(0).usedSources};

  // Store full entry in primary_entry_map_
  const auto key = util::append(name, ":", source);
  auto [_, success] = primary_entry_map_.emplace(key, PrimaryEntry{std::move(src_list), {}});
  if (!success) {
    std::cerr << "failed adding primary entry " << key << std::endl;
  }
  assert(success);

  // Store compat data in primary_compat_map_ (duplicates expected and OK)
  primary_compat_map_.emplace(source, std::move(src_compat));
}

auto KnownSources::get_primary_entry(const std::string& key) -> PrimaryEntry& {
  auto it = primary_entry_map_.find(key);
  assert(it != primary_entry_map_.end());
  return it->second;
}

auto KnownSources::get_primary_entry(const std::string& key) const
    -> const PrimaryEntry& {
  auto it = primary_entry_map_.find(key);
  assert(it != primary_entry_map_.end());
  return it->second;
}

void KnownSources::init_compound_entry(int count, const std::string& source,
    DeferredSourceDataList&& src_combo_list) {
  assert(count > 1 && "init_compound_entry: count must be > 1");
  auto& map = get_combo_map(count, true);  // true = create if doesn't exist
  auto [_, success] = map.emplace(source, CompoundEntry{std::move(src_combo_list), {}});
  if (!success) {
    std::cerr << "failed adding combo entry " << source << ":" << count << std::endl;
  }
  assert(success);
}

// TODO: move to inline in header, const_cast<>
auto KnownSources::get_compound_entry(int count, const std::string& source_csv)
    -> CompoundEntry& {
  assert(count > 1);
  auto& map = get_combo_map(count);
  auto it = map.find(source_csv);
  assert(it != map.end());
  return it->second;
}

auto KnownSources::get_compound_entry(int count, const std::string& source_csv) const
    -> const CompoundEntry& {
  assert(count > 1);
  const auto& map = get_combo_map(count);
  auto it = map.find(source_csv);
  assert(it != map.end());
  return it->second;
}

const std::set<std::string>& KnownSources::get_entry_clue_names(int count,
    const std::string& source_csv) const {
  if (count == 1) {
    return get_primary_entry(source_csv).clue_names;
  }
  return get_compound_entry(count, source_csv).clue_names;
}

size_t KnownSources::get_num_clue_sources(int count,
    const std::string& key) const {
  if (count == 1) {
    return get_primary_entry(key).src_list.size();
  }
  return get_compound_entry(count, key).dfer_list.size();
}

void KnownSources::dump_memory() const {
  size_t total_entries{};
  size_t total_sources{};
  size_t total_size{};

  std::cerr << "KnownSources memory:" << std::endl;

  // Primary entry map (count == 1)
  size_t primary_entry_size{};
  size_t primary_sources{};
  for (const auto& [key, entry] : primary_entry_map_) {
    primary_entry_size += key.capacity();
    primary_entry_size += entry.src_list.capacity() * sizeof(SourceData);
    for (const auto& src : entry.src_list) {
      primary_entry_size += src.primaryNameSrcList.capacity() * sizeof(NameCount);
      for (const auto& nc : src.primaryNameSrcList) {
        primary_entry_size += nc.name.capacity();
      }
      primary_entry_size += src.ncList.capacity() * sizeof(NameCount);
      for (const auto& nc : src.ncList) {
        primary_entry_size += nc.name.capacity();
      }
      primary_entry_size += src.nc_names.size() * (sizeof(void*) * 3 + 32);
      for (const auto& name : src.nc_names) {
        primary_entry_size += name.capacity();
      }
    }
    primary_sources += entry.src_list.size();
    primary_entry_size += entry.clue_names.size() * (sizeof(void*) * 3 + 32);
    for (const auto& name : entry.clue_names) {
      primary_entry_size += name.capacity();
    }
  }
  std::cerr << "  count 1 (primary): " << primary_entry_map_.size() << " entries, "
            << primary_sources << " sources, "
            << util::pretty_bytes(primary_entry_size) << std::endl;
  total_entries += primary_entry_map_.size();
  total_sources += primary_sources;
  total_size += primary_entry_size;

  // Primary compat map
  size_t primary_compat_size{};
  for (const auto& [key, src_compat] : primary_compat_map_) {
    primary_compat_size += key.capacity();
    primary_compat_size += sizeof(SourceCompatibilityData);
  }
  std::cerr << "  primary_compat_map: " << primary_compat_map_.size() << " entries, "
            << util::pretty_bytes(primary_compat_size) << std::endl;
  total_size += primary_compat_size;

  // Combo entry maps (count > 1)
  for (size_t count{2}; count <= compound_entry_maps_.size() + 1; ++count) {
    const auto& map = compound_entry_maps_.at(count - 2);
    if (map.empty()) continue;

    size_t map_size{};
    size_t map_combos{};
    for (const auto& [key, entry] : map) {
      map_size += key.capacity();
      map_size += entry.dfer_list.capacity() * sizeof(DeferredSourceData);
      for (const auto& combo : entry.dfer_list) {
        map_size += combo.known_nci_list.capacity() * sizeof(NameCountIndex);
        for (const auto& known_nci : combo.known_nci_list) {
          map_size += known_nci.nc.name.capacity();
        }
        map_size += combo.nc.name.capacity();
      }
      map_combos += entry.dfer_list.size();
      map_size += entry.clue_names.size() * (sizeof(void*) * 3 + 32);
      for (const auto& name : entry.clue_names) {
        map_size += name.capacity();
      }
    }
    std::cerr << "  count " << count << " (combo): " << map.size() << " entries, "
              << map_combos << " combos, "
              << util::pretty_bytes(map_size) << std::endl;
    total_entries += map.size();
    total_sources += map_combos;
    total_size += map_size;
  }

  std::cerr << "  total: " << total_entries << " entries, "
            << total_sources << " sources/combos, "
            << util::pretty_bytes(total_size) << std::endl;
}

}  // namespace cm
