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

// Helper to get a specific primary source by (name, count, idx)
// Returns the SourceData for a parent reference
const SourceData& get_primary_source_by_ref(const SourceParent& ref) {
  assert(ref.count == 1 && "get_primary_source_by_ref: only count==1 supported");
  const SourceData* result = nullptr;
  KnownSources::for_each_primary_source(ref.name,
      [&](const SourceData& src, index_t idx) {
        if (idx == ref.idx) {
          result = &src;
        }
      });
  if (!result) {
    std::cerr << "ERROR: get_primary_source_by_ref source not found: "
              << ref.name << ":" << ref.count << " idx " << ref.idx << std::endl;
  }
  assert(result != nullptr);
  return *result;
}

// Helper to get a SourceCombo by reference
const SourceCombo& get_combo_source_by_ref(const SourceParent& ref) {
  if (ref.count <= 1) {
    std::cerr << "ERROR: get_combo_source_by_ref called with invalid count: "
              << ref.count << ", name: " << ref.name << ", idx: " << ref.idx << std::endl;
  }
  assert(ref.count > 1 && "get_combo_source_by_ref: only count>1 supported");
  const SourceCombo* result = nullptr;
  KnownSources::for_each_combo_source(ref.name, ref.count,
      [&](const SourceCombo& combo, index_t idx) {
        if (idx == ref.idx) {
          result = &combo;
        }
      });
  if (!result) {
    std::cerr << "ERROR: get_combo_source_by_ref source not found: "
              << ref.name << ":" << ref.count << " idx " << ref.idx << std::endl;
  }
  assert(result != nullptr);
  return *result;
}

// Recursively collect primaryNameSrcList from a SourceCombo
void collect_primary_name_src_list(const SourceCombo& combo,
    NameCountList& result) {
  for (const auto& parent_ref : combo.parents) {
    if (parent_ref.count <= 0) {
      std::cerr << "ERROR: collect_primary_name_src_list: invalid parent count: "
                << parent_ref.count << ", name: " << parent_ref.name
                << ", idx: " << parent_ref.idx << std::endl;
      assert(false && "invalid parent count");
    }
    if (parent_ref.count == 1) {
      // Base case: primary source has full data
      const auto& primary = get_primary_source_by_ref(parent_ref);
      result.insert(result.end(), primary.primaryNameSrcList.begin(),
          primary.primaryNameSrcList.end());
    } else {
      // Recurse into compound parents
      const auto& parent_combo = get_combo_source_by_ref(parent_ref);
      collect_primary_name_src_list(parent_combo, result);
    }
  }
}

// Recursively collect nc_names from a SourceCombo
void collect_nc_names(const SourceCombo& combo, std::set<std::string>& result) {
  for (const auto& parent_ref : combo.parents) {
    if (parent_ref.count == 1) {
      // Base case: primary source has nc_names
      const auto& primary = get_primary_source_by_ref(parent_ref);
      result.insert(primary.nc_names.begin(), primary.nc_names.end());
    } else {
      // Recurse into compound parents
      const auto& parent_combo = get_combo_source_by_ref(parent_ref);
      collect_nc_names(parent_combo, result);
    }
  }
  // Include this combo's clue name
  result.insert(combo.nc.name);
}

}  // anonymous namespace

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

// THE RECONSTRUCTION POINT: Convert compact SourceCombo to full SourceData
/*static*/ SourceData KnownSources::reconstruct(const SourceCombo& combo) {
  // Build primaryNameSrcList by traversing parent tree
  NameCountList primaryNameSrcList;
  collect_primary_name_src_list(combo, primaryNameSrcList);

  // Build ncList from the combo's NC
  NameCountList ncList;
  ncList.emplace_back(combo.nc.name, combo.nc.count);

  // Build nc_names by collecting all names from parents + this combo's name
  std::set<std::string> nc_names;
  collect_nc_names(combo, nc_names);

  return SourceData(combo.usedSources, std::move(primaryNameSrcList),
      std::move(ncList), std::move(nc_names));
}

// Minimal reconstruction: only populates ncList from combo.nc
// Skips all recursive parent tree traversal
/*static*/ SourceData KnownSources::reconstruct_nclist(const SourceCombo& combo) {
  NameCountList ncList;
  ncList.emplace_back(combo.nc.name, combo.nc.count);
  return SourceData(combo.usedSources,
      NameCountList{},  // empty primaryNameSrcList
      std::move(ncList),
      std::set<std::string>{});  // empty nc_names
}

// NB: NOT threadsafe (but shouldn't matter)
// TODO: i think this can be eliminated; check src/tools/todo
/*static*/ bool KnownSources::add_compound_clue(const NameCount& nc,
    const std::string& sources_csv) {
  assert(nc.count > 1 && "add_compound_clue: count must be > 1");
  get().get_combo_entry(nc.count, sources_csv).clue_names.insert(nc.name);
  return true;
}

auto KnownSources::get_combo_map(int count, bool force_create /* = false */)
    -> ComboEntryMap& {
  // Compound sources start at count 2, so index = count - 2
  const auto idx = count - 2;
  assert(idx >= 0 && "get_combo_map: count must be >= 2");
  if (force_create && (int(combo_entry_maps_.size()) == idx)) {
    combo_entry_maps_.push_back(ComboEntryMap{});
  }
  return combo_entry_maps_.at(idx);
}

bool KnownSources::has_entries_for(int count) const {
  assert(count > 0);
  if (count == 1) {
    return !primary_entry_map_.empty();
  }
  const auto idx = count - 2;
  return int(combo_entry_maps_.size()) > idx;
}

bool KnownSources::has_entries_for(int count, const std::string& source) const {
  assert(count > 0);
  if (count == 1) {
    return primary_entry_map_.contains(source);
  }
  const auto idx = count - 2;
  if (int(combo_entry_maps_.size()) > idx) {
    return combo_entry_maps_.at(idx).contains(source);
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
  auto [_, success] = primary_entry_map_.emplace(key, Entry{std::move(src_list), {}});
  if (!success) {
    std::cerr << "failed adding primary entry " << key << std::endl;
  }
  assert(success);

  // Store compat data in primary_compat_map_ (duplicates expected and OK)
  primary_compat_map_.emplace(source, std::move(src_compat));
}

auto KnownSources::get_primary_entry(const std::string& key) -> Entry& {
  auto it = primary_entry_map_.find(key);
  assert(it != primary_entry_map_.end());
  return it->second;
}

auto KnownSources::get_primary_entry(const std::string& key) const
    -> const Entry& {
  auto it = primary_entry_map_.find(key);
  assert(it != primary_entry_map_.end());
  return it->second;
}

void KnownSources::init_combo_entry(int count, const std::string& source,
    SourceComboList&& src_combo_list) {
  assert(count > 1 && "init_combo_entry: count must be > 1");
  auto& map = get_combo_map(count, true);  // true = create if doesn't exist
  auto [_, success] = map.emplace(source, ComboEntry{std::move(src_combo_list), {}});
  if (!success) {
    std::cerr << "failed adding combo entry " << source << ":" << count << std::endl;
  }
  assert(success);
}

// TODO: move to inline in header, const_cast<>
auto KnownSources::get_combo_entry(int count, const std::string& source_csv)
    -> ComboEntry& {
  assert(count > 1);
  auto& map = get_combo_map(count);
  auto it = map.find(source_csv);
  assert(it != map.end());
  return it->second;
}

auto KnownSources::get_combo_entry(int count, const std::string& source_csv) const
    -> const ComboEntry& {
  assert(count > 1);
  const auto& map = get_combo_map(count);
  auto it = map.find(source_csv);
  assert(it != map.end());
  return it->second;
}

// Legacy compatibility: get_entry only works for count == 1 now
auto KnownSources::get_entry(int count, const std::string& source_csv)
    -> Entry& {
  assert(count == 1 && "get_entry: only count == 1 supported, use get_combo_entry for count > 1");
  return get_primary_entry(source_csv);
}

auto KnownSources::get_entry(int count, const std::string& source_csv) const
    -> const Entry& {
  assert(count == 1 && "get_entry: only count == 1 supported, use get_combo_entry for count > 1");
  return get_primary_entry(source_csv);
}

const std::set<std::string>& KnownSources::get_entry_clue_names(int count,
    const std::string& source_csv) const {
  if (count == 1) {
    return get_primary_entry(source_csv).clue_names;
  }
  return get_combo_entry(count, source_csv).clue_names;
}

size_t KnownSources::get_num_clue_sources(int count,
    const std::string& key) const {
  if (count == 1) {
    return get_primary_entry(key).src_list.size();
  }
  return get_combo_entry(count, key).src_combo_list.size();
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
  for (size_t count{2}; count <= combo_entry_maps_.size() + 1; ++count) {
    const auto& map = combo_entry_maps_.at(count - 2);
    if (map.empty()) continue;

    size_t map_size{};
    size_t map_combos{};
    for (const auto& [key, entry] : map) {
      map_size += key.capacity();
      map_size += entry.src_combo_list.capacity() * sizeof(SourceCombo);
      for (const auto& combo : entry.src_combo_list) {
        map_size += combo.parents.capacity() * sizeof(SourceParent);
        for (const auto& parent : combo.parents) {
          map_size += parent.name.capacity();
        }
        map_size += combo.nc.name.capacity();
      }
      map_combos += entry.src_combo_list.size();
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
