#ifndef INCLUDE_CLUE_MANAGER_H
#define INCLUDE_CLUE_MANAGER_H

#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <vector>
#include "cuda-types.h"
// not thrilled about this. it's really a header organization issue though
#include "combo-maker.h"

namespace cm::clue_manager {

// types/aliases

// name -> list of "sources"
// for primary clues, a "source" is a name:primary_source_packed_integer, for
//   which I really need a better, more distinct and descriptive name.
// for compound clues, a source is a comma-separated-value: "name,name"
using NameSourcesMap =
    std::unordered_map<std::string, std::vector<std::string>>;

// functions

// nameSourcesMaps

auto get_name_sources_map(int count) -> const NameSourcesMap&;

void set_name_sources_map(int count, NameSourcesMap&& name_sources_map);

bool is_known_name_count(const std::string& name, int count);  // known_nc

bool are_known_name_counts(const std::vector<std::string>& name_list,
    const std::vector<int>& count_list);

const std::vector<std::string>& get_nc_sources(const NameCount& nc);

// uniqueClueNames

int get_num_unique_clue_names(int count);

const NameCount& get_unique_clue_nc(int count, int idx);

const std::string& get_unique_clue_name(int count, int idx);

const SourceCompatibilityData& get_unique_clue_source(int count, int idx);

SourceCompatibilityList make_unique_clue_source_list(int count);

const SourceCompatibilityList& get_primary_unique_clue_source_list();

// TODO: needs rewrite
int get_unique_clue_starting_source_index(int count, int unique_name_idx);

// debugging
int get_num_unique_clue_sources(int count, const std::string& name);
inline int get_num_unique_clue_sources(int count, int unique_name_idx) {
  return get_num_unique_clue_sources(count,
      get_unique_clue_name(count, unique_name_idx));
}

// misc

void init_primary_clues(std::vector<std::string>&& names,
    std::vector<IndexList>&& idx_lists);

void dump_memory(std::string_view header = "clue-manager memory:");

}  // namespace cm::clue_manager

#endif // INCLUDE_CLUE_MANAGER_H
