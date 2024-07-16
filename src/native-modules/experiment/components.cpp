// show-components.cpp

#include <algorithm>
#include <functional>
#include <iostream>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>
#include "clue-manager.h"
#include "cm-precompute.h"
#include "combo-maker.h"
#include "components.h"
#include "merge.h"
#include "log.h"
#include "util.h"
using namespace std::literals;

namespace cm::components {

struct NamesAndCounts {
  std::vector<std::string> names;
  std::vector<int> counts;
};

struct Results {
  std::set<int> sums;
  std::vector<std::vector<int>> valid;
  std::vector<std::vector<int>> invalid;
  std::vector<NamesAndCounts> known;
  std::vector<NamesAndCounts> clues;
};

namespace {

// -t word1
void add_result_for_nc(const NameCount& nc, Results& results) {
  auto sources_list = clue_manager::get_nc_sources(nc);
  if (sources_list.size()) {
    std::vector<int> count_list = { nc.count };
    results.clues.emplace_back(sources_list, count_list);
  } else {
    std::cerr << "hmm, no sources for " << nc.name << ":" << nc.count
              << std::endl;
  }
}

// poorly named
// -t word1,word2
auto add_result_for_sources(const std::string& sources_csv,
    const std::vector<int>& counts, int sum, Results& results) {
  using namespace clue_manager;
  if (has_known_source_map(sum)) {
    if (is_known_source_map_entry(sum, sources_csv)) {
      const auto& names =
        get_known_source_map_entry(sum, sources_csv).clue_names;
      results.known.emplace_back(names, counts);
    } else {
      results.valid.emplace_back(counts);
    }
    return true;
  } else {
    if (log_level(Verbose)) {
      std::cerr << "!knownSourceMap(" << sum << "), sources: " << sources_csv
                << std::endl;
    }
    return false;
  }
}

auto get_results(
    const std::vector<std::string>& name_list, const SourceList& xor_src_list) {
  Results results;
  std::unordered_set<std::string> hash;
  for (const auto& src: xor_src_list) {
    const auto count_list = NameCount::listToCountList(src.ncList);
    const auto key = util::join(count_list, ",");
    if (hash.contains(key)) {
      continue;
    }
    hash.insert(key);
    const auto sum = util::sum(count_list);
    if (name_list.size() == 1u) {
      add_result_for_nc(NameCount{name_list.at(0), sum}, results);
    } else if (add_result_for_sources(util::join(name_list, ","),  //
                   count_list, sum, results)) {
      results.sums.insert(sum);
    }
  }
  return results;
}

auto get_source_clues(const std::vector<std::string>& source_list,
    const NamesAndCounts& names_counts) {
  const auto sum = util::sum(names_counts.counts);
  if (!clue_manager::is_known_source_map_entry(
        sum, util::join(source_list, ","))) {
    std::string sources{};
    // honestly i don't understand this at all
    for (const auto& source : source_list) {
      // [source] is wrong at least for primary clue case, need actual list of
      // sources.
      sources += source;  //  ??
    }
    return sources;
  }
  return util::join(names_counts.names, ",");
}

void display(const std::vector<int>& count_list, const std::string& text,
    const std::string& sources = "") {
  std::cout << util::join(count_list, ",") << " " << text << " " << sources
            << std::endl;
}

void display(
    const std::vector<std::vector<int>>& count_lists, const std::string& text) {
  for (const auto& count_list : count_lists) {
    display(count_list, text);
  }
}

void display(const std::vector<NamesAndCounts>& names_counts_list,
    const std::string& text, const std::vector<std::string>& source_list) {
  for (const auto& names_counts : names_counts_list) {
    std::string sources{};
    if (names_counts.counts.size() > 1u) {
      // -t name1,name2[,name3,...] (multiple names)
      sources += get_source_clues(source_list, names_counts);
    } else {
      // -t name (one name only)
      sources += util::join(names_counts.names, " - ");
    }
    display(names_counts.counts, text, sources);
  }
}

void display_results(
    const std::vector<std::string>& name_list, const Results& results) {
  display(results.invalid, "INVALID");
  display(results.known, "PRESENT as", name_list);
  display(results.clues, "PRESENT as clue with source:", name_list);
  display(results.valid, "VALID");
}

using opt_str_list_cref_t =
  std::optional<std::reference_wrapper<const std::vector<std::string>>>;

opt_str_list_cref_t get_clue_names_for_sources(int sum, const std::string& sources_csv) {
  static const std::vector<std::string> empty;
  using namespace clue_manager;
  if (has_known_source_map(sum)) {
    if (is_known_source_map_entry(sum, sources_csv)) {
      return std::make_optional(std::cref(get_known_source_map_entry(sum, sources_csv).clue_names));
    } else {
      return std::make_optional(std::cref(empty));
    }
  } else {
    if (log_level(Verbose)) {
      std::cerr << "!knownSourceMap(" << sum << "), sources: " << sources_csv
                << std::endl;
    }
    return {};
  }
}

bool are_sources_consistent(
    const std::vector<std::string>& name_list, const SourceList& xor_src_list) {
  assert(name_list.size() > 1u);
  const auto sources_csv = util::join(name_list, ",");
  std::unordered_set<std::string> hash;
  std::unordered_set<std::string> names;
  for (const auto& src: xor_src_list) {
    const auto count_list = NameCount::listToCountList(src.ncList);
    const auto key = util::join(count_list, ",");
    if (hash.contains(key)) {
      continue;
    }
    bool first = hash.empty();
    hash.insert(key);
    const auto sum = util::sum(count_list);
    auto opt_names = get_clue_names_for_sources(sum, sources_csv);
    if (!opt_names.has_value()) {
      continue;
    }
    if (first) {
      for (const auto& name: opt_names.value().get()) {
        names.insert(name);
      }
      continue;
    }
    if (names.size() != opt_names.value().get().size()) {
      return false;
    }
    for (const auto& name : opt_names.value().get()) {
      if (!names.contains(name)) {
        return false;
      }
    }
  }
  return true;
}

//////////
// v2

void display(const std::vector<std::string>& name_list) {
  std::cerr << "\nname_list: " << util::join(name_list, ","s) << std::endl;
}

void display(const std::vector<NCDataList>& nc_data_lists) {
  std::cerr << " nc_data_lists(" << nc_data_lists.size() << "):\n";
  for (const auto& nc_data_list : nc_data_lists) {
    std::cerr << " nc_lists(" << nc_data_list.size() << "):\n";
    for (const auto& nc_data : nc_data_list) {
      std::cerr << "  ";
      for (const auto& nc : nc_data.ncList) {
        std::cerr << nc.toString() << ", ";
      }
      std::cerr << std::endl;
    }
  }
}

void display(const SourceList& src_list) {
  std::cerr << " src_list(" << src_list.size() << "):\n";
  for (const auto& src : src_list) {
    std::cerr << "  ncl: ";
    for (const auto& nc : src.ncList) {
      std::cerr << nc.toString() << ", ";
    }
    std::cerr << std::endl;
    std::cerr << "  pnsl: ";
    for (const auto& nc : src.primaryNameSrcList) {
      std::cerr << nc.toString() << ", ";
    }
    std::cerr << std::endl;
  }
}

auto get_known_source_idx_list(const std::string& source_csv, int max_sources) {
  std::vector<int> idx_list;
  for (int i{2}; i <= max_sources; ++i) {
    if (clue_manager::is_known_source_map_entry(i, source_csv)) {
      idx_list.push_back(i);
    }
  }
  return idx_list;
}

auto get_all_known_source_clue_names(
    const std::string& source_csv, const std::vector<int>& idx_list) {
  std::unordered_set<std::string> clue_names;
  for (const auto idx : idx_list) {
    if (clue_manager::is_known_source_map_entry(idx, source_csv)) {
      const auto& entry =
          clue_manager::get_known_source_map_entry(idx, source_csv);
      for (const auto& name : entry.clue_names) {
        clue_names.insert(name);
      }
    }
  }
  return clue_names;
}

auto all_known_sources_have_clue_names(const std::string& source_csv,
    const std::vector<int>& idx_list,
    std::unordered_set<std::string>& clue_names) {
  for (const auto idx : idx_list) {
    if (!clue_manager::is_known_source_map_entry(idx, source_csv)) return false;
    const auto& entry =
        clue_manager::get_known_source_map_entry(idx, source_csv);
    for (const auto& clue_name : entry.clue_names) {
      if (!clue_names.contains(clue_name)) return false;
    }
  }
  return true;
}

auto get_addends(const std::vector<std::string>& name_list, int max_sources) {
  std::vector<std::vector<int>> result;
  for (int sum{2}; sum <= max_sources; ++sum) {
    auto addends = Peco::make_addends(sum, name_list.size());
    util::move_append(result, std::move(addends));
  }
  return result;
}

auto has_names_at_counts(const std::vector<std::string>& name_list,
    const std::vector<int>& count_list) {
  for (size_t i{}; i < name_list.size(); ++i) {
    if (!clue_manager::is_known_name_count(name_list.at(i), count_list.at(i))) {
      return false;
    }
  }
  return true;
}

// given a vector of count listss, ex. [ [1,2,3], .. ] generate all
// permutations of each list and test if the provided names exist at those
// counts. return all resulting count lists where that condition is true.
auto filter_valid_addend_perms(std::vector<std::vector<int>>& addends,
    const std::vector<std::string>& name_list) {
  std::vector<std::vector<int>> result;
  for (auto& count_list : addends) {
    do {
      if (has_names_at_counts(name_list, count_list)) {
        result.emplace_back(std::move(count_list));
        // once we get one valid perm for a count_list, we can break out,
        // because the NC (name:total_count) will be the same.
        break;
      }
    } while (std::next_permutation(count_list.begin(), count_list.end()));
  }
  return result;
}

auto make_nc_data_lists(const std::vector<std::vector<int>>& addends,
    const std::vector<std::string>& name_list) {
  std::vector<NCDataList> result;
  for (const auto& count_list : addends) {
    NCDataList nc_data_list;
    for (size_t i{}; i < name_list.size(); ++i) {
      NCData nc_data;
      nc_data.ncList.emplace_back(name_list.at(i), count_list.at(i));
      nc_data_list.emplace_back(std::move(nc_data));
    }
    result.emplace_back(std::move(nc_data_list));
  }
  return result;
}

}  // anonymous namespace

auto show(const std::vector<std::string>& name_list,
    const SourceList& xor_src_list) -> std::set<int> {
  auto results = get_results(name_list, xor_src_list);
  display_results(name_list, results);
  return results.sums;
}

auto consistency_check(const std::vector<std::string>& name_list,
    const SourceList& src_list, bool force_dump /*= false*/) -> bool {
  static bool dump = false;
  auto result = are_sources_consistent(name_list, src_list);
  if (force_dump || (!result && dump)) {
    display(name_list);
    std::cerr << "v1 result: " << std::boolalpha << result << std::endl;
    display(src_list);
    dump = false;
  }
  return result;
}

auto consistency_check2(MergeFilterData& mfd, const std::vector<std::string>& name_list,
    int max_sources) -> bool {
  static bool dump = false;
  if (dump) {
    display(name_list);
  }
  // TODO: 3-source clues don't work currently.
  if (name_list.size() > 2) return true;
  auto addends = get_addends(name_list, max_sources);
  if (0 && dump) {
    std::cerr << " addends: ";
    for (const auto& v : addends) {
      std::cerr << util::join(v, ","s) << ", ";
    }
    std::cerr << std::endl;
  }
  auto filtered_addends = filter_valid_addend_perms(addends, name_list);
  if (dump) {
    std::cerr << " filtered: ";
    for (const auto& v : filtered_addends) {
      std::cerr << util::join(v, ","s) << ", ";
    }
    std::cerr << std::endl;
  }
  auto nc_data_lists = make_nc_data_lists(filtered_addends, name_list);
  if (dump) {
    display(nc_data_lists);
  }
  //display(name_list);
  auto src_lists = buildSourceListsForUseNcData(nc_data_lists);
  /*
  const auto& src_list = src_lists.back();
  if (dump) {
    display(src_list);
    }
  */
  merge_xor_src_lists(mfd, src_lists, true);
  auto result = consistency_check(name_list, mfd.host.merged_xor_src_list,  //
      dump);
  dump = false;
  return result;
}

/*
// old noise
auto find_sum(int sum, const std::vector<std::vector<int>>& count_lists,
    std::vector<std::vector<int>>::const_iterator iter) {
  auto count_list = *iter;
  if (std::next(iter) == count_lists.end()) {
    return std::find(count_list.begin(), count_list.end(), sum) !=
count_list.end();
  }
  for (auto count : count_list) {
    if (sum - count >= 0) {
      if (find_sum(sum - count, count_lists, std::next(iter))) return true;
    }
  }
  return false;
}

auto check_names_and_counts(int sum, const std::vector<std::string>& name_list,
    const std::vector<int>& count_list) {
  assert(name_list.size() == count_list.size());
  std::vector<std::vector<int>> counts_for_names;
  for (const auto& name: name_list) {
    // filter_count_lists_for_clue_name(name, count_list)
    std::vector<int> counts;
    for (auto count: count_list) {
      if (clue_manager::is_known_name_count(name, count)) {
        counts.push_back(count);
      }
    }
    if (!counts.size()) return false;
    counts_for_names.emplace_back(std::move(counts));
  }
  return find_sum(sum, counts_for_names, counts_for_names.begin());
}

// old get_addends noise
struct VectorHash {
  size_t operator()(const std::vector<int>& v) const {
    std::hash<int> hasher;
    size_t seed = 0;
    for (int i : v) {
      seed ^= hasher(i) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
  }
};
std::unordered_set<std::vector<int>, VectorHash> result;
std::unordered_set<int> result;
  for (auto& count_list : addends) {
    if (check_names_and_counts(sum, name_list, count_list)) {
      result.emplace(std::move(count_list));
      result.insert(std::accumulate(count_list.begin(), count_list.end(), 0,
          [](int sum, int count) { return sum + count; }));
    }
  }
return std::vector<int>(count_set.begin(), count_set.end());

//auto source_csv = util::join(name_list, ","s);
//auto clue_names = get_all_known_source_clue_names(source_csv,potential_counts);
//return all_known_sources_have_clue_names(source_csv, potential_counts,clue_names);
//return true;

// this did.. something.. but not exactly what i wanted.
auto potential_counts = get_addends(name_list, max_sources);
auto source_csv = util::join(name_list, ","s);
if (!potential_counts.empty()) {
  std::cerr << "\n"
            << source_csv << "(" << potential_counts.size()
            << "): " << util::join(potential_counts, ","s) << std::endl;
}
auto clue_names = get_all_known_source_clue_names(source_csv, potential_counts);
return all_known_sources_have_clue_names(source_csv, potential_counts,
clue_names);


// this only works for already-declared clue-sources, not for VALID but
// undeclared.
auto idx_list = get_known_source_idx_list(source_csv, max_sources);
auto clue_names = get_all_known_source_clue_names(source_csv, idx_list);
return all_known_sources_have_clue_names(source_csv, idx_list, clue_names);
*/

}  // namespace cm::components
