// show-components.cpp

#include <algorithm>
#include <functional>
#include <iostream>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "clue-manager.h"
#include "cm-precompute.h"
#include "combo-maker.h"
#include "components.h"
#include "merge.h"
#include "log.h"
#include "util.h"
#include "wtf_pool.h"
using namespace std::literals;

namespace cm::components {

struct NamesAndCounts {
  std::set<std::string> names;
  std::vector<int> counts;
};

struct ShowResults {
  std::set<int> sums;
  std::vector<std::vector<int>> valid;
  std::vector<std::vector<int>> invalid;
  std::vector<NamesAndCounts> known;
  std::vector<NamesAndCounts> clues;
};

namespace {

using consistency_t = std::pair<std::string, NameCountList>;
wtf::ThreadPool<consistency_t, 5> consistency_pool_;
std::unordered_map<std::string, NameCountList> consistency_results_;

// -t word1
void add_show_result_for_nc(const NameCount& nc, ShowResults& results) {
  auto sources_list = clue_manager::get_nc_sources(nc);
  if (sources_list.size()) {
    std::vector<int> count_list = { nc.count };
    std::set<std::string> sources_set(sources_list.begin(), sources_list.end());
    results.clues.emplace_back(std::move(sources_set), std::move(count_list));
  } else {
    std::cerr << "hmm, no sources for " << nc.toString() << std::endl;
  }
}

// poorly named
// -t word1,word2
auto add_show_result_for_sources(const std::string& sources_csv,
    const std::vector<int>& counts, int sum, ShowResults& results) {
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

auto get_show_results(
    const std::vector<std::string>& name_list, const SourceList& xor_src_list) {
  ShowResults results;
  std::unordered_set<std::string> hash;
  for (const auto& src: xor_src_list) {
    const auto count_list = NameCount::listToCountList(src.ncList);
    // TODO: hash sum?
    const auto key = util::join(count_list, ",");
    if (hash.contains(key)) {
      continue;
    }
    hash.insert(key);
    const auto sum = util::sum(count_list);
    if (name_list.size() == 1u) {
      add_show_result_for_nc(NameCount{name_list.at(0), sum}, results);
    } else if (add_show_result_for_sources(util::join(name_list, ","),  //
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

void display_show_results(
    const std::vector<std::string>& name_list, const ShowResults& results) {
  display(results.invalid, "INVALID");
  display(results.known, "PRESENT as", name_list);
  display(results.clues, "PRESENT as clue with source:", name_list);
  display(results.valid, "VALID");
}

using opt_str_set_cref_t =
  std::optional<std::reference_wrapper<const std::set<std::string>>>;

opt_str_set_cref_t get_clue_names_for_source(int sum, const std::string& source_csv) {
  static const std::set<std::string> empty;
  using namespace clue_manager;
  if (has_known_source_map(sum)) {
    if (is_known_source_map_entry(sum, source_csv)) {
      auto& clue_names = get_known_source_map_entry(sum, source_csv).clue_names;
      return std::make_optional(std::cref(clue_names));
    } else {
      return std::make_optional(std::cref(empty));
    }
  } else {
    if (log_level(Verbose)) {
      std::cerr << "!knownSourceMap(" << sum << "), sources: " << source_csv
                << std::endl;
    }
    return {};
  }
}

void dump(const std::set<std::string>& name_set, int set_sum,
    const NameCountList& nc_list, const std::vector<std::string>& name_list,
    const std::string& missing_name) {
  if (name_set.size() != name_list.size()) {
    const auto count_list = NameCount::listToCountList(nc_list);
    const auto list_sum = util::sum(count_list);
    std::cerr << "\ntest:\n name_set.size(" << name_set.size() << "), sum("
              << set_sum << ")" << std::endl
              << " name_list.size(" << name_list.size() << "), sum(" << list_sum
              << ")\n";
    std::cerr << " name_list: " << util::join(name_list, ",") << std::endl;
  }
  if (!missing_name.empty()) {
    std::cerr << "test: missing name: " << missing_name << std::endl;
  }
}

// TODO: poorly named function 
// check if all valid, compatible, instances of the supplied clue source name
// list (pair, actually) have the same clue names.
// e.g., if name_list is [dog, food], and dog:1,dog:2,food:3 exist along with
// dogfood:4=dog,food then dogfood:5=dog,food must also exist.
bool are_sources_consistent(
    const std::vector<std::string>& name_list, const SourceList& xor_src_list) {
  assert(name_list.size() > 1u);
  //const bool test = false;
  const auto source_csv = util::join(name_list, ",");
  std::unordered_set<std::string> hash; // TODO: could be <int>
  std::unordered_set<std::string> names;
  //int names_sum{};
  bool first = true;
  for (const auto& src: xor_src_list) {
    const auto count_list = NameCount::listToCountList(src.ncList);
    const auto sum = util::sum(count_list);
    const auto key = util::join(count_list, ","); // TODO: we could just use sum as key?
    if (hash.contains(key)) {
      continue;
    }
    hash.insert(key);
    auto opt_names = get_clue_names_for_source(sum, source_csv);
    if (!opt_names.has_value()) {
      continue;
    }
    auto& clue_names = opt_names.value().get();
    if (first) {
      for (const auto& name : clue_names) {
        names.insert(name);
      }
      //names_sum = sum;
      first = false;
      continue;
    }
    if (names.size() != clue_names.size()) {
      //if (test) { dump(names, names_sum, src.ncList, clue_names, {}); }
      return false;
    }
    for (const auto& name : clue_names) {
      if (!names.contains(name)) {
        //if (test) { dump(names, names_sum, src.ncList, clue_names, name); }
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

auto get_addends(const std::vector<std::string>& name_list, int max_sources) {
  std::vector<std::vector<int>> result;
  for (int sum{2}; sum <= max_sources; ++sum) {
    auto addends = Peco::make_addends(sum, name_list.size());
    util::move_append(result, std::move(addends));
  }
  return result;
}

// given a vector of (sorted) count lists, ex. [ [1,2,3], .. ] test if the
// providede names exist at any permutation of each list. return all count
// count lists which have such a permutation.
auto filter_valid_addend_perms(std::vector<std::vector<int>>& addends,
    const std::vector<std::string>& name_list) {
  std::vector<std::vector<int>> result;
  for (auto& count_list : addends) {
    do {
      if (clue_manager::are_known_name_counts(name_list, count_list)) {
        result.emplace_back(std::move(count_list));
        // once we get one valid perm for a count_list, we can break out,
        // because the NC (name:sum_of_counts) will be the same.
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

auto get_all_compatible_sources(
    const std::vector<std::string>& name_list, int max_sources) {
  auto addends = get_addends(name_list, max_sources);
  auto filtered_addends = filter_valid_addend_perms(addends, name_list);
  auto nc_data_lists = make_nc_data_lists(filtered_addends, name_list);
  auto src_lists = build_src_lists(nc_data_lists);
  return merge_xor_compatible_src_lists(src_lists);
}

auto get_all_clue_names(
    const std::string& source_csv, const SourceList& src_list) {
  std::set<std::string> result;
  for (const auto& src: src_list) {
    const auto sum = util::sum(NameCount::listToCountList(src.ncList));
    /*
    const auto key = util::join(count_list, ",");
    if (hash.contains(key)) {
      continue;
    }
    hash.insert(key);
    */
    auto opt_names = get_clue_names_for_source(sum, source_csv);
    if (!opt_names.has_value()) {
      continue;
    }
    auto& clue_names = opt_names.value().get();
    for (const auto& name : clue_names) {
      result.insert(name);
    }
  }
  return result;
}

auto get_missing_nc_list(
    const std::string& source_csv, const SourceList& src_list) {
  NameCountList missing_nc_list;
  auto all_clue_names = get_all_clue_names(source_csv, src_list);
  std::unordered_set<int> hash;
  for (const auto& src : src_list) {
    const auto sum = util::sum(NameCount::listToCountList(src.ncList));
    if (hash.contains(sum)) continue;
    hash.insert(sum);
    auto opt_names = get_clue_names_for_source(sum, source_csv);
    if (!opt_names.has_value()) continue;
    auto& clue_names = opt_names.value().get();
    std::vector<std::string> missing_names;
    std::ranges::set_difference(
        all_clue_names, clue_names, std::back_inserter(missing_names));
    std::ranges::transform(missing_names, std::back_inserter(missing_nc_list),
        [sum](const std::string& name) -> NameCount { return {name, sum}; });
  }
  return missing_nc_list;
}

void consistency_check_result_processor(consistency_t&& result) {
  if (!result.first.empty()) {
    auto it = consistency_results_.find(result.first);
    if (it == consistency_results_.end()) {
      consistency_results_.emplace(std::move(result));
    } else {
      std::ranges::move(result.second, std::back_inserter(it->second));
    }
  }
}

}  // anonymous namespace

auto show(const std::vector<std::string>& name_list,
    const SourceList& xor_src_list) -> std::set<int> {
  auto show_results = get_show_results(name_list, xor_src_list);
  display_show_results(name_list, show_results);
  return show_results.sums;
}

auto old_consistency_check(const std::vector<std::string>& name_list,
    const SourceList& src_list) -> bool {
  return are_sources_consistent(name_list,src_list);
}

void consistency_check(
    const std::vector<std::string>&& name_list, int max_sources) {
  consistency_pool_.execute(
      [name_list = std::move(name_list), max_sources]() -> consistency_t {
        std::string source_csv;
        NameCountList nc_list;
        // TODO: 3-source clues don't work currently, because reasons.
        if (name_list.size() == 2) {
          auto src_list = get_all_compatible_sources(name_list, max_sources);
          source_csv = util::join(name_list, ",");
          if (1 || !are_sources_consistent(name_list, src_list)) {
            nc_list = get_missing_nc_list(source_csv, src_list);
          }
        }
        if (nc_list.empty()) source_csv.clear();
        return std::make_pair(std::move(source_csv), std::move(nc_list));
      },
      consistency_check_result_processor);
}

auto get_consistency_check_results()
    -> const std::unordered_map<std::string, NameCountList>& {
  consistency_pool_.process_all_results(consistency_check_result_processor);
  return consistency_results_;
}

/*
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
