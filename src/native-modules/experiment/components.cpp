// show-components.cpp

#include <functional>
#include <iostream>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>
#include "clue-manager.h"
#include "combo-maker.h"
#include "merge-filter-data.h"
#include "components.h"
#include "util.h"

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
    std::cerr << "!knownSourceMap(" << sum << "), sources: " << sources_csv
              << std::endl;
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
    std::cerr << "!knownSourceMap(" << sum << "), sources: " << sources_csv
              << std::endl;
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

}  // anonymous namespace

auto show(const std::vector<std::string>& name_list,
    const SourceList& xor_src_list) -> std::set<int> {
  auto results = get_results(name_list, xor_src_list);
  display_results(name_list, results);
  return results.sums;
}

auto consistency_check(const std::vector<std::string>& name_list,
    const SourceList& xor_src_list) -> bool {
  return are_sources_consistent(name_list, xor_src_list);
}

}  // namespace cm::components
