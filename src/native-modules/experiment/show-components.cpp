// show-components.cpp

#include <iostream>
#include <unordered_set>
#include "clue-manager.h"
#include "combo-maker.h"
#include "merge-filter-data.h"
#include "show-components.h"
#include "util.h"

namespace cm::show_components {

struct NamesAndCounts {
  std::vector<std::string> names;
  std::vector<int> counts;
};

struct Results {
  std::unordered_set<int> sums;
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
void add_result_for_sources(const std::string& sources_csv,
  const std::vector<int>& counts, int sum, Results& results) {
  //
  using namespace clue_manager;
  if (has_known_source_map(sum)) {
    if (is_known_source_map_entry(sum, sources_csv)) {
      const auto& names =
        get_known_source_map_entry(sum, sources_csv).clue_names;
      results.known.emplace_back(names, counts);
    } else {
      results.valid.emplace_back(counts);
    }
  } else {
    std::cerr << "!knownSourceMap(" << sum << "), sources: " << sources_csv
              << std::endl;
  }
}

auto get_results(const std::vector<std::string>& name_list) {
  Results results;
  std::unordered_set<std::string> hash;
  for (const auto& src: MFD.merge_xor_src_list) {
    const auto count_list = NameCount::listToCountList(src.ncList);
    const auto key = util::join(count_list, ",");
    if (hash.contains(key)) {
      continue;
    }
    hash.insert(key);
    const auto sum = util::sum(count_list);
    if (name_list.size() == 1u) {
      add_result_for_nc(NameCount{name_list.at(0), sum}, results);
    } else {
      add_result_for_sources(
        util::join(name_list, ","), count_list, sum, results);
      results.sums.insert(sum);
    }
  }
  return results;
}

auto get_source_clues(const std::vector<std::string>& source_list,
  const NamesAndCounts& names_counts) {
  //
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
  //
  for (const auto& names_counts : names_counts_list) {
    std::string sources{};
    if (names_counts.counts.size() > 1u) {
      // -t name1,name2[,name3,...] (multiple names)
      sources += get_source_clues(source_list, names_counts);
    } else {
      // -t name (one name only)
      sources += util::join(names_counts.names, ",");
    }
    display(names_counts.counts, text, sources);
  }
}

void display_results(
  const std::vector<std::string>& name_list, const Results& results) {
  //
  display(results.invalid, "INVALID");
  display(results.known, "PRESENT as", name_list);
  display(results.clues, "PRESENT as clue with source:", name_list);
  display(results.valid, "VALID");
}

}  // namespace

void of(const std::vector<std::string>& name_list) {
  auto results = get_results(name_list);
  display_results(name_list, results);
}

}  // namespace cm::show_components
