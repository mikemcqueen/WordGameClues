// validator.cpp

#include <vector>
#include "clue-manager.h"
#include "combo-maker.h"
#include "peco.h"
#include "util.h"
#include "validator.h"

using namespace cm;

namespace validator {

namespace {

auto buildNcSourceIndexLists(const NameCountList& nc_list) {
  Peco::IndexListVector idx_lists;
  for (const auto& nc : nc_list) {
    if (nc.count == 1) {
      idx_lists.emplace_back(
        Peco::make_index_list(clue_manager::getPrimaryClueSrcIndices(nc.name)));
    } else {
      idx_lists.emplace_back(
        Peco::make_index_list(clue_manager::get_num_nc_sources(nc)));
    }
  }
  return idx_lists;
}

}  // namespace

auto mergeNcListCombo(const NameCountList& nc_list, const IndexList& idx_list,
  const NameCount& as_nc) -> std::optional<SourceData> {
  //
  SourceData src;
  for (size_t i{}; i < idx_list.size(); ++i) {
    const auto& nc = nc_list.at(i);
    if (nc.count > 1) {
      const auto& nc_src = clue_manager::get_nc_src_list(nc).at(idx_list.at(i));
      //auto nc_src = clue_manager::copy_nc_src_as_nc(nc, idx_list.at(i));
      if (!src.addCompoundSource(nc_src)) {
        return std::nullopt;
      }
    } else if (!src.addPrimaryNameSrc(nc, idx_list.at(i))) {
      return std::nullopt;
    }
  }
  NameCountList hax_nc_list = {as_nc};
  src.ncList = std::move(hax_nc_list);
  return {src};
}

auto mergeAllNcListCombinations(const NameCountList& nc_list,
  Peco::IndexListVector&& idx_lists, const std::string& clue_name)
  -> SourceList {
  //
  SourceList src_list;
  auto count = std::accumulate(nc_list.begin(), nc_list.end(), 0,
    [](int sum, const NameCount& nc) { return sum + nc.count; });
  NameCount nc{clue_name, count};
  Peco peco(std::move(idx_lists));
  for (auto idx_list = peco.first_combination(); idx_list;
       idx_list = peco.next_combination()) {
    auto opt_src = mergeNcListCombo(nc_list, *idx_list, nc);
    if (opt_src.has_value()) {
      src_list.emplace_back(std::move(opt_src.value()));
    }
  }
  return src_list;
}

auto mergeNcListResults(
  const NameCountList& nc_list, const std::string& clue_name) -> SourceList {
  //
  auto idx_lists = buildNcSourceIndexLists(nc_list);
  return mergeAllNcListCombinations(nc_list, std::move(idx_lists), clue_name);
}

NameCountList copyNcListAddNc(
  const NameCountList& nc_list, const std::string& name, int count) {
  // for non-primary check for duplicate name:count entry
  // technically this is allowable for count > 1 if the there are
  // multiple entries of this clue name in the clueList[count].
  // (at least as many entries as there are copies of name in ncList)
  // TODO: make knownSourceMapArray store a count instead of boolean
  if ((count > 1) && NameCount::listContains(nc_list, name, count)) {
    return {};
  }
  auto list_copy = nc_list;
  list_copy.emplace_back(name, count);
  return list_copy;
}

template <typename T>
// requires T = string | int
auto chop_copy(const std::vector<T>& list, const T& chop_value) {
  std::vector<T> result;
  bool chopped = false;
  for (const auto& value: list) {
    if (!chopped && (value == chop_value)) {
      chopped = true;
    } else {
      result.emplace_back(value);
    }
  }
  return result;
}

struct VSForNameAndCountListsArgs {
  NameCountList& nc_list;
  bool validate_all;
};

auto validateSourcesForNameAndCountLists(const std::string& clue_name,
  const std::vector<std::string>& name_list, std::vector<int> count_list,
  NameCountList& nc_list) -> SourceList;

struct VSForNameCountArgs {
  NameCountList& nc_list;
  const std::vector<std::string>& name_list;
  const std::vector<int>& count_list;
};

auto validateSourcesForNameCount(const std::string& clue_name,
  const std::string& name, int count, const VSForNameCountArgs& args) -> SourceList {
  //
  auto nc_list = copyNcListAddNc(args.nc_list, name, count);
  if (nc_list.empty()) {
    // TODO:
    // duplicate name:count entry. technically this is allowable for
    // count > 1 if the there are multiple entries of this clue name
    // in the clueList[count]. (at least as many entries as there are
    // copies of name in ncList). SEE ALSO: copyAddNcList()
    // NOTE: this should be fixable with some effort if it ever fires.
    std::cerr << " duplicate nc, " << name << ":" << count << std::endl;
    return {};
  }
  // If only one name & count remain, we're done.
  // (name & count lists are equal length, just test one)
  if (args.name_list.size() == 1u) {
    // NOTE leave this here and at entry point of validateSources
    //assert(args.validate_all && "!validateAll not implemented");
    SourceList src_list = mergeNcListResults(nc_list, clue_name);
    if (!src_list.empty()) {
      args.nc_list.emplace_back(name, count);
    }
    return src_list; // TODO: playing fast & loose with NRVO here
  }
  // name_list.length > 1, remove current name & count, and validate remaining
  auto src_list = validateSourcesForNameAndCountLists(clue_name,
    chop_copy(args.name_list, name), chop_copy(args.count_list, count),
    nc_list);
  if (!src_list.empty()) {
    args.nc_list = std::move(nc_list);
  }
  return src_list;
}

auto validateSourcesForNameAndCountLists(const std::string& clue_name,
  const std::vector<std::string>& name_list, std::vector<int> count_list,
  NameCountList& nc_list) -> SourceList {
  //  const VSForNameAndCountListsArgs& args) -> SourceList {
  //
  // optimization: could have a map of count:boolean entries here
  // on a per-name basis (new map for each outer loop; once a
  // count is checked for a name, no need to check it again
  SourceList src_list;
  const auto& name = name_list.at(0);
  // TODO: could do this test earlier, in calling function, check entire
  // name list.
  if (name != clue_name) {
    for (auto count : count_list) {
      if (clue_manager::is_known_name_count(name, count)) {
        auto src_list = validateSourcesForNameCount(clue_name, name, count,
          {nc_list, name_list, count_list});
        if (!src_list.empty()) {
          return src_list;
        }
      }
    }
  }
  return {};
}

void display_addends(int sum, const std::vector<std::vector<int>>& addends) {
  std::cout << "sum: " << sum << std::endl;
  for (const auto& combination : addends) {
    std::cout << "[ ";
    for (const auto& num : combination)
      std::cout << num << ' ';
    std::cout << "]" << std::endl;
  }
}

auto validateSources(const std::string& clue_name,
  const std::vector<std::string>& src_names, int sum, bool validate_all)
  -> SourceList {
  //
  SourceList results;
  const auto addends = Peco::make_addends(sum, src_names.size());
  //display_addends(sum, addends);
  for (const auto& count_list : addends) {
    NameCountList nc_list;
    auto src_list =
      validateSourcesForNameAndCountLists(clue_name, src_names, count_list, nc_list);
    if (!src_list.empty()) {
      util::move_append(results, std::move(src_list));
      if (!validate_all) {
        break;
      }
    }
  }
  return results;
};

}  // namespace validator

