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
      idx_lists.emplace_back(Peco::make_index_list(
        clue_manager::getSourcesForPrimaryClueName(nc.name)));
    } else {
      idx_lists.emplace_back(
        Peco::make_index_list(clue_manager::getNumNcResults(nc)));
    }
  }
  return idx_lists;
}

}  // namespace

auto mergeNcListCombo(const NameCountList& nc_list, const IndexList& idx_list)
  -> std::optional<SourceData> {
  //
  SourceData src;
  for (size_t i{}; i < idx_list.size(); ++i) {
    const auto& nc = nc_list.at(i);
    if (nc.count > 1) {
      const auto& known_nc_src =
        clue_manager::get_known_nc_source(nc, idx_list.at(i));
      if (!src.addCompoundSource(known_nc_src)) {
        return std::nullopt;
      }
    } else if (!src.addPrimaryNameSrc(nc, idx_list.at(i))) {
      return std::nullopt;
    }
  }
  // TODO: return {src} ?
  return std::make_optional(src);
}

auto mergeAllNcListCombinations(const NameCountList& nc_list,
  Peco::IndexListVector&& idx_lists) -> SourceList {
  //
  SourceList src_list;
  Peco peco(std::move(idx_lists));
  for (auto idx_list = peco.first_combination(); idx_list;
       idx_list = peco.next_combination()) {
    auto opt_src = mergeNcListCombo(nc_list, *idx_list);
    if (opt_src.has_value()) {
      src_list.emplace_back(std::move(opt_src.value()));
    }
  }
  return src_list;
}

auto mergeNcListResults(const NameCountList& nc_list) -> SourceList {
  auto idx_lists = buildNcSourceIndexLists(nc_list);
  return mergeAllNcListCombinations(nc_list, std::move(idx_lists));
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
    SourceList src_list = mergeNcListResults(nc_list);
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
    // args.ncList.clear();
    // nc_list.forEach(nc -> args.nc_list.emplace_back(std::move(nc)));
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

/*
export const validateSources = (clueName: string|undefined, args: any):
    ValidateSourcesResult =>
{
    Debug(`++validateSources(${clueName})` +
          `${indentNewline()}  nameList(${args.nameList.length}): ${args.nameList}` +
          `, sum(${args.sum})` +
          `, count(${args.count})` +
          `, validateAll: ${args.validateAll}`);

    let success = false;
    let resultList: ValidateResult[] = [];
    Peco.makeNew({
        sum:   args.sum,
        count: args.count,
        max:   args.max
    }).getCombinations().some((countList: number[]) => {
        let sourceList = Native.validateSourcesForNameAndCountLists(clueName,
            args.nameList, countList, []);
        if (sourceList.length) {
            Debug('validateSources: VALIDATE SUCCESS!');
            //if (rvsResult.list) {
            // TODO: return empty array, get rid of .success
            resultList.push(...sourceList);
            //}
            success = true;
            if (!args.validateAll) return true; // found a match; some.exit
            Debug('validateSources: validateAll set, continuing...');
        }
        return false; // some.continue
    });
    Debug('--validateSources');

    return {
        success,
        list: success ? resultList : undefined
    };
};
*/

void get_addends_helper(int sum, int count, int start,
  std::vector<int>& current, std::vector<std::vector<int>>& result) {
  if (!count) {
    if (!sum) {
      result.push_back(current);
    }
    return;
  }
  for (auto i = start; i <= sum; ++i) {
    current.push_back(i);
    get_addends_helper(sum - i, count - 1, i, current, result);
    current.pop_back();  // backtrack
  }
}

std::vector<std::vector<int>> get_addends(int sum, int count) {
    std::vector<int> current;
    std::vector<std::vector<int>> result;
    get_addends_helper(sum, count, 1, current, result);
    return result;
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
  const auto addends = get_addends(sum, src_names.size());
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

