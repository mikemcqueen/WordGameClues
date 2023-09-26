// validator.cpp

#include <vector>
#include "clue-manager.h"
#include "combo-maker.h"
#include "peco.h"
#include "validator.h"

using namespace cm;

namespace validator {

namespace {

// TODO: move to clue-manager
std::vector<NcResultMap> ncResultMaps;

// TODO: move to clue-manager
NcResultData& getNcResult(const NameCount& nc) {
  if ((int)ncResultMaps.size() <= nc.count) {
    ncResultMaps.resize(nc.count + 1);
  }
  auto& map = ncResultMaps.at(nc.count);
  // TODO: specialize std::hash for NameCount?
  const auto nc_str = nc.toString();
  auto it = map.find(nc_str);
  if (it == map.end()) {
    map.insert(std::make_pair(nc_str, NcResultData{}));
    it = map.find(nc_str);
  }
  return it->second;
}

auto buildNcSourceIndexLists(const NameCountList& nc_list) {
  Peco::IndexListVector idx_lists;
  for (const auto& nc : nc_list) {
    if (nc.count == 1) {
      idx_lists.emplace_back(Peco::make_index_list(
        clue_manager::getSourcesForPrimaryClueName(nc.name)));
    } else {
      idx_lists.emplace_back(Peco::make_index_list(getNumNcResults(nc)));
    }
  }
  return idx_lists;
}

}  // namespace

// TODO: move to clue-manager
auto getNumNcResults(const NameCount& nc) -> int {
  return getNcResult(nc).src_list.size();
}

// TODO: move to clue-manager
void appendNcResults(const NameCount& nc, SourceList& src_list) {
  auto& nc_result = getNcResult(nc);
  for (auto& src: src_list) {
    if (nc_result.src_compat_set.find(src) == nc_result.src_compat_set.end()) {
      // add to set *before* moving to list
      nc_result.src_compat_set.insert(src);
      nc_result.src_list.emplace_back(std::move(src));
    }
  }
}

auto mergeNcListCombo(const NameCountList& nc_list, const IndexList& idx_list)
  -> std::optional<SourceData> {
  //
  SourceData src;
  for (size_t i{}; i < idx_list.size(); ++i) {
    const auto& nc = nc_list.at(i);
    if (nc.count > 1) {
      const auto& nc_result_src = getNcResult(nc).src_list.at(idx_list.at(i));
      if (!src.addCompoundSource(nc_result_src)) {
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

/*
type VSForNameCountArgs = NameListContainer & CountListContainer
    & NcListContainer & VSFlags;

let validateSourcesForNameCount = (clueName: string|undefined, srcName: string,
    srcCount: number, args: VSForNameCountArgs): ValidateSourcesResult =>
{
    Debug(`++validateSourcesForNameCount(${clueName}), ${srcName}:${srcCount}` +
        `, validateAll: ${args.validateAll} ${indentNewline()}` +
        `  ncList: ${args.ncList}, nameList: ${args.nameList}`);

    let ncList = copyAddNcList(args.ncList, srcName, srcCount);
    if (_.isEmpty(ncList)) {
        // TODO:
        // duplicate name:count entry. technically this is allowable for
        // count > 1 if the there are multiple entries of this clue name
        // in the clueList[count]. (at least as many entries as there are
        // copies of name in ncList)
        // SEE ALSO: copyAddNcList()
        // NOTE: this should be fixable with some effort if it ever fires.
        console.error(`  duplicate nc, ${srcName}:{srcCount}`);
        return { success: false }; // fail
    }
    Debug(`  added nc ${srcName}:${srcCount}, ncList.length: ${ncList.length}`);
    // If only one name & count remain, we're done.
    // (name & count lists are equal length, just test one)
    if (args.nameList.length === 1) {
        let result: ValidateSourcesResult;
         // NOTE getting rid of this validateAll check might fix --copy-from, --add, etc.
        if (args.fast && args.validateAll) {
            result = mergeNcListResults(ncList, args);
        } else {
            Assert(0, "was curious if this was used, didn't think it was (it shouldn't be)");
            result = OldValidator.checkUniqueSources(ncList, args);
            Debug(`checkUniqueSources --- ${result.success ? 'success!' : 'failure'}`);
        }
        if (result.success) {
            args.ncList.push(NameCount.makeNew(srcName, srcCount));
            Debug(`  added ${srcName}:${srcCount}, ncList(${ncList.length}): ${ncList}`);
        }
        return result;
    }
    
    // nameList.length > 1, remove current name & count,
    // and validate remaining
    Debug(` calling validateSourcesForNameCountLists recursively, ncList: ${ncList}`);
    let rvsResult = validateSourcesForNameCountLists(clueName,
        chop_copy(args.nameList, srcName), chop_copy(args.countList, srcCount), {
            ncList,
            fast: args.fast,
            validateAll: args.validateAll
        });
    if (!rvsResult.success) {
        Debug('--validateSourcesForNameCount: validateSourcesForNameCountLists failed');
        return rvsResult;
    }
    // does this achieve anything? modifies args.ncList. answer: probably.
    // TODO: probably need to remove why that matters. answer: maybe.
    // TODO: use slice() (or clone()?)
    args.ncList.length = 0;
    ncList.forEach(nc => args.ncList.push(nc));
    Debug(`--validateSourcesForNameCount, add ${srcName}:${srcCount}` +
          `, ncList(${ncList.length}): ${ncList}`);
    return rvsResult;
};
*/

/*
let copyAddNcList = (ncList: NameCount.List, name: string, count: number): NameCount.List => {
    // for non-primary check for duplicate name:count entry
    // technically this is allowable for count > 1 if the there are
    // multiple entries of this clue name in the clueList[count].
    // (at least as many entries as there are copies of name in ncList)
    // TODO: make knownSourceMapArray store a count instead of boolean

    if (!ncList.every(nc => {
        if (nc.count > 1) {
            if ((name === nc.name) && (count === nc.count)) {
                return false;
            }
        }
        return true;
    })) {
        return [];
    }
    let newNcList = ncList.slice();
    newNcList.push(NameCount.makeNew(name, count));
    return newNcList;
}
*/

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

/*
let chop_copy = (list: any, removeValue: any): any[] => {
    let copy: any[] = [];
    list.forEach((value: any) => {
        if (value === removeValue) {
            removeValue = undefined;
        } else {
            copy.push(value);
        }
    });
    return copy;
};
*/

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

/*
type VSForNameCountListsArgs = NcListContainer & VSFlags;

let validateSourcesForNameAndCountLists = (clueName: string|undefined, nameList:
string[], countList: number[], args: VSForNameCountListsArgs):
 ValidateSourcesResult =>
{
 logLevel++;
 Debug(`++validateSourcesForNameCountLists, looking for [${nameList}] in
[${countList}]`);
 //if (xp) Expect(nameList.length).is.equal(countList.length);

 // optimization: could have a map of count:boolean entries here
 // on a per-name basis (new map for each outer loop; once a
 // count is checked for a name, no need to check it again

 let resultList: ValidateResult[] = [];
 const name = nameList[0];
 // TODO: could do this test earlier, like in calling function, check entire
// name list.
if (name === clueName) { return { success: false, list: undefined };
 }
 let success =
   countList.filter((count
                      : number) = > ClueManager.isKnownNc({name, count}))
     .some((count
             : number) = > {
       let rvsResult = validateSourcesForNameCount(clueName, name, count, {
         nameList,
         countList,
         ncList : args.ncList,
         fast : args.fast,
         validateAll : args.validateAll
       });
       if (!rvsResult.success)
         return false;  // some.continue;
         Debug(`  validateSourcesForNameCount output for: ${name}`+
             `, ncList(${args.ncList.length}): ${args.ncList}`);

         resultList = rvsResult.list !;
         return true;  // success: some.exit
     });
 --logLevel;
 return {success, list : success ? resultList : undefined};
};
*/

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

}  // namespace validator

