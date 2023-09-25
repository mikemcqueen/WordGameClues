// validator.cpp

#include <vector>
#include "validator.h"

using namespace cm;

namespace validator {

namespace {

std::vector<NcResultMap> ncResultMaps;

/*
let appendUniqueResults = (ncResult: NcResultData, ncStr: string, 
    fromResults: ValidateResult[], args: any) : void => 
{
    for (let result of fromResults) {
        const key = NameCount.listToCountList(result.nameSrcList).sort().join(',');
        if (ncResult.set.has(key)) {
            continue;
        }
        ncResult.set.add(key);
        ncResult.list.push(result);
    }
};
*/

/*
const addResultsToNcResultMap = (results: ValidateResult[], name: string,
    count: number, args: any) =>
{
    let ncResultMap = getNcResultMap(count);
    if (!ncResultMap) {
        ncResultMap = State.ncResultMaps[count] = {};
    }
    const ncStr = NameCount.makeCanonicalName(name, count);
    if (!ncResultMap[ncStr]) {
        ncResultMap[ncStr] = {
            list: [],
            set: new Set<string>()
        };
    }
    appendUniqueResults(ncResultMap[ncStr], ncStr, results, args);
};
*/

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

}  // namespace

auto getNumNcResults(const NameCount& nc) -> int {
  return getNcResult(nc).src_list.size();
}

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

/*
const mergeNcListCombo = (ncList: NameCount.List, indexList: number[]):
    MergeNcListComboResult =>
{
    ++merge_nclc;
    let validateResult = emptyValidateResult();
    // indexList value is either an index into a resultMap.list (compound clue)
    // or a primary source (primary clue)
    for (let i = 0; i < indexList.length; ++i) {
        const nc = ncList[i];
        if (nc.count > 1) { // compound clue
            const listIndex = indexList[i];
            const ncResult = ClueManager.getNcResultMap(nc.count)[nc.toString()].list[listIndex];
            if (!Source.isXorCompatible(validateResult, ncResult)) {
                return { success: false };
            }
            Source.mergeUsedSourcesInPlace(validateResult.usedSources, ncResult.usedSources);
            addCompoundNc(validateResult, nc, ncResult);
        } else { // primary clue
            const primarySrc = indexList[i];
            if (!Source.addUsedSource(validateResult.usedSources, primarySrc, true)) {
                return { success: false };
            }
            addPrimaryNameSrc(validateResult, NameCount.makeNew(nc.name, primarySrc), nc);
        }
    }
    return { success: true, validateResult };
};
*/

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

/*
  let resultList: ValidateResult[] = Peco.makeNew({ listArray })
      .getCombinations()
      .map((indexList: number[]) => {
          ++native_merge_nclc;
          return Native.mergeNcListCombo(ncListToMerge, indexList);
      })
      .filter((mergeResult: NativeMergeResult) => !!mergeResult);
*/

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

}  // namespace validator
