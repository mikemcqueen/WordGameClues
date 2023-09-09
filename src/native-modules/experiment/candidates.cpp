// candidates.cpp

#include <cassert>
#include <chrono>
#include <optional>
#include <unordered_map>
#include "combo-maker.h"
#include "candidates.h"

namespace {
}  // namespace

namespace cm {

auto add_candidate(int sum, const std::string&& combo, int index) -> int {
  auto& candidate_list = allSumsCandidateData.find(sum)->second;
  candidate_list.at(index).combos.emplace(std::move(combo));
  return index;
}

int add_candidate(int sum, std::string&& combo,
  std::reference_wrapper<const SourceCompatibilityList> src_list_cref) {
  //
  if (auto it = allSumsCandidateData.find(sum);
      it == allSumsCandidateData.end()) {
    allSumsCandidateData.emplace(std::make_pair(sum, CandidateList{}));
  }
  std::set<std::string> combos{};
  combos.emplace(std::move(combo));
  // TODO: this move probably doesn't work without a specific constructor
  CandidateData candidate_data{src_list_cref, std::move(combos)};
  auto& candidate_list = allSumsCandidateData.find(sum)->second;
  candidate_list.emplace_back(std::move(candidate_data));
  return candidate_list.size() - 1;
}

#if 0
type MergedSources = Source.ListContainer & Source.CompatibilityData & OptionalCloneOnMerge;
type MergedSourcesList = MergedSources[];

struct MergedSource : SourceCompatibilityData {
  SourceCompatibilityCRefList src_cref_list;
};
using MergedSourceList = std::vector<MergedSource>;
#endif

#if 0
const mergeSourceInPlace = (mergedSources: MergedSources,
    source: Source.Data): MergedSources =>
{
    ++ms_inplace;
    CountBits.orInPlace(mergedSources.sourceBits, source.sourceBits);
    Source.mergeUsedSourcesInPlace(mergedSources.usedSources, source.usedSources);
    mergedSources.sourceList.push(source);
    return mergedSources;
}

const mergeSource = (mergedSources: MergedSources, source: Source.Data):
    MergedSources =>
{
    ++ms_copy;
    return {
        //CountBits.or(mergedSources.sourceBits, source.sourceBits),
        sourceBits: mergedSources.sourceBits.clone().union(source.sourceBits),
        usedSources: Source.mergeUsedSources(mergedSources.usedSources, source.usedSources),
        sourceList: [...mergedSources.sourceList, source]
    };
};

const makeMergedSourcesList = (sourceList: Source.List) : MergedSourcesList => {
    let result: MergedSourcesList = [];
    for (const source of sourceList) {
        result.push({
            // CountBits.makeFrom(source.sourceBits),
            sourceBits: source.sourceBits, // .clone(),
            usedSources: source.usedSources, // Source.cloneUsedSources(source.usedSources),
            sourceList: [source],
            cloneOnMerge: true
        });
    }
    return result;
};
#endif

#if 0
interface MergeData {
    mergedSources: MergedSources;
    sourceList: Source.List;
}
#endif

struct MergeData {
  SourceCompatibilityData merged_src;
  SourceCRefList src_cref_list;
};

#if 0
const getCompatibleSourcesMergeData = (mergedSourcesList: MergedSourcesList,
    sourceList: Source.List): MergeData[] =>
{
    let result: MergeData[] = [];
    for (let mergedSources of mergedSourcesList) {
        let mergeData: MergeData = { mergedSources, sourceList: [] };
        for (const source of sourceList) {
            ++ms_comp;
            if (Source.isXorCompatible(mergedSources, source)) {
                ++ms_compat;
                mergeData.sourceList.push(source);
            }
        }
        if (!listIsEmpty(mergeData.sourceList)) {
            result.push(mergeData);
        }
    }
    return result;
};
#endif

auto make_merge_list(const SourceCompatibilityList& merged_src_list,
  const SourceList& src_list) {
  //
  std::vector<MergeData> merge_list;
  for (const auto& merged_src : merged_src_list) {
    SourceCRefList compat_src_list;
    for (const auto& src : src_list) {
      if (merged_src.isXorCompatibleWith(src)) {
        compat_src_list.emplace_back(std::cref(src));
      }
    }
    if (!compat_src_list.empty()) {
      merge_list.push_back(
        MergeData{merged_src, std::move(compat_src_list)});
    }
  }
  return merge_list;
}

#if 0
const mergeSourcesInMergeData = (mergeData: MergeData[]): MergedSourcesList => {
    let result: MergedSourcesList = [];
    // TODO: since indexing isn't used below, i could use for..of
    for (let i = 0; i < mergeData.length; ++i) {
        let data = mergeData[i];
        for (let j = 0; j < data.sourceList.length; j++) {
            const source = data.sourceList[j];
            result.push(mergeSource(data.mergedSources, source));
        }
    }
    return result;
}
#endif

auto make_merged_src_list(const std::vector<MergeData>& merge_list) {
  SourceCompatibilityList merged_src_list;
  for (const auto& merge_data : merge_list) {
    for (const auto src_cref : merge_data.src_cref_list) {
      merged_src_list.emplace_back(
        merge_data.merged_src.copyMerge(src_cref.get()));
    }
  }
  return merged_src_list;
}

#if 0
const mergeAllCompatibleSources3 = (ncList: NameCount.List,
    sourceListMap: Map<string, Source.AnyData[]>, flag?: number): MergedSourcesList =>
{
    // because **maybe** broken for > 2
    Assert(ncList.length <= 2, `${ncList} length > 2 (${ncList.length})`);
    let mergedSourcesList: MergedSourcesList = [];
    for (let nc of ncList) {
        const sources = sourceListMap.get(NameCount.toString(nc)) as Source.List;
        if (listIsEmpty(mergedSourcesList)) {
            mergedSourcesList = makeMergedSourcesList(sources);
            continue;
        }
        const mergeData = getCompatibleSourcesMergeData(mergedSourcesList, sources);
        if (listIsEmpty(mergeData)) {
            return [];
        }
        mergedSourcesList = mergeSourcesInMergeData(mergedSourcesList, mergeData, flag);
    }
    return mergedSourcesList;
};
#endif

SourceCompatibilityList merge_all_compatible_sources(
  const NameCountList& ncList, const SourceListMap& src_list_map) {
  //
  assert(ncList.size() <= 2 && "ncList.size() > 2");
  SourceCompatibilityList merged_src_list;
  for (const auto& nc : ncList) {
    const auto& src_list = src_list_map.find(nc.toString())->second;
    if (merged_src_list.empty()) {
      merged_src_list.assign(src_list.begin(), src_list.end()); // deep copy of elements
      continue;
    }
    // TODO: optimization opportunity. Walk through loop building merge_lists
    // first. If any are empty, bail. After all succeed, do actual merges.
    const auto merge_list = make_merge_list(merged_src_list, src_list);
    if (merge_list.empty()) {
      return {};
    }
    merged_src_list = make_merged_src_list(merge_list);
  }
  return merged_src_list;
}

#if 0
  const key : string = NameCount.listToString(result.ncList!);
  let mergedSourcesList : MergedSourcesList = [];
  if (!hash[key]) {
    mergedSourcesList =
      mergeAllCompatibleSources3(result.ncList !, pcd.sourceListMap);
    if (listIsEmpty(mergedSourcesList)) {
      ++numMergeIncompatible;
    }
    hash[key] = {mergedSourcesList};
  } else {
    mergedSourcesList = hash[key].mergedSourcesList;
    numCacheHits += 1;
  }

  // failed to find any compatible combos
  if (listIsEmpty(mergedSourcesList))
    continue;

  // const combo = result.nameList!.sort().toString();
  const listOrIndex =
    (hash[key].index === undefined) ? mergedSourcesList : hash[key].index;
  hash[key].index =
    NativeComboMaker.addCandidateForSum(sum, combo, listOrIndex);
  candidateCount++;
#endif

struct CandidateRepoValue {
  SourceCompatibilityList merged_src_list;
  std::optional<int> opt_idx;
};

std::unordered_map<std::string, CandidateRepoValue> candidate_repo;

void consider_candidate(const NameCountList& ncList, int sum) {
  auto key = NameCount::listToString(ncList);
  if (candidate_repo.find(key) == candidate_repo.end()) {
    CandidateRepoValue repo_value;
    repo_value.merged_src_list =
      std::move(merge_all_compatible_sources(ncList, PCD.sourceListMap));
    candidate_repo.emplace(std::make_pair(key, std::move(repo_value)));
  }
  auto& repo_value = candidate_repo.find(key)->second;
  if (repo_value.merged_src_list.empty()) {
    return;
  }
  auto combo = NameCount::listToString(NameCount::listToNameList(ncList));
  if (repo_value.opt_idx.has_value()) {
    add_candidate(sum, std::move(combo), repo_value.opt_idx.value());
  } else {
    repo_value.opt_idx = add_candidate(
      sum, std::move(combo), std::cref(repo_value.merged_src_list));
  }
}

} // namespace cm

