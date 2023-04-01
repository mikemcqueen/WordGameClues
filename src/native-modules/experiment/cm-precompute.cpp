#include <functional>
#include <vector>
#include <memory>
#include <napi.h>
#include <string>
#include <iostream>
#include <cassert>
#include <unordered_map>
#include <unordered_set>
#include "combo-maker.h"
#include "dump.h"
#include "peco.h"

namespace cm {

#if 0
//
//
let mergeSources = (source1: AnySourceData, source2: AnySourceData, lazy: boolean | undefined): AnySourceData => {
    const primaryNameSrcList = [...source1.primaryNameSrcList, ...source2.primaryNameSrcList];
    const ncList = [...source1.ncList, ...source2.ncList];
    if (lazy) {
        Assert(ncList.length === 2, `ncList.length(${ncList.length})`);
        source1 = source1 as LazySourceData;
        source2 = source2 as LazySourceData;
        const result: LazySourceData = {
            primaryNameSrcList,
            ncList,
            synonymCounts: Clue.PropertyCounts.merge(
                getSynonymCountsForValidateResult(source1.validateResultList[0]),
                getSynonymCountsForValidateResult(source2.validateResultList[0])),
            validateResultList: [
                (source1 as LazySourceData).validateResultList[0],
                (source2 as LazySourceData).validateResultList[0]
            ]
        };
        return result;
    }
    source1 = source1 as SourceData;
    source2 = source2 as SourceData;
    const mergedSource: SourceData = {
        primaryNameSrcList,
        ncList,
        synonymCounts: Clue.PropertyCounts.merge(source1.synonymCounts, source2.synonymCounts),
        sourceNcCsvList: [...source1.sourceNcCsvList, ...source2.sourceNcCsvList]
    };
    // TODO: still used?
    mergedSource.ncCsv = NameCount.listToSortedString(mergedSource.ncList);
    return mergedSource;
};
#endif

SourceData mergeSources(const SourceData& source1, const SourceData& source2) {
  NameCountList primaryNameSrcList(source1.primaryNameSrcList);
  primaryNameSrcList.insert(primaryNameSrcList.end(), source2.primaryNameSrcList.begin(),
			    source2.primaryNameSrcList.end());

  NameCountList ncList(source1.ncList);
  ncList.insert(ncList.end(), source2.ncList.begin(), source2.ncList.end());

  StringList sourceNcCsvList(source1.sourceNcCsvList);
  sourceNcCsvList.insert(sourceNcCsvList.end(), source2.sourceNcCsvList.begin(),
			 source2.sourceNcCsvList.end());
  
  //synonymCounts: Clue.PropertyCounts.merge(source1.synonymCounts, source2.synonymCounts),
  //mergedSource.ncCsv = NameCount.listToSortedString(ncList);
  return { primaryNameSrcList, ncList, sourceNcCsvList };
}

#if 0
//
//
let mergeCompatibleSources = (source1: AnySourceData, source2: AnySourceData, args: MergeArgs): AnySourceData[] => {
    // TODO: this logic could be part of mergeSources
    // also, uh, isn't there a primarySrcArray I can be using here?
    return allCountUnique2(source1.primaryNameSrcList, source2.primaryNameSrcList)
        ? [mergeSources(source1, source2, args.lazy)]
        : [];
};
#endif

#if 0
//
//
let mergeCompatibleSourceLists = (sourceList1: AnySourceData[], sourceList2: AnySourceData[], args: MergeArgs): AnySourceData[] => {
    let mergedSourcesList: AnySourceData[] = [];
    for (const source1 of sourceList1) {
        for (const source2 of sourceList2) {
            mergedSourcesList.push(...mergeCompatibleSources(source1, source2, args))
        }
    }
    return mergedSourcesList;
};
#endif

bool allCountsUnique(const NameCountList& ncList1, const NameCountList& ncList2) {
  std::unordered_set<int> hash{};
  for (const auto& nc : ncList1) {
    hash.insert(nc.count);
  }
  for (const auto& nc : ncList2) {
    if (hash.find(nc.count) != hash.end()) return false;
  }
  return true;
}

auto mergeCompatibleSourceLists(const SourceList& sourceList1, const SourceList& sourceList2) {
  SourceList result;
  for (const auto& s1 : sourceList1) {
    for (const auto& s2 : sourceList2) {
      if (allCountsUnique(s1.primaryNameSrcList, s2.primaryNameSrcList)) {
	result.emplace_back(std::move(mergeSources(s1, s2)));
      }
    }
  }
  return result;
}


#if 0
//
//
let mergeAllCompatibleSources = (ncList: NameCount.List, sourceListMap: Map<string, AnySourceData[]> | undefined,
				 args: MergeArgs): AnySourceData[] => {
    // because **maybe** broken for > 2 below
    Assert(ncList.length <= 2, `${ncList} length > 2 (${ncList.length})`);
    // TODO: reduce (or some) here
    let sourceList = sourceListMap
	? sourceListMap.get(NameCount.toString(ncList[0])) as AnySourceData[] 
	: getSourceList(ncList[0], args);
    for (let ncIndex = 1; ncIndex < ncList.length; ++ncIndex) {
        const nextSourceList: AnySourceData[] = sourceListMap
	    ? sourceListMap.get(NameCount.toString(ncList[ncIndex])) as AnySourceData[]
	    : getSourceList(ncList[ncIndex], args);
        sourceList = mergeCompatibleSourceLists(sourceList, nextSourceList, args);
        if (!sourceListHasPropertyCountInBounds(sourceList, args.synonymMinMax)) sourceList = [];
        // TODO BUG this is broken for > 2; should be something like: if (sourceList.length !== ncIndex + 1) 
        if (listIsEmpty(sourceList)) break;
    }
    return sourceList;
};
#endif

SourceList mergeAllCompatibleSources(const NameCountList& ncList, const SourceListMap& sourceListMap) {
  //const auto& firstSourceList 
  // TODO: unnecessary copy except in special case where ncList.size() == 1, rare
  SourceList sourceList = sourceListMap.at(ncList[0].toString());
  for (auto i = 1u; i < ncList.size(); ++i) {
    const auto& prevSourceList = /*(i == 1u) ? firstSourceList : */ sourceList;
    const auto& nextSourceList = sourceListMap.at(ncList[i].toString());
    sourceList = std::move(mergeCompatibleSourceLists(prevSourceList, nextSourceList));
    if (sourceList.empty()) break;
  }
  return sourceList;
}

#if 0
//
//
let buildSourceListsForUseNcData = (useNcDataLists: NCDataList[], sourceListMap: Map<string, AnySourceData[]>,
				    args: MergeArgs): SourceList[] => {
    let sourceLists: SourceList[] = [];
    // TODO: This is to prevent duplicate sourceLists. I suppose I could use a Set or Map, above?
    let hashList: StringBoolMap[] = [];
    for (let ncDataList of useNcDataLists) {
        for (let [sourceListIndex, useNcData] of ncDataList.entries()) {
            if (!sourceLists[sourceListIndex]) sourceLists.push([]);
            if (!hashList[sourceListIndex]) hashList.push({});
            // give priority to any min/max args specific to an NcData, for example, through --xormm,
            // but fallback to the values we were called with
            const mergeArgs = useNcData.synonymMinMax ? { synonymMinMax: useNcData.synonymMinMax } : args;
            const sourceList = mergeAllCompatibleSources(useNcData.ncList, sourceListMap, mergeArgs) as SourceList;
            for (let source of sourceList) {
                let key = NameCount.listToString(_.sortBy(source.primaryNameSrcList, NameCount.count));
                if (!hashList[sourceListIndex][key]) {
                    sourceLists[sourceListIndex].push(source as SourceData);
                    hashList[sourceListIndex][key] = true;
                }
            }
        }
    }
    return sourceLists;
};
#endif

std::vector<SourceList> buildSourceListsForUseNcData(
  const vector<NCDataList>& useNcDataLists, const SourceListMap& sourceListMap)
{
  using StringSet = std::unordered_set<std::string>;
  auto size = useNcDataLists[0].size();
  std::vector<StringSet> hashList(size);
  std::vector<SourceList> sourceLists(size);
  //cout << "useNcDataLists.size(" << useNcDataLists.size() << ")" << endl;
  for (const auto& ncDataList : useNcDataLists) {
    //cout << "  ncDataList.size(" << ncDataList.size() << ")" << endl;
    for (auto i = 0u; i < ncDataList.size(); ++i) {
      auto sourceList = mergeAllCompatibleSources(ncDataList[i].ncList, sourceListMap);
      //cout << "    sourceList.size(" << sourceList.size() << ")" << endl;
      for (auto& source : sourceList) {
	std::sort(source.primaryNameSrcList.begin(), source.primaryNameSrcList.end(),
		  [](const auto& a, const auto& b){ return a.count < b.count; });
	const auto key = NameCount::listToString(source.primaryNameSrcList);
	//cout << "key: " << key << endl;
	if (hashList[i].find(key) == hashList[i].end()) {
	  //cout << "  ADDING!: " << key << endl;
	  sourceLists[i].emplace_back(std::move(source));
	  hashList[i].insert(key);
	}
      }
    }
  }
  return sourceLists;
}

#if 0
const mergeCompatibleXorSources = (indexList: number[], sourceLists: SourceList[]): XorSource[] => {
    let compatible = true;

    let sources: SourceList = [];
    let srcSet = new Set<number>();

    // TODO: indexList.some()
    for (let [sourceListIndex, sourceIndex] of indexList.entries()) {
        if (ZZ) console.log(`XOR sourceListIndex(${sourceListIndex}) sourceIndex(${sourceIndex})`);
        const source = sourceLists[sourceListIndex][sourceIndex];
        if (!NameCount.listAddCountsToSet(source.primaryNameSrcList, srcSet)) {
            compatible = false;
            break;
        }
        sources.push(source);
    }
    if (compatible) {
        let primaryNameSrcList: NameCount.List = [];
        let ncList: NameCount.List = [];
        for (let source of sources) {
            primaryNameSrcList.push(...source.primaryNameSrcList);
            ncList.push(...source.ncList);
        }

        // I feel like this is still valid and worth removing or commenting
        //Assert(!_.isEmpty(primaryNameSrcList), 'empty primaryNameSrcList');
        let result: XorSource = {
            primaryNameSrcList,
            primarySrcArray: listToCountArray(primaryNameSrcList),
            ncList
        };
        if (ZZ) {
            console.log(` XOR compatible: true, adding` +
                ` pnsl[${NameCount.listToString(result.primaryNameSrcList)}]`);
        }
        return [result];
    } else {
        if (ZZ) console.log(` XOR compatible: false`);
    }
    return [];
}
#endif

bool addCountsToSet(const NameCountList& ncList, std::unordered_set<int>& countSet) {
  for (const auto& nc : ncList) {
    if (countSet.find(nc.count) != countSet.end()) return false;
    countSet.insert(nc.count);
  }
  return true;
}

XorSourceList mergeCompatibleXorSources(const std::vector<int>& indexList,
					const std::vector<SourceList>& sourceLists)
{
  bool compatible = true;
  std::vector<std::reference_wrapper<const SourceData>> sources{};
  /*
  std::unordered_set<int> srcSet{};
  for (auto i = 0u; i < indexList.size(); ++i) {
    const auto& source = sourceLists[i][indexList[i]];
    if (!addCountsToSet(source.primaryNameSrcList, srcSet)) {
      compatible = false;
      break;
    }
    sources.push_back(std::reference_wrapper(source));
  }
  */compatible = false;
  XorSourceList result{};
  if (compatible) {
    NameCountList primaryNameSrcList{};
    NameCountList ncList{};
    for (const auto sourceRef : sources) {
      // copy (by design?)
      const auto& pnsl = sourceRef.get().primaryNameSrcList;
      primaryNameSrcList.insert(primaryNameSrcList.end(), pnsl.begin(), pnsl.end());
      const auto& ncl = sourceRef.get().ncList;
      ncList.insert(ncList.end(), ncl.begin(), ncl.end());
    }
    // I feel like this is still valid and worth removing or commenting
    //Assert(!_.isEmpty(primaryNameSrcList), 'empty primaryNameSrcList');
    // TODO: can optimize with move constructor, only called ~1500 times tho (currently)
    XorSource mergedSource{ primaryNameSrcList, ncList,
      NameCount::listToCountSet(primaryNameSrcList) };
    result.emplace_back(std::move(mergedSource));
  }
  return result;
}

#if 0
// TODO function name,
let mergeCompatibleXorSourceCombinations = (sourceLists: SourceList[]/*, args: MergeArgs*/): XorSource[] => {
    if (listIsEmpty(sourceLists)) return [];
    let begin = new Date();
    const numEmptyLists = listGetNumEmptySublists(sourceLists);
    if (numEmptyLists > 0) {
        // TODO: sometimes a sourceList is empty, like if doing $(cat required) with a
        // low clue count range (e.g. -c2,4). should that even be allowed?
        Assert(false, `numEmpty(${numEmptyLists}), numLists(${sourceLists.length})`);
    }
    let listArray = sourceLists.map(sourceList => [...Array(sourceList.length).keys()]);
    if (ZZ) console.log(`listArray(${listArray.length}): ${Stringify2(listArray)}`);
    //console.log(`sourceLists(${sourceLists.length}): ${Stringify2(sourceLists)}`);
    const peco = Peco.makeNew({
        listArray,
        max: 99999
    });
    let sourceList: XorSource[] = [];
    for (let indexList = peco.firstCombination(); indexList; ) {
        //if (ZZ) console.log(`iter (${iter}), indexList: ${stringify(indexList)}`);
        const mergedSources: XorSource[] = mergeCompatibleXorSources(indexList, sourceLists);
        sourceList.push(...mergedSources);
        indexList = peco.nextCombination();
    }
    let end = new Duration(begin, new Date()).milliseconds;
    console.error(` merge(${PrettyMs(end)})`);
    return sourceList;
};
#endif

std::string vec_to_string(const vector<int>& v) {
  std::string result{};
  for (auto i : v) {
    result.append(std::to_string(i));
    result.append(" ");
  }
  return result;
}

auto getNumEmptySublists(const std::vector<SourceList>& sourceLists) {
  int count = 0;
  for (const auto& sl : sourceLists) {
    if (sl.empty()) count++;
  }
  return count;
}

XorSourceList mergeCompatibleXorSourceCombinations(const std::vector<SourceList>& sourceLists) {
  if (sourceLists.empty()) return {};
  assert((getNumEmptySublists(sourceLists) == 0) && "mergeCompatibleXorSourceCombinations: empty sublist");
  std::vector<int> lengths{};
  for (const auto& sl : sourceLists) {
    lengths.push_back(sl.size());
  }
  auto combos = 0;
  XorSourceList sourceList{};
  auto peco = std::make_unique<Peco>(lengths);
  for (auto indexList = peco->first_combination(); !indexList->empty();
       indexList = peco->next_combination())
  {
    //cout << "indexList(" << indexList.size() << "): " << vec_to_string(indexList) << endl;
    XorSourceList mergedSources{};// = mergeCompatibleXorSources(indexList, sourceLists);
    if (!mergedSources.empty()) {
      sourceList.emplace_back(std::move(mergedSources.back()));
    }
    ++combos;
  }
  cerr << " Native combos(" << combos << "), XorSources(" << sourceList.size() << ")" << endl;

  for (auto i = 0; i < 73939320; ++i ) ;

  return sourceList;
}

} // namespace cm
