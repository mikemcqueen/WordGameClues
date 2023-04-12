#include <algorithm>
#include <chrono>
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

int merges = 0;
int list_merges = 0;
//SourceData
void mergeSources(SourceData& source1, const SourceData& source2) {
  merges++;
  //NameCountList primaryNameSrcList = source1.primaryNameSrcList;
  //primaryNameSrcList.insert(primaryNameSrcList.end(),
  source1.primaryNameSrcList.insert(source1.primaryNameSrcList.end(),
    source2.primaryNameSrcList.begin(), source2.primaryNameSrcList.end());

  //SourceBits primarySrcBits = source1.primarySrcBits | source2.primarySrcBits;
  source1.primarySrcBits |= source2.primarySrcBits;

  //NameCountList ncList = source1.ncList;
  //ncList.insert(ncList.end(), source2.ncList.begin(), source2.ncList.end());
  source1.ncList.insert(source1.ncList.end(), source2.ncList.begin(), source2.ncList.end());

  //StringList sourceNcCsvList = source1.sourceNcCsvList;
  //sourceNcCsvList.insert(sourceNcCsvList.end(),
  source1.sourceNcCsvList.insert(source1.sourceNcCsvList.end(),
    source2.sourceNcCsvList.begin(), source2.sourceNcCsvList.end());
  
  //synonymCounts: Clue.PropertyCounts.merge(source1.synonymCounts, source2.synonymCounts),
  //mergedSource.ncCsv = NameCount.listToSortedString(ncList);
  //return { primaryNameSrcList, primarySrcBits, ncList, sourceNcCsvList };
}

auto mergeCompatibleSourceLists(const MergedSourcesList& mergedSourcesList, const SourceList& sourceList2) {
  list_merges++;
  MergedSourcesList result{};
  for (const auto& mergedSources : mergedSourcesList) {
    for (const auto& s2 : sourceList2) {
      if ((mergedSources.primarySrcBits & s2.primarySrcBits).none()) {
	MergedSources ms{
	  mergedSources.primarySrcBits | s2.primarySrcBits,
	  mergedSources.sourceCRefList
	};
	ms.sourceCRefList.push_back(SourceCRef{s2});
	result.emplace_back(std::move(ms));
      }
    }
  }
  return result;
}

auto makeMergedSourcesList(const SourceList& sourceList) {
  //cout << sourceList.size() << endl;
  MergedSourcesList mergedSourcesList;
  for (const auto& source : sourceList) {
    MergedSources ms;
    ms.primarySrcBits = source.primarySrcBits;
    ms.sourceCRefList = SourceCRefList{SourceCRef{source}};
    mergedSourcesList.emplace_back(std::move(ms));
  }
  return mergedSourcesList;
}

// NOTE: for ncList.size() >= 2
//
MergedSourcesList mergeAllCompatibleSources(const NameCountList& ncList, const SourceListMap& sourceListMap) {
  // because **maybe** broken for > 2 below
  assert(ncList.size() <= 2 && "ncList.length > 2");
  // TODO: find smallest sourcelist to copy first, then skip merge in loop?
  MergedSourcesList mergedSourcesList = std::move(makeMergedSourcesList(sourceListMap.at(ncList[0].toString())));
  for (auto i = 1u; i < ncList.size(); ++i) {
    const auto& nextSourceList = sourceListMap.at(ncList[i].toString());
    mergedSourcesList = std::move(mergeCompatibleSourceLists(mergedSourcesList, nextSourceList));
    // TODO BUG this is broken for > 2; should be something like: if (sourceList.length !== ncIndex + 1) 
    if (mergedSourcesList.empty()) break;
  }
  return mergedSourcesList;
}

const SourceList& getSourceList(const NameCountList& ncList, const SourceListMap& sourceListMap) {
  assert((ncList.size() == 1) && "not implemented, but could be easily");
  return sourceListMap.at(ncList[0].toString());
}

std::vector<SourceCRefList> buildSourceListsForUseNcData(
  const vector<NCDataList>& useNcDataLists, const SourceListMap& sourceListMap)
{
  using StringSet = std::unordered_set<std::string>;
  using BitsToStringSetMap = std::unordered_map<SourceBits, StringSet>;

  srand(-1);
  int total = 0;
  int hash_hits = 0;
  int synonyms = 0;
  auto size = useNcDataLists[0].size();
  std::vector<BitsToStringSetMap> hashList(size);
  std::vector<SourceCRefList> sourceCRefLists(size);
  for (const auto& ncDataList : useNcDataLists) {
    for (auto i = 0u; i < ncDataList.size(); ++i) {
      // for size == 2: return by value; could return reference to static local in a pinch
      //auto sourceList = mergeAllCompatibleSources(ncDataList[i].ncList, sourceListMap);
      const auto& sourceList = getSourceList(ncDataList[i].ncList, sourceListMap);
      total += sourceList.size();
      for (const auto& source : sourceList) {
	// TODO: NOT GOOD ENOUUGH. still need a set of strings in value type.
	// HOWEVER, instead of listToSring'ing 75 million lists, how about we
	// mark "duplicate count, different name", i.e. "aliased" sources in 
	// clue-manager, and only listToString and add those to a separate
	// map<bitset, set<String>> for ncLists with "shared" primarySrcBits.
	// probably don't need a separate map. just don't bother populating
	// and looking in the set unless its an aliased source.
	auto it = hashList[i].find(source.primarySrcBits);
	if (it != hashList[i].end()) {
	  hash_hits++;
#if 0
	  if (it->second.find(key) != it->second.end()) continue;
	  synonyms++;
	  cout << "synonyms:" << endl << " " << key << endl
	       << " " << *it->second.begin() << endl << endl;
#endif
	  continue;
	}
	hashList[i][source.primarySrcBits] = StringSet{};
	//it = hashList[i].find(source.primarySrcBits);
	//it->second.insert(key);
	//sourceLists[i].emplace_back(std::move(source));
	sourceCRefLists[i].emplace_back(SourceCRef{source});
      }
    }
  }
  cerr << " hash_hits: " << hash_hits << ", synonyms: " << synonyms
       << ", total: " << total << ", list_merges: " << list_merges 
       << ", source_merges: " << merges << ", sourceLists: " << sourceCRefLists.size() 
       << ", " << std::accumulate(sourceCRefLists.begin(), sourceCRefLists.end(), 0u,
	  [](size_t total, const SourceCRefList& list){ return total + list.size(); }) << endl;
  return sourceCRefLists;
}

XorSourceList mergeCompatibleXorSources(const std::vector<int>& indexList,
  const std::vector<SourceCRefList>& sourceLists)
{
  // this part is inner-loop (100s of millions potentially) and should be fast
  // probably should make it a separate function
  SourceCRefList sources{};
  SourceBits bits{};
  for (auto i = 0u; i < indexList.size(); ++i) {
    const auto sourceRef = sourceLists[i][indexList[i]]; // reference, uh, optional here?
    if ((bits & sourceRef.get().primarySrcBits).any()) return {};
    bits |= sourceRef.get().primarySrcBits;
    sources.push_back(sourceRef);
  }
  // below here is called much less often, less speed critical

  NameCountList primaryNameSrcList{};
  NameCountList ncList{};
  for (const auto sourceRef : sources) {
    const auto& pnsl = sourceRef.get().primaryNameSrcList;
    primaryNameSrcList.insert(primaryNameSrcList.end(), pnsl.begin(), pnsl.end()); // copy (by design?)
    const auto& ncl = sourceRef.get().ncList;
    ncList.insert(ncList.end(), ncl.begin(), ncl.end());                           // copy (by design?)
  }
  // I feel like this is still valid and worth removing or commenting
  assert(!primaryNameSrcList.empty() && "empty primaryNameSrcList");

  XorSourceList result{};
  XorSource mergedSource{
    primaryNameSrcList,
    NameCount::listToSourceBits(primaryNameSrcList),
    ncList
  };
  result.emplace_back(std::move(mergedSource));
  return result;
}

std::string vec_to_string(const vector<int>& v) {
  std::string result{};
  for (auto i : v) {
    result.append(std::to_string(i));
    result.append(" ");
  }
  return result;
}

auto getNumEmptySublists(const std::vector<SourceCRefList>& sourceLists) {
  auto count = 0;
  for (const auto& sl : sourceLists) {
    if (sl.empty()) count++;
  }
  return count;
}

XorSourceList mergeCompatibleXorSourceCombinations(
  const std::vector<SourceCRefList>& sourceLists)
{
  using namespace std::chrono;

  if (sourceLists.empty()) return {};
  assert((getNumEmptySublists(sourceLists) == 0) && "mergeCompatibleXorSourceCombinations: empty sublist");
  std::vector<int> lengths{};
  for (const auto& sl : sourceLists) {
    lengths.push_back(sl.size());
  }

  auto peco0 = high_resolution_clock::now();

  int combos = 0;
  XorSourceList sourceList{};
  Peco peco(lengths);
  for (auto indexList = peco.first_combination(); indexList;
       indexList = peco.next_combination())
  {
    XorSourceList mergedSources = mergeCompatibleXorSources(*indexList, sourceLists);
    if (!mergedSources.empty()) {
      sourceList.emplace_back(std::move(mergedSources.back()));
    }
    ++combos;
  }

  auto peco1 = high_resolution_clock::now();
  auto d_peco = duration_cast<milliseconds>(peco1 - peco0).count();
  cerr << " Native peco loop: " << d_peco << "ms" << ", combos: " << combos
       << ", XorSources: " << sourceList.size() << endl;
  
  return sourceList;
}

} // namespace cm
