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

int merges = 0;
int list_merges = 0;

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

// this part is inner-loop (100s of millions potentially) and should be fast
SourceCRefList getCompatibleXorSources(const std::vector<int>& indexList,
  const std::vector<SourceCRefList>& sourceLists)
{
  const auto firstSourceRef = sourceLists[0][indexList[0]]; // reference type, uh, optional here?
  SourceCRefList sourceList{};
  sourceList.emplace_back(std::move(firstSourceRef));
  SourceBits bits{};
  bits |= firstSourceRef.get().primarySrcBits;
  for (auto i = 1u; i < indexList.size(); ++i) {
    const auto sourceRef = sourceLists[i][indexList[i]]; // reference type, uh, optional here?
    if ((bits & sourceRef.get().primarySrcBits).any()) return {};
    bits |= sourceRef.get().primarySrcBits;
    sourceList.emplace_back(std::move(sourceRef));
  }
  return sourceList;
}

// here is called much less often, less speed critical
XorSourceList mergeCompatibleXorSources(const SourceCRefList& sourceList) {
  NameCountList primaryNameSrcList{};
  NameCountList ncList{};
  for (const auto sourceRef : sourceList) {
    const auto& pnsl = sourceRef.get().primaryNameSrcList;
    primaryNameSrcList.insert(primaryNameSrcList.end(), pnsl.begin(), pnsl.end()); // copy (by design?)
    const auto& ncl = sourceRef.get().ncList;
    ncList.insert(ncList.end(), ncl.begin(), ncl.end());                           // copy (by design?)
  }
  // I feel like this is still valid and worth removing or commenting
  assert(!primaryNameSrcList.empty() && "empty primaryNameSrcList");
  XorSource mergedSource(std::move(primaryNameSrcList),
			 std::move(NameCount::listToSourceBits(primaryNameSrcList)),
			 std::move(ncList));
  XorSourceList result{};
  result.emplace_back(std::move(mergedSource));
  return result;
}

bool anyCompatibleXorSources(const SourceCRef sourceRef, const Peco::IndexList& indexList,
  const SourceCRefList& sourceList)
{
  SourceBits sourceBits = sourceRef.get().primarySrcBits;
  for (auto it_index = indexList.cbegin(); it_index != indexList.cend(); ++it_index) {
    if ((sourceBits & sourceList[*it_index].get().primarySrcBits).none())
      return true;
  }
  return false;
}

bool filterXorIncompatibleIndices(Peco::IndexListVector& indexLists, int first, int second,
  const std::vector<SourceCRefList>& sourceLists)
{
  Peco::IndexList& firstList = indexLists[first];
  for (auto it_first = firstList.before_begin(); std::next(it_first) != firstList.end(); /* nothing */) {
    bool any_compat = anyCompatibleXorSources(sourceLists[first][*std::next(it_first)],
      indexLists[second], sourceLists[second]);
    if (!any_compat) {
      firstList.erase_after(it_first);
    } else {
      ++it_first;
    }
  }
  return !firstList.empty();
}

bool filterAllXorIncompatibleIndices(Peco::IndexListVector& indexLists,
  const std::vector<SourceCRefList>& sourceLists)
{
  if (indexLists.size() < 2u) return true;
  for (auto first = 0u; first < indexLists.size(); ++first) {
    for (auto second = 0u; second < indexLists.size(); ++second) {
      if (first == second) continue;
      //cerr << "  comparing " << first << " to " << second << endl;
      if (!filterXorIncompatibleIndices(indexLists, first, second, sourceLists)) {
	return false;
      }
    }
  }
  return true;
}

auto list_size(const Peco::IndexList& indexList) {
  int size = 0;
  std::for_each(indexList.cbegin(), indexList.cend(),
		[&size](int i){ ++size; });
  return size;
}

int vec_product(const vector<int>& v) {
  int result{1};
  for (auto i : v) {
    result *= i;
  }
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
  cerr << "  initial lengths: " << vec_to_string(lengths)
       << ", product: " << vec_product(lengths) << endl;
  auto indexLists = Peco::initial_indices(lengths);
  bool valid = filterAllXorIncompatibleIndices(indexLists, sourceLists);
  
#if 1
  lengths.clear();
  for (const auto& il : indexLists) {
    lengths.push_back(list_size(il));
  }
  cerr << "  filtered lengths: " << vec_to_string(lengths)
       << ", product: " << vec_product(lengths)
       << ", valid: " << boolalpha << valid << endl;
#endif

  auto peco0 = high_resolution_clock::now();

  int combos = 0;
  int compatible = 0;
  int merged = 0;
  XorSourceList xorSourceList{};
  Peco peco(std::move(indexLists));
  for (auto indexList = peco.first_combination(); indexList;
       indexList = peco.next_combination())
  {
    ++combos;
    SourceCRefList sourceList = getCompatibleXorSources(*indexList, sourceLists);
    if (sourceList.empty()) continue;
    ++compatible;
    XorSourceList mergedSources = mergeCompatibleXorSources(sourceList);
    if (mergedSources.empty()) continue;
    ++merged;
    xorSourceList.emplace_back(std::move(mergedSources.back()));
  }
  auto peco1 = high_resolution_clock::now();
  auto d_peco = duration_cast<milliseconds>(peco1 - peco0).count();
  cerr << " Native peco loop: " << d_peco << "ms" << ", combos: " << combos
       << ", compatible: " << compatible << ", merged: " << merged
       << ", XorSources: " << xorSourceList.size() << endl;
  
  return xorSourceList;
}

} // namespace cm
