#include <algorithm>
#include <cassert>
#include <chrono>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <napi.h>
#include "combo-maker.h"
#include "dump.h"
#include "peco.h"

namespace cm {

int merges = 0;
int list_merges = 0;

auto sourceBitsAreCompatible(const SourceBits& bits1, const SourceBits& bits2) {
  return (bits1 & bits2).none();
}
  
auto usedSourcesAreCompatible(const UsedSources& used1, const UsedSources& used2) {
  for (auto i = 1u; i < used1.size(); ++i) {
    if (used1[i] && used2[i] && (used1[i] != used2[i])) {
      return false;
    }
  }
  return true;
}

void mergeUsedSourcesInPlace(UsedSources& to, const UsedSources& from) {
#if 0
  auto isCandidate = [](int source) { return source > 0; };
  auto toCandidate = std::find_if(to.cbegin(), to.cend(), isCandidate) != to.cend();
  auto fromCandidate = std::find_if(from.cbegin(), from.cend(), isCandidate) != from.cend();
  if (toCandidate && fromCandidate) {
    std::cerr << "merging to [";
    std::copy(std::begin(to), std::end(to), std::ostream_iterator<int>(std::cerr, " "));
    std::cerr << "] with [";
    std::copy(std::begin(from), std::end(from), std::ostream_iterator<int>(std::cerr, " "));
    std::cerr << "]" << endl;
  }
#endif
  for (auto i = 1u; i < from.size(); ++i) {
    if (from[i]) {
      if (to[i]) {
	assert(to[i] == from[i]);
      } else {
	to[i] = from[i];
      }
    }
  }
#if 0
  if (toCandidate) { // && fromCandidate) {
    std::cerr << "  to after: [";
    std::copy(std::begin(to), std::end(to), std::ostream_iterator<int>(std::cerr, " "));
    std::cerr << "]" << endl;
  }
  //return to;
#endif
}

auto mergeUsedSources(const UsedSources& used1, const UsedSources& used2) {
  UsedSources result{used1}; // copy (ok)
  mergeUsedSourcesInPlace(result, used2);
  return result;
}

// TODO: potential template function for these two
auto sourcesAreCompatible(const SourceData& source1, const SourceData& source2) {
  if (!sourceBitsAreCompatible(source1.sourceBits, source2.sourceBits)) {
    return false;
  }
  return usedSourcesAreCompatible(source1.usedSources, source2.usedSources);
}

auto sourcesAreCompatible(const MergedSources& mergedSources, const SourceData& source) {
  if (!sourceBitsAreCompatible(mergedSources.sourceBits, source.sourceBits)) {
    return false;
  }
  return usedSourcesAreCompatible(mergedSources.usedSources, source.usedSources);
}

auto mergeCompatibleSourceLists(const MergedSourcesList& mergedSourcesList, const SourceList& sourceList) {
  list_merges++;
  MergedSourcesList result{};
  for (const auto& mergedSources : mergedSourcesList) {
    for (const auto& source : sourceList) {
      if (mergedSources.isCompatibleWith(source)) {
	// TODO:
	/*
	MergedSources copyOfMergedSources{mergedSources};
	copyOfMergedSources.mergeWith(source);
	*/
	MergedSources ms{};
	ms.sourceBits = std::move(mergedSources.sourceBits | source.sourceBits);
	ms.usedSources = std::move(mergeUsedSources(mergedSources.usedSources, source.usedSources));
	ms.sourceCRefList = mergedSources.sourceCRefList; // no way around this copy
	ms.sourceCRefList.emplace_back(SourceCRef{source}); // copy (ok)
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
    mergedSourcesList.emplace_back(std::move(MergedSources{source}));
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

auto makeUsedSourcesString(const UsedSources& usedSources) {
  std::string result;
  for (auto i = 1u; i < usedSources.size(); i++) {
    result.append(std::to_string(usedSources[i]));
  }
  return result;
};
  
auto makeBitString(const SourceData& source) {
  return source.sourceBits.to_string().append("+")
    .append(makeUsedSourcesString(source.usedSources));
};

std::vector<SourceCRefList> buildSourceListsForUseNcData(
  const vector<NCDataList>& useNcDataLists, const SourceListMap& sourceListMap)
{
  using StringSet = std::unordered_set<std::string>;
  using BitStringToStringSetMap = std::unordered_map<string, StringSet>;

  srand(-1);
  int total = 0;
  int hash_hits = 0;
  int synonyms = 0;
  auto size = useNcDataLists[0].size();
  std::vector<BitStringToStringSetMap> hashList(size);
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
	// map<bitset, set<String>> for ncLists with "shared" sourceBits.
	// probably don't need a separate map. just don't bother populating
	// and looking in the set unless its an aliased source.
	
	// Well,I had a strong opinion, above, there, and I don't think I ever
	// resolved it. Apparently sourceBits aren't good enough here.
	// I should probably figure out what I was thinking, especially now
	// that usedSources is being phased in (and is not used here).

	// So it seems the point of this hashlist is to outputing *duplicate*
	// sourcelists. No effort is made here to determine if the sourceLists
	// themselves are actually compatible, among their constituent sources,
	// which is what I think needs to happen. I'm not sure why it's
	// happening at this point though.  Wasn't there a BuildNcDataLists
	// process that happeend before this, was it completely unaware of
	// compatibiltiy of any kind?

	auto key = makeBitString(source);
	//std::cerr << key << endl;
	auto it = hashList[i].find(key);
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
	hashList[i][key] = StringSet{};
	//it = hashList[i].find(source.sourceBits);
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

auto hasCandidate(const SourceCRef sourceRef) {
  for (const auto& nameSrc : sourceRef.get().primaryNameSrcList) {
    if (nameSrc.count >= 1'000'000) return true;
  }
  return false;
}
  
auto hasCandidate(const SourceCRefList& sourceList) {
  for (const auto sourceRef : sourceList) {
    if (hasCandidate(sourceRef)) return true;
  }
  return false;
}

// this part is inner-loop and should be fast
// NOTE: used to be 100s of millions potentially, now more like 10s of thousands.
SourceCRefList getCompatibleXorSources(const std::vector<int>& indexList,
  const std::vector<SourceCRefList>& sourceLists)
{
  // auto reference type, uh, optional here?
  const auto firstSourceRef = sourceLists[0][indexList[0]];
  SourceCRefList sourceList{};
  sourceList.emplace_back(firstSourceRef);
  SourceBits sourceBits{ firstSourceRef.get().sourceBits };
  UsedSources usedSources{ firstSourceRef.get().usedSources };
  for (auto i = 1u; i < indexList.size(); ++i) {
    // auto reference type, uh, optional here?
    const auto sourceRef = sourceLists[i][indexList[i]];
    if (!sourceBitsAreCompatible(sourceBits, sourceRef.get().sourceBits) ||
	!usedSourcesAreCompatible(usedSources, sourceRef.get().usedSources))
    {
      return {};
    }
    sourceBits |= sourceRef.get().sourceBits;
    mergeUsedSourcesInPlace(usedSources, sourceRef.get().usedSources);
    if (hasCandidate(sourceList) && hasCandidate(sourceRef)) {
      std::cerr << "hi" << endl;
    }
    sourceList.emplace_back(sourceRef);
  }
  return sourceList;
}

// here is called much less often, less speed critical
XorSourceList mergeCompatibleXorSources(const SourceCRefList& sourceList) {
  // TODO: this is kind of a weird way of doing things that requires a
  // XorSource (SourceData) multi-type move constructor. couldn't I
  // just start with an XorSource here initialized to sourceList[0] values
  // and merge-in-place all the subsequent elements? could even be a
  // SourceData member function.
  // TODO: Also, why am I not just |='ing srcbits in the loop?
  NameCountList primaryNameSrcList{};
  NameCountList ncList{};
  UsedSources usedSources{};
  for (const auto sourceRef : sourceList) {
    const auto& pnsl = sourceRef.get().primaryNameSrcList;
    primaryNameSrcList.insert(primaryNameSrcList.end(), pnsl.begin(), pnsl.end()); // copy (by design?)
    const auto& ncl = sourceRef.get().ncList;
    ncList.insert(ncList.end(), ncl.begin(), ncl.end());                           // copy (by design?)
    mergeUsedSourcesInPlace(usedSources, sourceRef.get().usedSources);
  }
  // I feel like this is still valid and worth removing or commenting
  assert(!primaryNameSrcList.empty() && "empty primaryNameSrcList");
  XorSource mergedSource(std::move(primaryNameSrcList),
			 std::move(NameCount::listToSourceBits(primaryNameSrcList)),
			 std::move(usedSources),
			 std::move(ncList));
  XorSourceList result{};
  result.emplace_back(std::move(mergedSource));
  return result;
}

bool anyCompatibleXorSources(const SourceCRef sourceRef, const Peco::IndexList& indexList,
  const SourceCRefList& sourceList)
{
  for (auto it = indexList.cbegin(); it != indexList.cend(); ++it) {
    //    if (sourcesAreCompatible(sourceRef.get(), sourceList[*it].get())) {
    if (sourceRef.get().isCompatibleWith(sourceList[*it].get())) {
      return true;
    }
  }
  return false;
}

// TODO: comment this. code is tricky
bool filterXorIncompatibleIndices(Peco::IndexListVector& indexLists, int first,
  int second, const std::vector<SourceCRefList>& sourceLists)
{
  Peco::IndexList& firstList = indexLists[first];
  for (auto it_first = firstList.before_begin();
       std::next(it_first) != firstList.end();
       /* nothing */)
  {
    const auto& first_source = sourceLists[first][*std::next(it_first)];
    bool any_compat = anyCompatibleXorSources(first_source, indexLists[second],
      sourceLists[second]);
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
