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


auto getCandidateCount(const SourceData& source) {
  auto count = 0;
  for (const auto& nameSrc : source.primaryNameSrcList) {
    if (nameSrc.count >= 1'000'000) {
      ++count;
    }
  }
  return count;
}

auto hasConflictingCandidates(const SourceData& source) {
  if (getCandidateCount(source) < 2) return false;
  UsedSources usedSources = { -1 };
  std::cerr << "-----" << endl;
  for (const auto& nameSrc : source.primaryNameSrcList) {
    if (nameSrc.count < 1'000'000) continue;
    auto sentence = nameSrc.count / 1'000'000;
    auto variation = (nameSrc.count % 1'000'000) / 100;
    std::cerr << sentence << "," << variation << ":" << nameSrc.count << std::endl;
    if ((usedSources[sentence] >= 0) && (usedSources[sentence] != variation)) {
      std::cerr << "true" << endl;
      return true;
    }
    usedSources[sentence] = variation;
  }
  std::cerr << "false" << endl;
  return false;
}

auto hasCandidate(const SourceData& source) {
  return getCandidateCount(source) > 0;
}

auto getCandidateCount(const SourceList& sourceList) {
  auto count = 0;
  for (const auto& source : sourceList) {
    count += getCandidateCount(source);
  }
  return count;
}

auto getCandidateCount(const SourceCRefList& sourceCRefList) {
  auto count = 0;
  for (const auto sourceRef : sourceCRefList) {
    count += getCandidateCount(sourceRef.get());
  }
  return count;
}

auto hasCandidate(const SourceCRefList& sourceCRefList) {
  return getCandidateCount(sourceCRefList) > 0;
}

void debugSource(const SourceData& source, std::string_view sv) {
  if (getCandidateCount(source) < 2) return;
  std::cerr << sv << ": " << NameCount::listToString(source.primaryNameSrcList) << std::endl;
}

void debugAddSourceToList(const SourceData& source,
  const SourceList& sourceList)
{
  //  if (!hasCandidate(source) || !hasCandidate(sourceCRefList)) return;
  if (getCandidateCount(source) + getCandidateCount(sourceList) < 2) return;
  std::string sources{};
  for (const auto& other : sourceList) {
    if (getCandidateCount(other)) {
      if (!sources.empty()) sources += " - ";
      sources += NameCount::listToString(other.primaryNameSrcList);
    }
  }
  std::cerr << "adding " << NameCount::listToString(source.primaryNameSrcList)
	    <<" to " << sources << endl;
}

void debugAddSourceToList(const SourceData& source,
  const SourceCRefList& sourceCRefList)
{
  //  if (!hasCandidate(source) || !hasCandidate(sourceCRefList)) return;
  if (getCandidateCount(source) + getCandidateCount(sourceCRefList) < 2) return;
  std::string sources{};
  for (const auto sourceCRef : sourceCRefList) {
    if (getCandidateCount(sourceCRef)) {
      if (!sources.empty()) sources += " - ";
      sources += NameCount::listToString(sourceCRef.get().primaryNameSrcList);
    }
  }
  // try:
  //std::cerr << "adding " << std::to_string(source.primaryNameSrcList) << 
  std::cerr << "adding " << NameCount::listToString(source.primaryNameSrcList)
	    <<" to " << sources << std::endl;
}

void debugSourceList(const SourceCRefList& sourceCRefList, std::string_view sv) {
  auto first = true;
  for (const auto sourceCRef : sourceCRefList) {
    if (!hasConflictingCandidates(sourceCRef.get())) continue;
    if (first) {
      std::cerr << sv << ":" << std::endl;
      first = false;
    }
    std::cerr << "  " << NameCount::listToString(sourceCRef.get().primaryNameSrcList) << std::endl;
  }
}

void debugSourceList(const SourceList& sourceList, std::string_view sv) {
  auto first = true;
  for (const auto& source : sourceList) {
    if (!hasConflictingCandidates(source)) continue;
    if (first) {
      std::cerr << sv << ":" << std::endl;
      first = false;
    }
    std::cerr << "  " << NameCount::listToString(source.primaryNameSrcList) << std::endl;
  }
}

void debugMergedSource(const SourceData& mergedSource, const SourceData& source1,
  const SourceData& source2)
{
  auto candidates = getCandidateCount(mergedSource);
  if (candidates > 1) {
    std::cerr << "candidates(" << candidates << "),"
	      << " merged " << NameCount::listToString(source1.primaryNameSrcList)
	      << " with " << NameCount::listToString(source2.primaryNameSrcList)
	      << std::endl;
  }
}

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
    std::cerr << "]" << std::endl;
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
    std::cerr << "]" << std::endl;
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

const SourceList& getSourceList(const NameCountList& ncList, const SourceListMap& sourceListMap) {
  assert((ncList.size() == 1) && "not implemented, but could be easily");
  return sourceListMap.at(ncList[0].toString());
}


auto mergeSources(const SourceData& source1, const SourceData& source2) {
  SourceData result{};
  result.primaryNameSrcList = std::move(NameCount::listMerge(source1.primaryNameSrcList,
    source2.primaryNameSrcList));
  //debugSource(result, "merged source");
  result.ncList = std::move(NameCount::listMerge(source1.ncList, source2.ncList));
  result.sourceBits = std::move(source1.sourceBits | source2.sourceBits);
  result.usedSources = std::move(mergeUsedSources(source1.usedSources, source2.usedSources));
  return result;
}

auto mergeCompatibleSourceLists(const SourceList& sourceList1,
  const SourceList& sourceList2)
{
  SourceList result{};
  for (const auto& source1 : sourceList1) {
    for (const auto& source2 : sourceList2) {
      if (source1.isCompatibleWith(source2)) {
	auto mergedSource = mergeSources(source1, source2);
	//debugMergedSource(mergedSource, source1, source2);
	result.emplace_back(std::move(mergedSource));
      }
    }
  }
  return result;
}

// NOTE: for ncList.size() >= 2
//
auto mergeAllCompatibleSources(const NameCountList& ncList,
  const SourceListMap& sourceListMap) -> SourceList
{
  // because **maybe** broken for > 2 below
  assert(ncList.size() <= 2 && "ncList.length > 2");
  // TODO: find smallest sourcelist to copy first, then skip merge in loop?
  SourceList sourceList{sourceListMap.at(ncList[0].toString())}; // copy
  //debugSourceList(sourceList, "initial copy");
  for (auto i = 1u; i < ncList.size(); ++i) {
    const auto& nextSourceList = sourceListMap.at(ncList[i].toString());
    sourceList = std::move(mergeCompatibleSourceLists(sourceList, nextSourceList));
    // TODO BUG this is broken for > 2; should be something like: if (sourceList.length !== ncIndex + 1) 
    if (sourceList.empty()) break;
  }
  //debugSourceList(sourceList, "merged list");
  return sourceList;
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

auto buildSourceListsForUseNcData(const vector<NCDataList>& useNcDataLists,
  const SourceListMap& sourceListMap) -> std::vector<SourceList>
{
  // possible optimization:
  // instead of constructing a new sourcelist in mergeAllCompatible,
  // we could have a new data structure like a SourceData but that
  // contains a list of NcCRefLists for both ncList and primaryNameSrcList,
  // and only merge sourceBits and usedSources (for the purposes of
  // determining compatibility). Then, at return/wrap time, we marshal
  // the lists-of-NcCRefLists into a single NcList.

  using StringSet = std::unordered_set<std::string>;
  using BitStringToStringSetMap = std::unordered_map<string, StringSet>;

  srand(-1);
  int total = 0;
  int hash_hits = 0;
  int synonyms = 0;
  auto size = useNcDataLists[0].size();
  std::vector<BitStringToStringSetMap> hashList(size);
  std::vector<SourceList> sourceLists(size);
  for (const auto& ncDataList : useNcDataLists) {
    for (auto i = 0u; i < ncDataList.size(); ++i) {
      // for size == 2: return by value; could return reference to static local in a pinch
      auto sourceList = mergeAllCompatibleSources(ncDataList[i].ncList, sourceListMap);
      //const auto& sourceList = getSourceList(ncDataList[i].ncList, sourceListMap);
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
	auto key = makeBitString(source);
	auto it = hashList[i].find(key);
	if (it != hashList[i].end()) {
	  hash_hits++;
	  continue;
	}
	hashList[i][key] = StringSet{};
	//it = hashList[i].find(source.sourceBits);
	//it->second.insert(key);
	//debugSource(source, "source after mergeAll");
	sourceLists[i].emplace_back(std::move(source));
      }
    }
  }
  cerr << " hash_hits: " << hash_hits << ", synonyms: " << synonyms
       << ", total: " << total << ", list_merges: " << list_merges 
       << ", source_merges: " << merges << ", sourceLists: " << sourceLists.size() 
       << ", " << std::accumulate(sourceLists.begin(), sourceLists.end(), 0u,
         [](size_t total, const SourceList& list){ return total + list.size(); })
       << std::endl;
  return sourceLists;
}

//////////

auto getNumEmptySublists(const std::vector<SourceList>& sourceLists) {
  auto count = 0;
  for (const auto& sl : sourceLists) {
    if (sl.empty()) count++;
  }
  return count;
}

auto anyCompatibleXorSources(const SourceData& source,
  const Peco::IndexList& indexList, const SourceList& sourceList)
{
  for (auto it = indexList.cbegin(); it != indexList.cend(); ++it) {
    if (source.isCompatibleWith(sourceList[*it])) {
      return true;
    }
  }
  return false;
}

// TODO: comment this. code is tricky
bool filterXorIncompatibleIndices(Peco::IndexListVector& indexLists, int first,
  int second, const std::vector<SourceList>& sourceLists)
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
  const std::vector<SourceList>& sourceLists)
{
  if (indexLists.size() < 2u) return true;
  for (auto first = 0u; first < indexLists.size(); ++first) {
    for (auto second = 0u; second < indexLists.size(); ++second) {
      if (first == second) continue;
      if (!filterXorIncompatibleIndices(indexLists, first, second, sourceLists)) {
	return false;
      }
    }
  }
  return true;
}

// this part is inner-loop and should be fast
// NOTE: used to be 100s of millions potentially, now more like 10s of thousands.
SourceCRefList getCompatibleXorSources(const std::vector<int>& indexList,
  const std::vector<SourceList>& sourceLists)
{
  // auto reference type, uh, optional here?
  SourceCRefList sourceCRefList{};
  const auto& firstSource = sourceLists[0][indexList[0]];
  sourceCRefList.emplace_back(SourceCRef{firstSource});
  SourceBits sourceBits{ firstSource.sourceBits }; // copy (ok)
  UsedSources usedSources{ firstSource.usedSources }; // copy (ok)
  for (auto i = 1u; i < indexList.size(); ++i) {
    const auto& source = sourceLists[i][indexList[i]];
    if (!sourceBitsAreCompatible(sourceBits, source.sourceBits) ||
	!usedSourcesAreCompatible(usedSources, source.usedSources))
    {
      return {};
    }
    sourceBits |= source.sourceBits;
    mergeUsedSourcesInPlace(usedSources, source.usedSources);
    //debugAddSourceToList(source, sourceCRefList);
    sourceCRefList.emplace_back(SourceCRef{source});
  }
  return sourceCRefList;
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

XorSourceList mergeCompatibleXorSourceCombinations(
  const std::vector<SourceList>& sourceLists)
{
  using namespace std::chrono;

  if (sourceLists.empty()) return {};
  assert((getNumEmptySublists(sourceLists) == 0) && "mergeCompatibleXorSourceCombinations: empty sublist");
  std::vector<int> lengths{};
  for (const auto& sl : sourceLists) {
    lengths.push_back(sl.size());
  }
  cerr << "  initial lengths: " << vec_to_string(lengths)
       << ", product: " << vec_product(lengths) << std::endl;
  auto indexLists = Peco::initial_indices(lengths);
  bool valid = filterAllXorIncompatibleIndices(indexLists, sourceLists);
  
#if 1
  lengths.clear();
  for (const auto& il : indexLists) {
    lengths.push_back(list_size(il));
  }
  cerr << "  filtered lengths: " << vec_to_string(lengths)
       << ", product: " << vec_product(lengths)
       << ", valid: " << boolalpha << valid << std::endl;
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
    SourceCRefList sourceCRefList = getCompatibleXorSources(*indexList, sourceLists);
    if (sourceCRefList.empty()) continue;
    ++compatible;
    XorSourceList mergedSources = mergeCompatibleXorSources(sourceCRefList);
    if (mergedSources.empty()) continue;
    ++merged;
    xorSourceList.emplace_back(std::move(mergedSources.back()));
  }
  auto peco1 = high_resolution_clock::now();
  auto d_peco = duration_cast<milliseconds>(peco1 - peco0).count();
  cerr << " Native peco loop: " << d_peco << "ms" << ", combos: " << combos
       << ", compatible: " << compatible << ", merged: " << merged
       << ", XorSources: " << xorSourceList.size() << std::endl;
  
  return xorSourceList;
}

} // namespace cm
