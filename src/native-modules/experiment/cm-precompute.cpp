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

#if !USEDSOURCES_BITSET
// TODO: move to a UsedSources type
auto makeUsedSourcesString(const UsedSources& usedSources) {
  std::string result{ "hi" };
  for (auto i = 1u; i < usedSources.size(); i++) {
    if (usedSources[i].empty()) {
      result.append("-");
    } else {
      auto first = true;
      for (const auto source: usedSources[i]) {
        if (!first) {
          result.append(",");
        }
        first = false;
        result.append(std::to_string(source));
      }
    }
  }
  return result;
};

auto makeBitString(const SourceData& source) {
  return source.sourceBits.to_string().append("+")
    .append(makeUsedSourcesString(source.usedSources));
};
#endif
  
/*
auto getCandidateCount(const SourceData& source) {
  auto count = 0;
  for (const auto& nameSrc : source.primaryNameSrcList) {
    if (Source::isCandidate(nameSrc.count)) {
      ++count;
    }
  }
  return count;
}

auto hasConflictingCandidates(const SourceData& source) {
  if (getCandidateCount(source) < 2) return false;
  UsedSources usedSources{};
  std::cerr << "-----" << endl;
  for (const auto& nameSrc : source.primaryNameSrcList) {
    if (nameSrc.count < 1'000'000) continue;
    auto sentence = nameSrc.count / 1'000'000;
    auto variation = (nameSrc.count % 1'000'000);
    std::cerr << sentence << "," << variation << ":" << nameSrc.count << std::endl;
    if (!usedSources[sentence].empty() && (usedSources[sentence].find(variation) != usedSources[sentence].end())) {
      std::cerr << "true" << endl;
      return true;
    }
    usedSources[sentence].insert(variation);
  }
  std::cerr << "false" << endl;
  return false;
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

void debugSource(const SourceData& source, std::string_view sv) {
  if (getCandidateCount(source) < 2) return;
  std::cerr << sv << ": " << NameCount::listToString(source.primaryNameSrcList) << std::endl;
}

void debugAddSourceToList(const SourceData& source,
  const SourceList& sourceList)
{
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
*/

const SourceList& getSourceList(const NameCountList& ncList,
  const SourceListMap& sourceListMap)
{
  assert((ncList.size() == 1) && "not implemented, but could be easily");
  return sourceListMap.at(ncList[0].toString());
}

auto mergeSources(const SourceData& source1, const SourceData& source2) {
  SourceData result{};
  result.primaryNameSrcList = std::move(NameCount::listMerge(
    source1.primaryNameSrcList, source2.primaryNameSrcList));
  result.ncList = std::move(NameCount::listMerge(source1.ncList, source2.ncList));
  result.sourceBits = std::move(source1.sourceBits | source2.sourceBits);
  result.usedSources = std::move(source1.merge(source2.usedSources));
  return result;
}

auto mergeCompatibleSourceLists(const SourceList& sourceList1,
  const SourceList& sourceList2)
{
  SourceList result{};
  for (const auto& source1 : sourceList1) {
    for (const auto& source2 : sourceList2) {
      if (!source1.isXorCompatibleWith(source2)) continue;
      result.emplace_back(std::move(mergeSources(source1, source2)));
    }
  }
  return result;
}

// NOTE: for ncList.size() <= 2
//
auto mergeAllCompatibleSources(const NameCountList& ncList,
  const SourceListMap& sourceListMap) -> SourceList
{
  // because **maybe** broken for > 2 below
  assert(ncList.size() <= 2 && "ncList.length > 2");
  constexpr auto log = false;
  if (log) {
    std::cerr << "nc[0]: " << ncList[0].toString() << std::endl;
  }
  // TODO: find smallest sourcelist to copy first, then skip merge in loop?
  SourceList sourceList{sourceListMap.at(ncList[0].toString())}; // copy
  for (auto i = 1u; i < ncList.size(); ++i) {
    if (log) {
      std::cerr << " nc[" << i << "]: " << ncList[1].toString() << std::endl;
    }
    const auto& nextSourceList = sourceListMap.at(ncList[i].toString());
    sourceList = std::move(mergeCompatibleSourceLists(sourceList, nextSourceList));
    // TODO BUG this is broken for > 2; should be something like:
    // if (sourceList.length !== ncIndex + 1) 
    if (sourceList.empty()) break;
  }
  return sourceList;
}

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
#if !USEDSOURCES_BITSET
  using HashMap = std::unordered_map<string, StringSet>;
#else
  using HashMap = std::unordered_map<SourceCompatibilityData, StringSet>;
#endif

  srand(-1);
  int total = 0;
  int hash_hits = 0;
  const auto size = useNcDataLists[0].size();
  std::vector<HashMap> hashList(size);
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
        
        // usedSources is being used (in makeBitString). and there are lots
        // of hash hits, so I know this is necessary. i'm not sure why its
        // still not good enough though, according to above. Might have been
        // due to fear of "name variations" with same source? that, we should
        // should actually allow those name variations, and they are not
        // currently allowed because we are checking sources only? That sounds
        // right.

#if !USEDSOURCES_BITSET
        const auto key = makeBitString(source);
#else
        const auto& key = source;
#endif
        if (hashList[i].find(key) != hashList[i].end()) {
          hash_hits++;
          continue;
        }
        hashList[i][key] = StringSet{}; // std::move ?
        //it = hashList[i].find(source.sourceBits);
        //it->second.insert(key);
        //debugSource(source, "source after mergeAll");
        sourceLists[i].emplace_back(std::move(source));
      }
    }
  }
#if USEDSOURCES_BITSET
  std::cerr << "  hash: " << hash_called << ", equal_to: "
    << equal_to_called << std::endl;
#endif
  std::cerr << "  total sources: " << total << ", hash_hits: " << hash_hits
    << ", sourceLists(" << sourceLists.size() << "): "
    << std::accumulate(sourceLists.begin(), sourceLists.end(), 0u,
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
    if (source.isXorCompatibleWith(sourceList[*it])) {
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

auto list_to_string(const std::vector<int>& v) {
  std::string r;
  auto first = true;
  for (auto e : v) {
    if (!first) r.append(",");
    first = false;
    r.append(std::to_string(e));
  }
  return r;
}

/*
auto set_to_string(const std::set<uint32_t>& s) {
  std::string r;
  auto first = true;
  for (auto e : s) {
    if (!first) r.append(",");
    first = false;
    r.append(std::to_string(e));
  }
  return r;
}
*/

// This is called in an inner-loop and should be fast.
// NOTE: used to be 100s of millions potentially, now more like 10s of thousands.
SourceCRefList getCompatibleXorSources(const std::vector<int>& indexList,
  const std::vector<SourceList>& sourceLists)
{
  // auto reference type, uh, optional here?
  SourceCRefList sourceCRefList{};
  const auto& firstSource = sourceLists[0][indexList[0]];
  sourceCRefList.emplace_back(SourceCRef{ firstSource });
  SourceCompatibilityData compatData(firstSource.sourceBits, firstSource.usedSources);
  for (auto i = 1u; i < indexList.size(); ++i) {
    const auto& source = sourceLists[i][indexList[i]];
    if (!compatData.isXorCompatibleWith(source)) {
      return {};
    }
    compatData.sourceBits |= source.sourceBits;
    compatData.mergeInPlace(source.usedSources);
    sourceCRefList.emplace_back(SourceCRef{ source });
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
    SourceCompatibilityData::mergeInPlace(usedSources, sourceRef.get().usedSources);
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
  assert((getNumEmptySublists(sourceLists) == 0)
    && "mergeCompatibleXorSourceCombinations: empty sublist");
  std::vector<int> lengths{};
  for (const auto& sl : sourceLists) {
    lengths.push_back(sl.size());
  }
  std::cerr << "  initial lengths: " << vec_to_string(lengths)
            << ", product: " << vec_product(lengths) << std::endl;
  auto indexLists = Peco::initial_indices(lengths);
  bool valid = filterAllXorIncompatibleIndices(indexLists, sourceLists);
  
#if 1
  lengths.clear();
  for (const auto& il : indexLists) {
    lengths.push_back(list_size(il));
  }
  std::cerr << "  filtered lengths: " << vec_to_string(lengths)
            << ", product: " << vec_product(lengths)
            << ", valid: " << boolalpha << valid << std::endl;
#endif
  if (!valid) return {};

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
  std::cerr << " native peco loop: " << d_peco << "ms" << ", combos: " << combos
            << ", compatible: " << compatible << ", merged: " << merged
            << ", XorSources: " << xorSourceList.size() << std::endl;
  
  return xorSourceList;
}

//////////

auto buildVariationIndicesMaps(const XorSourceList& xorSourceList)
  -> std::array<VariationIndicesMap, kNumSentences>
{
  auto variationIndicesMaps = std::array<VariationIndicesMap, kNumSentences>{};
  for (size_t i = 0; i < xorSourceList.size(); ++i) {
    std::array<int, kNumSentences> variations = { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
    for (const auto& nc : xorSourceList[i].primaryNameSrcList) {
      using namespace Source;
      if (isCandidate(nc.count)) {
        auto sentence = getSentence(nc.count) - 1;
        auto variation = getVariation(nc.count);
        // could sanity check compare equal if not -1 here
        variations[sentence] = variation;
      }
    }
    for (auto s = 0; s < kNumSentences; ++s) {
      //if (variations[s] > -1) {
      auto& map = variationIndicesMaps[s];
      if (map.find(variations[s]) == map.end()) {
        map.insert(std::make_pair(variations[s], std::vector<int>{})); // TODO: {}, emplace?
      }
      map[variations[s]].push_back(i);
      // could assert(true) on ibPair.second here
      //}
    }
  }
  if (1) {
    for (auto s = 0; s < kNumSentences; ++s) {
      const auto& map = variationIndicesMaps[s];
      if (map.size() > 1) {
        std::cerr << "S" << s << ": variations(" << map.size() << ")"
                  << std::endl;
        for (auto it = map.begin(); it != map.end(); ++it) {
          auto [key, value] = *it;
          std::cerr << "  v" << key << ": indices(" << value.size() << ")"
                    << std::endl;
        }
      }
    }
  }
  return variationIndicesMaps;
}

} // namespace cm
