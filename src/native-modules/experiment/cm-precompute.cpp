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
  // might be faster here to call NameCount::listToLegacySources(result.ncList)p
  result.legacySources = std::move(source1.copyMerge(source2.legacySources));
  result.usedSources = std::move(source1.usedSources.copyMerge(source2.usedSources));
  return result;
}

auto mergeCompatibleSourceLists(const SourceList& sourceList1,
  const SourceList& sourceList2)
{
  SourceList result{};
  for (const auto& source1 : sourceList1) {
    for (const auto& source2 : sourceList2) {
      if (source1.isXorCompatibleWith(source2)) {
        result.emplace_back(std::move(mergeSources(source1, source2)));
      }
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
  auto log = false;
  if (log) {
    std::cerr << "nc[0]: " << ncList[0].toString() << std::endl;
  }
  // TODO: find smallest sourcelist to copy first, then skip merge in loop?
  SourceList sourceList{ sourceListMap.at(ncList[0].toString()) }; // copy
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
  using HashMap = std::unordered_map<SourceCompatibilityData, StringSet>;

  srand(-1); // why?
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

        const auto& key = source;
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
  std::cerr << "  hash: " << hash_called << ", equal_to: "
    << equal_to_called << std::endl;
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
auto filterXorIncompatibleIndices(Peco::IndexListVector& indexLists, int first,
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

auto filterAllXorIncompatibleIndices(Peco::IndexListVector& indexLists,
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
  SourceCompatibilityData compatData(firstSource.sourceBits,
    firstSource.usedSources, firstSource.legacySources);
  for (auto i = 1u; i < indexList.size(); ++i) {
    const auto& source = sourceLists[i][indexList[i]];
    if (!compatData.isXorCompatibleWith(source)) {
      return {};
    }
    // i changed this up
    compatData.mergeInPlace(source);
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
    usedSources.mergeInPlace(sourceRef.get().usedSources);
  }
  // I feel like this is still valid and worth removing or commenting
  assert(!primaryNameSrcList.empty() && "empty primaryNameSrcList");
  XorSource mergedSource(std::move(primaryNameSrcList),
    std::move(NameCount::listToSourceBits(primaryNameSrcList)),
    std::move(usedSources),
    std::move(NameCount::listToLegacySources(primaryNameSrcList)),
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

auto mergeCompatibleXorSourceCombinations(
  const std::vector<SourceList>& sourceLists) -> XorSourceList
{
  using namespace std::chrono;

  if (sourceLists.empty()) return {};
  assert(!getNumEmptySublists(sourceLists)
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

namespace {
  void dumpSentenceVariationIndices(
    const SentenceVariationIndices& sentenceVariationIndices)
  {
    for (int s{}; s < kNumSentences; ++s) {
      const auto& variationIndicesList = sentenceVariationIndices.at(s);
      if (!variationIndicesList.empty()) {
        std::cerr << "S" << s << ": variations(" << variationIndicesList.size()
                  << ")" << std::endl;
        for (int v{}; v < (int)variationIndicesList.size(); ++v) {
          const auto& indices = variationIndicesList.at(v);
          std::cerr << "  v" << v - 1 << ": indices(" << indices.size() << ")"
                    << std::endl;
        }
      }
    }
  }
} // anon namespace
  
auto buildSentenceVariationIndices(const XorSourceList& xorSourceList,
  const std::vector<int>& xorSourceIndices) -> SentenceVariationIndices
{
  auto sentenceVariationIndices = SentenceVariationIndices{};
  for (size_t src_index = 0; src_index < xorSourceList.size(); ++src_index) {
    std::array<int, kNumSentences> variations =
      { -1, -1, -1, -1, -1, -1, -1, -1, -1 };
    const auto& source = xorSourceList.at(xorSourceIndices.at(src_index));
    for (const auto& nc : source.primaryNameSrcList) {
      using namespace Source;
      if (isCandidate(nc.count)) {
        auto sentence = getSentence(nc.count) - 1;
        auto variation = getVariation(nc.count);
        // could sanity check compare equal if not -1 here
        variations.at(sentence) = variation;
      }
    }
    for (int s{}; s < kNumSentences; ++s) {
      auto& variationIndicesList = sentenceVariationIndices.at(s);
      const size_t variation_index = variations.at(s) + 1;
      if (variationIndicesList.size() <= variation_index) {
        variationIndicesList.resize(variation_index + 1);
      }
      variationIndicesList.at(variation_index).push_back(src_index);
    }
  }
  // Some sentences may contain no variations across all xorSources.
  // At least, this is true in the current case when not all sentences use
  // variations. TODO: TBD if this is still true after all sentences have
  // been converted to use variations.
  // Until that time, destroy the variationIndicesLists for those sentences
  // with no variations, since these lists only contain a single element (0)
  // representing the "-1" variation that contains all indices.
  // It's redundant/unnecessary data and it's cleaner to be able to just test
  // if a variationIndicesList is empty.
  // Depending on resolution of TBD above, the "empty" check may eventually
  // become redundant/unnecessary.
  //for (auto& variationIndicesList : sentenceVariationIndices) {
  std::for_each(sentenceVariationIndices.begin(),
    sentenceVariationIndices.end(), [](auto& variationIndicesList)
  {
    if (variationIndicesList.size() == 1) {
      variationIndicesList.clear();
    }
  });
  if (1) {
    dumpSentenceVariationIndices(sentenceVariationIndices);
  }
  return sentenceVariationIndices;
}

//////////

namespace {
  using SourceSet = std::unordered_set<int>;
  using SourceCountMap = std::unordered_map<int, int>;
  using SourceFreqMap = std::unordered_map<int, double>;
  
  struct SourceCounts {
    void addToTotals(int source) {
      auto it = totals_map.find(source);
      if (it == totals_map.end()) {
        totals_map.insert(std::make_pair(source, 1));
      } else {
        it->second++;
      }
    }

    void addSources(int index, const Sources& sources) {
      auto& source_set = source_sets[index];
      for (int s{ 1 }; s <= kNumSentences; ++s) {
        auto start = Source::getFirstIndex(s);
        for (int i{}; i < kMaxUsedSourcesPerSentence; ++i) {
          auto source = sources[start + i];
          if (source == -1) break;
          source_set.insert(source);
          addToTotals(source);
        }
      }
    }
    
    void addSources(int index, const LegacySources& legacySources) {
      auto& source_set = source_sets[index];
      for (int i{}; i < kMaxLegacySources; ++i) {
        auto source = legacySources[i];
        if (!source) continue;
        source_set.insert(source);
        addToTotals(source);
      }
    }
    
    std::vector<SourceSet> source_sets;
    SourceCountMap totals_map;
  }; // struct SourceCounts
  
  /*
  auto getTotalCount(const SourceCountMap& countMap) {
    return std::accumulate(countMap.begin(), countMap.end(), 0,
      [](int total, int count) { // TODO: probably std::pair here
        total += count;
        return total;
      });
  }
  */
  
  auto makeSourceCounts(const XorSourceList& xorSources) {
    SourceCounts sourceCounts;
    sourceCounts.source_sets.resize(xorSources.size());
    std::for_each(xorSources.begin(), xorSources.end(),
      [idx = 0, &sourceCounts](const XorSource& xorSource) mutable {
        sourceCounts.addSources(idx, xorSource.usedSources.sources);
        sourceCounts.addSources(idx, xorSource.legacySources);
        idx++;
      });
    return sourceCounts;
  }
  
  auto makeSourceFreqMap(const SourceCounts& sourceCounts) {
    SourceFreqMap freqMap;
    return freqMap;
  }

  auto makeSortedIndices(const XorSourceList& xorSourceList,
    const SourceCounts& sourceCounts,
    const SourceFreqMap& sourceFreqMap)
  {
    std::vector<int> indices(xorSourceList.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
      [&sourceCounts/*, &sourceFreqMap*/](int i, int j) {
        return sourceCounts.source_sets[i].size() <
          sourceCounts.source_sets[j].size();
      });
    return indices;
  }
} // anon namespace
  
auto getSortedXorSourceIndices(const XorSourceList& xorSourceList)
  -> std::vector<int>
{
  auto sourceCounts = makeSourceCounts(xorSourceList);
  auto sourceFreqMap = makeSourceFreqMap(sourceCounts);
  auto indices = makeSortedIndices(xorSourceList, sourceCounts, sourceFreqMap);
  return indices;
}

} // namespace cm
