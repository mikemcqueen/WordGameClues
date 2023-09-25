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
#include "cm-precompute.h"
#include "peco.h"
#include "merge.h"

namespace cm {

namespace {

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
  result.usedSources =
    std::move(source1.usedSources.copyMerge(source2.usedSources));
  return result;
}

auto mergeCompatibleSourceLists(const SourceList& sourceList1,
  const SourceList& sourceList2)
{
  SourceList result{};
  for (const auto& source1 : sourceList1) {
    for (const auto& source2 : sourceList2) {
      if (source1.isXorCompatibleWith(source2)) {
        result.emplace_back(mergeSources(source1, source2));
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
  constexpr auto log = false;
  if constexpr (log) {
    std::cerr << "nc[0]: " << ncList[0].toString() << std::endl;
  }
  // TODO: find smallest sourcelist to copy first, then skip merge in loop?
  SourceList sourceList{ sourceListMap.at(ncList[0].toString()) }; // copy
  for (auto i = 1u; i < ncList.size(); ++i) {
    if constexpr (log) {
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

}  // namespace

auto buildSourceListsForUseNcData(const std::vector<NCDataList>& useNcDataLists,
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

  srand(-1); // why? hash?
  int total = 0;
  int hash_hits = 0;
  const auto size = useNcDataLists[0].size();
  std::vector<HashMap> hashList(size);
  std::vector<SourceList> sourceLists(size);
  for (const auto& ncDataList : useNcDataLists) {
    for (size_t i{}; i < ncDataList.size(); ++i) {
      // for size == 2: return by value; could return reference to static local in a pinch
      auto sourceList = mergeAllCompatibleSources(ncDataList[i].ncList, sourceListMap);
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
#if defined(PRECOMPUTE_LOGGING)
  std::cerr << "  hash: " << hash_called << ", equal_to: "
    << equal_to_called << std::endl;
  std::cerr << "  total sources: " << total << ", hash_hits: " << hash_hits
    << ", sourceLists(" << sourceLists.size() << "): "
    << std::accumulate(sourceLists.begin(), sourceLists.end(), 0u,
      [](size_t total, const SourceList& list){ return total + list.size(); })
    << std::endl;
#endif
  return sourceLists;
}

//////////

namespace {

void dumpSentenceVariationIndices(
  const SentenceVariationIndices& sentenceVariationIndices) {
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

}  // namespace

auto buildSentenceVariationIndices(const std::vector<SourceList>& xor_src_lists,
  const std::vector<IndexList>& compat_idx_lists,
  const std::vector<uint64_t>& compat_indices) -> SentenceVariationIndices {
  //
  auto sentenceVariationIndices = SentenceVariationIndices{};
  for (size_t i = 0; i < compat_indices.size(); ++i) {
    std::array<int, kNumSentences> variations =
      { -1, -1, -1, -1, -1, -1, -1, -1, -1 };

    auto combo_idx = compat_indices.at(i);
    for (int j{(int)compat_idx_lists.size() - 1}; j >= 0; --j) {
      const auto& idx_list = compat_idx_lists.at(j);
      auto src_idx = idx_list.at(combo_idx % idx_list.size());
      combo_idx /= idx_list.size();
      for (const auto& nc :
        xor_src_lists.at(j).at(src_idx).primaryNameSrcList) {
        using namespace Source;
        assert(isCandidate(nc.count));
        auto sentence = getSentence(nc.count) - 1;
        auto variation = getVariation(nc.count);
        // sanity check
        assert((variations.at(sentence) < 0)
               || (variations.at(sentence) == variation));
        variations.at(sentence) = variation;
      }
    }
    for (int s{}; s < kNumSentences; ++s) {
      auto& variationIndicesList = sentenceVariationIndices.at(s);
      const size_t variation_idx = variations.at(s) + 1;
      if (variationIndicesList.size() <= variation_idx) {
        variationIndicesList.resize(variation_idx + 1);
      }
      variationIndicesList.at(variation_idx).push_back(compat_indices.at(i));
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
  if (0) {
    dumpSentenceVariationIndices(sentenceVariationIndices);
  }
  return sentenceVariationIndices;
}

}  // namespace cm
