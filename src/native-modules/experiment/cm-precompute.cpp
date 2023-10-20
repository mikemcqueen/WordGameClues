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
#include "clue-manager.h"
#include "cm-precompute.h"
#include "merge.h"

namespace cm {

namespace {

SourceData mergeSources(const SourceData& source1, const SourceData& source2) {
  auto primaryNameSrcList = NameCount::listMerge(
    source1.primaryNameSrcList, source2.primaryNameSrcList);
  // ncList merge probably not necessary
  auto ncList = NameCount::listMerge(source1.ncList, source2.ncList);
  auto usedSources = source1.usedSources.copyMerge(source2.usedSources);
  return {
    std::move(primaryNameSrcList), std::move(ncList), std::move(usedSources)};
}

auto mergeCompatibleSourceLists(
  const SourceList& src_list, const SourceCRefList& src_cref_list) {
  //
  SourceList result{};
  for (const auto& src : src_list) {
    for (const auto& src_cref : src_cref_list) {
      if (src.isXorCompatibleWith(src_cref.get())) {
        result.emplace_back(mergeSources(src, src_cref.get()));
      }
    }
  }
  return result;
}

// NOTE: for ncList.size() <= 2
//
auto mergeAllCompatibleSources(const NameCountList& ncList) -> SourceList {
  // because **maybe** broken for > 2 below
  assert(ncList.size() <= 2 && "ncList.length > 2");
  constexpr auto log = false;
  if constexpr (log) {
    std::cerr << "nc[0]: " << ncList[0].toString() << std::endl;
  }
  // TODO: find smallest sourcelist to copy first, then skip merge in loop?
  SourceList src_list{clue_manager::make_src_list_for_nc(ncList[0])};  // copy
  for (auto i = 1u; i < ncList.size(); ++i) {
    if constexpr (log) {
      std::cerr << " nc[" << i << "]: " << ncList[i].toString() << std::endl;
    }
    const auto& src_cref_list{
      clue_manager::make_src_cref_list_for_nc(ncList[i])};
    src_list = std::move(mergeCompatibleSourceLists(src_list, src_cref_list));
    // TODO BUG this is broken for > 2; should be something like:
    // if (sourceList.length !== ncIndex + 1) 
    if (src_list.empty()) break;
  }
  return src_list;
}

bool every_combo_idx(combo_index_t combo_idx,
  const std::vector<IndexList>& idx_lists, const auto& fn) {
  //
  for (int i{(int)idx_lists.size() - 1}; i >= 0; --i) {
    const auto& idx_list = idx_lists.at(i);
    auto src_idx = idx_list.at(combo_idx % idx_list.size());
    if (!fn(i, src_idx))
      return false;
    combo_idx /= idx_list.size();
  }
  return true;
}

}  // namespace

auto buildSourceListsForUseNcData(const std::vector<NCDataList>& useNcDataLists)
  -> std::vector<SourceList> {
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
      // for size == 2: return by value; or reference to static local in a pinch
      auto sourceList = mergeAllCompatibleSources(ncDataList[i].ncList);
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

auto buildOrArg(SourceList& src_list) {
  OrArgData or_arg;
  for (auto&& src : src_list) {
    or_arg.or_src_list.emplace_back(OrSourceData{ std::move(src) });
  }
  return or_arg;
};

auto count_or_sources(const OrArgList& or_arg_list) {
  uint32_t total{};
  for (const auto& or_arg: or_arg_list) {
    total += or_arg.or_src_list.size();
  }
  return total;
}

auto buildOrArgList(std::vector<SourceList>&& or_src_list) -> OrArgList {
  using namespace std::chrono;
  
  OrArgList or_arg_list;
  auto t0 = high_resolution_clock::now();
  for (auto& src_list : or_src_list) {
    or_arg_list.emplace_back(buildOrArg(src_list));
  }
  auto t1 = high_resolution_clock::now();
  [[maybe_unused]] auto t_dur = duration_cast<milliseconds>(t1 - t0).count();
  std::cerr << "  buildOrArgList args(" << or_arg_list.size() << ")"
            << ", sources(" << count_or_sources(or_arg_list) << ") - "
            << t_dur << "ms" << std::endl;

  return or_arg_list;
}

bool isXorCompatibleWithAnySource(const OrSourceData& or_src,
  const std::vector<SourceList>& xor_src_lists,
  const std::vector<IndexList>& compat_idx_lists,
  const ComboIndexList& compat_indices) {
  //
  for (size_t i{}; i < compat_indices.size(); ++i) {
    if (every_combo_idx(compat_indices.at(i), compat_idx_lists,
          [&src = or_src.src, &xor_src_lists](
            index_t list_idx, index_t src_idx) {
            return src.isXorCompatibleWith(
              xor_src_lists.at(list_idx).at(src_idx));
          })) {
      return true;
    }
  }
  return false;
}

void markAllXorCompatibleOrSources(OrArgList& or_arg_list,
  const std::vector<SourceList>& xor_src_lists,
  const std::vector<IndexList>& compat_idx_lists,
  const ComboIndexList& compat_indices) {
  // whee nesting
  uint32_t num_compat{};
  using namespace std::chrono;
  auto t0 = high_resolution_clock::now();
  for (auto& or_arg : or_arg_list) {
    for (auto& or_src : or_arg.or_src_list) {
      if (isXorCompatibleWithAnySource(
            or_src, xor_src_lists, compat_idx_lists, compat_indices)) {
        or_src.xor_compat = true;
        ++num_compat;
      }
    }
  }
  auto t1 = high_resolution_clock::now();
  [[maybe_unused]] auto t_dur = duration_cast<milliseconds>(t1 - t0).count();
  std::cerr << "  markAllXorCompatibleOrSources(" << num_compat << ") - "
            << t_dur << "ms" << std::endl;
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
  const ComboIndexList& compat_indices) -> SentenceVariationIndices {
  //
  auto sentenceVariationIndices = SentenceVariationIndices{};
  for (size_t i = 0; i < compat_indices.size(); ++i) {
    std::array<int, kNumSentences> variations = {
      -1, -1, -1, -1, -1, -1, -1, -1, -1};
    every_combo_idx(compat_indices.at(i), compat_idx_lists,
      [&xor_src_lists, &variations](index_t list_idx, index_t src_idx) {
        for (const auto& nc :
          xor_src_lists.at(list_idx).at(src_idx).primaryNameSrcList) {
          using namespace Source;
          assert(isCandidate(nc.count));
          auto sentence = getSentence(nc.count) - 1;
          auto variation = getVariation(nc.count);
          // sanity check
          assert((variations.at(sentence) < 0)
                 || (variations.at(sentence) == variation));
          variations.at(sentence) = variation;
        }
        return true;
      });
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
  // for (auto& variationIndicesList : sentenceVariationIndices) {
  std::for_each(sentenceVariationIndices.begin(),
    sentenceVariationIndices.end(), [](auto& variationIndicesList) {
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
