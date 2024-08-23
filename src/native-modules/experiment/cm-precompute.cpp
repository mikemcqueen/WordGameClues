#include <algorithm>
#include <array>
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
#include "log.h"
#include "util.h"
#include "cm-hash.h"
#include "filter-types.h"

namespace cm {

namespace {

SourceData mergeSources(const SourceData& source1, const SourceData& source2) {
  auto primaryNameSrcList = NameCount::listMerge(
      source1.primaryNameSrcList, source2.primaryNameSrcList);
  // MAYBE TODO: ncList merge probably not necessary (??)
  auto ncList = NameCount::listMerge(source1.ncList, source2.ncList);
  auto usedSources = source1.usedSources.copyMerge(source2.usedSources);
  return {
      std::move(primaryNameSrcList), std::move(ncList), std::move(usedSources)};
}

auto mergeCompatibleSourceLists(
    const SourceList& src_list, const SourceCRefList& src_cref_list) {
  SourceList result{};
  for (const auto& src : src_list) {
    for (const auto& src_cref : src_cref_list) {
      if (src.isXorCompatibleWith(src_cref.get())) {
        result.push_back(std::move(mergeSources(src, src_cref.get())));
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
  const auto logging = false; // log_level(Ludicrous);
  // TODO: find smallest sourcelist to copy first, then skip merge in loop?

  // PROBLEM:
  // This src_cref_list is missing information about what the NAMEs were
  // that the src_list was generated from. We can't fully determine
  // compatibility without all of the names.

  SourceList src_list = clue_manager::make_src_list(ncList[0]);
  if (logging) {
    std::cerr << "nc[0]: " << ncList[0].toString() << " (" << src_list.size()
              << ")" << std::endl;
    SourceData::dumpList(src_list);
  }
  // TODO: std::next() or something.
  for (auto i = 1u; i < ncList.size(); ++i) {
    const auto src_cref_list = clue_manager::make_src_cref_list(ncList[i]);
    if (logging) {
      std::cerr << " nc[" << i << "]: " << ncList[i].toString() << " ("
                << src_cref_list.size() << ")" << std::endl;
      SourceData::dumpList(src_cref_list);
    }
    src_list = mergeCompatibleSourceLists(src_list, src_cref_list);
    // MAYBE BUG: this might be broken for > 2; should be something like:
    // if (sourceList.length !== ncIndex + 1)
    if (src_list.empty()) break;
  }
  return src_list;
}

void dumpSentenceVariationIndices(const SentenceVariationIndices& svi) {
  uint64_t total_indices{};
  for (int s{}; s < kNumSentences; ++s) {
    const auto& variation_idx_lists = svi.at(s);
    if (!variation_idx_lists.empty()) {
      const auto num_indices = util::sum_sizes(variation_idx_lists);
      if (log_level(ExtraVerbose)) {
        std::cerr << "S" << s + 1 << ": variations("
                  << variation_idx_lists.size() << "), indices(" << num_indices
                  << ")" << std::endl;
      }
      total_indices += num_indices;
      if (log_level(Ludicrous)) {
        size_t sum{};
        for (size_t i{}; i < variation_idx_lists.size(); ++i) {
          const auto& idx_list = variation_idx_lists.at(i);
          std::cerr << "  v" << int(i) - 1 << ": indices(" << idx_list.size() << ")"
                    << std::endl;
          sum += idx_list.size();
        }
        std::cerr << "  sum(" << sum << ")" << std::endl;
      }
    }
  }
  const auto MB = (total_indices * 8) / 1'000'000;
  std::cerr << "XOR variation indices: " << total_indices << " (" << MB << "MB)\n";
  if (MB > 2000) {
    std::cerr << "**** WARNING: variationIndices is getting big! ****\n";
  }
}

}  // namespace

auto build_src_lists(const std::vector<NCDataList>& nc_data_lists)
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
  int total_sources = 0;
  int hash_hits = 0;
  // all nc_data_lists are the same length, so just grab length of first
  const auto size = nc_data_lists[0].size();
  std::vector<HashMap> hashList(size);
  std::vector<SourceList> sourceLists(size);
  for (const auto& nc_data_list : nc_data_lists) {
    assert(nc_data_list.size() == size);
    for (size_t i{}; i < nc_data_list.size(); ++i) {
      // for size == 2: return by value; or reference to static local in a pinch
      auto sourceList = mergeAllCompatibleSources(nc_data_list[i].ncList);
      total_sources += sourceList.size();
      for (const auto& source : sourceList) {
        // TODO: NOT GOOD ENOUGH. still need a set of strings in value type.
        // HOWEVER, instead of listToSring'ing 75 million lists, how about we
        // mark "duplicate count, different name", i.e. "aliased" sources in
        // clue-manager, and only listToString and add those to a separate
        // map<bitset, set<String>> for ncLists with "shared" sourceBits.
        // probably don't need a separate map. just don't bother populating
        // and looking in the set unless its an aliased source.

        // i think the problem i'm referring to above is that we are removing
        // duplicate sources here without regards to what the clue names are
        // that were used to generate the source. I think this would actually
        // require a new field in NcData, "string clue_name". And then I'm not
        // sure what we'd do with it exactly, maybe only de-dupe if clue-names
        // are the same, and hope memory and src_list size don't explode?
        //
        // this is a problem for both show-components, and combo-maker (as well
        // as validator, but it doesn't use this function). in both former
        // cases, compatibility between two sources (say, an arbitrary source
        // and an --xor source, or two -t sources) can only be determined when
        // we know that they don't share any clue names.
        // e.g. if polar = white,bear then polar,bear is not a compatible pair.
        //
        // NOTE that if the above is true, I will probably have to add something
        // to SourceCompatibilityData so name compatibility can be determined.
        // * I can imagine a "unique clue name index" that gets incremented every
        // time a new clue name is introduced. That doesn't solve the synonym/
        // homonym problem however. Two numbers maybe? One for the actual clue
        // name, one for the "original" name from which it is derived? The chain
        // might be longer than two though? Synonym of a homonym?
        //
        // If I had stuff set up properly, "tooth" would be a synonym of "cog"
        // which would in turn be a prefix of "cogburn".
        //
        // I'm not sure what the "listToString" stuff was referring to above,
        // but the idea of having an aliased-sources map *might* work.
        const auto& key = source;
        if (hashList[i].find(key) != hashList[i].end()) {
          hash_hits++;
          continue;
        }
        hashList[i][key] = StringSet{}; // TODO: emplace? std::move?
        sourceLists[i].push_back(std::move(source));
      }
    }
  }
  if (log_level(Verbose)) {
    // std::cerr << " hash: " << hash_called //
    //           << ", equal_to: " << equal_to_called << std::endl;
    std::cerr << " total sources: " << total_sources
              << ", hash_hits: " << hash_hits             //
              << ", sourceLists: " << sourceLists.size()  //
              << ", sources: " << util::sum_sizes(sourceLists) << std::endl;
  }
  return sourceLists;
}

auto buildSentenceVariationIndices(const std::vector<SourceList>& xor_src_lists,
    const std::vector<IndexList>& compat_idx_lists,
    const FatIndexList& compat_indices)
    -> SentenceVariationIndices {
  SentenceVariationIndices svi;
  for (size_t idx{}; idx < compat_indices.size(); ++idx) {
    // = make_array<variation_index_t, kNumSentences>(-1);
    UsedSources::Variations variations = {-1, -1, -1, -1, -1, -1, -1, -1, -1};
    util::for_each_source_index(compat_indices.at(idx), compat_idx_lists,
        [&xor_src_lists, &variations](index_t list_idx, index_t src_idx) {
          const auto& src = xor_src_lists.at(list_idx).at(src_idx);
          for (const auto& nc : src.primaryNameSrcList) {
            assert(Source::isCandidate(nc.count));
            auto b = UsedSources::merge_one_variation(variations, nc.count);
            assert(b);
          }
        });
    for (int s{}; s < kNumSentences; ++s) {
      auto& variation_idx_lists = svi.at(s);
      const auto variation_idx = variations.at(s) + 1u;
      if (variation_idx >= variation_idx_lists.size()) {
        variation_idx_lists.resize(variation_idx + 1u);
      }
      variation_idx_lists.at(variation_idx).push_back(idx);
    }
  }
  // When the list of xor_sources is very small, there may be no xor_source
  // that contain a primary source from one or more sentences. Destroy the
  // variation_idx_listss for those sentences with no variations, since they
  // only contain a single element (index 0) representing the "-1" or "no"
  // variation, that contains all indices. It's redundant and unnecessary data.
  // TODO: for (auto& variation_idx_lists : svi) {
  std::for_each(svi.begin(),
    svi.end(), [](auto& variation_idx_lists) {
      if (variation_idx_lists.size() == 1) {
        variation_idx_lists.clear();
      }
    });
  dumpSentenceVariationIndices(svi);
  return svi;
}

// get list of indices into variations_list sorted by variation index for
// supplied sentence. supplied sentence is in range [0,kNumSentences).
auto get_indices_sorted_by_variation(
    const UsedSources::VariationsList& variations_list, int s) {
  IndexList indices(variations_list.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(), [&](index_t a, index_t b) {
    return variations_list.at(a).at(s) < variations_list.at(b).at(s);
  });
  return indices;
}

// supplied sentence is in range [0,kNumSentences).
auto get_variation_index_offsets(const IndexList& idx_list,
    const UsedSources::VariationsList& variations_list, int sentence) {
  std::vector<VariationIndexOffset> vi_offsets;
  variation_index_t last_vi{-2};  // intentional impossible value
  for (index_t idx{}; idx < idx_list.size(); ++idx) {
    auto vi = variations_list.at(idx_list.at(idx)).at(sentence);
    if (vi != last_vi) vi_offsets.emplace_back(vi, idx);
  }
  return vi_offsets;
}

  /*
auto build_OR_variation_indices(
    const UsedSources::VariationsList& variations_list,
    const FatIndexList& compat_indices)
    -> SentenceOrVariationIndices {
  SentenceOrVariationIndices all_or_vi;
  for (int s{}; s < kNumSentences; ++s) {
    auto& or_vi = all_or_vi.at(s);
    or_vi.indices = std::move(get_indices_sorted_by_variation(variations_list, s));
    or_vi.index_offsets =
        std::move(get_variation_index_offsets(or_vi.indices, variations_list, s));
  }
  return all_or_vi;
}
  */

}  // namespace cm
