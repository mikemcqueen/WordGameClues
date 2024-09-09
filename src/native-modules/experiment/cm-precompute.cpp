#include <cassert>
#include <functional>
#include <iostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "clue-manager.h"
#include "cm-precompute.h"
#include "log.h"
#include "util.h"
#include "cm-hash.h"

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
  size_t total_sources{};
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

}  // namespace cm
