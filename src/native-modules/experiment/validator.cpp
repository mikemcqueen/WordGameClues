// validator.cpp

#include <vector>
#include "clue-manager.h"
#include "peco.h"
#include "validator.h"

using namespace cm;

namespace validator {

namespace {

std::vector<NcResultMap> ncResultMaps;

NcResultData& getNcResult(const NameCount& nc) {
  if ((int)ncResultMaps.size() <= nc.count) {
    ncResultMaps.resize(nc.count + 1);
  }
  auto& map = ncResultMaps.at(nc.count);
  // TODO: specialize std::hash for NameCount?
  const auto nc_str = nc.toString();
  auto it = map.find(nc_str);
  if (it == map.end()) {
    map.insert(std::make_pair(nc_str, NcResultData{}));
    it = map.find(nc_str);
  }
  return it->second;
}


/*
mergeNcListResults(ncListToMerge);
let listArray: number[][] = ncListToMerge.map(nc => {
    if (nc.count === 1) {
        // TODO: optimization: these could be cached. i'm not sure it'd
        // matter too much.
        return getAllSourcesForPrimaryClueName(nc.name);
    } else {
        ++native_get_num_ncr;
        //const count =
ClueManager.getNcResultMap(nc.count)[nc.toString()].list.length; const count =
Native.getNumNcResults(nc); return [...Array(count).keys()].map(_.toNumber);
    }
});
*/

auto buildNcSourceIndexLists(const NameCountList& nc_list) {
  Peco::IndexListVector idx_lists;
  for (const auto& nc : nc_list) {
    if (nc.count == 1) {
      idx_lists.emplace_back(Peco::make_index_list(
        clue_manager::getSourcesForPrimaryClueName(nc.name)));
    } else {
      idx_lists.emplace_back(Peco::make_index_list(getNumNcResults(nc)));
    }
  }
  return idx_lists;
}

}  // namespace

// TODO: move to clue-manager
auto getNumNcResults(const NameCount& nc) -> int {
  return getNcResult(nc).src_list.size();
}

// TODO: move to clue-manager
void appendNcResults(const NameCount& nc, SourceList& src_list) {
  auto& nc_result = getNcResult(nc);
  for (auto& src: src_list) {
    if (nc_result.src_compat_set.find(src) == nc_result.src_compat_set.end()) {
      // add to set *before* moving to list
      nc_result.src_compat_set.insert(src);
      nc_result.src_list.emplace_back(std::move(src));
    }
  }
}

auto mergeNcListCombo(const NameCountList& nc_list, const IndexList& idx_list)
  -> std::optional<SourceData> {
  //
  SourceData src;
  for (size_t i{}; i < idx_list.size(); ++i) {
    const auto& nc = nc_list.at(i);
    if (nc.count > 1) {
      const auto& nc_result_src = getNcResult(nc).src_list.at(idx_list.at(i));
      if (!src.addCompoundSource(nc_result_src)) {
        return std::nullopt;
      }
    } else if (!src.addPrimaryNameSrc(nc, idx_list.at(i))) {
      return std::nullopt;
    }
  }
  // TODO: return {src} ?
  return std::make_optional(src);
}

auto mergeAllNcListCombinations(const NameCountList& nc_list,
  Peco::IndexListVector&& idx_lists) -> SourceList {
  //
  SourceList src_list;
  Peco peco(std::move(idx_lists));
  for (auto idx_list = peco.first_combination(); idx_list;
       idx_list = peco.next_combination()) {
    auto opt_src = mergeNcListCombo(nc_list, *idx_list);
    if (opt_src.has_value()) {
      src_list.emplace_back(std::move(opt_src.value()));
    }
  }
  return src_list;
}

auto mergeNcListResults(const NameCountList& nc_list) -> SourceList {
  auto idx_lists = buildNcSourceIndexLists(nc_list);
  return mergeAllNcListCombinations(nc_list, std::move(idx_lists));
}

}  // namespace validator

