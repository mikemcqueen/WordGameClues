#ifndef INCLUDE_VALIDATOR_H
#define INCLUDE_VALIDATOR_H

#include <optional>
#include "combo-maker.h"
#include "cuda-types.h"
#include "peco.h"

namespace validator {

// ugly. make types exist outside of NS, or in "common" or "native" or
// something
using namespace cm;

// types

struct NcResultData {
  SourceList src_list;
  std::unordered_set<SourceCompatibilityData> src_compat_set;
};
using NcResultMap = std::unordered_map<std::string, NcResultData>;

// functions

auto getNumNcResults(const NameCount& nc) -> int;

void appendNcResults(const NameCount& nc, SourceList& src_list);

auto mergeNcListCombo(const NameCountList& nc_list,
  const IndexList& idx_list) -> std::optional<SourceData>;

auto mergeAllNcListCombinations(const NameCountList& nc_list,
  Peco::IndexListVector&& idx_lists) -> SourceList;

auto mergeNcListResults(const NameCountList& nc_list) -> SourceList;

};  // namespace validator

#endif
