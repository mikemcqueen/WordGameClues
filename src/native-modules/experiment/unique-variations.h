#pragma once
#include <string_view>
#include <vector>
#include "candidates.h"
#include "merge-filter-data.h"
//#include "source-index.h"
//#include "variations.h"

namespace cm {

void build_unique_variations(
    FilterData::HostCommon& host, std::string_view name);

auto make_variations_sorted_idx_lists(
    const CandidateList& candidates) -> std::vector<IndexList>;

auto make_compat_src_indices(const CandidateList& candidates,
    const std::vector<IndexList>& idx_lists) -> CompatSourceIndicesList;

/*
auto make_src_compat_list(const CandidateList& candidates,
    const std::vector<IndexList>& idx_lists) -> SourceCompatibilityList;

auto make_unique_variations(
    const SourceCompatibilityList& src_compat_list)
    -> std::vector<UniqueVariations>;
*/

}  // namespace cm
