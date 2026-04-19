#pragma once
#include <vector>
#include "merge-filter-data.h"
#include "merge-type.h"
#include "source-core.h"

namespace cm {

auto get_merge_data(const std::vector<SourceComboList>& combo_lists,
    MergeData::Host& host, MergeData::Device& device, MergeType merge_type,
    cudaStream_t stream, bool merge_only = false) -> bool;

auto merge_xor_compatible_src_lists(
    const std::vector<SourceComboList>& combo_lists) -> SourceList;

auto merge_xor_compatible_src_lists_minimal(
    const std::vector<SourceComboList>& combo_lists) -> SourceList;

}  // namespace cm
