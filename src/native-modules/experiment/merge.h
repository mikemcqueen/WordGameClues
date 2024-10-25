#ifndef INCLUDE_MERGE_H
#define INCLUDE_MERGE_H

#pragma once
#include <vector>
#include "combo-maker.h"
#include "merge-filter-data.h"
#include "merge-type.h"

namespace cm {

auto get_merge_data(const std::vector<SourceList>& src_lists,
    MergeData::Host& host, MergeData::Device& device, MergeType merge_type,
    cudaStream_t stream, bool merge_only = false) -> bool;

auto merge_xor_compatible_src_lists(
    const std::vector<SourceList>& src_lists) -> SourceList;

}  // namespace cm

#endif  // INCLUDE_MERGE_H
