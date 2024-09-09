#pragma once
#include <vector>
#include "combo-maker.h"

namespace cm {

// functions

auto build_src_lists(
    const std::vector<NCDataList>& nc_data_lists) -> std::vector<SourceList>;

}  // namespace cm
