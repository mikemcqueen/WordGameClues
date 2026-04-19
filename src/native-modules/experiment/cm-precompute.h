#pragma once
#include <vector>
#include "source-core.h"

namespace cm {

// functions

auto build_src_lists(
    const std::vector<NCDataList>& nc_data_lists)
    -> std::vector<DeferredSourceDataList>;

}  // namespace cm
