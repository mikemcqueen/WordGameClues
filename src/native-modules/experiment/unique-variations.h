#pragma once
#include "merge-filter-data.h"

namespace cm {

void build_unique_variations(
    FilterData::HostCommon& host, std::string_view name);

}
