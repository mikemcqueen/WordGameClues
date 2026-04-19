#pragma once

#include <cstdint>
#include <vector>

namespace cm {

using index_t = uint32_t;
using fat_index_t = uint64_t;

template <typename T> using IndexListBase = std::vector<T>;

using IndexList = IndexListBase<index_t>;
using FatIndexList = IndexListBase<fat_index_t>;

}  // namespace cm
