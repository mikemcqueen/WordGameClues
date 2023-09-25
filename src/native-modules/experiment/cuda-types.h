#ifndef INCLUDE_CUDA_TYPES_H
#define INCLUDE_CUDA_TYPES_H

#include <span>
#include <cstdint>
#include <utility>
#include <vector>

namespace cm {

using result_t = uint8_t;
using index_t = uint32_t;
using combo_index_t = uint64_t;

using IndexList = std::vector<index_t>;
using ComboIndexList = std::vector<combo_index_t>;

using ComboIndexSpan = std::span<const combo_index_t>;
using ComboIndexSpanPair = std::pair<ComboIndexSpan, ComboIndexSpan>;

}  // namespace cm

#endif //  INCLUDE_CUDA_TYPES_H
