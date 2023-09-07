#ifndef INCLUDE_CUDA_TYPES_H
#define INCLUDE_CUDA_TYPES_H

#include <span>
#include <utility>
#include <vector>

namespace cm {

using result_t = uint8_t;

using index_t = uint32_t;
using IndexList = std::vector<index_t>;

using IndexSpan = std::span<const index_t>;
using IndexSpanPair = std::pair<IndexSpan, IndexSpan>;

}  // namespace cm

#endif //  INCLUDE_CUDA_TYPES_H
