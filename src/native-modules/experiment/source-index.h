#pragma once
#include <vector>
#include "cuda-types.h"

namespace cm {

// should be called ResultSourceIndices
// or ResultFlatIdx or ListFlatIdx or something....
struct SourceIndex {
  constexpr bool operator<(const SourceIndex& rhs) const {
    return (listIndex < rhs.listIndex)
           || ((listIndex == rhs.listIndex) && (index < rhs.index));
  }

  constexpr const char* as_string(char* buf) const {
    sprintf(buf, "%d:%d", listIndex, index);
    return buf;
  }

  index_t listIndex{};  // result_idx
  index_t index{};      // src_idx
};

using SourceIndexList = std::vector<SourceIndex>;

}  // namespace cm
