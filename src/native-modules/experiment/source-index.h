#pragma once
#include "cuda-types.h"

namespace cm {

struct SourceIndex {
  constexpr bool operator<(const SourceIndex& rhs) const {
    return (listIndex < rhs.listIndex)
           || ((listIndex == rhs.listIndex) && (index < rhs.index));
  }

  constexpr const char* as_string(char* buf) const {
    sprintf(buf, "%d:%d", listIndex, index);
    return buf;
  }

  index_t listIndex{};
  index_t index{};
};

}  // namespace cm
