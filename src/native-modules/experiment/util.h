#ifndef INCLUDE_UTIL_H
#define INCLUDE_UTIL_H

#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include "peco.h"

namespace cm {

inline int list_size(const Peco::IndexList& indexList) {
  // TODO: use std::distance?
  int size = 0;
  std::for_each(indexList.cbegin(), indexList.cend(),
                [&size](int i){ ++size; });
  return size;
}

template <typename T> auto sum_sizes(const std::vector<T>& vecs) {
  size_t sum{};
  for (const auto& v : vecs) {
    sum += v.size();
  }
  return sum;
}

inline std::string vec_to_string(const std::vector<size_t>& v) {
  std::string result{};
  for (auto i : v) {
    result.append(std::to_string(i));
    result.append(" ");
  }
  return result;
}

template <typename T, typename R = uint64_t>
R multiply_with_overflow_check(const std::vector<T>& values) {
  R total{1};
  for (auto v : values) {
    if (v > std::numeric_limits<R>::max() / total) {
      throw std::overflow_error("Multiplication overflow");
    }
    total *= v;
  }
  return total;
}

}  // namespace cm

#endif // INCLUDE_UTIL_H
