#ifndef INCLUDE_UTIL_H
#define INCLUDE_UTIL_H

#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include "peco.h"

namespace util {

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

template <typename T>
typename std::vector<T>::const_iterator move_append(
  std::vector<T>& dst, std::vector<T>&& src) {
  //
  typename std::vector<T>::const_iterator result;
  if (dst.empty()) {
    dst = std::move(src);
    result = std::cbegin(dst);
  } else {
    result = dst.insert(std::end(dst), std::make_move_iterator(std::begin(src)),
      std::make_move_iterator(std::end(src)));
  }
  src.clear();
  //src.shrink_to_fit();
  return result;
}

inline std::string join(
  const std::vector<std::string>& strings, const std::string& delim) {
  if (strings.empty()) {
    return {};
  }
  if (strings.size() == 1u) {
    return strings.front();
  }
  return std::accumulate(std::next(strings.begin()), strings.end(), strings.front(),
    [delim](const std::string& acc, const std::string& elem) {
      return acc + delim + elem;
    });
}

}  // namespace util

#endif // INCLUDE_UTIL_H
