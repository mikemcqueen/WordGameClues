#ifndef INCLUDE_UTIL_H
#define INCLUDE_UTIL_H

#include <chrono>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include "cuda-types.h" // IndexList
#include "peco.h" // Peco::IndexList

namespace util {

inline auto make_list_sizes(const std::vector<cm::IndexList>& idx_lists) {
  cm::IndexList sizes;
  sizes.reserve(idx_lists.size());
  for (const auto& idx_list : idx_lists) {
    sizes.push_back(idx_list.size());
  }
  return sizes;
}

template <typename T> inline auto sum_sizes(const std::vector<T>& vecs) {
  size_t sum{};
  for (const auto& v : vecs) {
    sum += v.size();
  }
  return sum;
}

template <typename T>
// TODO: is_integral_type<T>
inline std::string vec_to_string(const std::vector<T>& v) {
  std::string result{};
  for (T i : v) {
    result.append(std::to_string(i));
    result.append(" ");
  }
  return result;
}

template <typename T, typename R = T>
// TODO: is_integral_type<T>
R sum(const std::vector<T>& vals) {
  return std::accumulate(vals.begin(), vals.end(), 0, [](R total, T val) {
    total += val;
    return total;
  });
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
      // well i think this is probably more expensive than multiple += followed
      // by a return. would like to profile.
      return acc + delim + elem;
    });
}

inline std::string join(
  const std::vector<int>& nums, const std::string& delim) {
  if (nums.empty()) {
    return {};
  }
  if (nums.size() == 1u) {
    return std::to_string(nums.front());
  }
  return std::accumulate(std::next(nums.begin()), nums.end(),
    std::to_string(nums.front()), [delim](const std::string& acc, int num) {
      // well i think this is probably more expensive than multiple += followed
      // by a return. would like to profile.
      return acc + delim + std::to_string(num);
    });
}

class Timer {
public:
  void start() {
    start_ = std::chrono::high_resolution_clock::now();
  }

  void stop() {
    stop_ = std::chrono::high_resolution_clock::now();
  }

  template <typename TimeUnit = std::chrono::milliseconds>
  auto count() {
    return std::chrono::duration_cast<TimeUnit>(stop_ - start_).count();
  }

  static Timer start_timer() {
    Timer t;
    t.start();
    return t;
  }

private:
  using time_point_t = decltype(std::chrono::high_resolution_clock::now());

  time_point_t start_{};
  time_point_t stop_{};
};

}  // namespace util

#endif // INCLUDE_UTIL_H
