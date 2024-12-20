#ifndef INCLUDE_UTIL_H
#define INCLUDE_UTIL_H

#pragma once
#include <algorithm>
#include <array>
#include <chrono>
#include <format>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>
#include "cuda-types.h" // IndexList
#include "peco.h" // Peco::IndexList
#include "log.h"

namespace cm::util {

inline auto make_list_sizes(const std::vector<cm::IndexList>& idx_lists) {
  cm::IndexList sizes;
  sizes.reserve(idx_lists.size());
  for (const auto& idx_list : idx_lists) {
    sizes.push_back(index_t(idx_list.size()));
  }
  return sizes;
}

inline auto for_each_source_index(uint64_t flat_idx,
    const std::vector<IndexList>& idx_lists, const auto& func) {
  for (int list_idx{int(idx_lists.size()) - 1}; list_idx >= 0; --list_idx) {
    const auto& idx_list = idx_lists.at(list_idx);
    const auto src_idx = idx_list.at(flat_idx % idx_list.size());
    func(list_idx, src_idx);
    flat_idx /= idx_list.size();
  }
}

// TODO: 2nd return type template param, overflow check boolean function param
template <typename T>
requires requires(T t) { t.size(); }
inline auto sum_sizes(const std::vector<T>& v) {
  return std::accumulate(v.begin(), v.end(), 0u,
      [](size_t total, const T& t) { return total + t.size(); });
}

template <typename T>
requires std::is_integral_v<T>
inline std::string vec_to_string(const std::vector<T>& v) {
  std::string result{};
  for (T i : v) {
    result.append(std::to_string(i));
    result.append(" ");
  }
  return result;
}

template <typename T, typename R = T>
requires std::is_integral_v<T> && std::is_integral_v<R>
R sum(const std::vector<T>& vals) {
  return std::accumulate(vals.begin(), vals.end(), 0, [](R total, T val) {
    return total + val;
  });
}

// TOOD: "product"
template <typename T, typename R = uint64_t>
requires std::is_integral_v<T> && std::is_integral_v<R>
R multiply_with_overflow_check(const std::vector<T>& values) {
  // TODO: std::accumulate
  R total{1};
  for (auto v : values) {
    if (v > std::numeric_limits<R>::max() / total) {
      throw std::overflow_error("Multiplication overflow");
    }
    total *= v;
  }
  return total;
}

//
template <typename... T, template <typename...> class C>  // , typename R = uint64_t>
uint64_t multiply_sizes_with_overflow_check(const C<T...>& containers) {
  using R = uint64_t;
  R total{1};
  for (auto c : containers) {
    if (c.size() > std::numeric_limits<R>::max() / total) {
      throw std::overflow_error("Multiplication overflow");
    }
    total *= c.size();
  }
  return total;
}

#if 1 // TODO: eliminate this abomination
template <typename T>
typename std::vector<T>::const_iterator move_append(
    std::vector<T>& dst, std::vector<T>&& src) {
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
#endif

template <typename T, template <typename> class C>
  requires std::is_same_v<std::string, T>
inline std::string join(const C<T>& strings, const std::string& delim) {
  if (strings.empty()) {
    return {};
  }
  if (strings.size() == 1u) {
    return *strings.begin();
  }
  return std::accumulate(std::next(strings.begin()), strings.end(), *strings.begin(),
    [&delim](const std::string& acc, const std::string& elem) {
      return acc + delim + elem;
    });
}

template <typename T, template <typename> class C>
requires std::is_integral_v<T>
inline std::string join(const C<T>& nums, const std::string& delim) {
  if (nums.empty()) {
    return {};
  }
  if (nums.size() == 1u) {
    return std::to_string(nums.front());
  }
  return std::accumulate(std::next(nums.begin()), nums.end(),
    std::to_string(nums.front()), [&delim](const std::string& acc, int num) {
      return acc + delim + std::to_string(num);
    });
}

template <typename T, size_t N>
requires std::is_integral_v<T>
inline std::string join(
    const std::array<T, N>& nums, const std::string& delim) {
  if (nums.empty()) {
    return {};
  }
  if (nums.size() == 1u) {
    return std::to_string(nums.front());
  }
  return std::accumulate(std::next(nums.begin()), nums.end(),
    std::to_string(nums.front()), [&delim](const std::string& acc, int num) {
      return acc + delim + std::to_string(num);
    });
}

inline std::string append(const std::string& s1, const std::string& s2,
    const std::string& s3) {
  return s1 + s2 + s3;
}

class Timer {
public:
  static Timer start_timer() {
    Timer t;
    t.start();
    return t;
  }

  void start() {
    start_ = std::chrono::steady_clock::now();
  }

  template <typename TimeUnit = std::chrono::milliseconds>
  auto count() const {
    return std::chrono::duration_cast<TimeUnit>(stop_ - start_).count();
  }

  auto stop() {
    stop_ = std::chrono::steady_clock::now();
    return count();
  }

  template <typename TimeUnit = std::chrono::milliseconds>
  auto reset() {
    stop();
    auto result = count<TimeUnit>();
    start();
    return result;
  }

  auto microseconds() const {
    return count<std::chrono::microseconds>();
  }

  auto nanoseconds() const {
    return count<std::chrono::nanoseconds>();
  }

private:
  using time_point_t = decltype(std::chrono::steady_clock::now());

  time_point_t start_{};
  time_point_t stop_{};
}; // class Timer

template <typename TimeUnit> std::string_view time_unit_abbrev() {
  using namespace std::chrono;
  if constexpr (std::is_same_v<TimeUnit, milliseconds>)
    return "ms";
  else if constexpr (std::is_same_v<TimeUnit, microseconds>)
    return "µs";
  else if constexpr (std::is_same_v<TimeUnit, nanoseconds>)
    return "ns";
  return "units";
}

template <typename TimeUnit = std::chrono::milliseconds> class LogDuration {
public:
  LogDuration() = delete;
  LogDuration(std::string_view msg, LogLevel log_level = Normal)
      : msg_(msg), level_(log_level), logged_(false) {
    t_.start();
  }
  ~LogDuration() {
    log();
  }

  void log() {
    if (!logged_ && log_level(level_)) {
      t_.stop();
      // TODO: t.pretty() - get count<nanoseconds>() and auto determine best
      // unit and suffix
      std::cerr << msg_ << " - " << t_.count<TimeUnit>()
                << time_unit_abbrev<TimeUnit>() << std::endl;
    }
  }

private:
  Timer t_;
  std::string_view msg_;
  LogLevel level_;
  bool logged_;
};  // class LogDuration

inline auto pretty_bytes(size_t bytes) {
  const char* suffixes[7];
  suffixes[0] = "B";
  suffixes[1] = "KB";
  suffixes[2] = "MB";
  suffixes[3] = "GB";
  int s = 0;  // which suffix to use
  auto count = double(bytes);
  while (count >= 1024.0 && s < 4) {
    s++;
    count /= 1024.0;
  }
  if (count - floor(count) == 0.0) {
    return std::format("{} {}", int(count), suffixes[s]);
  } else {
    return std::format("{:.1f} {}", count, suffixes[s]);
  }
};

}  // namespace cm::util

#endif // INCLUDE_UTIL_H
