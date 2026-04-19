#pragma once

#include "variations.h"

namespace cm {

struct PrimarySourceId {
  int value{};

  constexpr PrimarySourceId() = default;
  constexpr PrimarySourceId(int value) : value(value) {}

  constexpr bool is_candidate() const noexcept { return value >= 1'000'000; }
  constexpr int sentence() const noexcept { return value / 1'000'000; }
  constexpr variation_index_t variation() const noexcept {
    return static_cast<variation_index_t>(sentence_masked() / 100);
  }
  constexpr int index() const noexcept { return sentence_masked() % 100; }

private:
  constexpr int sentence_masked() const noexcept { return value % 1'000'000; }
};

}  // namespace cm
