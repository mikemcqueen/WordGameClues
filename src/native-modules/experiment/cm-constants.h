#pragma once

namespace cm {

inline constexpr auto kMaxSources = 32u;
inline constexpr auto kMaxSourcesPerSentence = 32;
inline constexpr auto kNumSentences = 9;

static_assert(std::has_single_bit(kMaxSources),
    "kMaxSources must be a power of two");

}  // namespace cm
