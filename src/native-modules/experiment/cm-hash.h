#pragma once
#include <functional>
#include "combo-maker.h"
#include "cuda-types.h"

namespace cm {

template <typename SizeT>
inline void hash_combine(SizeT& seed, SizeT value) {
  seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

inline auto hash_called = 0;
inline auto equal_to_called = 0;

}  // namespace cm

//template struct std::hash<cm::UsedSources::SourceBits>;

namespace std {

// not technically hash related ... but needed for sorting.
template <> struct less<cm::Variations> {
  bool operator()(const cm::Variations& lhs,
      const cm::Variations& rhs) const noexcept {
    for (size_t i{}; i < lhs.size(); ++i) {
      if (lhs[i] < rhs[i]) return true;
    }
    return false;
  }
};

template <> struct equal_to<cm::Variations> {
  bool operator()(const cm::Variations& lhs,
      const cm::Variations& rhs) const noexcept {
    for (size_t i{}; i < lhs.size(); ++i) {
      if (lhs[i] != rhs[i]) return false;
    }
    return true;
  }
};

template <> struct hash<cm::Variations> {
  size_t operator()(const cm::Variations& variations) const noexcept {
    size_t seed = 0;
    for (const auto v: variations) {
      cm::hash_combine(seed, hash<cm::variation_index_t>()(v));
    }
    return seed;
  }
};

template <> struct equal_to<cm::UsedSources> {
  bool operator()(const cm::UsedSources& lhs,
      const cm::UsedSources& rhs) const noexcept {
    if (lhs.getBits() != rhs.getBits()) return false;
    if (lhs.variations != rhs.variations) return false;
    return true;
  }
};

template <> struct hash<cm::UsedSources> {
  size_t operator()(const cm::UsedSources& us) const noexcept {
    size_t bits_seed = 0;
    cm::hash_combine(bits_seed, us.getBits().hash());
    size_t seed = 0;
    cm::hash_combine(seed, bits_seed);
    cm::hash_combine(seed, hash<cm::Variations>()(us.variations));
    return seed;
  }
};

template <> struct equal_to<cm::SourceCompatibilityData> {
  bool operator()(const cm::SourceCompatibilityData& lhs,
    const cm::SourceCompatibilityData& rhs) const noexcept
  {
    return equal_to<cm::UsedSources>{}(lhs.usedSources, rhs.usedSources);
  }
};

template <> struct hash<cm::SourceCompatibilityData> {
  size_t operator()(const cm::SourceCompatibilityData& src) const noexcept {
#if 0
    size_t seed = 0;
    cm::hash_combine(seed, hash<cm::UsedSources>()(src.usedSources));
    return seed;
#else
    return hash<cm::UsedSources>{}(src.usedSources);
#endif
  }
};

template <> struct equal_to<cm::SourceCompatibilityDataCRef> {
  bool operator()(const cm::SourceCompatibilityDataCRef lhs_cref,
    const cm::SourceCompatibilityDataCRef rhs_cref) const noexcept
  {
    return equal_to<cm::SourceCompatibilityData>{}(lhs_cref, rhs_cref);
  }
};

template <> struct hash<cm::SourceCompatibilityDataCRef> {
  size_t operator()(const cm::SourceCompatibilityDataCRef src_cref) const noexcept {
#if 0
    size_t seed = 0;
    cm::hash_combine(seed, hash<cm::SourceCompatibilityData>()(src_cref);
    return seed;
#else
    return hash<cm::SourceCompatibilityData>{}(src_cref);
#endif
  }
};

template <> struct equal_to<cm::CompatSourceIndices> {
  bool operator()(const cm::CompatSourceIndices lhs,
      const cm::CompatSourceIndices rhs) const noexcept {
    return lhs.data() == rhs.data();
  }
};

template <> struct hash<cm::CompatSourceIndices> {
  size_t operator()(const cm::CompatSourceIndices csi) const noexcept {
    return hash<uint64_t>()(csi.data());
  }
};

} // namespace std

