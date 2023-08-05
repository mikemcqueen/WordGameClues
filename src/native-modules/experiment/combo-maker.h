#ifndef INCLUDE_COMBO_MAKER_H
#define INCLUDE_COMBO_MAKER_H

#include <algorithm>
#include <array>
#include <cassert>
#include <cstring>
#include <iostream>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#define USE_MMEBITSET 1

#if USE_MMEBITSET
#include "mmebitset.h"
using mme::bitset;
#else
#include <bitset>
using std::bitset;
#endif

namespace cm {

constexpr auto kMaxLegacySources = 111; // bits
constexpr auto kMaxSourcesPerSentence = 32; // bits - old: 128
constexpr auto kNumSentences = 9;
constexpr auto kMaxUsedSourcesPerSentence = 32;
constexpr auto kMaxUsedSources = kMaxUsedSourcesPerSentence * kNumSentences;

template<typename T, size_t N>
constexpr auto make_array(T value) -> std::array<T, N> {
  std::array<T, N> a{};
  for (auto& e : a) {
    e = value;
  }
  return a;
}

namespace Source {
  constexpr inline auto isCandidate(int src) noexcept { return src >= 1'000'000; }
  constexpr inline auto isLegacy(int src) noexcept { return !isCandidate(src); }
  constexpr inline auto getSentence(int src) noexcept { return src / 1'000'000; }
  constexpr inline auto getSource(int src) noexcept { return src % 1'000'000; }
  constexpr inline auto getVariation(int src) noexcept { return getSource(src) / 100; }
  constexpr inline auto getIndex(int src) noexcept { return getSource(src) % 100; }
#if 0 // unused
  constexpr inline auto getFirstIndex(int sentence) {
    assert(sentence > 0);
    return (sentence - 1) * kMaxUsedSourcesPerSentence;
  }
#endif
} // namespace Source

using LegacySourceBits = bitset<kMaxLegacySources>;
#if 0
using LegacySources = std::array<int8_t, kMaxLegacySources>;
// 32 bytes per sentence * 9 sentences = 2304? bits, 288 bytes, 36? 64-bit words
using Sources = std::array<int8_t, kMaxUsedSources>;
#endif

struct UsedSources {
  using VariationIndex_t = int16_t;
  // 32 bits per sentence * 9 sentences = 1152 bits, 144 bytes, 18 64-bit words
  using SourceBits = bitset<kMaxSourcesPerSentence * kNumSentences>;
  using Variations = std::array<VariationIndex_t, kNumSentences>;

  constexpr static auto getFirstBitIndex(int sentence) {
    assert(sentence > 0);
    return (sentence - 1) * kMaxSourcesPerSentence;
  }

  constexpr SourceBits& getBits() noexcept { return bits; }
  constexpr const SourceBits& getBits() const noexcept { return bits; }

  constexpr int getVariation(int sentence) const { return variations.at(sentence - 1); }
  void setVariation(int sentence, int value) { variations.at(sentence - 1) = value; }
  constexpr bool hasVariation(int sentence) const { return getVariation(sentence) > -1; }

  /*
  auto andBits(const SourceBits& other) const noexcept {
#if 1
    static SourceBits result{};
    result.reset();
    result |= getBits(); // possibly use static here
#else
    SourceBits result(getBits()); // possibly use static here
#endif
    result &= other;
    return result;
  }
  */

private:

  void addVariations(const UsedSources& other) {
    for (int sentence{ 1 }; sentence <= kNumSentences; ++sentence) {
      if (!other.hasVariation(sentence)) continue;
      // ensure variations for this sentence are compatible
      if (hasVariation(sentence)) {
        assert(getVariation(sentence) == other.getVariation(sentence));
      } else {
        setVariation(sentence, other.getVariation(sentence));
      }
    }
  }

public:
  constexpr static auto allVariationsMatch(
    const Variations& v1, const Variations& v2) {
    for (size_t i{}; i < v1.size(); ++i) {
      if ((v1[i] > -1) && (v2[i] > -1) && (v1[i] != v2[i])) {
        return false;
      }
    }
    return true;
  }

  constexpr static bool allVariationsMatch2(
    const Variations& v1, const Variations& v2) {
    int mismatches{};
    for (auto i{ 0u }; i < v1.size(); ++i) {
      auto first = v1[i] + 1;
      auto second = v2[i] + 1;
      mismatches += first && second && (first != second);
    }
    return !mismatches;
  }

  constexpr auto isXorCompatibleWith(const UsedSources& other) const {
    // compare bits
    if (getBits().intersects(other.getBits()))
      return false;
    // compare variations
    if (!allVariationsMatch(variations, other.variations))
      return false;
    return true;
  }

  // there is a potential optimization here, in the case where we
  // are checking OR compatibility. we may have failed due to XOR
  // compatibilty of variations failed, and here we are, potentially
  // checking variations again. we should only check variations
  // once in such a condition.
  // Logically, something like:
  //
  // if (allVariationsMatch(a,b)) {
  //   if (a.isXorCompatibleWith(b, NoVariationCheck)
  //       a.isAndCompatibleWith(b, NoVariationCheck)) {
  //     return true;
  //   }
  // }
  // return false;
  //
  // Another optimizatoin is that rather than testing Xor + And
  // separately, we have new bitset function something like
  // "is_disjoint_from_or_subset_of()" which I think covers
  // both cases.  "disjoint_from" is just the oppposite of
  // intersects(), right?
  // So we could specialize "Or" compatibility testing with that
  // bitset function, (and a variations check), rather than
  // calling two separate Xor/And functions here.

  constexpr auto isAndCompatibleWith(
    const UsedSources& other, bool /*useBits*/ = true) const {
    if (!getBits().is_subset_of(other.getBits()))
      return false;
    // compare variations
    if (!allVariationsMatch(variations, other.variations))
      return false;
    return true;
  }

#ifdef __CUDA_ARCH__
  constexpr auto isXorCompatibleWith(
    const UsedSources& other, uint32_t* other_src_bits) const {
    // compare bits (gpu shared memory)
    if (getBits().shared_intersects(other_src_bits)) {
      return false;
    }
    // compare variations
    if (!allVariationsMatch(variations, other.variations)) {
      return false;
    }
    return true;
  }
#endif

  /*
  // TODO: private
  void addSource(int src) {
    auto sentence = Source::getSentence(src);
    int first = Source::getFirstSourceIndex(sentence);
    int offset{};
    while (sources[first + offset] != -1) ++offset;
    assert(offset < kMaxUsedSourcesPerSentence);
    sources[first + offset] = Sources::getIndex(src);
  }
  */

  void addSource(int src) {
    auto sentence = Source::getSentence(src);
    assert(sentence > 0);
    auto variation = Source::getVariation(src);
    if (hasVariation(sentence) && (getVariation(sentence) != variation)) {
      std::cerr << "variation(" << sentence << "), this: "
                << getVariation(sentence) << ", src: " << variation
                << std::endl;
      assert(false && "addSource() variation mismatch");
    }
    assert(Source::getIndex(src) < kMaxSourcesPerSentence);

    // variation
    setVariation(sentence, variation);

    // bits
    auto bit_pos = Source::getIndex(src) + getFirstBitIndex(sentence);
    assert(!bits.test(bit_pos));
    bits.set(bit_pos);
  }

  auto mergeInPlace(const UsedSources& other) {
    // merge bits
    getBits() |= other.getBits();
    // merge variation. haxoid. should be separate function.
    addVariations(other);
  }

  auto copyMerge(const UsedSources& other) const {
    UsedSources result{ *this }; // copy
    result.mergeInPlace(other);
    return result;
  }

  constexpr void dump(bool device = false, char* buf = nullptr, char* smolbuf = nullptr) const {
    char big_buf[256];
    char smol_buf[32];
    if (!buf) buf = big_buf;
    if (!smolbuf) smolbuf = smol_buf;
    *buf = 0;
    *smolbuf = 0;
    auto first{true};
    if (device) {
      printf("sources:");
    } else {
      std::cerr << "sources:";
    }
    for (auto s{1}; s <= kNumSentences; ++s) {
      if (getVariation(s) > -1) {
        if (first) {
          if (device) {
            printf("\n");
          } else {
            std::cerr << std::endl;
          }
          first = false;
        }
        if (device) {
          printf("  s%d v%d:", s, getVariation(s));
        } else {
          std::cerr << "  s" << s << " v" << getVariation(s) << ":";
        }
        for (int i{}; i < kMaxSourcesPerSentence; ++i) {
          if (bits.test((s - 1) * kMaxSourcesPerSentence + i)) {
            if (device) {
              printf(" %d", i);
            } else {
              std::cerr << " " << i;
            }
          }
        }
        if (device) {
            printf("\n");
        } else {
          std::cerr << std::endl;
        }
      }
    }
    if (first) {
      if (device) {
        printf(" none\n");
      } else {
        std::cerr << " none" << std::endl;
      }
    }
  }

  constexpr void assert_valid() const {
    for (int s{1}; s <= kNumSentences; ++s) {
      if (hasVariation(s)) {
        assert(bits.count() > 0); // sources[Source::getFirstIndex(s)] != -1);
      }
    }
  }

  SourceBits bits{};
  Variations variations = make_array<VariationIndex_t, kNumSentences>(-1);
 };  // UsedSources

struct SourceCompatibilityData {
  SourceCompatibilityData() = default;
  // copy consruct/assign allowed for now, precompute.mergeAllCompatibleXorSources
  SourceCompatibilityData(const SourceCompatibilityData&) = default;
  SourceCompatibilityData& operator=(const SourceCompatibilityData&) = default;
  SourceCompatibilityData(SourceCompatibilityData&&) = default;
  SourceCompatibilityData& operator=(SourceCompatibilityData&&) = default;

  // copy components
  SourceCompatibilityData(
    const LegacySourceBits& legacySourceBits, const UsedSources& usedSources)
      : legacySourceBits(legacySourceBits),
        usedSources(usedSources) {
  }

  // move components
  SourceCompatibilityData(
    LegacySourceBits&& legacySourceBits, UsedSources&& usedSources)
      : legacySourceBits(std::move(legacySourceBits)),
        usedSources(std::move(usedSources)) {
  }

  constexpr auto isXorCompatibleWith(
    const SourceCompatibilityData& other) const {
    if (legacySourceBits.intersects(other.legacySourceBits)) {
      return false;
    }
    return usedSources.isXorCompatibleWith(other.usedSources);
  }

#ifdef __CUDA_ARCH__
  constexpr auto isXorCompatibleWith(const SourceCompatibilityData& other,
    uint32_t* other_src_bits, uint32_t* other_legacy_src_bits) const {
    if (legacySourceBits.shared_intersects(other_legacy_src_bits)) {
      return false;
    }
    return usedSources.isXorCompatibleWith(other.usedSources, other_src_bits);
  }
#endif

  constexpr auto isAndCompatibleWith(
    const SourceCompatibilityData& other) const {
    if (!legacySourceBits.is_subset_of(other.legacySourceBits))
      return false;
    return usedSources.isAndCompatibleWith(other.usedSources);
  }

  // OR == XOR || AND
  constexpr auto isOrCompatibleWith(
    const SourceCompatibilityData& other) const {
    return isXorCompatibleWith(other) || isAndCompatibleWith(other);
  }

  void addSource(int src) {
    if (cm::Source::isLegacy(src)) {
      assert(!legacySourceBits.test(src));
      legacySourceBits.set(src);
    } else {
      usedSources.addSource(src);
    }
  }

  void mergeInPlace(const SourceCompatibilityData& other) {
    auto count = legacySourceBits.count();
    legacySourceBits |= other.legacySourceBits;
    assert(legacySourceBits.count() == count + other.legacySourceBits.count());
    usedSources.mergeInPlace(other.usedSources);
  }

  constexpr void dump(const char* header = nullptr, bool device = false,
    char* buf = nullptr, char* smolbuf = nullptr) const
  {
    char big_buf[256] = "-";
    char smol_buf[32] = { 0 };
    if (!buf) buf = big_buf;
    if (!smolbuf) smolbuf = smol_buf;
    *buf = 0;
    *smolbuf = 0;
    if (header) {
      if (device) {
        sprintf(buf, "%s\n", header);
      } else {
        std::cerr << header << std::endl;
      }
    }
    usedSources.dump(device);
    if (device) {
      printf("legacy sources:");
    } else {
      std::cerr << "legacy sources:";
    }
    auto any{false};
    for (int i{}; i < kMaxLegacySources; ++i) {
      if (legacySourceBits.test(i)) {
        if (device) {
          printf(" %d", i);
        } else {
          std::cerr << " " << i;
        }
        any = true;
      }
    }
    if (!any) {
      if (device) {
        printf(" none");
      } else {
        std::cerr << " none";
      }
    }
    if (device) {
      printf("\n");
    } else {
      std::cerr << std::endl;
    }
  }

  constexpr void assert_valid() const {
    usedSources.assert_valid();
  }

  LegacySourceBits legacySourceBits;
  UsedSources usedSources;
  };  // SourceCompatibilityData
using SourceCompatibilityList = std::vector<SourceCompatibilityData>;

struct NameCount;
using NameCountList = std::vector<NameCount>;

struct NameCount {
  NameCount(std::string&& name, int count) :
    name(std::move(name)), count(count) {}
  NameCount() = default;
  NameCount(const NameCount&) = default;
  NameCount& operator=(const NameCount&) = default;
  NameCount(NameCount&&) = default;
  NameCount& operator=(NameCount&&) = default;

  std::string toString() const {
    char buf[128] = { 0 };
    snprintf(buf, sizeof(buf), "%s:%d", name.c_str(), count);
    return buf;
  }

  static std::string listToString(const NameCountList& list) {
    char buf[1280] = { 0 };
    for (auto it = list.cbegin(); it != list.cend(); ++it) {
      std::strcat(buf, it->toString().c_str());
      if ((it + 1) != list.cend()) {
        std::strcat(buf, ",");
      }
    }
    return buf;
  }

  static std::string listToString(const std::vector<const NameCount*>& list) {
    char buf[1280] = { 0 };
    for (auto it = list.cbegin(); it != list.cend(); ++it) {
      std::strcat(buf, (*it)->toString().c_str());
      if ((it + 1) != list.cend()) {
        std::strcat(buf, ",");
      }
    }
    return buf;
  }

  static auto listToCountSet(const NameCountList& list) {
    std::unordered_set<int> count_set;
    for (const auto& nc : list) {
      count_set.insert(nc.count);
    }
    return count_set;
  }

  static auto listToLegacySourceBits(const NameCountList& list) {
    LegacySourceBits bits{};
    for (const auto& nc : list) {
      if (Source::isLegacy(nc.count)) {
        bits.set(nc.count);
      }
    }
    return bits;
  }

  static auto listToUsedSources(const NameCountList& list) {
    UsedSources usedSources{};
    for (const auto& nc : list) {
      if (Source::isCandidate(nc.count)) {
        usedSources.addSource(nc.count);
      }
    }
    return usedSources;
  }

  static auto listMerge(const NameCountList& list1,
    const NameCountList& list2)
  {
    auto result = list1; // copy (ok)
    result.insert(result.end(), list2.begin(), list2.end()); // copy (ok)
    return result;
  }

  std::string name;
  int count;
};

struct NCData {
  NameCountList ncList;
};
using NCDataList = std::vector<NCData>;

  //struct NameCount;

struct SourceData : SourceCompatibilityData {
  SourceData() = default;
  SourceData(NameCountList&& primaryNameSrcList,
    LegacySourceBits&& legacySourceBits, UsedSources&& usedSources,
    NameCountList&& ncList)
      : SourceCompatibilityData(
        std::move(legacySourceBits), std::move(usedSources)),
        primaryNameSrcList(std::move(primaryNameSrcList)),
        ncList(std::move(ncList)) {
  }

  // copy assign allowed for now for precompute.mergeAllCompatibleXorSources
  SourceData(const SourceData&) = default;
  SourceData& operator=(const SourceData&) = default;
  SourceData(SourceData&&) = default;
  SourceData& operator=(SourceData&&) = default;

  NameCountList primaryNameSrcList;
  NameCountList ncList;
};

/*
struct SourceData : SourceBase {
  //std::vector<std::string> sourceNcCsvList; // TODO: I don't think this is even used anymore
  // synonymCounts

  SourceData() = default;
  SourceData(NameCountList&& primaryNameSrcList, SourceBits&& primarySrcBits,
      UsedSources&& usedSources, NameCountList&& ncList): //, std::vector<std::string>&& sourceNcCsvList) :
    SourceBase(std::move(primaryNameSrcList), std::move(primarySrcBits),
      std::move(usedSources), std::move(ncList))
      //,sourceNcCsvList(std::move(sourceNcCsvList))
  {}

  SourceData(const SourceData&) = delete;
  SourceData& operator=(const SourceData&) = delete;
  SourceData(SourceData&&) = default;
  SourceData& operator=(SourceData&&) = default;
};
*/

using SourceList = std::vector<SourceData>;
using SourceListMap = std::unordered_map<std::string, SourceList>;
using SourceCRef = std::reference_wrapper<const SourceData>;
using SourceCRefList = std::vector<SourceCRef>;

using XorSource = SourceData;
using XorSourceList = std::vector<XorSource>;

struct OrSourceData {
  SourceCompatibilityData source;
  bool xorCompatible = false;
  bool andCompatible = false;
};
using OrSourceList = std::vector<OrSourceData>;

// One OrArgData contains all of the data for a single --or argument.
//
struct OrArgData {
  OrSourceList orSourceList;
  bool compatible{false};
};
using OrArgList = std::vector<OrArgData>;

namespace device {
  struct OrArgData {
    OrSourceData* or_sources;
    unsigned num_or_sources;
    bool compatible;
  };
}

// These are precomputed on xorSourceList, to identify only those sources
// which share the same per-sentence variation.
// One list of indices per variation, plus '-1' (no) variation.
// indices to outer vector are offset by 1; variation -1 is index 0.
using VariationIndicesList = std::vector<std::vector<int>>;
// one variationIndicesLists per sentence
using SentenceVariationIndices = std::array<VariationIndicesList, kNumSentences>;

// on-device version of above
namespace device {
  struct VariationIndices {
    int* device_data;      // one chunk of allocated data; other pointers below
                           // point inside this chunk. only this gets freed.
    int* sourceIndices;    // -1 terminated for each variation
    int* variationOffsets; // offsets into sourceIndices
    int num_variations;
  };
};

struct PreComputedData {
  XorSourceList xorSourceList;
  std::vector<int> xorSourceIndices;
  SourceCompatibilityData* device_xorSources{nullptr};
  OrArgList orArgList;
  device::OrArgData* device_or_args{nullptr};
  SourceListMap sourceListMap;
  SentenceVariationIndices sentenceVariationIndices;
  device::VariationIndices* device_sentenceVariationIndices{nullptr};
};

struct MergedSources : SourceCompatibilityData {
  MergedSources() = default;
  MergedSources(const MergedSources&) = default; // allow, dangerous?
  MergedSources& operator=(const MergedSources&) = delete;
  MergedSources(MergedSources&&) = default;
  MergedSources& operator=(MergedSources&&) = default;

  // copy from SourceData
  MergedSources(const SourceData& source)
      : SourceCompatibilityData(source.legacySourceBits, source.usedSources),
        sourceCRefList(SourceCRefList{SourceCRef{source}}) {
  }

  SourceCRefList sourceCRefList;
};

using MergedSourcesList = std::vector<MergedSources>;

using StringList = std::vector<std::string>;

struct PerfData {
  int calls;       // # of function calls
  int range_calls; // # of calls with range
  int64_t comps;   // # of compares
  int compat;      // # of compatible results. # of incompatible = calls - compat
  int ss_attempt;  // # of short-circuit attempts
  int ss_fail;     // # of short-circute failures; # of successes = ss_attempt - ss_fail
  int full;        // # of full range calls; eventually this should = calls - ss_attempt
};

struct CandidateStats {
  int sum;
  int sourceLists;
  int totalSources;
  int comboMapIndices;
  int totalCombos;
};

// functions

void debugSourceList(const SourceList& sourceList, std::string_view sv);

auto buildSourceListsForUseNcData(const std::vector<NCDataList>& useNcDataLists,
  const SourceListMap& sourceListMap) -> std::vector<SourceList>;

XorSourceList mergeCompatibleXorSourceCombinations(
  const std::vector<SourceList>& sourceLists);

auto buildSentenceVariationIndices(const XorSourceList& xorSourceList,
  const std::vector<int>& xorSourceIndices) -> SentenceVariationIndices;

void mergeUsedSourcesInPlace(UsedSources& to, const UsedSources& from);

inline constexpr void assert_valid(const SourceList& src_list) {
  for (const auto& src: src_list) {
    src.assert_valid();
  }
}

// globals

inline PerfData isany_perf{};
inline PreComputedData PCD;

} // namespace cm

template <typename SizeT>
inline void hash_combine(SizeT& seed, SizeT value) {
  seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

inline auto hash_called = 0;
inline auto equal_to_called = 0;

template struct std::hash<cm::UsedSources::SourceBits>;

namespace std {

template<>
struct equal_to<cm::UsedSources> {
  bool operator()(const cm::UsedSources& lhs,
    const cm::UsedSources& rhs) const noexcept
  {
    if (lhs.getBits() != rhs.getBits()) return false;
    for (auto i = 0u; i < lhs.variations.size(); ++i) {
      if (lhs.variations[i] != rhs.variations[i]) return false;
    }
    return true;
  }
};

template<>
struct hash<cm::UsedSources> {
  size_t operator()(const cm::UsedSources& usedSources) const noexcept {
    size_t bits_seed = 0;
    hash_combine(bits_seed, usedSources.getBits().hash());
    size_t variation_seed = 0;
    for (const auto variation: usedSources.variations) {
      hash_combine(variation_seed, hash<int>()(variation));
    }
    size_t seed = 0;
    hash_combine(seed, bits_seed);
    hash_combine(seed, variation_seed);
    return seed;
  }
};

template<>
struct equal_to<cm::SourceCompatibilityData> {
  bool operator()(const cm::SourceCompatibilityData& lhs,
    const cm::SourceCompatibilityData& rhs) const noexcept
  {
    ++equal_to_called;
    return equal_to<cm::LegacySourceBits>{}(lhs.legacySourceBits, rhs.legacySourceBits) &&
      equal_to<cm::UsedSources>{}(lhs.usedSources, rhs.usedSources);
  }
};

template<>
struct hash<cm::SourceCompatibilityData> {
  size_t operator()(const cm::SourceCompatibilityData& data) const noexcept {
    ++hash_called;
    size_t seed = 0;
    hash_combine(seed, data.legacySourceBits.hash());
    hash_combine(seed, hash<cm::UsedSources>()(data.usedSources));
    return seed;
  }
};
} // namespace std

#endif // INCLUDE_COMBO_MAKER_H
