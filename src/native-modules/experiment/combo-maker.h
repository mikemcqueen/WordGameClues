#ifndef INCLUDE_COMBO_MAKER_H
#define INCLUDE_COMBO_MAKER_H

#if 1
#define CONSTEXPR_BITSET
#define HOST_DEVICE_ATTRIBUTES constexpr
#endif

#ifndef HOST_DEVICE_ATTRIBUTES
#define HOST_DEVICE_ATTRIBUTES
#endif

#include <algorithm>
#include <array>
#ifdef CONSTEXPR_BITSET
#include "constexpr_bitset.h"
using hax::bitset;
#else
#include <bitset>
using std::bitset;
#endif
#include <cassert>
#include <cstring>
#include <iostream>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cm {

constexpr auto kMaxLegacySources = 111; // bits
constexpr auto kMaxSourcesPerSentence = 128; // bits
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
  constexpr inline auto getFirstIndex(int sentence) {
    assert(sentence > 0);
    return (sentence - 1) * kMaxUsedSourcesPerSentence;
  }
} // namespace Source

using /*Legacy*/SourceBits = bitset<kMaxLegacySources>;
using LegacySources = std::array<int8_t, kMaxLegacySources>;
// 32 bytes per sentence * 9 sentences = 2304? bits, 288 bytes, 36? 64-bit words
using Sources = std::array<int8_t, kMaxUsedSources>;

#define USEDSOURCES_BITSET 1

struct UsedSources {
  using VariationIndex_t = int16_t;
  // 128 bits per sentence * 9 sentences = 1152 bits, 144 bytes, 18 64-bit words
  using Bits = bitset<kMaxSourcesPerSentence * kNumSentences>;
  using Variations = std::array<VariationIndex_t, kNumSentences>;

  static /*constexpr*/ auto getFirstBitIndex(int sentence) {
    assert(sentence > 0);
    return (sentence - 1) * kMaxSourcesPerSentence;
  }

  constexpr Bits& getBits() noexcept { return bits; }
  constexpr const Bits& getBits() const noexcept { return bits; }

  constexpr int getVariation(int sentence) const { return variations.at(sentence - 1); }
  void setVariation(int sentence, int value) { variations.at(sentence - 1) = value; }
  constexpr bool hasVariation(int sentence) const { return getVariation(sentence) > -1; }

  /*
  auto andBits(const Bits& other) const noexcept {
#if 1
    static Bits result{};
    result.reset();
    result |= getBits(); // possibly use static here
#else
    Bits result(getBits()); // possibly use static here
#endif
    result &= other;
    return result;
  }
  */

private:
  void addSources(const UsedSources& other) {
    for (int sentence{ 1 }; sentence <= kNumSentences; ++sentence) {
      const auto first = Source::getFirstIndex(sentence);
      int other_offset{};
      // does 'other' have sources for this sentence?
      if (other.sources[first + other_offset] == -1) continue;

      // ensure variations for this sentence are compatible
      if (hasVariation(sentence)) {
        assert(getVariation(sentence) == other.getVariation(sentence));
      } else {
        assert(other.hasVariation(sentence));
        setVariation(sentence, other.getVariation(sentence));
      }

      // determine our starting offset for merging in new sources
      int offset{};
      while (sources[first + offset] != -1) ++offset;
      
      // copy sources from other to us
      while (other.sources[first + other_offset] != -1) {
        sources[first + offset++] = other.sources[first + other_offset++];
      }
      // sanity check
      assert(offset < kMaxUsedSourcesPerSentence);
    }
  }

public:
  constexpr static bool anySourcesMatch(const Sources& s1,
    const Sources& s2)
  {
    // TODO: sanity check if sorted 

    for (int s{ 1 }; s <= kNumSentences; ++s) {
      auto start = Source::getFirstIndex(s);
      for (int i{start}, j{start}, remain{kMaxUsedSourcesPerSentence};
           remain > 0; --remain)
      {
        if ((s1[i] == -1) || (s2[j] == -1)) {
          break;
        }
        if (s1[i] == s2[j]) {
          return true;
        }
        else if (s1[i] > s2[j]) {
          i++;
        }
        else {
          j++;
        }
      }
    }
    return false;
  }

  constexpr static bool anySourcesMatch2(const Sources& s1,
    const Sources& s2)
  {
    // TODO: sanity check if sorted 
    int matches{};
    for (int s{ 1 }; s <= kNumSentences; ++s) {
      int8_t sources[kMaxUsedSourcesPerSentence + 1] = { 0 };
      auto start = Source::getFirstIndex(s);
      for (int i{}; i < kMaxUsedSourcesPerSentence + 1; ++i) {
        auto index = s1[start + i] + 1;
        sources[index] = index;
      }
      for (int i{}; i < kMaxUsedSourcesPerSentence + 1; ++i) {
        auto index = s2[start + i] + 1;
        matches += index && sources[index];
      }
    }
    return matches;
  }

  constexpr
  static auto allVariationsMatch(const Variations& v1,
    const Variations& v2, bool /*native*/ = true)
  {
    for (auto i{ 0u }; i < v1.size(); ++i) {
      if ((v1[i] > -1) && (v2[i] > -1)
          && (v1[i] != v2[i]))
      {
        return false;
      }
    }
    return true;
  }

  constexpr
  static bool allVariationsMatch2(const Variations& v1,
    const Variations& v2, bool /*native*/ = true)
  {
    int mismatches{};
    for (auto i{ 0u }; i < v1.size(); ++i) {
      auto first = v1[i] + 1;
      auto second = v2[i] + 1;
      mismatches += first && second && (first != second);
    }
    return !mismatches;
  }

  constexpr
  auto isXorCompatibleWith(const UsedSources& other,
    bool native = true, int* reason = nullptr) const
  {
    if (native) {
      // compare bits (cpu)
      if ((getBits() & other.getBits()).any()) {
        if (reason) *reason = 1;
        return false;
      }
    } else {
      // compare sources (gpu)
      if (anySourcesMatch(sources, other.sources)) {
        if (reason) *reason = 2;
        return false;
      }
    }
    // compare variations
    if (!allVariationsMatch(variations, other.variations, native)) {
      if (reason) *reason = 3;
      return false;
    }
    return true;
  }

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

    // source
    int first = Source::getFirstIndex(sentence);
    int offset{};
    while (sources[first + offset] != -1) ++offset;
    assert(offset < kMaxUsedSourcesPerSentence);
    sources[first + offset] = Source::getIndex(src);
  }

  auto mergeInPlace(const UsedSources& other) {
    // merge bits
    // TODO: option for compatibility checking
    getBits() |= other.getBits();

    addSources(other);
    sortSources();
  }

  auto copyMerge(const UsedSources& other) const {
    UsedSources result{ *this }; // copy
    result.mergeInPlace(other);
    return result;
  }

  constexpr int getFirstSource(int sentence) const {
    return sources.at(Source::getFirstIndex(sentence));
  }

  constexpr int countSources(int sentence) const {
    int count{};
    if (getVariation(sentence) > -1) {
      for (int i{}; i < kMaxUsedSourcesPerSentence; ++i) {
        auto src = sources.at(Source::getFirstIndex(sentence) + i);
        if (src < 0) break;
        ++count;
      }
    }
    return count;
  }

  void sortSources() {
    for (int s{1}; s <= kNumSentences; ++s) {
      std::sort(sources.data() + Source::getFirstIndex(s),
                sources.data() + Source::getFirstIndex(s + 1),
                std::greater<int8_t>());
    }
  }
  
  void dump() const {
    auto first{true};
    std::cerr << "sources:";
    for (auto s{1}; s <= kNumSentences; ++s) {
      if (getVariation(s) > -1) {
        if (first) {
          std::cerr << std::endl;
          first = false;
        }
        std::cerr << "  s" << s << " v" << getVariation(s) << ":";
        for (int i{}; i < kMaxUsedSourcesPerSentence; ++i) {
          auto src = sources.at(Source::getFirstIndex(s) + i);
          if (src < 0) break;
          std::cerr << " " << int(src);
        }
        std::cerr << std::endl;
      }
    }
    if (first) std::cerr << " none" << std::endl;
  }

  constexpr void assert_valid() const {
    for (int s{1}; s <= kNumSentences; ++s) {
      if (hasVariation(s)) {
        assert(sources[Source::getFirstIndex(s)] != -1);
      }
    }
  }

  Bits bits{};
  Sources sources = make_array<int8_t, kMaxUsedSources>(-1);
  Variations variations = make_array<VariationIndex_t, kNumSentences>(-1);
}; // UsedSources

struct SourceCompatibilityData {
  SourceCompatibilityData() = default;
  // copy consruct/assign allowed for now, precompute.mergeAllCompatibleXorSources
  SourceCompatibilityData(const SourceCompatibilityData&) = default;
  SourceCompatibilityData& operator=(const SourceCompatibilityData&) = default;
  SourceCompatibilityData(SourceCompatibilityData&&) = default;
  SourceCompatibilityData& operator=(SourceCompatibilityData&&) = default;

  // copy components
  SourceCompatibilityData(const SourceBits& sourceBits,
      const UsedSources& usedSources, const LegacySources& legacySources):
    sourceBits(sourceBits), usedSources(usedSources),
    legacySources(legacySources)
  {}

  // move components
  SourceCompatibilityData(SourceBits&& sourceBits,
      UsedSources&& usedSources, LegacySources&& legacySources):
    sourceBits(std::move(sourceBits)), usedSources(std::move(usedSources)),
    legacySources(std::move(legacySources))
  {}

  constexpr static bool anyLegacySourcesMatch(const LegacySources& ls1,
    const LegacySources& ls2)
  {
    for (int i{}; i < kMaxLegacySources; ++i) {
      if (ls1[i] && ls2[i]) return true;
    }
    return false;
  }
  
  constexpr auto isXorCompatibleWith(const SourceCompatibilityData& other,
    bool useBits = true, int* reason = nullptr) const
  {
    if (useBits) {
      if ((sourceBits & other.sourceBits).any()) {
        if (reason) *reason = 4;
        return false;
      }
    } else if (anyLegacySourcesMatch(legacySources, other.legacySources)) { 
      if (reason) *reason = 5;
      return false;
    }
    return usedSources.isXorCompatibleWith(other.usedSources, useBits, reason);
  }

  constexpr auto isAndCompatibleWith(const SourceCompatibilityData& other,
    bool /*useBits*/ = true) const
  {
    auto andBits = sourceBits & other.sourceBits;
    if (andBits != other.sourceBits) return false;
    return false; // TODO: usedSources.isAndCompatibleWith(other.usedSources);
  }

  // OR == XOR || AND
  constexpr auto isOrCompatibleWith(const SourceCompatibilityData& other,
    bool useBits = true) const
  {
    return isXorCompatibleWith(other, useBits)
      || isAndCompatibleWith(other, useBits);
  }

  static void addLegacySource(LegacySources& sources, int src) {
    assert(!sources[src]);
    sources[src] = 1;
  }

  void addSource(int src) {
    if (cm::Source::isLegacy(src)) {
      assert(!sourceBits.test(src));
      sourceBits.set(src);
      addLegacySource(legacySources, src);
    } else {
      usedSources.addSource(src);
    }
  }

  static void merge(LegacySources& to, const LegacySources& from) {
    for (int i{}; i < kMaxLegacySources; ++i) {
      if (from[i]) {
        addLegacySource(to, i);
      }
    }
  }

  auto copyMerge(const LegacySources& other) const {
    LegacySources sources{ legacySources };
    merge(sources, other);
    return sources;
  }

  void mergeInPlace(const LegacySources& other) {
    merge(legacySources, other);
  }

  void mergeInPlace(const SourceCompatibilityData& other) {
    auto count = sourceBits.count();
    sourceBits |= other.sourceBits;
    assert(sourceBits.count() == count + other.sourceBits.count());
    usedSources.mergeInPlace(other.usedSources);
    mergeInPlace(other.legacySources);
  }

  // used for debug logging
  constexpr int getFirstLegacySource() const {
    for (int i{}; i < kMaxLegacySources; ++i) {
      const auto src = legacySources.at(i);
      if (src) return i;
    }
    return -1;
  }

  constexpr int countLegacySources() const {
    int count{};
    for (int i{}; i < kMaxLegacySources; ++i) {
      const auto src = legacySources.at(i);
      if (src) ++count;
    }
    return count;
  }

  void dump(const char* header = nullptr) const {
    if (header) std::cerr << header << std::endl;
    usedSources.dump();
    std::cerr << "legacy sources:";
    auto any{false};
    for (int i{}; i < kMaxLegacySources; ++i) {
      if (legacySources.at(i)) {
        std::cerr << " " << i;
        any = true;
      }
    }
    if (!any) std::cerr << " none";
    std::cerr << std::endl;
  }

  constexpr void assert_valid() const {
    usedSources.assert_valid();
  }

  SourceBits sourceBits;
  UsedSources usedSources;
  // TODO: could be array of bool too
  LegacySources legacySources = make_array<int8_t, kMaxLegacySources>(0);
}; // SourceCompatibilityData
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

  static auto listToSourceBits(const NameCountList& list) {
    SourceBits bits{};
    for (const auto& nc : list) {
      if (Source::isLegacy(nc.count)) {
        bits.set(nc.count);
      }
    }
    return bits;
  }

  static auto listToLegacySources(const NameCountList& list) {
    auto sources = make_array<int8_t, kMaxLegacySources>(0);
    for (const auto& nc : list) {
      if (Source::isLegacy(nc.count)) {
        sources[nc.count] = 1;
      }
    }
    return sources;
  }

  static auto listToUsedSources(const NameCountList& list) {
    UsedSources usedSources{};
    for (const auto& nc : list) {
      if (Source::isCandidate(nc.count)) {
        usedSources.addSource(nc.count);
      }
    }
    usedSources.sortSources();
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
  SourceData(NameCountList&& primaryNameSrcList, SourceBits&& sourceBits,
      UsedSources&& usedSources, LegacySources&& legacySources,
      NameCountList&& ncList) :
    SourceCompatibilityData(std::move(sourceBits), std::move(usedSources),
      std::move(legacySources)),
    primaryNameSrcList(std::move(primaryNameSrcList)),
    ncList(std::move(ncList))
  {}

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
  SourceData source;
  bool xorCompatible = false;
  bool andCompatible = false;
};
using OrSourceList = std::vector<OrSourceData>;

// One OrArgData contains all of the data for a single --or argument.
//
struct OrArgData {
  OrSourceList orSourceList;
  bool compatible = false;
};
using OrArgDataList = std::vector<OrArgData>;

// These are precomputed on xorSourceList, to identify only those sources
// which share the same per-sentence variation.

// one list of indices per variation, plus '-1' (no) variation.
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
  XorSource* device_xorSources{ nullptr };
  OrArgDataList orArgDataList;
  SourceListMap sourceListMap;
  SentenceVariationIndices sentenceVariationIndices;
  device::VariationIndices* device_sentenceVariationIndices{ nullptr };
};

struct MergedSources : SourceCompatibilityData {
  MergedSources() = default;
  MergedSources(const MergedSources&) = default; // allow, dangerous?
  MergedSources& operator=(const MergedSources&) = delete;
  MergedSources(MergedSources&&) = default;
  MergedSources& operator=(MergedSources&&) = default;

  // copy from SourceData
  MergedSources(const SourceData& source) :
    SourceCompatibilityData(source.sourceBits, source.usedSources,
      source.legacySources),
    sourceCRefList(SourceCRefList{SourceCRef{source}})
  {}

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

// functions TODO: precompute.h
 
void debugSourceList(const SourceList& sourceList, std::string_view sv);

auto buildSourceListsForUseNcData(const std::vector<NCDataList>& useNcDataLists,
  const SourceListMap& sourceListMap) -> std::vector<SourceList>;

XorSourceList mergeCompatibleXorSourceCombinations(
  const std::vector<SourceList>& sourceLists);

auto buildSentenceVariationIndices(const XorSourceList& xorSourceList,
  const std::vector<int>& xorSourceIndices) -> SentenceVariationIndices;

auto getSortedXorSourceIndices(const XorSourceList& xorSourceList)
  -> std::vector<int>;

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

template struct std::hash<cm::UsedSources::Bits>;

namespace std {

template<>
struct equal_to<cm::UsedSources> {
  HOST_DEVICE_ATTRIBUTES
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
    hash_combine(bits_seed, hash<cm::UsedSources::Bits>()(usedSources.getBits()));
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
  //constexpr
  bool operator()(const cm::SourceCompatibilityData& lhs,
    const cm::SourceCompatibilityData& rhs) const noexcept
  {
    ++equal_to_called;
    return equal_to<cm::SourceBits>{}(lhs.sourceBits, rhs.sourceBits) &&
      equal_to<cm::UsedSources>{}(lhs.usedSources, rhs.usedSources);
  }
};

template<>
struct hash<cm::SourceCompatibilityData> {
  size_t operator()(const cm::SourceCompatibilityData& data) const noexcept {
    ++hash_called;
    size_t seed = 0;
    hash_combine(seed, hash<cm::SourceBits>()(data.sourceBits));
    hash_combine(seed, hash<cm::UsedSources>()(data.usedSources));
    return seed;
  }
};
} // namespace std

#endif // INCLUDE_COMBO_MAKER_H
