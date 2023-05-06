#ifndef include_combo_maker_h
#define include_combo_maker_h

#include <bitset>
#include <cassert>
#include <iostream>
#include <napi.h>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace cm {

constexpr auto kMaxLegacySources = 111; // bits
constexpr auto kMaxSourcesPerSentence = 128; // bits
constexpr auto kSentences = 9;

template<typename T, size_t N>
constexpr auto make_array(T value) -> std::array<T, N> {
  std::array<T, N> a{};
  for (auto& e : a) {
    e = value;
  }
  return a;
}

  namespace Source {
    constexpr inline auto isCandidate(int src) { return src >= 1'000'000; }
    constexpr inline auto isLegacy(int src) { return !isCandidate(src); }
    constexpr inline auto getSentence(int src) { return src / 1'000'000; }
    constexpr inline auto getSource(int src) { return src % 1'000'000; }
    constexpr inline auto getVariation(int src) { return getSource(src) / 100; }
    constexpr inline auto getIndex(int src) { return getSource(src) % 100; }
  } // namespace Source

using SourceBits = std::bitset<kMaxLegacySources>;
using SourceBitsList = std::vector<SourceBits>;

#define USEDSOURCES_BITSET 0

#if !USEDSOURCES_BITSET
using UsedSources = std::array<std::set<uint32_t>, 10>;
#else
struct UsedSources {
  static constexpr auto kMaxUsedSourcesPerSentence = 32;

  // 128 * 32 = 4096 bits, 512 bytes, 64 uint64_t
  using Bits = std::bitset<kMaxSourcesPerSentence * kMaxUsedSourcesPerSentence>;
  using Sentences = std::array<Bits, kSentences>;
  using Variations = std::array<int, kSentences>;

  Sentences sentences{};
  Variations variations = make_array<int, kSentences>( -1 );

  Bits& getBits(int sentence) { return sentences[sentence - 1]; }
  const Bits& getBits(int sentence) const { return sentences[sentence - 1]; }

  constexpr int getVariation(int sentence) const { return variations[sentence - 1]; }
  void setVariation(int sentence, int value) { variations[sentence - 1] = value; }
  constexpr bool hasVariation(int sentence) const { return getVariation(sentence) > -1; }

  auto isXorCompatibleWith(const UsedSources& other) const {
    for (auto i = 0u; i < variations.size(); ++i) {
      if ((variations[i] == -1) || (other.variations[i] == -1)) continue;
      if (variations[i] != other.variations[i]) return false;
      if ((sentences[i] & other.sentences[i]).any()) return false;
    }
    return true;
  }
 
  void addSource(int src) {
    auto sentence = Source::getSentence(src);
    auto variation = Source::getVariation(src);
    //auto thisVariation = getVariation(sentence);
    if (hasVariation(sentence) && (getVariation(sentence) != variation)) {
      std::cerr << "variation(" << sentence << "), this: " << getVariation(sentence)
                << ", src: " << variation << std::endl;
      assert(true && "addSource() variation mismatch");
    }
    auto& bits = getBits(sentence);
    auto pos = Source::getIndex(src) * kMaxSourcesPerSentence;
    assert(!bits.test(pos));
    setVariation(sentence, variation);
    bits.set(pos);
  }

  auto mergeInPlace(const UsedSources& other) {
    for (auto i = 1u; i <= sentences.size(); ++i) {
      auto& bits = getBits(i);
      // could assert bits here
      bits |= other.getBits(i);
      if (hasVariation(i) && other.hasVariation(i)) {
        assert(getVariation(i) == other.getVariation(i));
      } else if (!hasVariation(i)) {
        setVariation(i, other.getVariation(i));
      }
    }
  }

  auto merge(const UsedSources& other) const {
    UsedSources result{ *this }; // copy
    result.mergeInPlace(other);
    return other;
  }
};
#endif // USED_SOURCES_BITSET

struct SourceCompatibilityData {
  SourceBits sourceBits;
  UsedSources usedSources;

  SourceCompatibilityData() = default;
  // copy consruct/assign allowed for now for precompute.mergeAllCompatibleXorSources
  SourceCompatibilityData(const SourceCompatibilityData&) = default;
  SourceCompatibilityData& operator=(const SourceCompatibilityData&) = default;
  SourceCompatibilityData(SourceCompatibilityData&&) = default;
  SourceCompatibilityData& operator=(SourceCompatibilityData&&) = default;

  // copy components
  SourceCompatibilityData(const SourceBits& sourceBits,
      const UsedSources& usedSources):
    sourceBits(sourceBits),
    usedSources(usedSources)
  {}

  // move components
  SourceCompatibilityData(SourceBits&& sourceBits,
      UsedSources&& usedSources):
    sourceBits(std::move(sourceBits)),
    usedSources(std::move(usedSources))
  {}

#if !USEDSOURCES_BITSET
  static auto areUsedSourcesCompatible(const UsedSources& usedSources, const UsedSources& other) {
    for (auto i = 1u; i < usedSources.size(); ++i) {
      if (usedSources[i].empty() || other[i].empty()) continue;
      if (Source::getVariation(*usedSources[i].cbegin()) != 
          Source::getVariation(*other[i].cbegin()))
      {
        return false;
      }
      for (auto it = usedSources[i].cbegin(); it != usedSources[i].cend(); ++it) {
        if (other[i].find(*it) != other[i].end()) {
          return false;
        }
      }
    }
    return true;
  }

  static auto addUsedSource(UsedSources& usedSources, int src) -> bool {
    auto nothrow = false;
    if (!Source::isCandidate(src)) throw "poop";
    auto sentence = Source::getSentence(src);
    auto source = Source::getSource(src);
    auto& set = usedSources[sentence];
    // defensive incompatible variation index check
    if (!set.empty()) {
      if (Source::getVariation(*set.begin()) != Source::getVariation(source)) {
        if (nothrow) return false;
        throw "oopsie1"; // new Error(`oopsie ${anyElem}, ${source}`);
      }
      if (set.find(source) != set.end()) {
        if (nothrow) return false;
        throw "oopsie2"; // new Error(`poopsie ${source}, [${[...set]}]`);
      }
    }
    set.insert(source);
    return true;
  }
#endif

  auto isXorCompatibleWith(const SourceCompatibilityData& other) const {
    if ((sourceBits & other.sourceBits).any()) {
      return false;
    }
#if !USEDSOURCES_BITSET
    return areUsedSourcesCompatible(usedSources, other.usedSources);
#else
    return usedSources.isXorCompatibleWith(other.usedSources);
#endif
  }

  auto addUsedSource(int src) {
#if !USEDSOURCES_BITSET
    return addUsedSource(usedSources, src);
#else
    return usedSources.addSource(src);
#endif
  }
};

struct NameCount;
using NameCountList = std::vector<NameCount>;

using SourceCompatibilityList = std::vector<SourceCompatibilityData>;

struct SourceData : SourceCompatibilityData {
  NameCountList primaryNameSrcList;
  NameCountList ncList;

  SourceData() = default;
  SourceData(NameCountList&& primaryNameSrcList, SourceBits&& sourceBits,
      UsedSources&& usedSources, NameCountList&& ncList) :
    SourceCompatibilityData(std::move(sourceBits), std::move(usedSources)),
    primaryNameSrcList(std::move(primaryNameSrcList)),
    ncList(std::move(ncList))
  {}

  // copy assign allowed for now for precompute.mergeAllCompatibleXorSources
  SourceData(const SourceData&) = default;
  SourceData& operator=(const SourceData&) = default;
  SourceData(SourceData&&) = default;
  SourceData& operator=(SourceData&&) = default;
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

struct PreComputedData {
  XorSourceList xorSourceList;
  OrArgDataList orArgDataList;
  SourceListMap sourceListMap;
};

struct MergedSources : SourceCompatibilityData {
  SourceCRefList sourceCRefList;

  MergedSources() = default;
  MergedSources(const MergedSources&) = default; // allow, dangerous?
  MergedSources& operator=(const MergedSources&) = delete;
  MergedSources(MergedSources&&) = default;
  MergedSources& operator=(MergedSources&&) = default;

  // copy from SourceData
  MergedSources(const SourceData& source) :
      SourceCompatibilityData(source.sourceBits, source.usedSources),
      sourceCRefList(SourceCRefList{SourceCRef{source}})
  {}
};

using MergedSourcesList = std::vector<MergedSources>;

using StringList = std::vector<std::string>;

struct NameCount {
  std::string name;
  int count;

  NameCount(std::string&& name, int count) : name(std::move(name)), count(count) {}
  NameCount() = default;
  NameCount(const NameCount&) = default;
  NameCount& operator=(const NameCount&) = default;
  NameCount(NameCount&&) = default;
  NameCount& operator=(NameCount&&) = default;

  std::string toString() const {
    char buf[128] = { 0 };
    sprintf(buf, "%s:%d", name.c_str(), count);
    return buf;
  }

  static std::string listToString(const std::vector<NameCount>& list) {
    char buf[1280] = { 0 };
    for (auto it = list.cbegin(); it != list.cend(); ++it) {
      strcat(buf, it->toString().c_str());
      if ((it + 1) != list.cend()) {
        strcat(buf, ",");
      }
    }
    return buf;
  }

  static std::string listToString(const std::vector<const NameCount*>& list) {
    char buf[1280] = { 0 };
    for (auto it = list.cbegin(); it != list.cend(); ++it) {
      strcat(buf, (*it)->toString().c_str());
      if ((it + 1) != list.cend()) {
        strcat(buf, ",");
      }
    }
    return buf;
  }

  static auto listToCountSet(const std::vector<NameCount>& list) {
    std::unordered_set<int> count_set;
    for (const auto& nc : list) {
      count_set.insert(nc.count);
    }
    return count_set;
  }

  static auto listToSourceBits(const std::vector<NameCount>& list) {
    SourceBits bits{};
    for (const auto& nc : list) {
      if (Source::isLegacy(nc.count)) {
        bits.set(nc.count);
      }
    }
    return bits;
  }

  static auto listToUsedSources(const std::vector<NameCount>& list) {
    UsedSources usedSources{};
    for (const auto& nc : list) {
      if (Source::isCandidate(nc.count)) {
#if !USEDSOURCES_BITSET
        SourceCompatibilityData::addUsedSource(usedSources, nc.count);
#else
        usedSources.addSource(nc.count);
#endif
      }
    }
    return usedSources;
  }

  static auto listMerge(const std::vector<NameCount>& list1,
    const std::vector<NameCount>& list2)
  {
    auto result = list1; // copy (ok)
    result.insert(result.end(), list2.begin(), list2.end()); // copy (ok)
    return result;
  }
};

struct NCData {
  NameCountList ncList;
};

using NCDataList = std::vector<NCData>;

// functions
 
void debugSourceList(const SourceList& sourceList, std::string_view sv);

auto buildSourceListsForUseNcData(const std::vector<NCDataList>& useNcDataLists,
  const SourceListMap& sourceListMap) -> std::vector<SourceList>;

XorSourceList mergeCompatibleXorSourceCombinations(
  const std::vector<SourceList>& sourceLists);

bool isAnySourceCompatibleWithUseSources(const SourceCompatibilityList& sourceCompatList);

void mergeUsedSourcesInPlace(UsedSources& to, const UsedSources& from);

} // namespace cm

template <typename SizeT>
inline void hash_combine(SizeT& seed, SizeT value) {
  seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

#if USEDSOURCES_SETBIT
template<>
struct std::equal_to<cm::UsedSources> {
  constexpr bool operator()(const cm::UsedSources& lhs,
    const cm::UsedSources& rhs) const noexcept
  {
    for (auto i = 0u; i < lhs.sentences.size(); ++i) {
      if (!std::equal_to<cm::UsedSources::Bits>{}(lhs.sentences[i], rhs.sentences[i])) return false;
      if (lhs.variations[i] != rhs.variations[i]) return false;
    }
    return true;
  }
};

template<>
struct std::hash<cm::UsedSources> {
  std::size_t operator()(const cm::UsedSources& usedSources) const noexcept {
    std::size_t sentence_seed = 0;
    for (const auto& sentence: usedSources.sentences) {
      hash_combine(sentence_seed, std::hash<cm::UsedSources::Bits>{}(sentence));
    }
    std::size_t variation_seed = 0;
    for (const auto variation: usedSources.variations) {
      hash_combine(variation_seed, std::hash<int>{}(variation));
    }
    std::size_t seed = 0;
    hash_combine(seed, sentence_seed);
    hash_combine(seed, variation_seed);
    return seed;
  }
};

template<>
struct std::equal_to<cm::SourceCompatibilityData> {
  constexpr bool operator()(const cm::SourceCompatibilityData& lhs,
    const cm::SourceCompatibilityData& rhs) const noexcept
  {
    return std::equal_to<cm::SourceBits>{}(lhs.sourceBits, rhs.sourceBits) &&
      std::equal_to<cm::UsedSources>{}(lhs.usedSources, rhs.usedSources);
  }
};

template<>
struct std::hash<cm::SourceCompatibilityData> {
  std::size_t operator()(const cm::SourceCompatibilityData& data) const noexcept {
    std::size_t seed = 0;
    hash_combine(seed, std::hash<cm::SourceBits>{}(data.sourceBits));
    hash_combine(seed, std::hash<cm::UsedSources>{}(data.usedSources));
    return seed;
  }
};
#endif // USEDSOURCES_SETBIT

#endif // include_combo_maker_h
