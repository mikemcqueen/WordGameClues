#ifndef INCLUDE_COMBO_MAKER_H
#define INCLUDE_COMBO_MAKER_H

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "cuda-string.h"
#include "mmebitset.h"
#include "variations.h"

namespace cm {

  //constexpr auto kMaxSourcesPerSentence = 32;
  //constexpr auto kNumSentences = 9;

template<typename T, size_t N>
constexpr auto make_array(T value) -> std::array<T, N> {
  std::array<T, N> a{};
  for (auto& e : a) {
    e = value;
  }
  return a;
}

namespace Source {

// clang-format off
constexpr inline auto isCandidate(int src) noexcept  { return src >= 1'000'000; }
constexpr inline auto getSentence(int src) noexcept  { return src / 1'000'000; }
constexpr inline auto getSource(int src) noexcept    { return src % 1'000'000; }
constexpr inline auto getVariation(int src) noexcept { return variation_index_t(getSource(src) / 100); }
constexpr inline auto getIndex(int src) noexcept     { return getSource(src) % 100; }
// clang-format on

}  // namespace Source

struct UsedSources {
  //  using variation_index_t = int16_t;

  struct SourceDescriptor {
    SourceDescriptor() = default;

    SourceDescriptor(int sentence, int bit_pos, variation_index_t variation)
        : sentence{static_cast<int8_t>(sentence)},
          bit_pos{static_cast<int8_t>(bit_pos)}, variation{variation} {
      validate();
    }

    constexpr void dump() const {
      printf("sentence %d, variation %d, bit_pos %d\n", (int)sentence, (int)variation, (int)bit_pos);
    }

    constexpr void validate() const {
      bool valid = true;
      if ((sentence < 0) || (sentence > kNumSentences)) {
        printf("invalid sentence\n");
        valid = false;
      }
      if ((bit_pos < 0) || (bit_pos >= kNumSentences * kMaxSourcesPerSentence)) {
        printf("invalid bit_pos\n");
        valid = false;
      }
      if (variation < 0) {
        printf("invalid variation\n");
        valid = false;
      }
      if (!valid) {
        dump();
        assert(0 && "SourceDescriptor::validate");
      }
    }

    int8_t sentence{};
    int8_t bit_pos{};
    variation_index_t variation{};
  };
  //  using SourceDescriptorPair = std::pair<SourceDescriptor, SourceDescriptor>;
  struct SourceDescriptorPair {
    SourceDescriptor first;
    SourceDescriptor second;
  };

  // 32 bits per sentence * 9 sentences = 288 bits, 36 bytes
  using SourceBits = mme::bitset<kMaxSourcesPerSentence * kNumSentences>;

  constexpr static auto getFirstBitIndex(int sentence) {
    assert(sentence > 0);
    return (sentence - 1) * kMaxSourcesPerSentence;
  }

  constexpr SourceBits& getBits() noexcept { return bits; }
  constexpr const SourceBits& getBits() const noexcept { return bits; }

  constexpr variation_index_t getVariation(int sentence) const {
    return variations.at(sentence - 1);
  }

  void setVariation(int sentence, variation_index_t value) {
    variations.at(sentence - 1) = value;
  }

  constexpr bool hasVariation(int sentence) const {
    return getVariation(sentence) > -1;
  }

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
    for (int sentence{1}; sentence <= kNumSentences; ++sentence) {
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
  static constexpr auto merge_one_variation(
      Variations& to, int sentence, variation_index_t from_value) {
    if (from_value == -1) return true;
    sentence -= 1;
    const auto to_value = to[sentence];
    if (to_value == -1) {
      to[sentence] = from_value;
      return true;
    }
    return from_value == to_value;
  }

  static auto merge_one_variation(Variations& to, int src) {
    return merge_one_variation(
        to, Source::getSentence(src), Source::getVariation(src));
  }

  // TODO: maybe should have a fast version fo this for CUDA
  static constexpr auto merge_variations(Variations& to, const Variations& from) {
    for (int sentence{1}; sentence <= kNumSentences; ++sentence) {
      const auto from_value = from[sentence - 1];
      if (!merge_one_variation(to, sentence, from_value)) return false;
    }
    return true;
  }

  static constexpr auto allVariationsMatch(
      const Variations& v1, const Variations& v2) {
    for (size_t i{}; i < v1.size(); ++i) {
      if ((v1[i] > -1) && (v2[i] > -1) && (v1[i] != v2[i])) {
        return false;
      }
    }
    return true;
  }

#if 1
  static constexpr auto are_variations_compatible(
      variation_index_t vi1, variation_index_t vi2) {
    vi1 += 1;
    vi2 += 1;
    if (vi1 && vi2 && (vi1 != vi2)) return false;
    return true;
  }

  static constexpr auto are_variations_compatible(
      const Variations& v1, const Variations& v2) {
    for (size_t i{}; i < v1.size(); ++i) {
      if (!are_variations_compatible(v1[i], v2[i])) return false;
    }
    return true;
  }
#endif

  constexpr auto isXorCompatibleWith(
      const UsedSources& other, bool check_variations = true) const {
    // compare bits
    if (getBits().intersects(other.getBits()))
      return false;
    // compare variations
    if (check_variations && !allVariationsMatch(variations, other.variations))
      return false;
    return true;
  }

  // NB! order of (a,b) in a.is(b) here matters!
  constexpr auto isCompatibleSubsetOf(
      const UsedSources& other, bool check_variations = true) const {
    if (!getBits().is_subset_of(other.getBits()))
      return false;
    // compare variations
    if (check_variations && !allVariationsMatch(variations, other.variations))
      return false;
    return true;
  }

  constexpr auto isOrCompatibleWith(const UsedSources& other) const {
    return are_variations_compatible(variations, other.variations)
           && (getBits().intersects(other.getBits())
               || getBits().is_subset_of(other.getBits()));
  }

  bool addSource(int src, bool nothrow = false) {
    auto sentence = Source::getSentence(src);
    assert(sentence > 0);
    auto variation = Source::getVariation(src);
    if (hasVariation(sentence) && (getVariation(sentence) != variation)) {
      if (nothrow) { return false; }
      std::cerr << "sentence " << sentence
                << " variation: " << getVariation(sentence)
                << ", src variation: " << variation << std::endl;
      assert(false && "addSource() variation mismatch");
    }
    assert(Source::getIndex(src) < kMaxSourcesPerSentence);

    // variation
    setVariation(sentence, variation);

    // bits
    auto bit_pos = Source::getIndex(src) + getFirstBitIndex(sentence);
    if (bits.test(bit_pos)) {
      if (nothrow) { return false; }
      assert(0 && "addSource: bit already set");
    }
    bits.set(bit_pos);
    return true;
  }

  auto mergeInPlace(const UsedSources& other) {
    // merge bits
    getBits() |= other.getBits();
    // merge variations.
    addVariations(other);
  }

  auto copyMerge(const UsedSources& other) const {
    UsedSources result{*this};  // copy
    result.mergeInPlace(other);
    return result;
  }

  constexpr void dump(const char* header = nullptr) const {
    char buf[512];
    char smolbuf[32];
    auto first{true};
    cuda_strcpy(buf, header ? header : "sources:");
    for (int s{1}; s <= kNumSentences; ++s) {
      if (hasVariation(s)) {
        if (first) {
          cuda_strcat(buf, "\n");
          first = false;
        }
        // sprintf(smolbuf, "  s%d v%d:", s, (int)getVariation(s));
        cuda_strcat(buf, "s");
        cuda_itoa(s, smolbuf);
        cuda_strcat(buf, smolbuf);
        cuda_strcat(buf, " v");
        cuda_itoa(getVariation(s), smolbuf);
        cuda_strcat(buf, smolbuf);
        cuda_strcat(buf, ":");
        for (int i{}; i < kMaxSourcesPerSentence; ++i) {
          if (bits.test((s - 1) * kMaxSourcesPerSentence + i)) {
            // sprintf(smolbuf, " %d", i);
            cuda_strcat(buf, " ");
            cuda_itoa(i, smolbuf);
            cuda_strcat(buf, smolbuf);
          }
        }
        cuda_strcat(buf, "\n");
      }
    }
    if (first) { cuda_strcat(buf, " none"); }
    printf("%s", buf);
  }

  constexpr void assert_valid() const {
    for (int s{1}; s <= kNumSentences; ++s) {
      if (hasVariation(s)) {
        assert(bits.count() > 0);  // sources[Source::getFirstIndex(s)] != -1);
      }
    }
  }

  SourceDescriptorPair get_source_descriptor_pair() const {
    SourceDescriptor first;
    for (int i{}; i < bits.wc(); ++i) {
      auto word = bits.word(i);
      while (word) {
        auto new_word = word & (word - 1);  // word with LSB removed
        // probably should have commented this when i did it.
        auto bit_pos = int(lrint(log2(word ^ new_word)));
        SourceDescriptor sd{i + 1, bit_pos, getVariation(i + 1)};
        if (!first.sentence) {
          first = sd;
        } else {
          return {first, sd};
        }
        word = new_word;
      }
    }
    assert(0 && "two sources not found");
    return {};
  }

  constexpr auto has(SourceDescriptor sd) const {
    return (getVariation(sd.sentence) == sd.variation)
           && getBits().test(sd.sentence - 1, sd.bit_pos);
  }

  constexpr auto has(SourceDescriptorPair sd_pair) const {
    return has(sd_pair.first) && has(sd_pair.second);
  }

  void reset() {
    bits.reset();
    std::memset(&variations, -1, sizeof(variations));
  }

  SourceBits bits{};
  Variations variations = make_array<variation_index_t, kNumSentences>(-1);
};  // UsedSources

struct SourceCompatibilityData {
  SourceCompatibilityData() = default;
  // copy consruct/assign allowed for now, precompute.mergeAllCompatibleXorSources
  SourceCompatibilityData(const SourceCompatibilityData&) = default;
  SourceCompatibilityData& operator=(const SourceCompatibilityData&) = default;
  SourceCompatibilityData(SourceCompatibilityData&&) = default;
  SourceCompatibilityData& operator=(SourceCompatibilityData&&) = default;

  // copy components
  SourceCompatibilityData(const UsedSources& usedSources)
      : usedSources(usedSources) {
  }

  // move components
  SourceCompatibilityData(UsedSources&& usedSources)
      : usedSources(std::move(usedSources)) {
  }

  constexpr auto isXorCompatibleWith(const SourceCompatibilityData& other,
      bool check_variations = true) const {
    return usedSources.isXorCompatibleWith(other.usedSources, check_variations);
  }

  constexpr auto isCompatibleSubsetOf(const SourceCompatibilityData& other,
      bool check_variations = true) const {
    return usedSources.isCompatibleSubsetOf(
        other.usedSources, check_variations);
  }

  constexpr auto hasCompatibleVariationsWith(
      const SourceCompatibilityData& other) const {
    // TODO: add hasCompatibleVariationsWith() member function (non-static) to
    // UsedSources?
    return UsedSources::are_variations_compatible(
        usedSources.variations, other.usedSources.variations);
  }

  // OR == XOR || AND
  // Another optimization is that rather than testing Xor + And
  // separately, we have new bitset function something like
  // "is_disjoint_from_or_subset_of()" which I think covers
  // both cases.  "disjoint_from" is just the oppposite of
  // intersects(), right?
  // So we could specialize "Or" compatibility testing with that
  // bitset function, (and a variations check), rather than
  // calling two separate Xor/And functions here.
  constexpr auto isOrCompatibleWith(
      const SourceCompatibilityData& other) const {
    if (!hasCompatibleVariationsWith(other)) return false;
    return isXorCompatibleWith(other, false)
           || isCompatibleSubsetOf(other, false);
  }

  bool isXorCompatibleWithAnySource(const auto& src_list) {
    for (const auto& src : src_list) {
      if (isXorCompatibleWith(src))
        return true;
    }
    return src_list.empty();
  }

  bool addSource(int src, bool nothrow = false) {
    return usedSources.addSource(src, nothrow);
  }

  void mergeInPlace(const SourceCompatibilityData& other) {
    usedSources.mergeInPlace(other.usedSources);
  }

  auto copyMerge(const SourceCompatibilityData& other) const {
    auto src_copy{*this};
    src_copy.mergeInPlace(other);
    return src_copy;
  }

  constexpr void dump(const char* header = nullptr) const {
    usedSources.dump(header);
  }

  constexpr void assert_valid() const {
    usedSources.assert_valid();
  }

  UsedSources usedSources;
};  // SourceCompatibilityData

using SourceCompatibilityList = std::vector<SourceCompatibilityData>;
using SourceCompatibilitySet = std::unordered_set<SourceCompatibilityData>;
using SourceCompatibilityDataCRef =
  std::reference_wrapper<const SourceCompatibilityData>;
using SourceCompatibilityCRefList = std::vector<SourceCompatibilityDataCRef>;

struct NameCount;
using NameCountList = std::vector<NameCount>;

struct NameCount {
  NameCount(const std::string& name, int count) : name(name), count(count) {}
  /*
  NameCount() = default;
  NameCount(const NameCount&) = default;
  NameCount(NameCount&&) = default;
  NameCount& operator=(const NameCount&) = default;
  NameCount& operator=(NameCount&&) = default;
  */

  std::string toString() const {
    char buf[128] = { 0 };
    snprintf(buf, sizeof(buf), "%s:%d", name.c_str(), count);
    return buf;
  }

  static std::string makeString(const std::string& name, int count) {
    return name + ":" + std::to_string(count);
  }

  static std::vector<std::string> listToNameList(const NameCountList& list) {
    std::vector<std::string> names;
    for (auto it = list.cbegin(); it != list.cend(); ++it) {
      names.emplace_back(it->name);
    }
    return names;
  }

  static void listSort(NameCountList& list) {
    std::ranges::sort(list,
        [](const NameCount& a, const NameCount& b) { return a.name < b.name; });
  }

  static std::string listToString(const std::vector<std::string>& list) {
    char buf[1280] = { 0 };
    for (auto it = list.cbegin(); it != list.cend(); ++it) {
      std::strcat(buf, it->c_str());
      if ((it + 1) != list.cend()) { // TODO std::next() ?
        std::strcat(buf, ",");
      }
    }
    return buf;
  }

  static std::string listToString(const NameCountList& list) {
    char buf[1280] = { 0 };
    for (auto it = list.cbegin(); it != list.cend(); ++it) {
      std::strcat(buf, it->toString().c_str());
      if ((it + 1) != list.cend()) { // TODO std::next() ?
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

  static auto listToCountList(const NameCountList& list) {
    std::vector<int> result;
    result.reserve(list.size());
    for (const auto& nc : list) {
      result.push_back(nc.count);
    }
    return result;
  }

  static auto listToUsedSources(const NameCountList& list) {
    UsedSources usedSources{};
    for (const auto& nc : list) {
      // TDOO: assert this is true
      if (Source::isCandidate(nc.count)) {
        usedSources.addSource(nc.count);
      }
    }
    return usedSources;
  }

  static auto listMerge(
    const NameCountList& list1, const NameCountList& list2) {
    //
    auto result = list1;                                     // copy (ok)
    result.insert(result.end(), list2.begin(), list2.end()); // copy (ok)
    return result;
  }

  static auto listContains(
    const NameCountList& list, const std::string& name, int count) {
    //
    for (const auto& nc : list) {
      if ((nc.name == name) && (nc.count == count)) {
        return true;
      }
    }
    return false;
  }

  std::string name;
  int count{};
};

struct NCData {
  NameCountList ncList;
};
using NCDataList = std::vector<NCData>;

struct SourceData;
using SourceList = std::vector<SourceData>;
using SourceCRef = std::reference_wrapper<const SourceData>;
using SourceCRefList = std::vector<SourceCRef>;

struct SourceData : SourceCompatibilityData {
  SourceData() = default;
  SourceData(NameCountList&& primaryNameSrcList, NameCountList&& ncList,
      UsedSources&& usedSources)
      : SourceCompatibilityData(std::move(usedSources)),
        primaryNameSrcList(std::move(primaryNameSrcList)),
        ncList(std::move(ncList)) {}

  // constructor for list.emplace
  // used_sources is const-ref because it doesn't benefit from move
  SourceData(const UsedSources& used_sources,
      NameCountList&& primary_name_src_list, NameCountList&& nc_list,
      std::set<std::string>&& nc_names)
      : SourceCompatibilityData(used_sources),
        primaryNameSrcList(std::move(primary_name_src_list)),
        ncList(std::move(nc_list)), nc_names(std::move(nc_names)) {}

  // copy assign allowed for now for precompute.mergeAllCompatibleXorSources
  SourceData(const SourceData&) = default;
  SourceData& operator=(const SourceData&) = default;
  SourceData(SourceData&&) = default;
  SourceData& operator=(SourceData&&) = default;

  enum class AddLists {
    Yes,
    No,
    Only
  };

  bool addCompoundSource(const SourceData& src,
      AddLists add_lists = AddLists::Yes, bool check_variations = true) {
    using enum AddLists;
    if (add_lists != Only) {
      if (!isXorCompatibleWith(src, check_variations)) {
        return false;
      }
      mergeInPlace(src);
    }
    if (add_lists == Yes || add_lists == Only) {
      primaryNameSrcList.insert(primaryNameSrcList.end(),
          src.primaryNameSrcList.begin(), src.primaryNameSrcList.end());
      ncList.insert(ncList.end(), src.ncList.begin(), src.ncList.end());
    }
    return true;
  }

  bool addPrimaryNameSrc(const NameCount& nc, int primary_src,
      AddLists add_lists = AddLists::Yes) {
    using enum AddLists;
    if ((add_lists != Only) && !addSource(primary_src, true)) {
      return false;
    }
    if (add_lists == Yes || add_lists == Only) {
      primaryNameSrcList.emplace_back(nc.name, primary_src);
      ncList.emplace_back(nc.name, nc.count);
    }
    return true;
  }

  auto merge_nc_name(const std::string& name) {
    return nc_names.insert(name).second;
  }

  auto merge_nc_names(
      const std::set<std::string>& from_names, bool allow_duplicates = false) {
    bool duplicate{};
    for (const auto& name : from_names) {
      if (!merge_nc_name(name)) {
        if (!allow_duplicates) return false;
        duplicate = true;
      }
    }
    return !duplicate;
  }

  static void dumpList(const SourceList& src_list) {
    for (const auto& src : src_list) {
      std::cerr << " " << NameCount::listToString(src.ncList) << " - "
                << NameCount::listToString(src.primaryNameSrcList) << std::endl;
    }
  }

  static void dumpList(const SourceCRefList& src_cref_list) {
    // TODO: auto& or just auto here?
    for (const auto src_cref : src_cref_list) {
      std::cerr << " " << NameCount::listToString(src_cref.get().ncList)
                << " - "
                << NameCount::listToString(src_cref.get().primaryNameSrcList)
                << std::endl;
    }
  }

  NameCountList primaryNameSrcList;
  NameCountList ncList;
  std::set<std::string> nc_names;
};

using SourceListCRef = std::reference_wrapper<const SourceList>;
using SourceListCRefList = std::vector<SourceListCRef>;
using SourceListMap = std::unordered_map<std::string, SourceList>;

using XorSource = SourceData;
using XorSourceList = std::vector<XorSource>;

  /*
// TODO comment/rename
// These are precomputed on xorSourceList, to identify only those sources
// which share the same per-sentence variation.
// One list of indices per variation, plus '-1' (no) variation.
// indices are offset by 1; variation -1 is index 0.
// one vector of IndexLists per sentence
using SentenceVariationIndices =
    std::array<std::vector<IndexList>, kNumSentences>;
using FatSentenceVariationIndices =
    std::array<std::vector<FatIndexList>, kNumSentences>;
  */

struct MergedSources : SourceCompatibilityData {
  MergedSources() = default;
  MergedSources(const MergedSources&) = default;  // allow, dangerous?
  MergedSources& operator=(const MergedSources&) = delete;
  MergedSources(MergedSources&&) = default;
  MergedSources& operator=(MergedSources&&) = default;

  // copy from SourceData
  MergedSources(const SourceData& source)
      : SourceCompatibilityData(source.usedSources),
        sourceCRefList(SourceCRefList{SourceCRef{source}}) {}

  SourceCRefList sourceCRefList;
};

using MergedSourcesList = std::vector<MergedSources>;
using StringList = std::vector<std::string>;

// functions

inline constexpr void assert_valid(const SourceList& src_list) {
  for (const auto& src : src_list) {
    src.assert_valid();
  }
}

#if 0
inline std::vector<SourceCompatibilityData> makeCompatibleSources(
    const SourceList& sources) {
  std::vector<SourceCompatibilityData> compat_sources;
  for (const auto& src : sources) {
    compat_sources.push_back(src);
  }
  return compat_sources;
}
#endif

}  // namespace cm

#endif // INCLUDE_COMBO_MAKER_H
