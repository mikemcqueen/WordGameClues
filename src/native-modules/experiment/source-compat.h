#pragma once

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <unordered_set>
#include <utility>
#include <vector>
#include "cuda-string.h"
#include "mmebitset.h"
#include "primary-source-id.h"
#include "source-desc.h"
#include "variations.h"

namespace cm {

template<typename T, size_t N>
constexpr auto make_array(T value) -> std::array<T, N> {
  std::array<T, N> a{};
  for (auto& e : a) {
    e = value;
  }
  return a;
}

struct UsedSources {
  //  using variation_index_t = int16_t;

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

  constexpr void setVariation(int sentence, variation_index_t value) {
    variations.at(sentence - 1) = value;
  }

  constexpr bool hasVariation(int sentence) const {
    return getVariation(sentence) > static_cast<variation_index_t>(-1);
  }

private:
  constexpr auto addVariations(const UsedSources& other) {
    for (int sentence{1}; sentence <= kNumSentences; ++sentence) {
      if (!other.hasVariation(sentence)) continue;
      // ensure variations for this sentence are compatible
      if (hasVariation(sentence)) {
        if (getVariation(sentence) != other.getVariation(sentence)) {
          printf("Variation Mismatch sentence %d, this %hd, other %hd\n",
              sentence, getVariation(sentence), other.getVariation(sentence));
          return false;
        }
        //assert(getVariation(sentence) == other.getVariation(sentence));
      } else {
        setVariation(sentence, other.getVariation(sentence));
      }
    }
    return true;
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

  static constexpr auto merge_one_variation(Variations& to, PrimarySourceId src) {
    return merge_one_variation(to, src.sentence(), src.variation());
  }

  // TODO: maybe should have a fast version of this for CUDA
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

  // OR == XOR || AND
  // old code && (getBits().intersects(other.getBits())
  constexpr auto isOrCompatibleWith(const UsedSources& other) const {
    return are_variations_compatible(variations, other.variations)
        && (!getBits().intersects(other.getBits())
            || getBits().is_subset_of(other.getBits()));
  }

  bool addSource(PrimarySourceId src, bool nothrow = false) {
    auto sentence = src.sentence();
    assert(sentence > 0);
    auto variation = src.variation();
    if (hasVariation(sentence) && (getVariation(sentence) != variation)) {
      if (nothrow) { return false; }
      std::cerr << "sentence " << sentence
                << " variation: " << getVariation(sentence)
                << ", src variation: " << variation << std::endl;
      assert(false && "addSource() variation mismatch");
    }
    assert(src.index() < kMaxSourcesPerSentence);

    // variation
    setVariation(sentence, variation);

    // bits
    auto bit_pos = src.index() + getFirstBitIndex(sentence);
    if (bits.test(bit_pos)) {
      if (nothrow) { return false; }
      assert(0 && "addSource: bit already set");
    }
    bits.set(bit_pos);
    return true;
  }

  constexpr auto mergeInPlace(const UsedSources& other) {
    // merge bits
    getBits() |= other.getBits();
    // merge variations.
    return addVariations(other);
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

  // This works because it is only used (currently) for incompatible
  // sources resulting from the synchronous `Sum == 2` filter kernel
  // execution. That implies always exactly one primary source.
  SourceDescriptor get_source_descriptor() const {
    for (int i{}; i < bits.wc(); ++i) {
      auto word = bits.word(i);
      if (word) {
        auto bit_pos = int(lrint(log2(word)));
        return {i + 1, bit_pos, getVariation(i + 1)};
      }
    }
    assert(0 && "source not found");
    return {};
  }

  // TODO: remove, probably.
  // This works because it is only used (currently) for incompatible
  // sources resulting from the synchronous `Sum == 2` filter kernel
  // execution. That implies always exactly two primary sources.
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

  // TODO: remove
  constexpr auto has(SourceDescriptorPair sd_pair) const {
    return has(sd_pair.first) && has(sd_pair.second);
  }

  void reset() {
    bits.reset();
    std::fill(variations.begin(), variations.end(), NoVariation);
  }

  SourceBits bits{};
  Variations variations = make_array<variation_index_t, kNumSentences>(-1);
};  // UsedSources

struct alignas(16) SourceCompatibilityData {
  SourceCompatibilityData() = default;
  // copy consruct/assign allowed for now, precompute.mergeAllCompatibleXorSources
  SourceCompatibilityData(const SourceCompatibilityData&) = default;

  SourceCompatibilityData& operator=(const SourceCompatibilityData&) = default;
  SourceCompatibilityData(SourceCompatibilityData&&) = default;
  SourceCompatibilityData& operator=(SourceCompatibilityData&&) = default;

  // conversion from usedSources
  SourceCompatibilityData(const UsedSources& usedSources)
      : usedSources(usedSources) {}

  SourceCompatibilityData(UsedSources&& usedSources)
      : usedSources(std::move(usedSources)) {}

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

  bool addSource(PrimarySourceId src, bool nothrow = false) {
    return usedSources.addSource(src, nothrow);
  }

  constexpr auto mergeInPlace(const SourceCompatibilityData& other) {
    return usedSources.mergeInPlace(other.usedSources);
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
using SourceCompatCRef =
    std::reference_wrapper<const SourceCompatibilityData>;
using SourceCompatCRefList = std::vector<SourceCompatCRef>;

}  // namespace cm
