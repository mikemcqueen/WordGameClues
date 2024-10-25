#pragma once

#include <cassert>
#include <cstring>
#include "variations.h"

namespace cm {

struct SourceDescriptor {
  SourceDescriptor() = default;

  SourceDescriptor(int sentence, int bit_pos, variation_index_t variation)
      : sentence{static_cast<int8_t>(sentence)},
        bit_pos{static_cast<int8_t>(bit_pos)}, variation{variation} {
    validate();
  }

  constexpr void dump() const {
    printf("sentence %d, variation %d, bit_pos %d\n", (int)sentence,
        (int)variation, (int)bit_pos);
  }

  std::string toString() const {
    char buf[32];
    snprintf(buf, sizeof(buf), "s%d v%d b%d", (int)sentence, (int)variation,
        (int)bit_pos);
    return buf;
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

// I think this was bad juju:
//  using SourceDescriptorPair = std::pair<SourceDescriptor, SourceDescriptor>;
struct SourceDescriptorPair {
  SourceDescriptor first;
  SourceDescriptor second;
};

}  // namespace cm
