#ifndef INCLUDE_VARIATION_INDICES_H
#define INCLUDE_VARIATION_INDICES_H

#include <array>
#include "filter-types.h"

namespace cm {

namespace device {

// TODO: I need to modify SentenceVariationIndices to use index_t

struct VariationIndices {
  struct SentenceInfo {
    index_t* variation_offsets{};  // num_variations index_t offsets
    index_t* num_src_indices{};    // followed by num_variations index_t counts
    index_t* src_indices{};        // followed by variable # of index_t indices
    index_t num_variations{};
  };

  using Sentences = std::array<SentenceInfo, kNumSentences>;

  index_t* device_data;          // one block of allocated data includes all
                                 // data for  all sentences; pointers in
                                 // SentenceInfo point inside this block.
  Sentences* device_sentences;

  std::vector<index_t> host_data;
  Sentences host_sentences;

  static auto from(const SentenceVariationIndices& sentenceVariationIndices) {

  }
};


}  // namespace device
}  // namespace cm

#endif  // INCLUDE_VARIATION_INDICES_H
