#ifndef INCLUDE_FILTER_H
#define INCLUDE_FILTER_H

#include <utility>
#include <vector>
#include "filter-types.h"

namespace cm {

  [[nodiscard]] SourceCompatibilityData* cuda_allocCopyXorSources(
    const XorSourceList& xorSourceList);

  [[nodiscard]] std::pair<device::OrSourceData*, unsigned>
  cuda_allocCopyOrSources(const OrArgList& orArgList);

  [[nodiscard]] auto cuda_allocCopySentenceVariationIndices(
    const SentenceVariationIndices& sentenceVariationIndices)
    -> device::VariationIndices*;

  [[nodiscard]] index_t* cuda_allocCopyXorSourceIndices(
    const std::vector<index_t> xorSourceIndices);
}

#endif  // INCLUDE_FILTER_H
