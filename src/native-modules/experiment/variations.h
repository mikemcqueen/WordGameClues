#pragma once
#include <array>
#include <unordered_set>
#include <vector>
#include "cm-constants.h"

namespace cm {

using variation_index_t = int16_t;

using Variations = std::array<variation_index_t, kNumSentences>;
using VariationsList = std::vector<Variations>;
using VariationsSet = std::unordered_set<Variations>;

}
