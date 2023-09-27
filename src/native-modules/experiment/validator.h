#ifndef INCLUDE_VALIDATOR_H
#define INCLUDE_VALIDATOR_H

#include <optional>
#include "combo-maker.h"
#include "cuda-types.h"
#include "peco.h"

namespace validator {

// ugly. make types exist outside of NS, or in "common" or "native" or
// something
using namespace cm;

auto validateSources(const std::string& clue_name,
  const std::vector<std::string>& src_names, int sum, bool validate_all)
  -> SourceList;

};  // namespace validator

#endif  // INCLUDE_VALIDATOR_H
