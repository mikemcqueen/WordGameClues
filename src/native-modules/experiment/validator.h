#ifndef INCLUDE_VALIDATOR_H
#define INCLUDE_VALIDATOR_H

#include <optional>
#include <string>
#include <unordered_set>
#include <vector>
#include "combo-maker.h"
#include "cuda-types.h"
#include "peco.h"

namespace cm::validator {

auto validateSources(const std::string& clue_name,
    const std::vector<std::string>& src_names, int sum,
    bool validate_all) -> SourceList;

void show_validator_durations();

};  // namespace cm::validator

#endif  // INCLUDE_VALIDATOR_H
