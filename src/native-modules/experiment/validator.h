#pragma once

#include <optional>
#include <string>
#include <unordered_set>
#include <vector>
#include "cuda-types.h"
#include "peco.h"
#include "source-core.h"

namespace cm::validator {

auto validate_sources(const std::string& clue_name,
    const std::vector<std::string>& src_names, int sum,
    bool validate_all) -> DeferredSourceDataList;

auto is_xor_compatible(const std::vector<std::string>& src_names,
    const std::vector<int>& count_list) -> bool;

void show_validator_durations();

};  // namespace cm::validator
