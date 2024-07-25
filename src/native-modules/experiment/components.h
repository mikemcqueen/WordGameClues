// show-components.h

#ifndef INCLUDE_SHOW_COMPONENTS_H
#define INCLUDE_SHOW_COMPONENTS_H

#pragma once
#include <set>
#include <string>
#include <unordered_map>
#include <vector>
#include "combo-maker.h"

namespace cm::components {

auto show(const std::vector<std::string>& name_list,
    const SourceList& xor_src_list) -> std::set<int>;

auto old_consistency_check(const std::vector<std::string>& name_list,
    const SourceList& xor_src_list) -> bool;

void consistency_check(
    std::vector<std::string>&& name_list, int max_sources);

auto get_consistency_check_results()
    -> const std::unordered_map<std::string, NameCountList>&;

}  // namespace cm::components

#endif // INCLUDE_SHOW_COMPONENTS_H
