// show-components.h

#ifndef INCLUDE_SHOW_COMPONENTS_H
#define INCLUDE_SHOW_COMPONENTS_H

#include <set>
#include <string>
#include <vector>

namespace cm::components {

auto show(const std::vector<std::string>& name_list,
    const SourceList& xor_src_list) -> std::set<int>;

auto consistency_check(const std::vector<std::string>& name_list,
    const SourceList& xor_src_list) -> bool;

auto consistency_check2(
    const std::vector<std::string>& name_list, int max_sources) -> bool;

}  // namespace cm::components

#endif // INCLUDE_SHOW_COMPONENTS_H
