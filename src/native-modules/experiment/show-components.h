// show-components.h

#ifndef INCLUDE_SHOW_COMPONENTS_H
#define INCLUDE_SHOW_COMPONENTS_H

#include <set>
#include <string>
#include <vector>

namespace cm::show_components {

auto of(const std::vector<std::string>& name_list) -> std::set<int>;

}  // namespace cm::show_components

#endif // INCLUDE_SHOW_COMPONENTS_H
