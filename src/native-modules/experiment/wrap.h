#ifndef INCLUDE_WRAP_H
#define INCLUDE_WRAP_H

#include <set>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <napi.h>
#include "combo-maker.h"
#include "candidates.h"
#include "clue-manager.h"

namespace cm {

Napi::Object wrap(Napi::Env& env, const NameCount& nc);
Napi::Array wrap(Napi::Env& env, const NameCountList& ncList);
Napi::Object wrap(
    Napi::Env& env, const std::unordered_map<std::string, NameCountList>& map);

Napi::Object wrap(Napi::Env& env, const XorSource& xorSource);
Napi::Array wrap(Napi::Env& env, const XorSourceList& xorSourceList);
Napi::Array wrap(Napi::Env& env,
    std::vector<clue_manager::KnownSourceMapValueCRef> cref_entries);

// combine two or three of these with templated forward delcarations?
// TODO: combine int one as well
/*
Napi::Array wrap(Napi::Env& env, const std::unordered_set<std::string>&
string_set); Napi::Array wrap(Napi::Env& env, const std::vector<std::string>&
strList);
*/
template <class T, template <class> class C>
  requires std::is_same_v<T, std::string>  // || is_integral_t<T>
Napi::Array wrap(Napi::Env& env, const C<T>& container) {
  Napi::Array jsList = Napi::Array::New(env, container.size());
  for (int i{}; const auto& elem : container) {
    if constexpr (std::is_same_v<T, std::string>) {
      jsList.Set(i++, Napi::String::New(env, elem));
    }
  }
  return jsList;
}

Napi::Array wrap(Napi::Env& env, const std::set<int>& values);

}  // namespace cm

#endif  // INCLUDE_WRAP_H
