#ifndef INCLUDE_WRAP_H
#define INCLUDE_WRAP_H

#include <set>
#include <string>
#include <unordered_set>
#include <vector>
#include <napi.h>
#include "combo-maker.h"
#include "candidates.h"
#include "clue-manager.h"

namespace cm {

Napi::Object wrap(Napi::Env& env, const NameCount& nc);
Napi::Array wrap(Napi::Env& env, const std::vector<NameCount>& ncList);

Napi::Object wrap(Napi::Env& env, const XorSource& xorSource);
Napi::Array wrap(Napi::Env& env, const XorSourceList& xorSourceList);
Napi::Array wrap(Napi::Env& env,
  std::vector<clue_manager::KnownSourceMapValueCRef> cref_entries);

// TODO: combine two or three of these with templated forward delcarations?
Napi::Array wrap(Napi::Env& env, const std::unordered_set<std::string>& string_set);
Napi::Array wrap(Napi::Env& env, const std::vector<std::string>& strList);
Napi::Array wrap(Napi::Env& env, const std::set<int>& values);

}  // namespace cm

#endif  // INCLUDE_WRAP_H
