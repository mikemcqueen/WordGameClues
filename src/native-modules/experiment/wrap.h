#ifndef INCLUDE_WRAP_H
#define INCLUDE_WRAP_H

#include <unordered_set>
#include <string_view>
#include <vector>
#include <napi.h>
#include "combo-maker.h"
#include "candidates.h"
#include "filter-types.h"

namespace cm {
  Napi::Object wrap(Napi::Env& env, const NameCount& nc); 
  Napi::Array wrap(Napi::Env& env, const std::vector<NameCount>& ncList);

  Napi::Object wrap(Napi::Env& env, const XorSource& xorSource);
  Napi::Array wrap(Napi::Env& env, const XorSourceList& xorSourceList);

  Napi::Array wrap(Napi::Env& env, const filter_result_t& filter_result);

  Napi::Object wrap(Napi::Env& env, const PerfData& perf);
  Napi::Object wrap(Napi::Env& env, const CandidateStats& cs);
} // namespace cm

#endif  // INCLUDE_WRAP_H
