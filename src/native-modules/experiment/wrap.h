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

  Napi::Object wrap(Napi::Env& env, const XorSource& xorSource,
    std::string_view primaryNameSrcList = "primaryNameSrcList");
  Napi::Array wrap(Napi::Env& env, const XorSourceList& xorSourceList,
    std::string_view primaryNameSrcList = "primaryNameSrcList");

#if 0
  Napi::Object wrap(Napi::Env& env, const SourceData& source);
  Napi::Array wrap(Napi::Env& env, const SourceList& sourceList);
  
  Napi::Array wrap(Napi::Env& env, const MergedSourcesList& mergedSourcesList);
#endif
  
  Napi::Object wrap(Napi::Env& env, const PerfData& perf);
  Napi::Object wrap(Napi::Env& env, const CandidateStats& cs);

  Napi::Array wrap(Napi::Env& env, const filter_result_t& filter_result);
} // namespace cm

#endif  // INCLUDE_WRAP_H
