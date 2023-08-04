#ifndef include_wrap_h
#define include_wrap_h

#include <unordered_set>
#include <vector>
#include <napi.h>
#include "combo-maker.h"
#include "candidates.h"

namespace cm {
  Napi::Object wrap(Napi::Env& env, const NameCount& nc); 
  Napi::Array wrap(Napi::Env& env, const std::vector<NameCount>& ncList); 

  Napi::Object wrap(Napi::Env& env, const XorSource& xorSource);
  Napi::Array wrap(Napi::Env& env, const XorSourceList& xorSourceList);

#if 0
  Napi::Object wrap(Napi::Env& env, const SourceData& source);
  Napi::Array wrap(Napi::Env& env, const SourceList& sourceList);
  
  Napi::Array wrap(Napi::Env& env, const MergedSourcesList& mergedSourcesList);
#endif
  
  Napi::Object wrap(Napi::Env& env, const PerfData& perf);
  Napi::Object wrap(Napi::Env& env, const CandidateStats& cs);

  Napi::Array wrap(Napi::Env& env, const filter_result_t& filter_result);
} // namespace cm

#endif // include_wrap_h
