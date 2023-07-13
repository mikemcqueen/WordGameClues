#ifndef include_wrap_h
#define include_wrap_h

#include <unordered_set>
#include <vector>
#include <napi.h>
#include "combo-maker.h"

namespace cm {

Napi::Object wrap(Napi::Env& env, const NameCount& nc); 
Napi::Array wrap(Napi::Env& env, const std::vector<NameCount>& ncList); 
  //Napi::Array wrap(Napi::Env& env, const std::unordered_set<int>& set);

Napi::Object wrap(Napi::Env& env, const XorSource& xorSource);
Napi::Array wrap(Napi::Env& env, const XorSourceList& xorSourceList);

Napi::Object wrap(Napi::Env& env, const SourceData& source);
Napi::Array wrap(Napi::Env& env, const SourceList& sourceList);

Napi::Array wrap(Napi::Env& env, const MergedSourcesList& mergedSourcesList);

Napi::Object wrap(Napi::Env& env, const PerfData& perf);

}

#endif // include_wrap_h
