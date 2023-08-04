#include <chrono>
#include "wrap.h"

using namespace Napi;

namespace cm {

Object wrap(Env& env, const NameCount& nc) {
  Object jsObj = Object::New(env);
  jsObj.Set("name", String::New(env, nc.name));
  jsObj.Set("count", Number::New(env, nc.count));
  return jsObj;
}

Array wrap(Env& env, const std::vector<NameCount>& ncList) {
  Array jsList = Array::New(env, ncList.size());
  for (size_t i{}; i < ncList.size(); ++i) {
    jsList.Set(i, wrap(env, ncList[i]));
  }
  return jsList;
}

Array wrap(Env& env, const std::vector<std::string>& strList) {
  Array jsList = Array::New(env, strList.size());
  for (size_t i{}; i < strList.size(); ++i) {
    jsList.Set(i, String::New(env, strList[i]));
  }
  return jsList;
}

Object wrap(Env& env, const XorSource& xorSource) {
  Object jsObj = Object::New(env);
  jsObj.Set("primaryNameSrcList", wrap(env, xorSource.primaryNameSrcList));
  jsObj.Set("ncList", wrap(env, xorSource.ncList));
  return jsObj;
}

Array wrap(Env& env, const XorSourceList& xorSourceList) {
  using namespace std::chrono;

  auto t0 = high_resolution_clock::now();

  Array jsList = Array::New(env, xorSourceList.size());
  for (size_t i{}; i < xorSourceList.size(); ++i) {
    jsList.Set(i, wrap(env, xorSourceList[i]));
  }

  auto t1 = high_resolution_clock::now();
  auto d_wrap = duration_cast<milliseconds>(t1 - t0).count();

  std::cerr << " wrap xor sources (" << cm::PCD.xorSourceList.size() << ")"
            << " - " << d_wrap << "ms" << std::endl;

  return jsList;
}

Napi::Object wrap(Napi::Env& env, const PerfData& perf) {
  Object jsObj = Object::New(env);
  jsObj.Set("calls", Number::New(env, perf.calls));
  jsObj.Set("comps", Number::New(env, perf.comps));
  jsObj.Set("compat", Number::New(env, perf.compat));
  jsObj.Set("range_calls", Number::New(env, perf.range_calls));
  jsObj.Set("ss_attempt", Number::New(env, perf.ss_attempt));
  jsObj.Set("ss_fail", Number::New(env, perf.ss_fail));
  jsObj.Set("full", Number::New(env, perf.full));
  return jsObj;
}

Napi::Object wrap(Napi::Env& env, const CandidateStats& cs) {
  Object jsObj = Object::New(env);
  jsObj.Set("sum", Number::New(env, cs.sum));
  jsObj.Set("sourceLists", Number::New(env, cs.sourceLists));
  jsObj.Set("totalSources", Number::New(env, cs.totalSources));
  jsObj.Set("comboMapIndices", Number::New(env, cs.comboMapIndices));
  jsObj.Set("totalCombos", Number::New(env, cs.totalCombos));
  return jsObj;
};

// unordered_set<std::string>
Array wrap(Env& env, const filter_result_t& filter_result) {
  Array jsList = Array::New(env, filter_result.size());
  int i{};
  for (const auto& combo : filter_result) {
    jsList.Set(i++, String::New(env, combo));
  }
  return jsList;
}

#if 0
Object wrap(Env& env, const SourceData& source) {
  Object jsObj = Object::New(env);
  jsObj.Set("primaryNameSrcList", wrap(env, xorSource.primaryNameSrcList));
  jsObj.Set("ncList", wrap(env, xorSource.ncList));
  return jsObj;
  auto jsObj = wrap(env, (const SourceBase&)source);
  jsObj.Set("sourceNcCsvList", wrap(env, source.sourceNcCsvList));
  return jsObj;
}

Array wrap(Env& env, const SourceList& sourceList) {
  Array jsList = Array::New(env, sourceList.size());
  for (size_t i{}; i < sourceList.size(); ++i) {
    jsList.Set(i, wrap(env, sourceList[i]));
  }
  return jsList;
}

Object wrapMergedSource(Env& env, const SourceCRefList& sourceCRefList) {
  Object jsObj = Object::New(env);
  Array jsPnsl = Array::New(env);
  Array jsNcl = Array::New(env);
  //Array jsSncl = Array::New(env);
  for (const auto sourceCRef : sourceCRefList) {
    const auto& source = sourceCRef.get();
    for (const auto& nc : source.primaryNameSrcList) {
      jsPnsl.Set(jsPnsl.Length(), wrap(env, nc));
    }
    for (const auto& nc : source.ncList) {
      jsNcl.Set(jsNcl.Length(), wrap(env, nc));
    }
    /*
    for (const auto& str : source.sourceNcCsvList) {
      jsSncl.Set(jsSncl.Length(), String::New(env, str));
    }
    */
  }
  jsObj.Set("primaryNameSrcList", jsPnsl);
  jsObj.Set("ncList", jsNcl);
  //jsObj.Set("sourceNcCsvList", jsSncl);
  return jsObj;
}

Array wrap(Env& env, const MergedSourcesList& mergedSourcesList) {
  Array jsList = Array::New(env, mergedSourcesList.size());
  for (size_t i{}; i < mergedSourcesList.size(); ++i) {
    jsList.Set(i, wrapMergedSource(env, mergedSourcesList[i].sourceCRefList));
  }
  return jsList;
}
#endif

} // namespace cm
