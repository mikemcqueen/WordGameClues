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
  for (auto i = 0u; i < ncList.size(); ++i) {
    jsList.Set(i, wrap(env, ncList[i]));
  }
  return jsList;
}

Array wrap(Env& env, const std::vector<std::string>& strList) {
  Array jsList = Array::New(env, strList.size());
  for (auto i = 0u; i < strList.size(); ++i) {
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
  Array jsList = Array::New(env, xorSourceList.size());
  for (auto i = 0u; i < xorSourceList.size(); ++i) {
    jsList.Set(i, wrap(env, xorSourceList[i]));
  }
  return jsList;
}

Object wrap(Env& env, const SourceData& source) {
#if 0
  Object jsObj = Object::New(env);
  jsObj.Set("primaryNameSrcList", wrap(env, xorSource.primaryNameSrcList));
  jsObj.Set("ncList", wrap(env, xorSource.ncList));
  return jsObj;
#endif
  auto jsObj = wrap(env, (const SourceBase&)source);
  jsObj.Set("sourceNcCsvList", wrap(env, source.sourceNcCsvList));
  return jsObj;
}

Array wrap(Env& env, const SourceList& sourceList) {
  Array jsList = Array::New(env, sourceList.size());
  for (auto i = 0u; i < sourceList.size(); ++i) {
    jsList.Set(i, wrap(env, sourceList[i]));
  }
  return jsList;
}

Object wrapMergedSource(Env& env, const SourceCRefList& sourceCRefList) {
  Object jsObj = Object::New(env);
  Array jsPnsl = Array::New(env);
  Array jsNcl = Array::New(env);
  Array jsSncl = Array::New(env);
  for (const auto sourceCRef : sourceCRefList) {
    const auto& source = sourceCRef.get();
    for (const auto& nc : source.primaryNameSrcList) {
      jsPnsl.Set(jsPnsl.Length(), wrap(env, nc));
    }
    for (const auto& nc : source.ncList) {
      jsNcl.Set(jsNcl.Length(), wrap(env, nc));
    }
    for (const auto& str : source.sourceNcCsvList) {
      jsSncl.Set(jsSncl.Length(), String::New(env, str));
    }
  }
  jsObj.Set("primaryNameSrcList", jsPnsl);
  jsObj.Set("ncList", jsNcl);
  jsObj.Set("sourceNcCsvList", jsSncl);
  return jsObj;
}

Array wrap(Env& env, const MergedSourcesList& mergedSourcesList) {
  Array jsList = Array::New(env, mergedSourcesList.size());
  for (auto i = 0u; i < mergedSourcesList.size(); ++i) {
    jsList.Set(i, wrapMergedSource(env, mergedSourcesList[i].sourceCRefList));
  }
  return jsList;
}

} // namespace cm
