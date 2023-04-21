#include <vector>
#include <memory>
#include <napi.h>
#include <string>
#include <iostream>
#include <unordered_map>
#include "greeting.h"
#include "combo-maker.h"
#include "dump.h"
#include "wrap.h"

using namespace Napi;

Napi::String greetHello(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();
  std::string result = helloUser("John");
  return Napi::String::New(env, result);
}

std::vector<std::string> makeStringList(Env& env, const Napi::Array& jsList) {
  std::vector<std::string> list{};
  for (auto i = 0u; i < jsList.Length(); ++i) {
    if (!jsList[i].IsString()) {
      Napi::TypeError::New(env, "makeStringList: non-string element").ThrowAsJavaScriptException();
      return {};
    }
    list.emplace_back(std::move(jsList[i].As<String>().Utf8Value()));
  }
  return list;
}

cm::NameCount makeNameCount(Env& env, const Napi::Object& jsObject) {
  auto jsName = jsObject.Get("name");
  auto jsCount = jsObject.Get("count");
  if (!jsName.IsString() || !jsCount.IsNumber()) {
    Napi::TypeError::New(env, "makeNameCount: invalid arguments").ThrowAsJavaScriptException();
    return {};
  }
  auto name = jsName.As<String>().Utf8Value();
  const int count = (int)jsCount.As<Number>().Int32Value();
  return cm::NameCount(std::move(name), count);
}

cm::NameCountList makeNameCountList(Env& env, const Napi::Array& jsList) {
  cm::NameCountList ncList{};
  for (auto i = 0u; i < jsList.Length(); ++i) {
    if (!jsList[i].IsObject()) {
      Napi::TypeError::New(env, "makeNameCountList: non-object element").ThrowAsJavaScriptException();
      return {};
    }
    ncList.emplace_back(std::move(makeNameCount(env, jsList[i].As<Object>())));
  }
  return ncList;
}

cm::SourceData makeSourceData(Env& env, const Napi::Object& jsSourceData) {
  auto jsPrimaryNameSrcList = jsSourceData.Get("primaryNameSrcList");
  auto jsSourceNcCsvList = jsSourceData.Get("sourceNcCsvList");
  auto jsNcList = jsSourceData.Get("ncList");
  if (!jsPrimaryNameSrcList.IsArray() || !jsSourceNcCsvList.IsArray() || !jsNcList.IsArray()) {
    Napi::TypeError::New(env, "makeSourceData: invalid arguments").ThrowAsJavaScriptException();
    return {};
  }
  auto primaryNameSrcList = makeNameCountList(env, jsPrimaryNameSrcList.As<Array>());
  auto primarySrcBits = cm::NameCount::listToSourceBits(primaryNameSrcList);
  auto sourceNcCsvList = makeStringList(env, jsSourceNcCsvList.As<Array>());
  auto ncList = makeNameCountList(env, jsNcList.As<Array>());
  return cm::SourceData(std::move(primaryNameSrcList), std::move(primarySrcBits),
			std::move(ncList), std::move(sourceNcCsvList));
}

cm::SourceList makeSourceList(Napi::Env& env, const Napi::Array& jsList) {
  cm::SourceList sourceList{};
  for (auto i = 0u; i < jsList.Length(); ++i) {
    if (!jsList[i].IsObject()) {
      Napi::TypeError::New(env, "makeSourceList: non-object element").ThrowAsJavaScriptException();
      return {};
    }
    sourceList.emplace_back(std::move(makeSourceData(env, jsList[i].As<Object>())));
  }
  return sourceList;
}

cm::NCData makeNcData(Napi::Env& env, const Napi::Object& jsObject) {
  auto jsNcList = jsObject.Get("ncList");
  if (!jsNcList.IsArray()) {
    Napi::TypeError::New(env, "makeNcData: ncList is non-array type").ThrowAsJavaScriptException();
    return {};
  }
  return { makeNameCountList(env, jsNcList.As<Array>()) };
}

cm::NCDataList makeNcDataList(Napi::Env& env, const Napi::Array& jsList) {
  cm::NCDataList list;
  for (auto i = 0u; i < jsList.Length(); ++i) {
    if (!jsList[i].IsObject()) {
      Napi::TypeError::New(env, "makeNcDataList: element is non-object type")
	.ThrowAsJavaScriptException();
      return {};
    }
    list.emplace_back(std::move(makeNcData(env, jsList[i].As<Object>())));
  }
  return list;
}

std::vector<cm::NCDataList> makeNcDataLists(Napi::Env& env, const Napi::Array& jsList) {
  std::vector<cm::NCDataList> lists;  
  for (auto i = 0u; i < jsList.Length(); ++i) {
    if (!jsList[i].IsArray()) {
      Napi::TypeError::New(env, "makeNcDataLists: element is non-array type")
	.ThrowAsJavaScriptException();
      return {};
    }
    lists.emplace_back(std::move(makeNcDataList(env, jsList[i].As<Array>())));
  }
  return lists;
}

cm::SourceListMap makeSourceListMap(Napi::Env& env, const Napi::Array& jsList) {
  cm::SourceListMap map{};
  for (auto i = 0u; i < jsList.Length(); ++i) {
    if (!jsList[i].IsArray()) {
      Napi::TypeError::New(env, "makeSourceListMap: mapEntry is non-array type")
	.ThrowAsJavaScriptException();
      return {};
    }
    const auto tuple = jsList[i].As<Array>();
    if (!tuple[0u].IsString() || !tuple[1u].IsArray()) {
      Napi::TypeError::New(env, "makeSourceListMap: invalid mapEntry key/value type")
	.ThrowAsJavaScriptException();
      return {};
    }
    const auto key = tuple[0u].As<String>().Utf8Value();
    auto sourceList = makeSourceList(env, tuple[1u].As<Array>());
    map.emplace(std::move(key), std::move(sourceList));
  }
  return map;
}

// FromSourceList
cm::SourceBits makeSourceBits(Napi::Env& env, const Napi::Array& jsList) {
  cm::SourceBits sourceBits{};
  for (auto i = 0u; i < jsList.Length(); ++i) {
    if (!jsList[i].IsObject()) {
      Napi::TypeError::New(env, "makeSourceBits: non-object element").ThrowAsJavaScriptException();
      return {};
    }
    const auto jsPnsl = jsList[i].As<Object>().Get("primaryNameSrcList").As<Array>();
    for (auto j = 0u; j < jsPnsl.Length(); ++j) {
      const auto jsNc = jsPnsl[j].As<Object>();
      sourceBits.set(jsNc.Get("count").As<Number>().Int32Value());
    }
  }
  return sourceBits;
}

// FromMergedSourcesList
cm::SourceBitsList makeSourceBitsList(Napi::Env& env, const Napi::Array& jsList) {
  cm::SourceBitsList sourceBitsList{};
  for (auto i = 0u; i < jsList.Length(); ++i) {
    if (!jsList[i].IsObject()) {
      Napi::TypeError::New(env, "makeSourceBitsList: non-object element").ThrowAsJavaScriptException();
      return {};
    }
    sourceBitsList.emplace_back(std::move(makeSourceBits(env,
        jsList[i].As<Object>().Get("sourceList").As<Array>())));
  }
  return sourceBitsList;
}

cm::OrSourceData makeOrSource(Napi::Env& env, const Napi::Object& jsObject) {
  cm::OrSourceData orSource;
  orSource.source = std::move(makeSourceData(env, jsObject["source"].As<Object>()));
  orSource.xorCompatible = jsObject["xorCompatible"].As<Boolean>();
  orSource.andCompatible = jsObject["andCompatible"].As<Boolean>();
  return orSource;
}

cm::OrSourceList makeOrSourceList(Napi::Env& env, const Napi::Array& jsList) {
  cm::OrSourceList orSourceList{};
  for (auto i = 0u; i < jsList.Length(); ++i) {
    if (!jsList[i].IsObject()) {
      Napi::TypeError::New(env, "makeOrSourceList: non-object element").ThrowAsJavaScriptException();
      return {};
    }
    orSourceList.emplace_back(std::move(makeOrSource(env, jsList[i].As<Object>())));
  }
  return orSourceList;
}

cm::OrArgData makeOrArgData(Napi::Env& env, const Napi::Object& jsObject) {
  cm::OrArgData orArgData{};
  orArgData.orSourceList = std::move(makeOrSourceList(env, jsObject["orSourceList"].As<Array>()));
  orArgData.compatible = jsObject["compatible"].As<Boolean>();
  return orArgData;
}

cm::OrArgDataList makeOrArgDataList(Napi::Env& env, const Napi::Array& jsList) {
  cm::OrArgDataList orArgDataList{};
  for (auto i = 0u; i < jsList.Length(); ++i) {
    if (!jsList[i].IsObject()) {
      Napi::TypeError::New(env, "makeOrArgDataList: non-object element").ThrowAsJavaScriptException();
      return {};
    }
    orArgDataList.emplace_back(std::move(makeOrArgData(env, jsList[i].As<Object>())));
  }
  return orArgDataList;
}

//
// unused? i think
//
Value buildSourceListsForUseNcData(const CallbackInfo& info) {
  Env env = info.Env();
  if (!info[0].IsArray() || !info[1].IsArray()) {
      Napi::TypeError::New(env, "buildSourceListsForUseNcData: non-array parameter")
	.ThrowAsJavaScriptException();
      return env.Null();
  }
  auto ncDataLists = makeNcDataLists(env, info[0].As<Array>());
  //dump(ncDataLists);
  //std::cout << "------------" << std::endl;
  auto sourceListMap = makeSourceListMap(env, info[1].As<Array>());
  //dump(sourceListMap);
  cm::buildSourceListsForUseNcData(ncDataLists, sourceListMap);
  return env.Null();
}

namespace cm {
  extern PreComputedData PCD;
}

//
// mergeAllCompatibleSources
//
Value mergeAllCompatibleSources(const CallbackInfo& info) {
  Env env = info.Env();
  if (!info[0].IsArray()) {
      Napi::TypeError::New(env, "mergeAllCompatibleSources: non-array parameter")
	.ThrowAsJavaScriptException();
      return env.Null();
  }
  auto ncList = makeNameCountList(env, info[0].As<Array>());
  auto mergedSourcesList = cm::mergeAllCompatibleSources(ncList, cm::PCD.sourceListMap);
  return cm::wrap(env, mergedSourcesList);
}

//
// mergeCompatibleXorSourceCombinations
//
Value mergeCompatibleXorSourceCombinations(const CallbackInfo& info) {
  using namespace std::chrono;

  Env env = info.Env();
  if (!info[0].IsArray() || !info[1].IsArray()) {
      Napi::TypeError::New(env, "mergeCompatibleXorSourceCombinations: non-array parameter")
	.ThrowAsJavaScriptException();
      return env.Null();
  }

  auto unwrap0 = high_resolution_clock::now();

  auto ncDataLists = makeNcDataLists(env, info[0].As<Array>());
  cm::PCD.sourceListMap = makeSourceListMap(env, info[1].As<Array>());

  auto unwrap1 = high_resolution_clock::now();
  auto d_unwrap = duration_cast<milliseconds>(unwrap1 - unwrap0).count();
  cerr << " native unwrap: " << d_unwrap << "ms" << endl;

  //--
    
  auto build0 = high_resolution_clock::now();

  auto sourceLists = cm::buildSourceListsForUseNcData(ncDataLists, cm::PCD.sourceListMap);

  auto build1 = high_resolution_clock::now();
  auto d_build = duration_cast<milliseconds>(build1 - build0).count();
  cerr << " native build: " << d_build << "ms" << endl;

  //--

  auto merge0 = high_resolution_clock::now();

  cm::PCD.xorSourceList = cm::mergeCompatibleXorSourceCombinations(sourceLists);

  auto merge1 = high_resolution_clock::now();
  auto d_merge = duration_cast<milliseconds>(merge1 - merge0).count();
  cerr << " native merge: " << d_merge << "ms" << endl;

  //--

  return cm::wrap(env, cm::PCD.xorSourceList);
}

//
// isAnySourceCompatibleWithUseSources
//
Value isAnySourceCompatibleWithUseSources(const CallbackInfo& info) {
  using namespace std::chrono;

  Env env = info.Env();
  if (!info[0].IsArray()) {
    TypeError::New(env, "isAnySourceCompatibleWithUseSources: non-array parameter")
      .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto sourceBitsList = makeSourceBitsList(env, info[0].As<Array>());
  auto flag = info[1].As<Boolean>();
  bool compatible = cm::isAnySourceCompatibleWithUseSources(sourceBitsList, flag);
  return Boolean::New(env, compatible);
}

//
// setOrArgDataList
//
Value setOrArgDataList(const CallbackInfo& info) {
  using namespace std::chrono;

  Env env = info.Env();
  if (!info[0].IsArray()) {
      Napi::TypeError::New(env, "mergeCompatibleXorSourceCombinations: non-array parameter")
	.ThrowAsJavaScriptException();
      return env.Null();
  }
  cm::PCD.orArgDataList = makeOrArgDataList(env, info[0].As<Array>());
  return env.Null();
}

Object Init(Env env, Object exports) {
  exports["greetHello"] = Function::New(env, greetHello);

  //  exports["buildSourceListsForUseNcData"] = Function::New(env, buildSourceListsForUseNcData);
  exports["mergeAllCompatibleSources"] = Function::New(env, mergeAllCompatibleSources);
  exports["mergeCompatibleXorSourceCombinations"] = Function::New(env, mergeCompatibleXorSourceCombinations);
  exports["isAnySourceCompatibleWithUseSources"] = Function::New(env, isAnySourceCompatibleWithUseSources);
  exports["setOrArgDataList"] = Function::New(env, setOrArgDataList);

  return exports;
}

NODE_API_MODULE(experiment, Init)
