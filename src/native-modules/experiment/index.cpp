#include <vector>
#include <memory>
#include <napi.h>
#include <string>
#include <iostream>
#include <unordered_map>
#include "greeting.h"
#include "combo-maker.h"
#include "dump.h"

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
  const auto name = jsName.As<String>().Utf8Value();
  const int count = (int)jsCount.As<Number>().Int32Value();
  return { name, count };
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
  auto sourceNcCsvList = makeStringList(env, jsSourceNcCsvList.As<Array>());
  auto ncList = makeNameCountList(env, jsNcList.As<Array>());
  return { primaryNameSrcList, ncList, sourceNcCsvList };
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
    map.emplace(std::make_pair(key, sourceList));
  }
  return map;
}

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

Value mergeCompatibleXorSourceCombinations(const CallbackInfo& info) {
  Env env = info.Env();
  if (!info[0].IsArray() || !info[1].IsArray()) {
      Napi::TypeError::New(env, "mergeCompatibleXorSourceCombinations: non-array parameter")
	.ThrowAsJavaScriptException();
      return env.Null();
  }
  cout << "++unwrap" << endl;
  cout << "  unwrapping ncDataLists" << endl;
  auto ncDataLists = makeNcDataLists(env, info[0].As<Array>());
  cout << "  unwrapping sourceListMap" << endl;
  auto sourceListMap = makeSourceListMap(env, info[1].As<Array>());
  cout << "--unwrap" << endl;

  auto sourceLists = cm::buildSourceListsForUseNcData(ncDataLists, sourceListMap);
  //dump(sourceLists);
  cm::mergeCompatibleXorSourceCombinations(sourceLists);
  return env.Null();
}

Napi::Object Init(Napi::Env env, Napi::Object exports) {
  exports.Set(Napi::String::New(env, "greetHello"),
	      Napi::Function::New(env, greetHello));

  exports.Set(Napi::String::New(env, "buildSourceListsForUseNcData"),
	      Napi::Function::New(env, buildSourceListsForUseNcData));

  exports.Set(Napi::String::New(env, "mergeCompatibleXorSourceCombinations"),
	      Napi::Function::New(env, mergeCompatibleXorSourceCombinations));

  return exports;
}

NODE_API_MODULE(experiment, Init)
