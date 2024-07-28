#include "unwrap.h"

namespace cm {
  
using namespace Napi;
using namespace cm;
using namespace cm::clue_manager;
//using namespace cm::validator;

std::vector<int> makeIntList(Env& env, const Array& jsList) {
  std::vector<int> int_list{};
  for (size_t i{}; i < jsList.Length(); ++i) {
    if (!jsList[i].IsNumber()) {
      TypeError::New(env, "makeIntList: non-number element")
        .ThrowAsJavaScriptException();
      return {};
    }
    int_list.emplace_back(jsList[i].As<Number>().Int32Value());
  }
  return int_list;
}

IndexList makeIndexList(Env& env, const Array& jsList) {
  IndexList idx_list{};
  for (size_t i{}; i < jsList.Length(); ++i) {
    if (!jsList[i].IsNumber()) {
      TypeError::New(env, "makeIndexList: non-number element")
        .ThrowAsJavaScriptException();
      return {};
    }
    idx_list.emplace_back(jsList[i].As<Number>().Uint32Value());
  }
  return idx_list;
}

std::vector<std::string> makeStringList(Env& env, const Array& jsList) {
  std::vector<std::string> list{};
  for (size_t i{}; i < jsList.Length(); ++i) {
    if (!jsList[i].IsString()) {
      TypeError::New(env, "makeStringList: non-string element")
        .ThrowAsJavaScriptException();
      return {};
    }
    list.emplace_back(std::move(jsList[i].As<String>().Utf8Value()));
  }
  return list;
}

NameCount makeNameCount(Env& env, const Object& jsObject) {
  auto jsName = jsObject.Get("name");
  auto jsCount = jsObject.Get("count");
  if (!jsName.IsString() || !jsCount.IsNumber()) {
    TypeError::New(env, "makeNameCount: invalid arguments")
      .ThrowAsJavaScriptException();
    return NameCount{"error", 1};
  }
  auto name = jsName.As<String>().Utf8Value();
  const int count = (int)jsCount.As<Number>().Int32Value();
  return NameCount{std::move(name), count};
}

NameCountList makeNameCountList(Env& env, const Array& jsList) {
  NameCountList ncList{};
  for (size_t i{}; i < jsList.Length(); ++i) {
    if (!jsList[i].IsObject()) {
      TypeError::New(env, "makeNameCountList: non-object element")
        .ThrowAsJavaScriptException();
      return {};
    }
    ncList.emplace_back(std::move(makeNameCount(env, jsList[i].As<Object>())));
  }
  return ncList;
}

SourceData makeSourceData(Env& env, const Object& jsSourceData,
    std::string_view nameSrcList /*= "primaryNameSrcList"*/) {
  auto jsPrimaryNameSrcList = jsSourceData.Get(nameSrcList.data());
  if (!jsPrimaryNameSrcList.IsArray()) {
    TypeError::New(env, "makeSourceData: primaryNameSrcList is not an array")
      .ThrowAsJavaScriptException();
    return {};
  }
  auto jsNcList = jsSourceData.Get("ncList");
  if (!jsNcList.IsArray()) {
    TypeError::New(env, "makeSourceData: ncList is not an array")
      .ThrowAsJavaScriptException();
    return {};
  }
  // TODO: declare SourceData result; assign result.xxx = std::move(yyy);;
  // return result (no move-all-params constructor required)
  auto primaryNameSrcList =
    makeNameCountList(env, jsPrimaryNameSrcList.As<Array>());
  auto ncList =
    makeNameCountList(env, jsNcList.As<Array>());
  auto usedSources = NameCount::listToUsedSources(primaryNameSrcList);
#if 0
  usedSources.assert_valid();
#endif
  return {
    std::move(primaryNameSrcList), std::move(ncList), std::move(usedSources)};
}

SourceList makeSourceList(Env& env, const Array& jsList,
    std::string_view nameSrcList /*= "primaryNameSrcList"*/) {
  SourceList sourceList{};
  for (size_t i{}; i < jsList.Length(); ++i) {
    if (!jsList[i].IsObject()) {
      TypeError::New(env, "makeSourceList: non-object element")
        .ThrowAsJavaScriptException();
      return {};
    }
    sourceList.emplace_back(
      std::move(makeSourceData(env, jsList[i].As<Object>(), nameSrcList)));
  }
  return sourceList;
}

NCData makeNcData(Env& env, const Object& jsObject) {
  auto jsNcList = jsObject.Get("ncList");
  if (!jsNcList.IsArray()) {
    TypeError::New(env, "makeNcData: ncList is non-array type")
      .ThrowAsJavaScriptException();
    return {};
  }
  return { makeNameCountList(env, jsNcList.As<Array>()) };
}

NCDataList makeNcDataList(Env& env, const Array& jsList) {
  NCDataList list;
  for (size_t i{}; i < jsList.Length(); ++i) {
    if (!jsList[i].IsObject()) {
      TypeError::New(env, "makeNcDataList: element is non-object type")
        .ThrowAsJavaScriptException();
      return {};
    }
    list.emplace_back(std::move(makeNcData(env, jsList[i].As<Object>())));
  }
  return list;
}

std::vector<NCDataList> makeNcDataLists(Env& env, const Array& jsList) {
  std::vector<NCDataList> lists;  
  for (size_t i{}; i < jsList.Length(); ++i) {
    if (!jsList[i].IsArray()) {
      TypeError::New(env, "makeNcDataLists: element is non-array type")
        .ThrowAsJavaScriptException();
      return {};
    }
    lists.emplace_back(std::move(makeNcDataList(env, jsList[i].As<Array>())));
  }
  return lists;
}

NameSourcesMap makeNameSourcesMap(Env& env, const Array& jsList) {
  NameSourcesMap map;
  for (size_t i{}; i < jsList.Length(); ++i) {
    if (!jsList[i].IsArray()) {
      TypeError::New(env, "makeNameSourcesMap: mapEntry is non-array type")
        .ThrowAsJavaScriptException();
      return {};
    }
    const auto tuple = jsList[i].As<Array>();
    if (!tuple[0u].IsString() || !tuple[1u].IsArray()) {
      TypeError::New(
        env, "makeNameSourcesMap: invalid mapEntry key/value type")
        .ThrowAsJavaScriptException();
      return {};
    }
    const auto name = tuple[0u].As<String>().Utf8Value();
    auto sources = makeStringList(env, tuple[1u].As<Array>());
    map.emplace(std::move(name), std::move(sources));
  }
  return map;
}

/*
SourceCompatibilityData makeSourceCompatibilityDataFromSourceData(
    Env& env, const Object& jsSourceData) {
  // TODO: addPnslToCompatData(jsSouceData, compatData);
  SourceCompatibilityData compatData{};
  const auto jsPnsl = jsSourceData.Get("primaryNameSrcList").As<Array>();
  for (size_t i{}; i < jsPnsl.Length(); ++i) {
    const auto count =
      jsPnsl[i].As<Object>().Get("count").As<Number>().Int32Value();
    compatData.addSource(count);
  }
  return compatData;
}

SourceCompatibilityData makeSourceCompatibilityDataFromSourceList(
    Env& env, const Array& jsSourceList) {
  SourceCompatibilityData compatData{};
  for (size_t i{}; i < jsSourceList.Length(); ++i) {
    if (!jsSourceList[i].IsObject()) {
      TypeError::New(env, "makeSourceCompatibilityData: non-object element")
        .ThrowAsJavaScriptException();
      return {};
    }
    // TODO: addPnslToCompatData(jsSourceList[i].As<Object>(), compatData);
    const auto jsPnsl =
      jsSourceList[i].As<Object>().Get("primaryNameSrcList").As<Array>();
    for (size_t j{}; j < jsPnsl.Length(); ++j) {
      const auto count =
        jsPnsl[j].As<Object>().Get("count").As<Number>().Int32Value();
      compatData.addSource(count);
    }
  }
  return compatData;
}

SourceCompatibilityList makeSourceCompatibilityListFromMergedSourcesList(
    Env& env, const Array& jsList) {
  SourceCompatibilityList sourceCompatList{};
  for (size_t i{}; i < jsList.Length(); ++i) {
    if (!jsList[i].IsObject()) {
      TypeError::New(env, "makeSourceCompatibiltyList: non-object element")
        .ThrowAsJavaScriptException();
      return {};
    }
    auto jsSourceList = jsList[i].As<Object>().Get("sourceList").As<Array>();
    SourceCompatibilityData compatData =
      makeSourceCompatibilityDataFromSourceList(env, jsSourceList);
    sourceCompatList.emplace_back(std::move(compatData));
  }
  return sourceCompatList;
}

OrSourceData makeOrSource(Env& env, const Object& jsObject) {
OrSourceData orSource;
orSource.src = std::move(makeSourceCompatibilityDataFromSourceData(
  env, jsObject["source"].As<Object>()));
orSource.is_xor_compat = jsObject["xorCompatible"].As<Boolean>();
//  orSource.and_compat = jsObject["andCompatible"].As<Boolean>();
return orSource;
}

OrSourceList makeOrSourceList(Env& env, const Array& jsList) {
  OrSourceList orSourceList{};
  for (size_t i{}; i < jsList.Length(); ++i) {
    if (!jsList[i].IsObject()) {
      TypeError::New(env, "makeOrSourceList: non-object element")
        .ThrowAsJavaScriptException();
      return {};
    }
    orSourceList.emplace_back(
      std::move(makeOrSource(env, jsList[i].As<Object>())));
  }
  return orSourceList;
}

OrArgData makeOrArgData(Env& env, const Object& jsObject) {
  OrArgData orArgData{};
  orArgData.or_src_list =
    std::move(makeOrSourceList(env, jsObject["orSourceList"].As<Array>()));
  orArgData.compat = jsObject["compatible"].As<Boolean>();
  return orArgData;
}

OrArgList makeOrArgList(Env& env, const Array& jsList) {
  OrArgList orArgList{};
  for (size_t i{}; i < jsList.Length(); ++i) {
    if (!jsList[i].IsObject()) {
      TypeError::New(env, "makeOrArgDataList: non-object element")
        .ThrowAsJavaScriptException();
      return {};
    }
    orArgList.emplace_back(
      std::move(makeOrArgData(env, jsList[i].As<Object>())));
  }
  return orArgList;
}
*/

}  // namespace cm
