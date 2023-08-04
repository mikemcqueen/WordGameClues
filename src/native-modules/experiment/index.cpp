#include <cassert>
#include <chrono>
#include <numeric>
#include <iostream>
#include <memory>
#include <napi.h>
#include <string>
#include <unordered_map>
#include <vector>
#include "greeting.h"
#include "combo-maker.h"
#include "candidates.h"
#include "dump.h"
#include "wrap.h"

using namespace Napi;

std::vector<std::string> makeStringList(Env& env, const Napi::Array& jsList) {
  std::vector<std::string> list{};
  for (auto i = 0u; i < jsList.Length(); ++i) {
    if (!jsList[i].IsString()) {
      Napi::TypeError::New(env, "makeStringList: non-string element")
        .ThrowAsJavaScriptException();
      return {};
    }
    list.emplace_back(std::move(jsList[i].As<String>().Utf8Value()));
  }
  return list;
}

/*
cm::UsedSources makeUsedSources(Env& env, const Napi::Array& jsList) {
  cm::UsedSources usedSources{};
  for (auto i = 0u; i < usedSources.size(); ++i) {
    std::int32_t value = 0;
    if (i < jsList.Length()) {
      if (!jsList[i].IsNumber() && !jsList[i].IsUndefined()) {
        Napi::TypeError::New(env, "makeUsedSources: non-(number or undefined) element")
          .ThrowAsJavaScriptException();
        return {};
      }
      if (jsList[i].IsNumber()) {
        value = jsList[i].As<Number>().Int32Value();
      }
    }
    usedSources[i] = value;
  }
  return usedSources;
}
*/

cm::NameCount makeNameCount(Env& env, const Napi::Object& jsObject) {
  auto jsName = jsObject.Get("name");
  auto jsCount = jsObject.Get("count");
  if (!jsName.IsString() || !jsCount.IsNumber()) {
    Napi::TypeError::New(env, "makeNameCount: invalid arguments")
      .ThrowAsJavaScriptException();
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
      Napi::TypeError::New(env, "makeNameCountList: non-object element")
        .ThrowAsJavaScriptException();
      return {};
    }
    ncList.emplace_back(std::move(makeNameCount(env, jsList[i].As<Object>())));
  }
  return ncList;
}

cm::SourceData makeSourceData(Env& env, const Napi::Object& jsSourceData) {
  auto jsPrimaryNameSrcList = jsSourceData.Get("primaryNameSrcList");
  if (!jsPrimaryNameSrcList.IsArray()) {
    Napi::TypeError::New(env, "makeSourceData: primaryNameSrcList is not an array")
      .ThrowAsJavaScriptException();
    return {};
  }
  auto jsNcList = jsSourceData.Get("ncList");
  if (!jsNcList.IsArray()) {
    Napi::TypeError::New(env, "makeSourceData: ncList is not an array")
      .ThrowAsJavaScriptException();
    return {};
  }
  auto jsUsedSources = jsSourceData.Get("usedSources");
  if (!jsUsedSources.IsArray()) {
    Napi::TypeError::New(env, "makeSourceData: usedSources is not an array")
      .ThrowAsJavaScriptException();
    return {};
  }
  // TODO: declare SourceData result; assign result.xxx = std::move(yyy);; return result
  // (no move-all-params constructor required)
  auto primaryNameSrcList =
    makeNameCountList(env, jsPrimaryNameSrcList.As<Array>());
  auto primarySrcBits =
    cm::NameCount::listToLegacySourceBits(primaryNameSrcList);
  auto usedSources = cm::NameCount::listToUsedSources(primaryNameSrcList);
#if 0
  usedSources.assert_valid();
#endif
  auto ncList = makeNameCountList(env, jsNcList.As<Array>());
  return cm::SourceData(std::move(primaryNameSrcList),
    std::move(primarySrcBits), std::move(usedSources), std::move(ncList));
}

cm::SourceList makeSourceList(Napi::Env& env, const Napi::Array& jsList) {
  cm::SourceList sourceList{};
  for (auto i = 0u; i < jsList.Length(); ++i) {
    if (!jsList[i].IsObject()) {
      Napi::TypeError::New(env, "makeSourceList: non-object element")
        .ThrowAsJavaScriptException();
      return {};
    }
    sourceList.emplace_back(std::move(makeSourceData(env, jsList[i].As<Object>())));
  }
  return sourceList;
}

cm::NCData makeNcData(Napi::Env& env, const Napi::Object& jsObject) {
  auto jsNcList = jsObject.Get("ncList");
  if (!jsNcList.IsArray()) {
    Napi::TypeError::New(env, "makeNcData: ncList is non-array type")
      .ThrowAsJavaScriptException();
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

std::vector<cm::NCDataList> makeNcDataLists(
  Napi::Env& env, const Napi::Array& jsList) {
  //
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
      Napi::TypeError::New(
        env, "makeSourceListMap: invalid mapEntry key/value type")
        .ThrowAsJavaScriptException();
      return {};
    }
    const auto key = tuple[0u].As<String>().Utf8Value();
    auto sourceList = makeSourceList(env, tuple[1u].As<Array>());
    //cm::debugSourceList(sourceList, "makeSourceListMap");
    map.emplace(std::move(key), std::move(sourceList));
  }
  return map;
}

// FromSourceData
cm::SourceCompatibilityData makeSourceCompatData(Env& env, const Napi::Object& jsSourceData) {
  auto jsPrimaryNameSrcList = jsSourceData.Get("primaryNameSrcList");
  if (!jsPrimaryNameSrcList.IsArray()) {
    Napi::TypeError::New(env, "makeSourceData: primaryNameSrcList is not an array")
      .ThrowAsJavaScriptException();
    return {};
  }
  // TODO: declare SourceData result; assign result.xxx = std::move(yyy);; return result
  // (no move-all-params constructor required)
  auto primaryNameSrcList =
    makeNameCountList(env, jsPrimaryNameSrcList.As<Array>());
  auto legacySrcBits =
    cm::NameCount::listToLegacySourceBits(primaryNameSrcList);
  auto usedSources = cm::NameCount::listToUsedSources(primaryNameSrcList);
#if 0
  usedSources.assert_valid();
#endif
  return cm::SourceCompatibilityData(
    std::move(legacySrcBits), std::move(usedSources));
}

// FromSourceList
cm::SourceCompatibilityData makeSourceCompatibilityData(Napi::Env& env,
  const Napi::Array& jsList)
{
  cm::SourceCompatibilityData compatData{};
  for (auto i = 0u; i < jsList.Length(); ++i) {
    if (!jsList[i].IsObject()) {
      Napi::TypeError::New(env, "makeSourceCompatibilityData: non-object element")
        .ThrowAsJavaScriptException();
      return {};
    }
    const auto jsPnsl = jsList[i].As<Object>().Get("primaryNameSrcList").As<Array>();
    for (auto j = 0u; j < jsPnsl.Length(); ++j) {
      const auto count = jsPnsl[j].As<Object>().Get("count").As<Number>().Int32Value();
      compatData.addSource(count);
    }
  }
  return compatData;
}

cm::SourceCompatibilityList makeSourceCompatibilityList(Napi::Env& env,
  const Napi::Array& jsList)
{
  cm::SourceCompatibilityList sourceCompatList{};
  for (auto i = 0u; i < jsList.Length(); ++i) {
    if (!jsList[i].IsObject()) {
      Napi::TypeError::New(env, "makeSourceCompatibiltyList: non-object element")
        .ThrowAsJavaScriptException();
      return {};
    }
    auto jsSourceList = jsList[i].As<Object>().Get("sourceList").As<Array>();
    cm::SourceCompatibilityData compatData =
      makeSourceCompatibilityData(env, jsSourceList);
    sourceCompatList.emplace_back(std::move(compatData));
  }
  return sourceCompatList;
}

/*
cm::SourceCompatDeviceList makeSourceCompatDeviceList(Napi::Env& env,
  const Napi::Array& jsList)
{
  cm::SourceCompatDeviceList sourceCompatList{};
  for (auto i = 0u; i < jsList.Length(); ++i) {
    if (!jsList[i].IsObject()) {
      Napi::TypeError::New(env, "makePmrSourceCompatDeviceList: non-object element")
        .ThrowAsJavaScriptException();
      return {};
    }
    auto jsSourceList = jsList[i].As<Object>().Get("sourceList").As<Array>();
    cm::SourceCompatibilityData compatData =
      makeSourceCompatibilityData(env, jsSourceList);
    sourceCompatList.emplace_back(std::move(compatData));
  }
  return sourceCompatList;
}
*/

cm::OrSourceData makeOrSource(Napi::Env& env, const Napi::Object& jsObject) {
  cm::OrSourceData orSource;
  orSource.source = std::move(
    makeSourceCompatData(env, jsObject["source"].As<Object>()));
  orSource.xorCompatible = jsObject["xorCompatible"].As<Boolean>();
  orSource.andCompatible = jsObject["andCompatible"].As<Boolean>();
  return orSource;
}

cm::OrSourceList makeOrSourceList(Napi::Env& env, const Napi::Array& jsList) {
  cm::OrSourceList orSourceList{};
  for (auto i = 0u; i < jsList.Length(); ++i) {
    if (!jsList[i].IsObject()) {
      Napi::TypeError::New(env, "makeOrSourceList: non-object element")
        .ThrowAsJavaScriptException();
      return {};
    }
    orSourceList.emplace_back(std::move(makeOrSource(env, jsList[i].As<Object>())));
  }
  return orSourceList;
}

cm::OrArgData makeOrArgData(Napi::Env& env, const Napi::Object& jsObject) {
  cm::OrArgData orArgData{};
  orArgData.orSourceList =
    std::move(makeOrSourceList(env, jsObject["orSourceList"].As<Array>()));
  orArgData.compatible = jsObject["compatible"].As<Boolean>();
  return orArgData;
}

cm::OrArgList makeOrArgList(Napi::Env& env, const Napi::Array& jsList) {
  cm::OrArgList orArgList{};
  for (auto i = 0u; i < jsList.Length(); ++i) {
    if (!jsList[i].IsObject()) {
      Napi::TypeError::New(env, "makeOrArgDataList: non-object element")
        .ThrowAsJavaScriptException();
      return {};
    }
    orArgList.emplace_back(
      std::move(makeOrArgData(env, jsList[i].As<Object>())));
  }
  return orArgList;
}

/*
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
*/

/*
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
*/

//
// mergeCompatibleXorSourceCombinations
//
Value mergeCompatibleXorSourceCombinations(const CallbackInfo& info) {
  using namespace std::chrono;

  Env env = info.Env();
  if (!info[0].IsArray() || !info[1].IsArray()) {
      Napi::TypeError::New(env,
        "mergeCompatibleXorSourceCombinations: non-array parameter")
        .ThrowAsJavaScriptException();
      return env.Null();
  }

  auto unwrap0 = high_resolution_clock::now();
  auto ncDataLists = makeNcDataLists(env, info[0].As<Array>());
  cm::PCD.sourceListMap =
    std::move(makeSourceListMap(env, info[1].As<Array>()));
  auto unwrap1 = high_resolution_clock::now();
  [[maybe_unused]] auto d_unwrap = 
    duration_cast<milliseconds>(unwrap1 - unwrap0).count();
  //std::cerr << " native unwrap - " << d_unwrap << "ms" << std::endl;

  //--
    
  auto build0 = high_resolution_clock::now();

  std::vector<cm::SourceList> sourceLists =
    cm::buildSourceListsForUseNcData(ncDataLists, cm::PCD.sourceListMap);

  auto build1 = high_resolution_clock::now();
  [[maybe_unused]] auto d_build =
    duration_cast<milliseconds>(build1 - build0).count();
  //std::cerr << " native build - " << d_build << "ms" << std::endl;

#if 1
  for (const auto& src_list: sourceLists) {
    cm::assert_valid(src_list);
  }
#endif

  //--

  if (sourceLists.size() > 1) {
    auto merge0 = high_resolution_clock::now();
    
    cm::PCD.xorSourceList =
      std::move(cm::mergeCompatibleXorSourceCombinations(sourceLists));
#if 1
    cm::assert_valid(cm::PCD.xorSourceList);
#endif
    
    auto merge1 = high_resolution_clock::now();
    auto d_merge = duration_cast<milliseconds>(merge1 - merge0).count();
    std::cerr << " native merge - " << d_merge << "ms" << std::endl;
  } else if (sourceLists.size() == 1) {
    cm::PCD.xorSourceList = std::move(sourceLists.back());
  }

  //--

  auto xs0 = high_resolution_clock::now();

  cm::PCD.device_xorSources = cm::cuda_allocCopyXorSources(
    cm::PCD.xorSourceList);

  auto xs1 = high_resolution_clock::now();
  auto d_xs = duration_cast<milliseconds>(xs1 - xs0).count();
  std::cerr << " copy xor sources (" << cm::PCD.xorSourceList.size() << ")"
            << " - " << d_xs << "ms" << std::endl;

  //--

  // NOTE that when I ressurect this I should be indexing via the
  // sorted (index) list generated above
  if (cm::PCD.xorSourceList.size()) {
#if 0
    auto svi0 = high_resolution_clock::now();

    cm::PCD.sentenceVariationIndices =
      std::move(cm::buildSentenceVariationIndices(cm::PCD.xorSourceList,
        cm::PCD.xorSourceIndices));
    cm::PCD.device_sentenceVariationIndices =
      cm::cuda_allocCopySentenceVariationIndices(
        cm::PCD.sentenceVariationIndices);
    
    auto svi1 = high_resolution_clock::now();
    auto d_svi = duration_cast<milliseconds>(svi1 - svi0).count();
    std::cerr << " variation indices - " << d_svi << "ms" << std::endl;
#endif
  }

  //--

  return cm::wrap(env, cm::PCD.xorSourceList);
}

/*
//
// isAnySourceCompatibleWithUseSources
//
Value isAnySourceCompatibleWithUseSources(const CallbackInfo& info) {
  Env env = info.Env();
  if (!info[0].IsArray()) {
    TypeError::New(env, "isAnySourceCompatibleWithUseSources: non-array parameter")
      .ThrowAsJavaScriptException();
    return env.Null();
  }
  const auto compatList = makeSourceCompatibilityList(env, info[0].As<Array>());
  const bool compatible = cm::isAnySourceCompatibleWithUseSources(compatList);
  return Boolean::New(env, compatible);
}
*/

Value addCandidateForSum(const CallbackInfo& info) {
  Env env = info.Env();
  if (!info[0].IsNumber() || !info[1].IsString()
      || !(info[2].IsArray() || info[2].IsNumber())) {
    Napi::TypeError::New(env, "addCandidateForSum: invalid parameter type")
      .ThrowAsJavaScriptException();
    return env.Null();
  }
  const auto sum = info[0].As<Number>().Int32Value();
  assert(sum >= 2);
  auto combo = info[1].As<String>().Utf8Value();
  int index{};
  if (info[2].IsArray()) {
    auto compatList = makeSourceCompatibilityList(env, info[2].As<Array>());
    index = cm::addCandidate(sum, std::move(combo), std::move(compatList));
  } else {
    index = info[2].As<Number>().Int32Value();
    cm::addCandidate(sum, combo, index);
  }
  return Number::New(env, index);
}

//
// setOrArgDataList
//
Value setOrArgs(const CallbackInfo& info) {
  using namespace std::chrono;
  Env env = info.Env();
  if (!info[0].IsArray()) {
      Napi::TypeError::New(env, "setOrArgs: non-array parameter")
        .ThrowAsJavaScriptException();
      return env.Null();
  }
  cm::PCD.orArgList = makeOrArgList(env, info[0].As<Array>());

  //--

  auto t0 = high_resolution_clock::now();

  cm::PCD.device_or_args = cm::cuda_allocCopyOrArgs(cm::PCD.orArgList);

  auto t1 = high_resolution_clock::now();
  auto d_t = duration_cast<milliseconds>(t1 - t0).count();
  std::cerr << " copy or_args (" << cm::PCD.orArgList.size() << ")"
            << " - " << d_t << "ms" << std::endl;

  return env.Null();
}

//
// getIsAnyPerfData
//
Value getIsAnyPerfData(const CallbackInfo& info) {
  Env env = info.Env();
  return cm::wrap(env, cm::isany_perf);
}

auto getCandidateStats(int sum) {
  cm::CandidateStats cs;
  cs.sum = sum;
  const auto& cd = cm::allSumsCandidateData.find(sum)->second;
  cs.sourceLists = (int)cd.sourceCompatLists.size();
  cs.totalSources = std::accumulate(cd.sourceCompatLists.cbegin(),
    cd.sourceCompatLists.cend(), 0, [](int sum, const auto& scl) {
      sum += (int)scl.size();
      return sum;
    });
  cs.comboMapIndices = (int)cd.indexComboListMap.size();
  cs.totalCombos = std::accumulate(cd.indexComboListMap.cbegin(),
    cd.indexComboListMap.cend(), 0, [](int sum, const auto& kv) -> int {
      sum += kv.second.size();
      return sum;
    });
  return cs;
}

//
// getCandidateStatsForSum
//
Value getCandidateStatsForSum(const CallbackInfo& info) {
  Env env = info.Env();
  if (!info[0].IsNumber()) {
      Napi::TypeError::New(env, "getCandidateStatsForSum: non-number parameter")
        .ThrowAsJavaScriptException();
      return env.Null();
  }
  auto sum = info[0].As<Number>().Int32Value();
  assert(sum >= 2);
  auto candidateStats = getCandidateStats(sum);
  return cm::wrap(env, candidateStats);
}

//
// filterCandidatesForSum
//
Value filterCandidatesForSum(const CallbackInfo& info) {
  Env env = info.Env();
  if (!info[0].IsNumber() || !info[1].IsNumber() || !info[2].IsNumber()
      || !info[3].IsNumber() || !info[3].IsNumber()) {
    Napi::TypeError::New(env, "fitlerCandidatesForSum: non-number parameter")
      .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto sum = info[0].As<Number>().Int32Value();
  assert(sum >= 2);
  auto threads_per_block = info[1].As<Number>().Int32Value();
  auto streams = info[2].As<Number>().Int32Value();
  auto stride = info[3].As<Number>().Int32Value();
  auto iters = info[4].As<Number>().Int32Value();
  cm::filterCandidates(sum, threads_per_block, streams, stride, iters);
  return env.Null();
}

//
// getResult
//
Value getResult(const CallbackInfo& info) {
  Env env = info.Env();
  cm::get_filter_results();
  return env.Null();
}

//
Object Init(Env env, Object exports) {
  //  exports["buildSourceListsForUseNcData"] = Function::New(env,
  //  buildSourceListsForUseNcData); exports["mergeAllCompatibleSources"] =
  //  Function::New(env, mergeAllCompatibleSources);

  //  exports["isAnySourceCompatibleWithUseSources"] = Function::New(env,
  //  isAnySourceCompatibleWithUseSources);
  exports["mergeCompatibleXorSourceCombinations"] =
    Function::New(env, mergeCompatibleXorSourceCombinations);
  exports["setOrArgDataList"] = Function::New(env, setOrArgs);
  exports["getIsAnyPerfData"] = Function::New(env, getIsAnyPerfData);
  exports["addCandidateForSum"] = Function::New(env, addCandidateForSum);
  exports["getCandidateStatsForSum"] =
    Function::New(env, getCandidateStatsForSum);
  exports["filterCandidatesForSum"] =
    Function::New(env, filterCandidatesForSum);
  exports["getResult"] = Function::New(env, getResult);

  return exports;
}

NODE_API_MODULE(experiment, Init)
