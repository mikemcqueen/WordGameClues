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
#include "filter.h"
#include "merge.h"

namespace {

using namespace Napi;

std::vector<std::string> makeStringList(Env& env, const Array& jsList) {
  std::vector<std::string> list{};
  for (auto i = 0u; i < jsList.Length(); ++i) {
    if (!jsList[i].IsString()) {
      TypeError::New(env, "makeStringList: non-string element")
        .ThrowAsJavaScriptException();
      return {};
    }
    list.emplace_back(std::move(jsList[i].As<String>().Utf8Value()));
  }
  return list;
}

/*
cm::UsedSources makeUsedSources(Env& env, const Array& jsList) {
  cm::UsedSources usedSources{};
  for (auto i = 0u; i < usedSources.size(); ++i) {
    std::int32_t value = 0;
    if (i < jsList.Length()) {
      if (!jsList[i].IsNumber() && !jsList[i].IsUndefined()) {
        TypeError::New(env, "makeUsedSources: non-(number or undefined) element")
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

cm::NameCount makeNameCount(Env& env, const Object& jsObject) {
  auto jsName = jsObject.Get("name");
  auto jsCount = jsObject.Get("count");
  if (!jsName.IsString() || !jsCount.IsNumber()) {
    TypeError::New(env, "makeNameCount: invalid arguments")
      .ThrowAsJavaScriptException();
    return {};
  }
  auto name = jsName.As<String>().Utf8Value();
  const int count = (int)jsCount.As<Number>().Int32Value();
  return cm::NameCount(std::move(name), count);
}

cm::NameCountList makeNameCountList(Env& env, const Array& jsList) {
  cm::NameCountList ncList{};
  for (auto i = 0u; i < jsList.Length(); ++i) {
    if (!jsList[i].IsObject()) {
      TypeError::New(env, "makeNameCountList: non-object element")
        .ThrowAsJavaScriptException();
      return {};
    }
    ncList.emplace_back(std::move(makeNameCount(env, jsList[i].As<Object>())));
  }
  return ncList;
}

cm::SourceData makeSourceData(Env& env, const Object& jsSourceData) {
  auto jsPrimaryNameSrcList = jsSourceData.Get("primaryNameSrcList");
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
  auto jsUsedSources = jsSourceData.Get("usedSources");
  if (!jsUsedSources.IsArray()) {
    TypeError::New(env, "makeSourceData: usedSources is not an array")
      .ThrowAsJavaScriptException();
    return {};
  }
  // TODO: declare SourceData result; assign result.xxx = std::move(yyy);;
  // return result (no move-all-params constructor required)
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

cm::SourceList makeSourceList(Env& env, const Array& jsList) {
  cm::SourceList sourceList{};
  for (auto i = 0u; i < jsList.Length(); ++i) {
    if (!jsList[i].IsObject()) {
      TypeError::New(env, "makeSourceList: non-object element")
        .ThrowAsJavaScriptException();
      return {};
    }
    sourceList.emplace_back(
      std::move(makeSourceData(env, jsList[i].As<Object>())));
  }
  return sourceList;
}

cm::NCData makeNcData(Env& env, const Object& jsObject) {
  auto jsNcList = jsObject.Get("ncList");
  if (!jsNcList.IsArray()) {
    TypeError::New(env, "makeNcData: ncList is non-array type")
      .ThrowAsJavaScriptException();
    return {};
  }
  return { makeNameCountList(env, jsNcList.As<Array>()) };
}

cm::NCDataList makeNcDataList(Env& env, const Array& jsList) {
  cm::NCDataList list;
  for (auto i = 0u; i < jsList.Length(); ++i) {
    if (!jsList[i].IsObject()) {
      TypeError::New(env, "makeNcDataList: element is non-object type")
        .ThrowAsJavaScriptException();
      return {};
    }
    list.emplace_back(std::move(makeNcData(env, jsList[i].As<Object>())));
  }
  return list;
}

std::vector<cm::NCDataList> makeNcDataLists(Env& env, const Array& jsList) {
  //
  std::vector<cm::NCDataList> lists;  
  for (auto i = 0u; i < jsList.Length(); ++i) {
    if (!jsList[i].IsArray()) {
      TypeError::New(env, "makeNcDataLists: element is non-array type")
        .ThrowAsJavaScriptException();
      return {};
    }
    lists.emplace_back(std::move(makeNcDataList(env, jsList[i].As<Array>())));
  }
  return lists;
}

cm::SourceListMap makeSourceListMap(Env& env, const Array& jsList) {
  cm::SourceListMap map{};
  for (auto i = 0u; i < jsList.Length(); ++i) {
    if (!jsList[i].IsArray()) {
      TypeError::New(env, "makeSourceListMap: mapEntry is non-array type")
        .ThrowAsJavaScriptException();
      return {};
    }
    const auto tuple = jsList[i].As<Array>();
    if (!tuple[0u].IsString() || !tuple[1u].IsArray()) {
      TypeError::New(
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
cm::SourceCompatibilityData makeSourceCompatData(
  Env& env, const Object& jsSourceData) {
  // TODO: addPnslToCompatData(jsSouceData, compatData);
  cm::SourceCompatibilityData compatData{};
  const auto jsPnsl = jsSourceData.Get("primaryNameSrcList").As<Array>();
  for (size_t i{}; i < jsPnsl.Length(); ++i) {
    const auto count =
      jsPnsl[i].As<Object>().Get("count").As<Number>().Int32Value();
    compatData.addSource(count);
  }
  return compatData;
#if 0
  auto primaryNameSrcList =
    makeNameCountList(env, jsPrimaryNameSrcList.As<Array>());
  auto legacySrcBits =
    cm::NameCount::listToLegacySourceBits(primaryNameSrcList);
  auto usedSources = cm::NameCount::listToUsedSources(primaryNameSrcList);
#if 1
  usedSources.assert_valid();
#endif
  return cm::SourceCompatibilityData(
    std::move(legacySrcBits), std::move(usedSources));
#endif
}

// FromSourceList
cm::SourceCompatibilityData makeSourceCompatibilityData(
  Env& env, const Array& jsSourceList) {

  cm::SourceCompatibilityData compatData{};
  for (auto i = 0u; i < jsSourceList.Length(); ++i) {
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

cm::SourceCompatibilityList makeSourceCompatibilityList(
  Env& env, const Array& jsList) {

  cm::SourceCompatibilityList sourceCompatList{};
  for (auto i = 0u; i < jsList.Length(); ++i) {
    if (!jsList[i].IsObject()) {
      TypeError::New(env, "makeSourceCompatibiltyList: non-object element")
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
cm::SourceCompatDeviceList makeSourceCompatDeviceList(Env& env,
  const Array& jsList)
{
  cm::SourceCompatDeviceList sourceCompatList{};
  for (auto i = 0u; i < jsList.Length(); ++i) {
    if (!jsList[i].IsObject()) {
      TypeError::New(env, "makePmrSourceCompatDeviceList: non-object element")
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

cm::OrSourceData makeOrSource(Env& env, const Object& jsObject) {
  cm::OrSourceData orSource;
  orSource.source = std::move(
    makeSourceCompatData(env, jsObject["source"].As<Object>()));
  orSource.xorCompatible = jsObject["xorCompatible"].As<Boolean>();
  orSource.andCompatible = jsObject["andCompatible"].As<Boolean>();
  return orSource;
}

cm::OrSourceList makeOrSourceList(Env& env, const Array& jsList) {
  cm::OrSourceList orSourceList{};
  for (auto i = 0u; i < jsList.Length(); ++i) {
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

cm::OrArgData makeOrArgData(Env& env, const Object& jsObject) {
  cm::OrArgData orArgData{};
  orArgData.orSourceList =
    std::move(makeOrSourceList(env, jsObject["orSourceList"].As<Array>()));
  orArgData.compatible = jsObject["compatible"].As<Boolean>();
  return orArgData;
}

cm::OrArgList makeOrArgList(Env& env, const Array& jsList) {
  cm::OrArgList orArgList{};
  for (auto i = 0u; i < jsList.Length(); ++i) {
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

/*
Value buildSourceListsForUseNcData(const CallbackInfo& info) {
  Env env = info.Env();
  if (!info[0].IsArray() || !info[1].IsArray()) {
      TypeError::New(env, "buildSourceListsForUseNcData: non-array parameter")
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
      TypeError::New(env, "mergeAllCompatibleSources: non-array parameter")
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
      TypeError::New(env,
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
  std::cerr << " native build - " << d_build << "ms" << std::endl;

#if 1
  for (const auto& src_list: sourceLists) {
    cm::assert_valid(src_list);
  }
#endif

  //--

  if (sourceLists.size() > 1) {
    auto merge0 = high_resolution_clock::now();
    
    cm::PCD.xorSourceList =
      std::move(cm::cuda_mergeCompatibleXorSourceCombinations(sourceLists));
#if 0
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
  std::cerr << " copy xor sources to device (" << cm::PCD.xorSourceList.size() << ")"
            << " - " << d_xs << "ms" << std::endl;

  //--

  // NOTE that when I ressurect this I should be indexing via the
  // sorted (index) list generated above
  if (cm::PCD.xorSourceList.size()) {
#if 1
    auto svi0 = high_resolution_clock::now();

    // for if/when i want to sort xor sources, which is probably a dumb idea.
    auto xorSourceIndices = []() {
      cm::IndexList v;
      v.resize(cm::PCD.xorSourceList.size());
      iota(v.begin(), v.end(), (cm::index_t)0);
      return v;
    }();
    cm::PCD.sentenceVariationIndices =
      std::move(cm::buildSentenceVariationIndices(
        cm::PCD.xorSourceList, xorSourceIndices));
    cm::PCD.device_sentenceVariationIndices =
      cm::cuda_allocCopySentenceVariationIndices(
        cm::PCD.sentenceVariationIndices);
    // TODO: temporary until all clues are converted to sentences
    cm::PCD.device_xor_src_indices =
      cm::cuda_allocCopyXorSourceIndices(xorSourceIndices);

    auto svi1 = high_resolution_clock::now();
    auto d_svi = duration_cast<milliseconds>(svi1 - svi0).count();
    std::cerr << " variation indices - " << d_svi << "ms" << std::endl;
#endif
  }

  //--

  //  return cm::wrap(env, cm::PCD.xorSourceList);
  return Number::New(env, cm::PCD.xorSourceList.size());
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
    TypeError::New(env, "addCandidateForSum: invalid parameter type")
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
      TypeError::New(env, "setOrArgs: non-array parameter")
        .ThrowAsJavaScriptException();
      return env.Null();
  }
  cm::PCD.orArgList = makeOrArgList(env, info[0].As<Array>());

  //--

  auto t0 = high_resolution_clock::now();

  auto sources_count_pair = cm::cuda_allocCopyOrSources(cm::PCD.orArgList);
  cm::PCD.device_or_sources = sources_count_pair.first;
  cm::PCD.num_or_sources = sources_count_pair.second;

  auto t1 = high_resolution_clock::now();
  auto d_t = duration_cast<milliseconds>(t1 - t0).count();
  std::cerr << " copy or_args (" << cm::PCD.orArgList.size() << ")"
            << " - " << d_t << "ms" << std::endl;

  return Number::New(env, sources_count_pair.second);
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
      TypeError::New(env, "getCandidateStatsForSum: non-number parameter")
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
      || !info[3].IsNumber() || !info[4].IsNumber()) {
    TypeError::New(env, "fitlerCandidatesForSum: non-number parameter")
      .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto sum = info[0].As<Number>().Int32Value();
  assert(sum >= 2);
  auto threads_per_block = info[1].As<Number>().Int32Value();
  auto streams = info[2].As<Number>().Int32Value();
  auto stride = info[3].As<Number>().Int32Value();
  auto iters = info[4].As<Number>().Int32Value();
  cm::filterCandidatesCuda(sum, threads_per_block, streams, stride, iters);
  return env.Null();
}

//
// getResult
//
Value getResult(const CallbackInfo& info) {
  Env env = info.Env();
  auto result = cm::get_filter_result();
  return cm::wrap(env, result);
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

}  // namespace

NODE_API_MODULE(experiment, Init)
