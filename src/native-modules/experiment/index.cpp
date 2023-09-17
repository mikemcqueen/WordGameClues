#include <cassert>
#include <chrono>
#include <numeric>
#include <iostream>
#include <memory>
#include <napi.h>
#include <string>
#include <unordered_map>
#include <vector>
#include "combo-maker.h"
#include "candidates.h"
#include "dump.h"
#include "wrap.h"
#include "filter.h"
#include "merge.h"

namespace {

using namespace Napi;
using namespace cm;

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

NameCount makeNameCount(Env& env, const Object& jsObject) {
  auto jsName = jsObject.Get("name");
  auto jsCount = jsObject.Get("count");
  if (!jsName.IsString() || !jsCount.IsNumber()) {
    TypeError::New(env, "makeNameCount: invalid arguments")
      .ThrowAsJavaScriptException();
    return {};
  }
  auto name = jsName.As<String>().Utf8Value();
  const int count = (int)jsCount.As<Number>().Int32Value();
  return NameCount(std::move(name), count);
}

NameCountList makeNameCountList(Env& env, const Array& jsList) {
  NameCountList ncList{};
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

SourceData makeSourceData(Env& env, const Object& jsSourceData) {
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
    NameCount::listToLegacySourceBits(primaryNameSrcList);
  auto usedSources = NameCount::listToUsedSources(primaryNameSrcList);
#if 0
  usedSources.assert_valid();
#endif
  auto ncList = makeNameCountList(env, jsNcList.As<Array>());
  return SourceData(std::move(primaryNameSrcList),
    std::move(primarySrcBits), std::move(usedSources), std::move(ncList));
}

SourceList makeSourceList(Env& env, const Array& jsList) {
  SourceList sourceList{};
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

std::vector<NCDataList> makeNcDataLists(Env& env, const Array& jsList) {
  std::vector<NCDataList> lists;  
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

SourceListMap makeSourceListMap(Env& env, const Array& jsList) {
  SourceListMap map{};
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
    map.emplace(std::move(key), std::move(sourceList));
  }
  return map;
}

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

SourceCompatibilityList makeSourceCompatibilityListFromMergedSourcesList(
  Env& env, const Array& jsList) {
  //
  SourceCompatibilityList sourceCompatList{};
  for (auto i = 0u; i < jsList.Length(); ++i) {
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
  orSource.source = std::move(makeSourceCompatibilityDataFromSourceData(
    env, jsObject["source"].As<Object>()));
  orSource.xorCompatible = jsObject["xorCompatible"].As<Boolean>();
  orSource.andCompatible = jsObject["andCompatible"].As<Boolean>();
  return orSource;
}

OrSourceList makeOrSourceList(Env& env, const Array& jsList) {
  OrSourceList orSourceList{};
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

OrArgData makeOrArgData(Env& env, const Object& jsObject) {
  OrArgData orArgData{};
  orArgData.orSourceList =
    std::move(makeOrSourceList(env, jsObject["orSourceList"].As<Array>()));
  orArgData.compatible = jsObject["compatible"].As<Boolean>();
  return orArgData;
}

OrArgList makeOrArgList(Env& env, const Array& jsList) {
  OrArgList orArgList{};
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

//
// mergeCompatibleXorSourceCombinations
//
Value mergeCompatibleXorSourceCombinations(const CallbackInfo& info) {
  using namespace std::chrono;

  Env env = info.Env();
  if (!info[0].IsArray() || !info[1].IsArray() || !info[2].IsBoolean()) {
      TypeError::New(env,
        "mergeCompatibleXorSourceCombinations: invalid parameter")
        .ThrowAsJavaScriptException();
      return env.Null();
  }

  auto unwrap0 = high_resolution_clock::now();
  auto ncDataLists = makeNcDataLists(env, info[0].As<Array>());
  PCD.sourceListMap =
    std::move(makeSourceListMap(env, info[1].As<Array>()));
  auto xor_wrap = info[2].As<Boolean>();

  auto unwrap1 = high_resolution_clock::now();
  [[maybe_unused]] auto d_unwrap = 
    duration_cast<milliseconds>(unwrap1 - unwrap0).count();
  //std::cerr << " native unwrap - " << d_unwrap << "ms" << std::endl;

  //--
    
  auto build0 = high_resolution_clock::now();

  std::vector<SourceList> sourceLists =
    buildSourceListsForUseNcData(ncDataLists, PCD.sourceListMap);

  auto build1 = high_resolution_clock::now();
  [[maybe_unused]] auto d_build =
    duration_cast<milliseconds>(build1 - build0).count();
  std::cerr << " native build - " << d_build << "ms" << std::endl;

#if 0
  for (const auto& src_list: sourceLists) {
    assert_valid(src_list);
  }
#endif

  //--

  if (sourceLists.size() > 1) {
    auto merge0 = high_resolution_clock::now();
    
    PCD.xorSourceList =
      std::move(cuda_mergeCompatibleXorSourceCombinations(sourceLists));
#if 0
    assert_valid(PCD.xorSourceList);
#endif
    
    auto merge1 = high_resolution_clock::now();
    auto d_merge = duration_cast<milliseconds>(merge1 - merge0).count();
    std::cerr << " native merge - " << d_merge << "ms" << std::endl;
  } else if (sourceLists.size() == 1) {
    PCD.xorSourceList = std::move(sourceLists.back());
  }

  //--

  auto xs0 = high_resolution_clock::now();

  PCD.device_xorSources = cuda_allocCopyXorSources(
    PCD.xorSourceList);

  auto xs1 = high_resolution_clock::now();
  auto d_xs = duration_cast<milliseconds>(xs1 - xs0).count();
  std::cerr << " copy xor sources to device (" << PCD.xorSourceList.size() << ")"
            << " - " << d_xs << "ms" << std::endl;

  //--

  if (PCD.xorSourceList.size()) {
    auto svi0 = high_resolution_clock::now();

    // for if/when i want to sort xor sources, which is probably a dumb idea.
    auto xorSourceIndices = []() {
      IndexList v;
      v.resize(PCD.xorSourceList.size());
      iota(v.begin(), v.end(), (index_t)0);
      return v;
    }();
    PCD.sentenceVariationIndices =
      std::move(buildSentenceVariationIndices(
        PCD.xorSourceList, xorSourceIndices));
    PCD.device_sentenceVariationIndices =
      cuda_allocCopySentenceVariationIndices(
        PCD.sentenceVariationIndices);
    // TODO: temporary until all clues are converted to sentences
    PCD.device_xor_src_indices =
      cuda_allocCopyXorSourceIndices(xorSourceIndices);

    auto svi1 = high_resolution_clock::now();
    auto d_svi = duration_cast<milliseconds>(svi1 - svi0).count();
    std::cerr << " variation indices - " << d_svi << "ms" << std::endl;
  }

  //--

  if (xor_wrap) {
    return wrap(env, PCD.xorSourceList);
  }
  return Number::New(env, PCD.xorSourceList.size());
}

// considerCandidate
Value considerCandidate(const CallbackInfo& info) {
  Env env = info.Env();
  if (!info[0].IsArray() || !info[1].IsNumber()) {
    TypeError::New(env, "considerCandidates: invalid parameter type")
      .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto ncList = makeNameCountList(env, info[0].As<Array>());
  const auto sum = info[1].As<Number>().Int32Value();
  assert(sum >= 2);
  consider_candidate(ncList, sum);
  return env.Null();
}

// addCandidateForSum
#if 0
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
    auto compatList = makeSourceCompatibilityListFromMergedSourcesList(
      env, info[2].As<Array>());
    index = addCandidate(sum, std::move(combo), std::move(compatList));
  } else {
    index = info[2].As<Number>().Int32Value();
    addCandidate(sum, combo, index);
  }
  return Number::New(env, index);
}
#endif
  
//
// setOrArgDataList
//
Value setOrArgDataList(const CallbackInfo& info) {
  using namespace std::chrono;
  Env env = info.Env();
  if (!info[0].IsArray()) {
      TypeError::New(env, "setOrArgs: non-array parameter")
        .ThrowAsJavaScriptException();
      return env.Null();
  }
  PCD.orArgList = makeOrArgList(env, info[0].As<Array>());

  //--

  auto t0 = high_resolution_clock::now();

  auto sources_count_pair = cuda_allocCopyOrSources(PCD.orArgList);
  PCD.device_or_sources = sources_count_pair.first;
  PCD.num_or_sources = sources_count_pair.second;

  auto t1 = high_resolution_clock::now();
  auto d_t = duration_cast<milliseconds>(t1 - t0).count();
  std::cerr << " copy or_args (" << PCD.orArgList.size() << ")"
            << " - " << d_t << "ms" << std::endl;

  return Number::New(env, sources_count_pair.second);
}

#if 0
auto getCandidateStats(int sum) {
  CandidateStats cs;
  cs.sum = sum;
  const auto& cd = allSumsCandidateData.find(sum)->second;
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
  return wrap(env, candidateStats);
}
#endif

//
// filterCandidatesForSum
//
Value filterCandidatesForSum(const CallbackInfo& info) {
  Env env = info.Env();
  if (!info[0].IsNumber() || !info[1].IsNumber() || !info[2].IsNumber()
      || !info[3].IsNumber() || !info[4].IsNumber() || !info[5].IsBoolean()) {
    TypeError::New(env, "fitlerCandidatesForSum: invalid parameter type")
      .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto sum = info[0].As<Number>().Int32Value();
  assert(sum >= 2);
  auto threads_per_block = info[1].As<Number>().Int32Value();
  auto streams = info[2].As<Number>().Int32Value();
  auto stride = info[3].As<Number>().Int32Value();
  auto iters = info[4].As<Number>().Int32Value();
  auto synchronous = info[5].As<Boolean>().Value();
  filterCandidatesCuda(
    sum, threads_per_block, streams, stride, iters, synchronous);
  return env.Null();
}

//
// getResult
//
Value getResult(const CallbackInfo& info) {
  Env env = info.Env();
  auto result = get_filter_result();
  return wrap(env, result);
}

//
Object Init(Env env, Object exports) {
  exports["mergeCompatibleXorSourceCombinations"] =
    Function::New(env, mergeCompatibleXorSourceCombinations);
  exports["setOrArgDataList"] = Function::New(env, setOrArgDataList);
  exports["considerCandidate"] = Function::New(env, considerCandidate);
  //  exports["getCandidateStatsForSum"] = Function::New(env, getCandidateStatsForSum);
  exports["filterCandidatesForSum"] =
    Function::New(env, filterCandidatesForSum);
  exports["getResult"] = Function::New(env, getResult);

  return exports;
}

}  // namespace

NODE_API_MODULE(experiment, Init)
