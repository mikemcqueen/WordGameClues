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
  // merge_only means "-t" mode, in which case no filter kernel will be called,
  // so we don't need to do additional work/copy additional device data
  auto merge_only = info[2].As<Boolean>();

  auto unwrap1 = high_resolution_clock::now();
  [[maybe_unused]] auto d_unwrap = 
    duration_cast<milliseconds>(unwrap1 - unwrap0).count();
  std::cerr << " build src_list_map - " << d_unwrap << "ms" << std::endl;

  //--
    
  auto build0 = high_resolution_clock::now();

  // TODO: I'm not convinced this data to hang around on host side.
  PCD.xor_src_lists =
    std::move(buildSourceListsForUseNcData(ncDataLists, PCD.sourceListMap));

  auto build1 = high_resolution_clock::now();
  [[maybe_unused]] auto d_build =
    duration_cast<milliseconds>(build1 - build0).count();
  std::cerr << " build xor_src_lists - " << d_build << "ms" << std::endl;

#if 0
  for (const auto& src_list: PCD.xor_src_lists) {
    assert_valid(src_list);
  }
#endif

  XorSourceList merge_only_xor_src_list;
  if (!merge_only || (PCD.xor_src_lists.size() > 1)) {
    // TODO: support for single-list compat indices
    auto idx_lists = get_compatible_indices(PCD.xor_src_lists);
    if (!idx_lists.empty()) {
      auto compat_indices =
        cuda_get_compat_xor_src_indices(PCD.xor_src_lists, idx_lists);
      assert(idx_lists.size() == compat_indices.size());
      if (merge_only) {
        // merge-only with multiple xor args uses compat indices for the sole
        // purpose of generating an xor_src_list.
        merge_only_xor_src_list = std::move(
          merge_xor_sources(PCD.xor_src_lists, idx_lists, compat_indices));
      } else {
        // filter will copy compat indices to device memory in a subsequent
        // call, though it *could* be done here to save a bit of host memory.
        PCD.compat_indices = std::move(compat_indices);
      }
    }
    if (!merge_only) {
      // filter returns number of compatible indices
      return Number::New(env, idx_lists.size());
    }
  } else if (PCD.xor_src_lists.size() == 1) {
    merge_only_xor_src_list = std::move(PCD.xor_src_lists.back());
  }
  // merge-only returns xor_src_lsit
  return wrap(env, merge_only_xor_src_list);
}

void prepare_filter_indices() {
  using namespace std::chrono;
  assert(!PCD.compat_indices.empty());

  auto svi0 = high_resolution_clock::now();

#if TODO
   PCD.device_compat_indices =
    std::move(cuda_alloc_copy_compat_indices(PCD.compat_indices));
     PCD.compat_indices = PCD.compat_indices.size();
#endif

  // for if/when i want to sort xor sources, which is probably a dumb idea.
  /*
  auto xorSourceIndices = []() {
    IndexList v;
    v.resize(PCD.xorSourceList.size());
    iota(v.begin(), v.end(), (index_t)0);
    return v;
  }();
  */
#if TODO
  auto variation_indices = buildSentenceVariationIndices(PCD.xorSourceList);
  PCD.device_variation_indices =
    cuda_allocCopySentenceVariationIndices(variation_indices);
#endif

  // TODO: temporary until all clues are converted to sentences
  // PCD.device_legacy_xor_src_indices =
  //    cuda_allocCopyXorSourceIndices(xorSourceIndices);

  auto svi1 = high_resolution_clock::now();
  auto d_svi = duration_cast<milliseconds>(svi1 - svi0).count();
  std::cerr << " prepare filter indices - " << d_svi << "ms" << std::endl;
}

//
// filterPreparation
//
Value filterPreparation(const CallbackInfo& info) {
  using namespace std::chrono;
  Env env = info.Env();
  if (!info[0].IsArray()) {
      TypeError::New(env, "filterPreparation: non-array parameter")
        .ThrowAsJavaScriptException();
      return env.Null();
  }
  auto t0 = high_resolution_clock::now();

  // TODO: needed beyond the scope of this function?
  auto orArgList = makeOrArgList(env, info[0].As<Array>());
  PCD.num_or_args = orArgList.size();
  auto sources_count_pair = cuda_allocCopyOrSources(orArgList);
  PCD.device_or_sources = sources_count_pair.first;
  PCD.num_or_sources = sources_count_pair.second;

  prepare_filter_indices();

  auto t1 = high_resolution_clock::now();
  auto d_t = duration_cast<milliseconds>(t1 - t0).count();
  std::cerr << " filter preparation, --or args(" << PCD.num_or_args << ")"
            << ", sources(" << PCD.num_or_sources << ") - " << d_t << "ms"
            << std::endl;

  return Number::New(env, sources_count_pair.second);
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
  exports["filterPreparation"] = Function::New(env, filterPreparation);
  exports["considerCandidate"] = Function::New(env, considerCandidate);
  //  exports["getCandidateStatsForSum"] = Function::New(env, getCandidateStatsForSum);
  exports["filterCandidatesForSum"] =
    Function::New(env, filterCandidatesForSum);
  exports["getResult"] = Function::New(env, getResult);

  return exports;
}

}  // namespace

NODE_API_MODULE(experiment, Init)
