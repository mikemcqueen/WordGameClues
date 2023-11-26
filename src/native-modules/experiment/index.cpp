#include <cassert>
#include <chrono>
#include <numeric>
#include <iostream>
#include <memory>
#include <napi.h>
#include <string>
#include <unordered_map>
#include <vector>
#include "candidates.h"
#include "clue-manager.h"
#include "combo-maker.h"
#include "cm-precompute.h"
#include "dump.h"
#include "filter.h"
#include "merge.h"
#include "merge-filter-common.h"
#include "merge-filter-data.h"
#include "components.h"
#include "util.h"
#include "validator.h"
#include "wrap.h"
#include "log.h"

namespace {

// types

using namespace Napi;
using namespace clue_manager;
using namespace cm;
using namespace validator;

// globals

MergeFilterData MFD;

// functions

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
    return {};
  }
  auto name = jsName.As<String>().Utf8Value();
  const int count = (int)jsCount.As<Number>().Int32Value();
  return NameCount(std::move(name), count);
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
  std::string_view nameSrcList = "primaryNameSrcList") {
  //
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
  std::string_view nameSrcList = "primaryNameSrcList") {
  //
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
  //
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

LogArgs makeLogArgs(Env& env, const Object& jsObject) {
  LogArgs log_args;
  auto jsQuiet = jsObject.Get("quiet");
  auto jsVerbose = jsObject.Get("verbose");
  if (!jsQuiet.IsUndefined()) {
    if (!jsQuiet.IsBoolean()) {
      TypeError::New(env, "makeLogArgs: invalid quiet arg")
        .ThrowAsJavaScriptException();
      return {};
    }
    log_args.quiet = jsQuiet.As<Boolean>();
  }
  if (!jsVerbose.IsUndefined()) {
    if (!jsVerbose.IsBoolean()) {
      TypeError::New(env, "makeLogArgs: invalid verbose arg")
        .ThrowAsJavaScriptException();
      return {};
    }
    log_args.verbose = jsVerbose.As<Boolean>();
  }
  return log_args;
}

//
// Clue-manager
//

// _.keys(nameSourcesMap), values_lists(nameSourcesMap)
Value setPrimaryNameSrcIndicesMap(const CallbackInfo& info) {
  Env env = info.Env();
  if (!info[0].IsArray() || !info[1].IsArray()) {
    TypeError::New(env, "setPrimaryClueNameSourcesMap: invalid parameter type")
      .ThrowAsJavaScriptException();
    return env.Null();
  }
  // arg0
  auto name_list = makeStringList(env, info[0].As<Array>());
  // arg1
  const auto& js_idx_lists = info[1].As<Array>();
  std::vector<IndexList> idx_lists;
  idx_lists.reserve(js_idx_lists.Length());
  for (size_t i{}; i < js_idx_lists.Length(); ++i) {
    if (!js_idx_lists[i].IsArray()) {
      TypeError::New(env, "setPrimaryClueNameSourcesMap: non-array element")
        .ThrowAsJavaScriptException();
      return {};
    }
    idx_lists.emplace_back(makeIndexList(env, js_idx_lists[i].As<Array>()));
  }
  // --
  using namespace clue_manager;
  clue_manager::setPrimaryNameSrcIndicesMap(
    buildPrimaryNameSrcIndicesMap(name_list, idx_lists));
  return env.Null();
}

// count, _.entries(clueMap[count])
Value setCompoundClueNameSourcesMap(const CallbackInfo& info) {
  Env env = info.Env();
  if (!info[0].IsNumber() || !info[1].IsArray()) {
    TypeError::New(env, "setCompoundClueNames: invalid parameter type")
      .ThrowAsJavaScriptException();
    return env.Null();
  }
  // arg0
  auto count{info[0].As<Number>().Int32Value()};
  // arg1
  auto map{makeNameSourcesMap(env, info[1].As<Array>())};
  // --
  clue_manager::setNameSourcesMap(count, std::move(map));
  return env.Null();
}

// count, nameCsv
Value isKnownSourceMapEntry(const CallbackInfo& info) {
  Env env = info.Env();
  if (!info[0].IsNumber() || !info[1].IsString()) {
    TypeError::New(env, "isKnownSourceMapEntry: invalid parameter type")
      .ThrowAsJavaScriptException();
    return env.Null();
  }
  // arg0
  auto count = info[0].As<Number>().Int32Value();
  // arg1
  auto src_csv = info[1].As<String>().Utf8Value();
  // --
  return Boolean::New(
    env, clue_manager::is_known_source_map_entry(count, src_csv));
}

void show_device_state() {
  static auto state_shown = false;
  if (!state_shown) {
    unsigned flags;
    auto err = cudaGetDeviceFlags(&flags);
    assert_cuda_success(err, "cudaGetDeviceFlags");
    auto blocking_sync = !!(flags & cudaDeviceScheduleBlockingSync);
    std::cerr << "cudaDeviceScheduleBlockingSync: " << std::boolalpha
              << blocking_sync << " (" << flags << ")" << std::endl;
    if (!blocking_sync) {
      err = cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
      std::cerr << "set blocking sync to true, error: " << err << std::endl;
    } else {
      state_shown = true;
    }
  }
}

// nc, name_csv
Value addCompoundClue(const CallbackInfo& info) {
  Env env = info.Env();
  if (!info[0].IsObject() || !info[1].IsString()) {
    TypeError::New(env, "appendNcResultsFromSrcMap: invalid parameter type")
      .ThrowAsJavaScriptException();
    return env.Null();
  }
  // arg0
  auto nc = makeNameCount(env, info[0].As<Object>());
  // arg1
  auto src_csv = info[1].As<String>().Utf8Value();
  // --
  clue_manager::add_compound_clue(nc, src_csv);

  // arbitrary to put it here; just somewhere early
  //show_device_state();

  return env.Null();
}

Value getSourcesForNc(const CallbackInfo& info) {
  Env env = info.Env();
  if (!info[0].IsObject()) {
    TypeError::New(env, "getSourcesForNc: invalid parameter type")
      .ThrowAsJavaScriptException();
    return env.Null();
  }
  // arg0
  auto nc = makeNameCount(env, info[0].As<Object>());
  // --
  assert(nc.count == 1);  // assert only known use case.
  return wrap(env, clue_manager::get_nc_sources(nc));
}

Value getSourceListsForNc(const CallbackInfo& info) {
  Env env = info.Env();
  if (!info[0].IsObject()) {
    TypeError::New(env, "getSourceListForNc: invalid parameter type")
      .ThrowAsJavaScriptException();
    return env.Null();
  }
  // arg0
  auto nc = makeNameCount(env, info[0].As<Object>());
  // --
  // assert(nc.count == 1);  // assert only known use case.
#if 0
  if (!clue_manager::is_known_name_count(nc.name, nc.count)) {
    std::cerr << "invalid nc" << std::endl;
    return env.Null();
  }
  const auto& nc_sources = clue_manager::get_nc_sources(nc);
  for (const auto& source_csv : nc_sources) {

  }
#endif
  auto cref_entries = clue_manager::get_known_source_map_entries(nc);
  return wrap(env, cref_entries);
}

//
// Validator
//

Value getNumNcResults(const CallbackInfo& info) {
  Env env = info.Env();
  if (!info[0].IsObject()) {
    TypeError::New(env, "getNumNcResults: invalid parameter type")
      .ThrowAsJavaScriptException();
    return env.Null();
  }
  auto nc = makeNameCount(env, info[0].As<Object>());
  return Number::New(env, clue_manager::get_num_nc_sources(nc));
}

Value validateSources(const CallbackInfo& info) {
  Env env = info.Env();
  if (!info[0].IsString() || !info[1].IsArray() || !info[2].IsNumber()
      || !info[3].IsBoolean()) {
    TypeError::New(
      env, "validateSources: invalid parameter type")
      .ThrowAsJavaScriptException();
    return env.Null();
  }
  // arg 0
  auto clue_name = info[0].As<String>().Utf8Value();
  // arg 1
  auto src_names = makeStringList(env, info[1].As<Array>());
  // arg 2
  auto sum = info[2].As<Number>().Int32Value();
  // arg 3
  auto validate_all = info[3].As<Boolean>();
  // --
  auto src_list =
    validator::validateSources(clue_name, src_names, sum, validate_all);
  auto is_valid_src_list = !src_list.empty();
  if (validate_all && is_valid_src_list) {
    clue_manager::init_known_source_map_entry(
      sum, src_names, std::move(src_list));
  }
  return Boolean::New(env, is_valid_src_list);
}

//
// Combo-maker
//

Value mergeCompatibleXorSourceCombinations(const CallbackInfo& info) {
  using namespace std::chrono;

  Env env = info.Env();
  if (!info[0].IsArray() || !info[1].IsBoolean()) {
    TypeError::New(
      env, "mergeCompatibleXorSourceCombinations: invalid parameter")
      .ThrowAsJavaScriptException();
    return env.Null();
  }
  // arg0
  auto ncDataLists = makeNcDataLists(env, info[0].As<Array>());
  // arg1
  auto merge_only = info[1].As<Boolean>();
  // --
  auto build0 = high_resolution_clock::now();
  // TODO: I'm not convinced this data needs to hang around on host side.
  // maybe for async copy?
  MFD.host.xor_src_lists =
    std::move(buildSourceListsForUseNcData(ncDataLists));

  auto build1 = high_resolution_clock::now();
  [[maybe_unused]] auto d_build =
    duration_cast<milliseconds>(build1 - build0).count();
  std::cerr << " build xor_src_lists - " << d_build << "ms" << std::endl;

#if 0
  for (const auto& src_list: MFD.host.xor_src_lists) {
    assert_valid(src_list);
  }
#endif

  //--
  if (!merge_only || (MFD.host.xor_src_lists.size() > 1)) {
    // TODO: support for single-list compat indices
    auto compat_idx_lists = get_compatible_indices(MFD.host.xor_src_lists);
    if (!compat_idx_lists.empty()) {
      // TODO: free if already set. set_src_lists?
      MFD.device.src_lists = alloc_copy_src_lists(MFD.host.xor_src_lists);
      MFD.device.idx_lists = alloc_copy_idx_lists(compat_idx_lists);
      const auto idx_list_sizes = util::make_list_sizes(compat_idx_lists);
      MFD.device.idx_list_sizes = alloc_copy_list_sizes(idx_list_sizes);
      auto combo_indices = cuda_get_compat_xor_src_indices(
        MFD.host.xor_src_lists, MFD.device.src_lists, compat_idx_lists,
        MFD.device.idx_lists, MFD.device.idx_list_sizes);
      if (merge_only) {
        // merge-only with multiple xor args uses compat index lists and combo
        // indices for the sole purpose of generating an xor_src_list - no need
        // to save them for later use
        MFD.host.merged_xor_src_list = std::move(merge_xor_sources(
          MFD.host.xor_src_lists, compat_idx_lists, combo_indices));
      } else {
        // filter otoh will need them both later
        MFD.host.compat_idx_lists = std::move(compat_idx_lists);
        MFD.host.combo_indices = std::move(combo_indices);
      }
    }
  } else if (MFD.host.xor_src_lists.size() == 1) {
    assert(merge_only);
    MFD.host.merged_xor_src_list = std::move(MFD.host.xor_src_lists.back());
  }
  auto result = (merge_only) ? MFD.host.merged_xor_src_list.size()
                             : MFD.host.combo_indices.size();
  return Number::New(env, (uint32_t)result);
}

void validate_marked_or_sources(
  const OrArgList& or_arg_list, const std::vector<result_t>& mark_results) {
  //
  size_t or_arg_idx{};
  size_t num_or_args = or_arg_list[or_arg_idx].or_src_list.size();
  bool is_arg_marked{false};
  for (size_t result_idx{}; result_idx < mark_results.size();) {
    if (mark_results[result_idx]) {
      is_arg_marked = true;
    }
    if (++result_idx == num_or_args) {
      if (!is_arg_marked) {
        std::cerr << "or_arg_idx " << or_arg_idx << " is not compatible"
                  << std::endl;
        assert(is_arg_marked);
      }
      if (++or_arg_idx == or_arg_list.size()) {
        return;
      }
      num_or_args += or_arg_list[++or_arg_idx].or_src_list.size();
      is_arg_marked = false;
    }
  }
}

void set_or_args(const std::vector<NCDataList>& ncDataLists) {
  using namespace std::chrono;
  auto build0 = high_resolution_clock::now();
  MFD.host.or_arg_list =
    buildOrArgList(buildSourceListsForUseNcData(ncDataLists));
  if (MFD.host.or_arg_list.size()) {
    // TODO: to eliminate sync call in allocCopyOrSources
    //  auto or_src_list = make_or_src_list(MFD.host.or_arg_list);
    auto [device_or_src_list, num_or_sources] =
      cuda_allocCopyOrSources(MFD.host.or_arg_list);
    MFD.device.or_src_list = device_or_src_list;
    MFD.device.num_or_sources = num_or_sources;

    // Thoughts on AND compatibility of OrSources:
    // Just because (one sourceList of) an OrSource is AND compatible with an
    // XorSource doesn't mean the OrSource is redundant and can be ignored
    // (i.e., the container cannot be marked as "compatible.") We still need
    // to check the possibility that any of the other XOR-but-not-AND-compatible
    // sourceLists could be AND-compatible with the generated-combo sourceList.
    // So, a container can be marked compatible if and only if there are no
    // no remaining XOR-compatible sourceLists.
    // TODO: markAllANDCompatibleOrSources(xorSourceList, orSourceList);

    if (num_or_sources) {
      if constexpr (0) {
        markAllXorCompatibleOrSources(MFD.host.or_arg_list,
          MFD.host.xor_src_lists, MFD.host.compat_idx_lists,
          MFD.host.combo_indices);
      }
      // TODO: name change. this doesn't mark anything. 
      auto mark_results = cuda_markAllXorCompatibleOrSources(MFD);
      validate_marked_or_sources(MFD.host.or_arg_list, mark_results);
      MFD.device.num_or_sources =
        move_marked_or_sources(MFD.device.or_src_list, mark_results);
    }
  }
  auto build1 = high_resolution_clock::now();
  [[maybe_unused]] auto d_build =
    duration_cast<milliseconds>(build1 - build0).count();
  std::cerr << " build/mark or_args(" << MFD.host.or_arg_list.size() << ")"
            << ", or_sources(" << MFD.device.num_or_sources << ") - " << d_build
            << "ms" << std::endl;
}

void alloc_copy_filter_indices() {
  using namespace std::chrono;
  assert(!MFD.host.combo_indices.empty());
  auto t0 = high_resolution_clock::now();

  auto src_list_start_indices = make_start_indices(MFD.host.xor_src_lists);
  MFD.device.src_list_start_indices =
    alloc_copy_start_indices(src_list_start_indices);
  auto idx_list_start_indices = make_start_indices(MFD.host.compat_idx_lists);
  MFD.device.idx_list_start_indices =
    alloc_copy_start_indices(idx_list_start_indices);
  auto variation_indices = buildSentenceVariationIndices(
    MFD.host.xor_src_lists, MFD.host.compat_idx_lists, MFD.host.combo_indices);
  MFD.device.variation_indices =
    cuda_allocCopySentenceVariationIndices(variation_indices);

  auto t1 = high_resolution_clock::now();
  auto t_dur = duration_cast<milliseconds>(t1 - t0).count();
  std::cerr << " prepare filter indices - " << t_dur << "ms" << std::endl;
}

//
// filterPreparation
//
Value filterPreparation(const CallbackInfo& info) {
  using namespace std::chrono;
  Env env = info.Env();
  if (!info[0].IsArray()) {
    TypeError::New(env, "filterPreparation: invalid parameter")
      .ThrowAsJavaScriptException();
    return env.Null();
  }
  // arg0
  auto orNcDataLists = makeNcDataLists(env, info[0].As<Array>());
  // --
  alloc_copy_filter_indices();
  set_or_args(orNcDataLists);
  return env.Null();
}

// considerCandidate
Value considerCandidate(const CallbackInfo& info) {
  Env env = info.Env();
  if (!info[0].IsArray()) {
    TypeError::New(env, "considerCandidates: invalid parameter type")
      .ThrowAsJavaScriptException();
    return env.Null();
  }
  // arg0
  auto ncList = makeNameCountList(env, info[0].As<Array>());
  // --
  consider_candidate(ncList);
  return env.Null();
}
  
//
// filterCandidatesForSum
//

auto make_source_descriptor_pairs(
  const SourceCompatibilitySet& incompatible_sources) {
  //
  std::vector<UsedSources::SourceDescriptorPair> src_desc_pairs;
  src_desc_pairs.reserve(incompatible_sources.size());
  for (const auto& src: incompatible_sources) {
    if constexpr (0) {
      auto pair = src.usedSources.get_source_descriptor_pair();
      printf("---\n");
      src.usedSources.dump();
      pair.first.dump();
      pair.second.dump();
      src_desc_pairs.emplace_back(pair);
    } else {
      src_desc_pairs.emplace_back(src.usedSources.get_source_descriptor_pair());
    }
  }
  return src_desc_pairs;
}

void set_incompatible_sources(
  const SourceCompatibilitySet& incompatible_sources) {
  // empty set technically possible; disallowed here as a canary
  assert(!incompatible_sources.empty());
  assert(MFD.host.incompatible_src_desc_pairs.empty());
  assert(!MFD.device.incompatible_src_desc_pairs);

  MFD.host.incompatible_src_desc_pairs =
    std::move(make_source_descriptor_pairs(incompatible_sources));
  MFD.device.incompatible_src_desc_pairs =
    cuda_alloc_copy_source_desc_pairs(MFD.host.incompatible_src_desc_pairs);
  MFD.device.num_incompatible_sources = incompatible_sources.size();
  std::cerr << " incompatible sources: " << incompatible_sources.size()
            << std::endl;
}

Value filterCandidatesForSum(const CallbackInfo& info) {
  Env env = info.Env();
  if (!info[0].IsNumber() || !info[1].IsNumber() || !info[2].IsNumber()
      || !info[3].IsNumber() || !info[4].IsNumber() || !info[5].IsBoolean()
      || !info[6].IsObject()) {
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
  // arg6
  set_log_args(makeLogArgs(env, info[6].As<Object>()));
  // --
  auto opt_incompatible_sources = filter_candidates_cuda(
    MFD, sum, threads_per_block, streams, stride, iters, synchronous);
  assert(synchronous == opt_incompatible_sources.has_value());
  if (opt_incompatible_sources.has_value()) {
    set_incompatible_sources(*opt_incompatible_sources);
  }
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
// showComponents
//
Value showComponents(const CallbackInfo& info) {
  Env env = info.Env();
  if (!info[0].IsArray()) {
    TypeError::New(env, "showComponents: invalid parameter type")
      .ThrowAsJavaScriptException();
    return env.Null();
  }
  // arg0:
  auto name_list = makeStringList(env, info[0].As<Array>());
  // --
  auto sums = components::show(name_list, MFD.host.merged_xor_src_list);
  return wrap(env, sums);
}

//
// checkClueConsistency
//
Value checkClueConsistency(const CallbackInfo& info) {
  Env env = info.Env();
  if (!info[0].IsArray()) {
    TypeError::New(env, "checkClueConsistency: invalid parameter type")
      .ThrowAsJavaScriptException();
    return env.Null();
  }
  // arg0:
  auto name_list = makeStringList(env, info[0].As<Array>());
  // --
  auto result =
    components::consistency_check(name_list, MFD.host.merged_xor_src_list);
  return Boolean::New(env, result);
}

//
Object Init(Env env, Object exports) {
  // clue-manager
  //
  exports["getNumNcResults"] = Function::New(env, getNumNcResults);
  exports["setPrimaryNameSrcIndicesMap"] = Function::New(env, setPrimaryNameSrcIndicesMap);
  exports["setCompoundClueNameSourcesMap"] = Function::New(env, setCompoundClueNameSourcesMap);
  exports["isKnownSourceMapEntry"] = Function::New(env, isKnownSourceMapEntry);
  exports["getSourcesForNc"] = Function::New(env, getSourcesForNc);
  exports["getSourceListsForNc"] = Function::New(env, getSourceListsForNc);
  exports["addCompoundClue"] = Function::New(env, addCompoundClue);

  // validator
  //
  exports["validateSources"] = Function::New(env, validateSources);

  // combo-maker
  //
  exports["mergeCompatibleXorSourceCombinations"] =
    Function::New(env, mergeCompatibleXorSourceCombinations);
  exports["filterPreparation"] = Function::New(env, filterPreparation);
  exports["considerCandidate"] = Function::New(env, considerCandidate);
  exports["filterCandidatesForSum"] =
    Function::New(env, filterCandidatesForSum);
  exports["getResult"] = Function::New(env, getResult);

  // components
  //
  exports["showComponents"] = Function::New(env, showComponents);

  // consistency
  //
  exports["checkClueConsistency"] = Function::New(env, checkClueConsistency);

  return exports;
}

}  // namespace

NODE_API_MODULE(experiment, Init)
