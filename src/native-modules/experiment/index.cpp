#include <cassert>
#include <iostream>
#include <memory>
#include <napi.h>
#include <numeric>
#include <ranges>
#include <set>
#include <string>
#include <stacktrace>
#include <unordered_map>
#include <vector>
#include "candidates.h"
#include "clue-manager.h"
#include "combo-maker.h"
#include "cm-precompute.h"
#include "dump.h"
#include "filter.cuh"
#include "filter.h"
#include "known-sources.h"
#include "merge.h"
#include "merge-filter-common.h"
#include "merge-filter-data.h"
#include "components.h"
#include "util.h"
#include "unique-variations.h"
#include "validator.h"
#include "unwrap.h"
#include "wrap.h"
#include "log.h"

namespace cm {
// combo-maker.cpp
void compute_combos_for_sum(int sum, int max);
#if 1
long get_compute_duration();
long get_make_indices_duration();
#endif
}

namespace {

// using

using namespace Napi;
using namespace cm;
using namespace cm::clue_manager;
using namespace cm::validator;

// globals

FilterData MFD;

// functions

//
// Clue-manager
//

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
  for (uint32_t i{}; i < js_idx_lists.Length(); ++i) {
    if (!js_idx_lists[i].IsArray()) {
      TypeError::New(env, "setPrimaryClueNameSourcesMap: non-array element")
        .ThrowAsJavaScriptException();
      return {};
    }
    idx_lists.push_back(makeIndexList(env, js_idx_lists[i].As<Array>()));
  }
  // --
  clue_manager::init_primary_clues(std::move(name_list), std::move(idx_lists));

#if 0
  KnownSources::test_keys();
#endif

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
  auto count = info[0].As<Number>().Int32Value();
  // arg1
  auto map = makeNameSourcesMap(env, info[1].As<Array>());
  // --
  clue_manager::set_name_sources_map(count, std::move(map));
  return env.Null();
}

// count, source
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
  return Boolean::New(env, KnownSources::get().has_entries_for(count, src_csv));
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
  KnownSources::add_compound_clue(nc, src_csv);

  // arbitrary to put it here; just somewhere early
  //show_device_state();

  return env.Null();
}

long validate_ns = 0;

Value validateSources(const CallbackInfo& info) {
  Env env = info.Env();
  if (!info[0].IsString() || !info[1].IsArray() || !info[2].IsNumber()
      || !info[3].IsBoolean()) {
    TypeError::New(env, "validateSources: invalid parameter type")
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
  auto t = util::Timer::start_timer();
  auto src_list =
      validator::validateSources(clue_name, src_names, sum, validate_all);
  t.stop();
  validate_ns += t.nanoseconds();
  const auto is_valid_src_list = !src_list.empty();
  if (validate_all && is_valid_src_list) {
    KnownSources::get().init_entry(sum, util::join(src_names, ","),
        std::move(src_list));
  }
  return Boolean::New(env, is_valid_src_list);
}

void show_clue_manager_durations(){
  static bool shown = false;
  if (!shown) {
    std::cerr << "(delayed clue_manager durations)\n"
              << " validateSources - " << int((double)validate_ns / 1e6) << "ms\n";
    show_validator_durations();
    shown = true;
  }
}

void display(const std::vector<NCDataList>& nc_data_lists) {
  std::cerr << " nc_data_lists(" << nc_data_lists.size() << "):\n";
  for (const auto& nc_data_list : nc_data_lists) {
    std::cerr << " nc_lists(" << nc_data_list.size() << "):\n";
    for (const auto& nc_data : nc_data_list) {
      std::cerr << "  ";
      for (const auto& nc : nc_data.ncList) {
        std::cerr << nc.toString() << ", ";
      }
      std::cerr << std::endl;
    }
  }
}

void host_memory_dump(
    const FilterData::HostXor& host_xor, std::string_view header = "post merge") {
  if (!log_level(MemoryDumps)) return;

  // MergeData
  // std::vector<IndexList> compat_idx_lists;
  // ComboIndexList combo_indices;
  auto idx_lists_size =
      host_xor.compat_idx_lists.size() * sizeof(IndexList::value_type);
  auto combo_indices_size =
      host_xor.compat_indices.size() * sizeof(fat_index_t);
  auto merged_src_list_size =
      host_xor.merged_xor_src_list.size() * sizeof(SourceList::value_type);

  std::cerr << "host memory dump " << header << std::endl
            << " idx_lists  :     " << util::pretty_bytes(idx_lists_size)
            << std::endl
            << " combo_indices:   " << util::pretty_bytes(combo_indices_size)
            << std::endl
            << " merged_src_list: " << util::pretty_bytes(merged_src_list_size)
            << std::endl
            << " total:           "
            << util::pretty_bytes(
                   idx_lists_size + combo_indices_size + merged_src_list_size)
            << std::endl;
}

//
// Combo-maker
//

Value mergeCompatibleXorSourceCombinations(const CallbackInfo& info) {
  Env env = info.Env();
  if (!info[0].IsArray() || !info[1].IsBoolean()) {
    TypeError::New(
        env, "mergeCompatibleXorSourceCombinations: invalid parameter")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  // arg0
  auto nc_data_lists = makeNcDataLists(env, info[0].As<Array>());
  // arg1
  auto merge_only = info[1].As<Boolean>();
  if (merge_only) the_log_args_.quiet = true;
  // --

  // arbitrary, want to do it somewhere
  show_clue_manager_durations();

#if 0
  static bool fifo_initialized{false};
  if (!fifo_initialized) {
    auto err = cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 50'000'000);
    assert_cuda_success(err, "get_compat_combos set fifo size");
    fifo_initialized = true;
  }
#endif

  auto xor_src_lists = build_src_lists(nc_data_lists);
  if (log_level(Normal)) {
    //std::cerr << "build xor_src_lists(" << xor_src_lists.size() << ")\n";
  }
#if 0
  for (const auto& src_list: xor_src_lists) {
    assert_valid(src_list);
  }
#endif
  //--
  size_t num_compat{};

  // temporarily handle merge_only case until it's eliminated
  if (merge_only) {
    // merge-only=true is used for showComponents() and consistencyCheck V1,
    // both of which can be eliminated.
    if (xor_src_lists.size() > 1) {
      MFD.host_xor.merged_xor_src_list =
          merge_xor_compatible_src_lists(xor_src_lists);
    } else {
      MFD.host_xor.merged_xor_src_list = std::move(xor_src_lists.back());
    }
    num_compat = MFD.host_xor.merged_xor_src_list.size();
    host_memory_dump(MFD.host_xor);
  }
  else {
    auto stream = cudaStreamPerThread;
    if (get_merge_data(xor_src_lists, MFD.host_xor, MFD.device_xor,
            MergeType::XOR, stream)) {
      num_compat = MFD.host_xor.compat_indices.size();
      // filter needs this later
      MFD.host_xor.src_lists = std::move(xor_src_lists);
    }
  }
  return Number::New(env, uint32_t(num_compat));
}

  /*
auto combo_to_str(fat_index_t combo_idx, const IndexList& list_sizes) {
  std::string s;
  for (size_t i{}; i < list_sizes.size(); ++i) {
    const auto list_size = list_sizes.at(i);
    auto idx = combo_idx % list_size;
    combo_idx /= list_size;
    if (!s.empty()) s += " ";
    s += std::to_string(idx);
  }
  return s;
}
  */

void dump_combos(const MergeData::Host& host, const MergeData::Device& device) {
  std::vector<Variations> variations;
  const auto list_sizes = util::make_list_sizes(host.compat_idx_lists);
  std::cout << "list_sizes: ";
  for (auto s : list_sizes) std::cout << s << " ";
  std::cout << std::endl;
  /*
  for (auto combo_idx : host.combo_indices) {
    std::cout << "host: " << combo_idx
              << " [" << combo_to_str(combo_idx, list_sizes) << "]" << std::endl;
  }
  */
}

void show_unique_XOR_variations(const MergeData::Host& host) {  // host_xor
  VariationsSet variations;
  for (auto combo_idx : host.compat_indices) {
    Variations v{-1, -1, -1, -1, -1, -1, -1, -1, -1};
    util::for_each_source_index(combo_idx, host.compat_idx_lists,
        [&host, &v, combo_idx](index_t list_idx, index_t src_idx) {
          const auto& src = host.src_lists.at(list_idx).at(src_idx);
          if (UsedSources::merge_variations(v, src.usedSources.variations)) {
            return true;
          }
          assert(false);
        });
    variations.insert(v);
  }
  std::cerr << "XOR unique variations: " << variations.size() << std::endl;
}

//
// filterPreparation
//

Value filterPreparation(const CallbackInfo& info) {
  Env env = info.Env();
  if (!info[0].IsArray()) {
    TypeError::New(env, "filterPreparation: invalid parameter")
      .ThrowAsJavaScriptException();
    return env.Null();
  }
  // arg0
  auto nc_data_lists = makeNcDataLists(env, info[0].As<Array>());
  // --
  auto success{true};
  MFD.host_or.src_lists = std::move(build_src_lists(nc_data_lists));
  if (MFD.host_or.src_lists.size()) {
    if (!get_merge_data(MFD.host_or.src_lists, MFD.host_or, MFD.device_or,
            MergeType::OR, cudaStreamPerThread)) {
      std::cerr << "failed to merge OR args" << std::endl;
      success = false;
    } else {
      build_unique_variations(MFD.host_or, "OR");
    }
  }
  if (success) {
    build_unique_variations(MFD.host_xor, "XOR");
    filter_init(MFD);
    cuda_memory_dump("filter preparation");
  }
  return Boolean::New(env, success);
}

//
// computeCombosForSum
//

Value computeCombosForSum(const CallbackInfo& info) {
  Env env = info.Env();
  if (!info[0].IsNumber() && !info[1].IsNumber() && !info[2].IsBoolean()) {
    TypeError::New(env, "computeCombosForSum: invalid parameter type")
      .ThrowAsJavaScriptException();
    return env.Null();
  }
  // arg0
  auto sum = info[0].As<Number>().Int32Value();
  // arg1
  auto max = info[1].As<Number>().Int32Value();
  // arg2
  //auto dump = info[2].As<Boolean>();
  // --
  compute_combos_for_sum(sum, max);
  // TODO: move save_current_candidate_counts() here?
  return env.Null();
}
  
//
// filterCandidatesForSum
//

Value filterCandidatesForSum(const CallbackInfo& info) {
  Env env = info.Env();
  if (!info[0].IsNumber() || !info[1].IsNumber() || !info[2].IsNumber()
      || !info[3].IsNumber() || !info[4].IsNumber() || !info[5].IsBoolean()
      || !info[6].IsBoolean()) {
    TypeError::New(env, "fitlerCandidatesForSum: invalid parameter type")
      .ThrowAsJavaScriptException();
    return env.Null();
  }
  const auto sum = info[0].As<Number>().Int32Value();
  assert(sum >= 2);
  [[maybe_unused]] FilterParams filter_params{sum,
      info[1].As<Number>().Int32Value(),  // threads_per_block
      info[2].As<Number>().Int32Value(),  // num_streams
      info[3].As<Number>().Int32Value(),  // stride
      info[4].As<Number>().Int32Value(),  // num_iters
      info[5].As<Boolean>().Value(),      // synchronous
      info[6].As<Boolean>().Value()};     // load_all_prior_sources
  // --
  // this function signifies the end of calls to consider_candidate() for a
  // particular sum. filter_candidates_cuda() will clear candidate data to
  // free memory. now is a good opportunity to save candidate counts for this
  // sum so we can access it later.
#if 1
  save_current_candidate_counts(sum);

  // const auto opt_incompat_sources =
  filter_candidates_cuda(MFD, filter_params);
#if 0
  assert(filter_params.synchronous == opt_incompat_sources.has_value());
  if (opt_incompat_sources.has_value()) {
    const auto stream = cudaStreamPerThread;
    set_incompatible_sources(MFD, opt_incompat_sources.value(), stream);
  }
#endif
#endif
  // NOTE: can only free device data here after synchronous call. 
  return env.Null();
}

//
// getResult
//
Value getResult(const CallbackInfo& info) {
  Env env = info.Env();
  auto result = get_filter_result();
  // free all device data
  MFD.device_xor.cuda_free();
  MFD.device_or.cuda_free();
  // mirrors filter_init() in filterPreparation()
  filter_cleanup();
  cuda_memory_dump("filter complete");
#if 1
  std::cerr
      << "total consider(C++) durations - compute: " << get_compute_duration()
      << "ms, make_indices: " << get_make_indices_duration() << "ms\n";
#endif
  return wrap(env, result);
}

//
// Components
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
  auto sums = components::show(name_list, MFD.host_xor.merged_xor_src_list);
  return wrap(env, sums);
}

Value checkClueConsistency(const CallbackInfo& info) {
  Env env = info.Env();
  if (!info[0].IsArray() && !info[1].IsNumber() && !info[2].IsNumber()
      && !info[3].IsBoolean()) {
    TypeError::New(env, "checkClueConsistency: invalid parameter type")
      .ThrowAsJavaScriptException();
    return env.Null();
  }
  // arg0:
  auto name_list = makeStringList(env, info[0].As<Array>());
  // arg1:
  auto max_sources = info[1].As<Number>().Int32Value();
  // arg2:
  auto version = info[2].As<Number>().Int32Value();
  // --
  bool result{true};
  switch (version) {
  case 1:
    result = components::old_consistency_check(
        name_list, MFD.host_xor.merged_xor_src_list);
    break;
  case 2:
    components::consistency_check(std::move(name_list), max_sources);
    break;
  default:
    assert(false);
  }
  return Boolean::New(env, result);
}

Value getConsistencyCheckResults(const CallbackInfo& info) {
  Env env = info.Env();
  return wrap(env, components::get_consistency_check_results());
}

LogOptions makeLogOptions(Env& env, const Object& jsObject) {
  LogOptions log_args;
  auto jsQuiet = jsObject.Get("quiet");
  if (!jsQuiet.IsUndefined()) {
    if (!jsQuiet.IsBoolean()) {
      TypeError::New(env, "makeLogOptions: invalid quiet arg")
        .ThrowAsJavaScriptException();
      return {};
    }
    log_args.quiet = jsQuiet.As<Boolean>();
  }
  auto jsMemory = jsObject.Get("memory");
  if (!jsMemory.IsUndefined()) {
    if (!jsMemory.IsBoolean()) {
      TypeError::New(env, "makeLogOptions: invalid memory arg")
        .ThrowAsJavaScriptException();
      return {};
    }
    log_args.mem_dumps = jsMemory.As<Boolean>();
  }
  auto jsAllocs = jsObject.Get("allocations");
  if (!jsAllocs.IsUndefined()) {
    if (!jsAllocs.IsBoolean()) {
      TypeError::New(env, "makeLogOptions: invalid allocations arg")
        .ThrowAsJavaScriptException();
      return {};
    }
    log_args.mem_allocs = jsAllocs.As<Boolean>();
  }
  auto jsVerbose = jsObject.Get("verbose");
  if (!jsVerbose.IsUndefined()) {
    if (!jsVerbose.IsNumber()) {
      TypeError::New(env, "makeLogOptions: invalid verbose arg")
        .ThrowAsJavaScriptException();
      return {};
    }
    log_args.level = static_cast<LogLevel>(jsVerbose.As<Number>().Int32Value());
  }
  return log_args;
}

//
// Misc
//

Value setOptions(const CallbackInfo& info) {
  Env env = info.Env();
  if (!info[0].IsObject()) {
    TypeError::New(env, "setOptions: invalid parameter type")
      .ThrowAsJavaScriptException();
    return env.Null();
  }
  // arg0
  set_log_options(makeLogOptions(env, info[0].As<Object>()));
  // --
  return env.Null();
}

Value dumpMemory(const CallbackInfo& info) {
  Env env = info.Env();
  if (!(info[0].IsString() || !info[0].IsUndefined())) {
    TypeError::New(env, "dumpMemory: invalid parameter type")
        .ThrowAsJavaScriptException();
    return env.Null();
  }
  // arg 0
  if (info[0].IsString()) {
    auto header = info[0].As<String>().Utf8Value();
    clue_manager::dump_memory(header);
  } else {
    clue_manager::dump_memory();
  }
  return env.Null();
}

Object Init(Env env, Object exports) {
  // clue-manager
  //
  exports["setPrimaryNameSrcIndicesMap"] = Function::New(env,
      setPrimaryNameSrcIndicesMap);
  exports["setCompoundClueNameSourcesMap"] = Function::New(env,
      setCompoundClueNameSourcesMap);
  exports["isKnownSourceMapEntry"] = Function::New(env, isKnownSourceMapEntry);
  exports["addCompoundClue"] = Function::New(env, addCompoundClue);

  // validator
  //
  exports["validateSources"] = Function::New(env, validateSources);

  // combo-maker
  //
  exports["mergeCompatibleXorSourceCombinations"] =
      Function::New(env, mergeCompatibleXorSourceCombinations);
  exports["filterPreparation"] = Function::New(env, filterPreparation);
  exports["computeCombosForSum"] = Function::New(env, computeCombosForSum);
  exports["filterCandidatesForSum"] = Function::New(env,
      filterCandidatesForSum);
  exports["getResult"] = Function::New(env, getResult);

  // components
  //
  exports["showComponents"] = Function::New(env, showComponents);
  exports["checkClueConsistency"] = Function::New(env, checkClueConsistency);
  exports["getConsistencyCheckResults"] = Function::New(env,
      getConsistencyCheckResults);

  // misc
  //
  exports["setOptions"] = Function::New(env, setOptions);
  exports["dumpMemory"] = Function::New(env, dumpMemory);

  return exports;
}

}  // namespace

NODE_API_MODULE(experiment, Init)
