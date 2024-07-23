#pragma once
#include <string>
#include <string_view>
#include <vector>
#include <napi.h>
#include "clue-manager.h"
#include "combo-maker.h"

namespace cm {

std::vector<int> makeIntList(Napi::Env& env, const Napi::Array& jsList);

IndexList makeIndexList(Napi::Env& env, const Napi::Array& jsList);

std::vector<std::string> makeStringList(
    Napi::Env& env, const Napi::Array& jsList);

NameCount makeNameCount(Napi::Env& env, const Napi::Object& jsObject);

NameCountList makeNameCountList(Napi::Env& env, const Napi::Array& jsList);

SourceData makeSourceData(Napi::Env& env, const Napi::Object& jsSourceData,
    std::string_view nameSrcList = "primaryNameSrcList");

SourceList makeSourceList(Napi::Env& env, const Napi::Array& jsList,
    std::string_view nameSrcList = "primaryNameSrcList");

NCData makeNcData(Napi::Env& env, const Napi::Object& jsObject);

NCDataList makeNcDataList(Napi::Env& env, const Napi::Array& jsList);

std::vector<NCDataList> makeNcDataLists(
    Napi::Env& env, const Napi::Array& jsList);

clue_manager::NameSourcesMap makeNameSourcesMap(
    Napi::Env& env, const Napi::Array& jsList);

SourceCompatibilityData makeSourceCompatibilityDataFromSourceData(
    Napi::Env& env, const Napi::Object& jsSourceData);

SourceCompatibilityData makeSourceCompatibilityDataFromSourceList(
    Napi::Env& env, const Napi::Array& jsSourceList);

SourceCompatibilityList makeSourceCompatibilityListFromMergedSourcesList(
    Napi::Env& env, const Napi::Array& jsList);

OrSourceData makeOrSource(Napi::Env& env, const Napi::Object& jsObject);

OrSourceList makeOrSourceList(Napi::Env& env, const Napi::Array& jsList);

OrArgData makeOrArgData(Napi::Env& env, const Napi::Object& jsObject);

OrArgList makeOrArgList(Napi::Env& env, const Napi::Array& jsList);

}  // namespace cm
