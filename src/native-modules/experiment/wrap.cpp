// wrap.cpp

#include <chrono>
#include <iostream>
#include "wrap.h"

namespace cm {

using namespace Napi;

Object wrap(Env& env, const NameCount& nc) {
  Object jsObj = Object::New(env);
  jsObj.Set("name", String::New(env, nc.name));
  jsObj.Set("count", Number::New(env, nc.count));
  return jsObj;
}

Array wrap(Env& env, const NameCountList& ncList) {
  Array jsList = Array::New(env, ncList.size());
  for (size_t i{}; i < ncList.size(); ++i) {
    jsList.Set(i, wrap(env, ncList[i]));
  }
  return jsList;
}

Object wrap(Env& env, const XorSource& xorSource) {
  Object jsObj = Object::New(env);
  jsObj.Set("primaryNameSrcList", wrap(env, xorSource.primaryNameSrcList));
  jsObj.Set("ncList", wrap(env, xorSource.ncList));
  return jsObj;
}

Array wrap(Env& env, const XorSourceList& xorSourceList) {
  using namespace std::chrono;
  auto t0 = high_resolution_clock::now();
  Array jsList = Array::New(env, xorSourceList.size());
  for (size_t i{}; i < xorSourceList.size(); ++i) {
    jsList.Set(i, wrap(env, xorSourceList[i]));
  }
  auto t1 = high_resolution_clock::now();
  [[maybe_unused]] auto t_dur = duration_cast<milliseconds>(t1 - t0).count();
  /*
  std::cout << " wrap src-list(" << xorSourceList.size() << ")"
            << " - " << t_dur << "ms" << std::endl;
  */
  return jsList;
}

Array wrap(Env& env,
  std::vector<clue_manager::KnownSourceMapValueCRef> cref_entries) {
  //
  Array jsList = Array::New(env, cref_entries.size());
  for (size_t i{}; i < cref_entries.size(); ++i) {
    jsList.Set(i, wrap(env, cref_entries.at(i).get().src_list));
  }
  return jsList;
}

  /*
template <template <class T> class C>
Array wrap(Env& env, const T<C>& container) {
  Array jsList = Array::New(env, container.size());
  for (int i{}; const auto& str : container) {
    jsList.Set(i++, String::New(env, str));
  }
  return jsList;
}
  */

  /*
Array wrap(Env& env, const std::unordered_set<std::string>& str_set) {
  Array jsList = Array::New(env, str_set.size());
  for (int i{}; const auto& str : str_set) {
    jsList.Set(i++, String::New(env, str));
  }
  return jsList;
}

Array wrap(Env& env, const std::vector<std::string>& strList) {
  Array jsList = Array::New(env, strList.size());
  for (size_t i{}; i < strList.size(); ++i) {
    jsList.Set(i, String::New(env, strList[i]));
  }
  return jsList;
}
  */

Array wrap(Napi::Env& env, const std::set<int>& values) {
  Array jsList = Array::New(env, values.size());
  int idx{};
  for (auto it = values.begin(); it != values.end(); ++it, ++idx) {
    jsList.Set(idx, Number::New(env, *it));
  }
  return jsList;
}

Object wrap(
    Napi::Env& env, const std::unordered_map<std::string, NameCountList>& map) {
  Object jsObj = Object::New(env);
  for (const auto& [key, value] : map) {
    jsObj.Set(key, wrap(env, value));
  }
  return jsObj;
}

}  // namespace cm
