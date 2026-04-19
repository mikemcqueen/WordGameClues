#include <algorithm>
#include <cstdio>
#include <cstring>
#include "name-count.h"

namespace cm {

std::string NameCount::toString() const {
  return makeString(name, count);
}

std::string NameCount::makeString(const std::string& name, int count) {
  char buf[128];
  snprintf(buf, sizeof(buf), "%s:%d", name.c_str(), count);
  return buf;
}

std::string NameCount::makeString(const NameCount& nc1, const NameCount& nc2) {
  char buf[64];
  snprintf(buf, sizeof(buf), "%s:%d,%s:%d", nc1.name.c_str(), nc1.count,
      nc2.name.c_str(), nc2.count);
  return buf;
}

std::string NameCount::makeString(const std::string& name1,
    const std::string& name2) {
  return name1 + "," + name2;
}

std::vector<std::string> NameCount::listToNameList(const NameCountList& list) {
  std::vector<std::string> names;
  for (auto it = list.cbegin(); it != list.cend(); ++it) {
    names.emplace_back(it->name);
  }
  return names;
}

void NameCount::listSort(NameCountList& list) {
  std::ranges::sort(list,
      [](const NameCount& a, const NameCount& b) { return a.name < b.name; });
}

void NameCount::listSort(NameCountCRefList& cref_list) {
  std::ranges::sort(cref_list,
      [](const NameCountCRef& a, const NameCountCRef& b) {
        return a.get().name < b.get().name;
      });
}

std::string NameCount::listToNameCsv(const NameCountCRefList& cref_list) {
  char buf[1280];
  buf[0] = 0;
  for (auto it = cref_list.cbegin(); it != cref_list.cend(); ++it) {
    std::strcat(buf, it->get().name.c_str());
    if ((it + 1) != cref_list.cend()) {  // TODO std::next() ?
      std::strcat(buf, ",");
    }
  }
  return buf;
}

std::string NameCount::listToString(const NameCountCRefList& cref_list) {
  char buf[1280];
  buf[0] = 0;
  for (auto it = cref_list.cbegin(); it != cref_list.cend(); ++it) {
    std::strcat(buf, it->get().toString().c_str());
    if ((it + 1) != cref_list.cend()) { // TODO std::next() ?
      std::strcat(buf, ",");
    }
  }
  return buf;
}

std::string NameCount::listToString(const std::vector<std::string>& list) {
  char buf[1280];
  buf[0] = 0;
  for (auto it = list.cbegin(); it != list.cend(); ++it) {
    std::strcat(buf, it->c_str());
    if ((it + 1) != list.cend()) { // TODO std::next() ?
      std::strcat(buf, ",");
    }
  }
  return buf;
}

std::string NameCount::listToString(const NameCountList& list) {
  char buf[1280];
  buf[0] = 0;
  for (auto it = list.cbegin(); it != list.cend(); ++it) {
    std::strcat(buf, it->toString().c_str());
    if ((it + 1) != list.cend()) { // TODO std::next() ?
      std::strcat(buf, ",");
    }
  }
  return buf;
}

std::string NameCount::listToString(const std::vector<const NameCount*>& list) {
  char buf[1280] = { 0 };
  for (auto it = list.cbegin(); it != list.cend(); ++it) {
    std::strcat(buf, (*it)->toString().c_str());
    if ((it + 1) != list.cend()) {
      std::strcat(buf, ",");
    }
  }
  return buf;
}

UsedSources NameCount::listToUsedSources(const NameCountList& list) {
  UsedSources usedSources{};
  for (const auto& nc : list) {
    // TODO: assert this is true
    const auto primary_src = PrimarySourceId{nc.count};
    if (primary_src.is_candidate()) {
      usedSources.addSource(primary_src);
    }
  }
  return usedSources;
}

}  // namespace cm
