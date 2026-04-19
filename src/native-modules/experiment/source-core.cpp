#include <iostream>
#include "source-core.h"

namespace cm {

void SourceData::dumpList(const SourceList& src_list) {
  for (const auto& src : src_list) {
    std::cerr << " " << NameCount::listToString(src.ncList) << " - "
              << NameCount::listToString(src.primaryNameSrcList) << std::endl;
  }
}

void SourceData::dumpList(const SourceCRefList& src_cref_list) {
  // TODO: auto& or just auto here?
  for (const auto src_cref : src_cref_list) {
    std::cerr << " " << NameCount::listToString(src_cref.get().ncList)
              << " - "
              << NameCount::listToString(src_cref.get().primaryNameSrcList)
              << std::endl;
  }
}

}  // namespace cm
