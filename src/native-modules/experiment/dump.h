#ifndef include_dump_h
#define include_dump_h

#include <unordered_map>
#include <string>
#include <iostream>
#include "combo-maker.h"

//using namespace std;

inline std::string indent = "  ";

inline void dump(
  const cm::NameCountList& list, std::string_view spaces, bool firstIndent = true) {
  if (firstIndent) std::cout << spaces;
  std::cout << '[' << std::endl;
  for (const auto& nc : list) {
    std::cout << spaces << indent << "{ '" << nc.name << "':" << nc.count << " },"
         << std::endl;
  }
  std::cout << spaces << ']' << std::endl;
}

inline void dump(
  const std::vector<std::string>& list, std::string_view spaces, bool firstIndent = true) {
  if (firstIndent) std::cout << spaces;
  std::cout << '[' << std::endl;
  for (const auto& str : list) {
    std::cout << spaces << indent << "'" << str << "'," << std::endl;
  }
  std::cout << spaces << ']' << std::endl;
}

inline void dump(const cm::SourceData& sd, std::string_view spaces) {
  std::cout << spaces << '{' << std::endl;
  std::cout << spaces << indent << "primaryNameSrcList: ";
  dump(sd.primaryNameSrcList, std::string(spaces).append(indent), false);
  //std::cout << spaces << indent << "sourceNcCsvList: ";
  //dump(sd.sourceNcCsvList, std::string(spaces).append(indent), false);
  std::cout << spaces << indent << "ncList: ";
  dump(sd.ncList, std::string(spaces).append(indent), false);
  std::cout << spaces << '}' << std::endl;
}

inline void dump(const cm::SourceList& sl, std::string_view spaces) {
  std::cout << spaces << '[' << std::endl;
  for (const auto& sd : sl) {
    dump(sd, std::string(spaces).append(indent));
  }
  std::cout << spaces << ']' << std::endl;
}

inline void dump(const std::vector<cm::SourceList>& vec) {
  std::string spaces{};
  std::cout << spaces << '[' << std::endl;
  for (const auto& sl : vec) {
    dump(sl, std::string(spaces).append(indent));
  }
  std::cout << spaces << ']' << std::endl;
}

inline void dump(const cm::SourceListMap& map) {
  for (auto it = map.cbegin(); it != map.cend(); ++it) {
    std::cout << '[' << std::endl;
    std::cout << indent << "'" << it->first << "'" << ',' << std::endl;
    dump(it->second, indent);
    std::cout << ']' << std::endl;
  }
}

inline void dump(const cm::NCData& data, std::string_view spaces) {
  std::cout << spaces << '{' << std::endl;
  dump(data.ncList, std::string(spaces).append(indent));
  std::cout << spaces << '}' << std::endl;
}

inline void dump(const cm::NCDataList& list, std::string_view spaces) {
  std::cout << spaces << '[' << std::endl;
  for (auto it = list.cbegin(); it != list.cend(); ++it) {
    dump(*it, std::string(spaces).append(indent));
  }
  std::cout << spaces << ']' << std::endl;
}

inline void dump(const std::vector<cm::NCDataList>& lists) {
  std::cout << '[' << std::endl;
  for (auto it = lists.cbegin(); it != lists.cend(); ++it) {
    dump(*it, indent);
  }
  std::cout << ']' << std::endl;
}

#endif // include_dump_h
