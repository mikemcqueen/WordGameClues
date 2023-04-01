#ifndef include_dump_h
#define include_dump_h

#include <unordered_map>
#include <string>
#include <iostream>
#include "combo-maker.h"

using namespace std;

inline string indent = "  ";

inline void dump(const cm::NameCountList& list, string_view spaces, bool firstIndent = true) {
  if (firstIndent) cout << spaces;
  cout << '[' << endl;
  for (const auto& nc : list) {
    cout << spaces << indent << "{ '" << nc.name << "':" << nc.count << " }," << endl;
  }
  cout << spaces << ']' << endl;
}

inline void dump(const vector<string>& list, string_view spaces, bool firstIndent = true) {
  if (firstIndent) cout << spaces;
  cout << '[' << endl;
  for (const auto& str : list) {
    cout << spaces << indent << "'" << str << "'," << endl;
  }
  cout << spaces << ']' << endl;
}

inline void dump(const cm::SourceData& sd, string_view spaces) {
  cout << spaces << '{' << endl;
  cout << spaces << indent << "primaryNameSrcList: ";
  dump(sd.primaryNameSrcList, string(spaces).append(indent), false);
  cout << spaces << indent << "sourceNcCsvList: ";
  dump(sd.sourceNcCsvList, string(spaces).append(indent), false);
  cout << spaces << indent << "ncList: ";
  dump(sd.ncList, string(spaces).append(indent), false);
  cout << spaces << '}' << endl;
}

inline void dump(const cm::SourceList& sl, string_view spaces) {
  cout << spaces << '[' << endl;
  for (const auto& sd : sl) {
    dump(sd, string(spaces).append(indent));
  }
  cout << spaces << ']' << endl;
}

inline void dump(const std::vector<cm::SourceList>& vec) {
  string spaces{};
  cout << spaces << '[' << endl;
  for (const auto& sl : vec) {
    dump(sl, string(spaces).append(indent));
  }
  cout << spaces << ']' << endl;
}

inline void dump(const cm::SourceListMap& map) {
  for (auto it = map.cbegin(); it != map.cend(); ++it) {
    cout << '[' << endl;
    cout << indent << "'" << it->first << "'" << ',' << endl;
    dump(it->second, indent);
    cout << ']' << endl;
  }
}

inline void dump(const cm::NCData& data, string_view spaces) {
  cout << spaces << '{' << endl;
  dump(data.ncList, string(spaces).append(indent));
  cout << spaces << '}' << endl;
}

inline void dump(const cm::NCDataList& list, string_view spaces) {
  cout << spaces << '[' << endl;
  for (auto it = list.cbegin(); it != list.cend(); ++it) {
    dump(*it, string(spaces).append(indent));
  }
  cout << spaces << ']' << endl;
}

inline void dump(const vector<cm::NCDataList>& lists) {
  cout << '[' << endl;
  for (auto it = lists.cbegin(); it != lists.cend(); ++it) {
    dump(*it, indent);
  }
  cout << ']' << endl;
}

#endif // include_dump_h
