// candidates.cpp

#include <chrono>
#include <span>
#include "combo-maker.h"
#include "candidates.h"

namespace cm {
  #define NATIVE_FILTER 0
  #if NATIVE_FILTER
  void filterCandidatesNative(int sum) {
    using namespace std::chrono;
    int num_compatible{};
    const auto& sourceCompatLists =
      allSumsCandidateData[sum - 2].sourceCompatLists;
    auto t0 = high_resolution_clock::now();
    for (const auto& compatList: sourceCompatLists) {
      if (isAnySourceCompatibleWithUseSources(compatList)) {
        ++num_compatible;
      }
    }
    auto t1 = high_resolution_clock::now();
    auto d = duration_cast<milliseconds>(t1 - t0).count();
    std::cerr << "  native filter: compatible(" << num_compatible << ") - "
              << d << "ms" << std::endl;
  }
  #endif

  void filterCandidates(int sum) {
    //filterCandidatesNative(sum);
    filterCandidatesCuda(sum);
  }

  auto addCandidate(int sum, std::string&& combo, int index) -> int {
    IndexComboListMap& indexCombosMap =
      allSumsCandidateData[sum - 2].indexComboListMap;
    auto it = indexCombosMap.find(index);
    assert(it != indexCombosMap.end());
    it->second.emplace_back(std::move(combo));
    return index;
  }
  
  auto addCandidate(int sum, std::string&& combo,
    cm::SourceCompatibilityList&& compatList) -> int
  {
    int index{};
    std::vector<std::string> comboList{};
    comboList.emplace_back(std::move(combo));
    
    if (sum == (int)allSumsCandidateData.size() + 2) {
      SourceCompatibilityLists sourceCompatLists{};
      sourceCompatLists.emplace_back(std::move(compatList));
      index = sourceCompatLists.size() - 1;
      
      IndexComboListMap indexComboListMap; 
      auto [ignore, success] =
        indexComboListMap.insert(std::make_pair(index, std::move(comboList)));
      assert(success);
      
      OneSumCandidateData oneSumData{ std::move(sourceCompatLists),
        std::move(indexComboListMap) };
      allSumsCandidateData.emplace_back(std::move(oneSumData));
    } else {
      auto& sourceCompatLists = allSumsCandidateData[sum - 2].sourceCompatLists;
      sourceCompatLists.emplace_back(std::move(compatList));
      index = sourceCompatLists.size() - 1;
      
      auto& indexComboListMap = allSumsCandidateData[sum - 2].indexComboListMap;
      auto [ignore, success] =
        indexComboListMap.insert(std::make_pair(index, std::move(comboList)));
      assert(success);
    }
    return index;
  }
} // namespace cm
