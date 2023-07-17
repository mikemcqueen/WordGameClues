// candidates.cpp

#include <chrono>
#include "combo-maker.h"
#include "candidates.h"

namespace cm {
  void filterCandidatesCuda(int sum);
  
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

  auto addCandidate(int sum, const std::string& combo, int index) -> int {
    IndexComboListMap& indexCombosMap =
      allSumsCandidateData[sum - 2].indexComboListMap;
    auto it = indexCombosMap.find(index);
    assert(it != indexCombosMap.end());
    it->second.insert(combo);
    return index;
  }
  
  auto addCandidate(int sum, std::string&& combo,
    cm::SourceCompatibilityList&& compatList) -> int
  {
    int index{};
    std::set<std::string> combos{};
    combos.emplace(std::move(combo));
    
    // TODO this could be simplified. add sumData if new sum, then common code.
    if (sum == (int)allSumsCandidateData.size() + 2) {
      SourceCompatibilityLists sourceCompatLists{};
      sourceCompatLists.emplace_back(std::move(compatList));
      
      IndexComboListMap indexComboListMap; 
      auto [ignore, success] =
        indexComboListMap.insert(std::make_pair(index, std::move(combos)));
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
        indexComboListMap.insert(std::make_pair(index, std::move(combos)));
      assert(success);
    }
    return index;
  }

  /*
  auto deviceAddCandidate(int sum, std::string&& combo, int index) -> int {
    IndexComboListMap& indexCombosMap =
      allSumsCandidateData[sum - 2].indexComboListMap;
    auto it = indexCombosMap.find(index);
    assert(it != indexCombosMap.end());
    it->second.emplace_back(std::move(combo));
    return index;
  }
  */
  
  #if 0
  auto deviceAddCandidate(int sum, std::string&& combo,
    cm::SourceCompatibilityList&& compatList) -> int
  {
    //int index{};
    /*
    const auto size = compatList.size();
    DeviceSourceCompatListAndSize listAndSize{ std::move(compatList), size };
    */
    if (sum == (int)allSumsCandidateData.size() + 2) {
      allSumsCandidateData.emplace_back(DeviceSourceCompatListAndSizes{});
    }
    auto& listAndSizes = allSumsCandidateData[sum - 2]
      .deviceSourceCompatListAndSizes;
    listAndSizes.sizes.push_back(compatList.size());
    for (auto& compatData: compatList) {
      listAndSizes.device_list.push_back(std::move(compatData));
    }
    
    std::set<std::string> combos{};
    combos.emplace(std::move(combo));
    
    IndexComboListMap indexComboListMap; 
    auto index = (int)listAndSizes.sizes.size() - 1;
    const auto [ignore, success] =
      indexComboListMap.insert(std::make_pair(index, std::move(comboList)));
    assert(success);
      
    /*
    OneSumCandidateData oneSumData{ std::move(listAndSizes),
      std::move(indexComboListMap) };
    allSumsCandidateData.emplace_back(std::move(oneSumData));
    } else {
      auto& sourceCompatLists =
        allSumsCandidateData[sum - 2].device_sourceCompatLists;
      sourceCompatLists.emplace_back(std::move(listAndSize));
      index = sourceCompatLists.size - 1;
      
      auto& indexComboListMap = allSumsCandidateData[sum - 2].indexComboListMap;
      auto [ignore, success] =
        indexComboListMap.insert(std::make_pair(index, std::move(comboList)));
      assert(success);
    }
    */
    return index;
  }
  #endif
} // namespace cm

