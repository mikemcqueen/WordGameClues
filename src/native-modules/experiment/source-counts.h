#ifndef INCLUDE_SOURCECOUNTS_H
#define INCLUDE_SOURCECOUNTS_H
 
#include "combo-maker.h"

namespace cm {

//////////

  using SourceSet = std::unordered_set<int>;
  using SourceCountMap = std::unordered_map<int, int>;
  using SourceFreqMap = std::unordered_map<int, double>;
  
  struct SourceCounts {
    void addToTotals(int source) {
      auto it = totals_map.find(source);
      if (it == totals_map.end()) {
        totals_map.insert(std::make_pair(source, 1));
      } else {
        it->second++;
      }
    }

    void addSources(int index, const Sources& sources) {
      auto& source_set = source_sets[index];
      for (int s{ 1 }; s <= kNumSentences; ++s) {
        auto start = Source::getFirstIndex(s);
        for (int i{}; i < kMaxUsedSourcesPerSentence; ++i) {
          auto source = sources.at(start + i);
          if (source == -1) break;
          source_set.insert(source);
          addToTotals(source);
        }
      }
    }
    
    void addSources(int index, const LegacySources& legacySources) {
      auto& source_set = source_sets[index];
      for (int i{}; i < kMaxLegacySources; ++i) {
        auto source = legacySources.at(i);
        if (!source) continue;
        source_set.insert(source);
        addToTotals(source);
      }
    }
    
    std::vector<SourceSet> source_sets;
    SourceCountMap totals_map;
  }; // struct SourceCounts
  
template<typename T>
auto makeSourceCounts(const T& sourceList) {
  SourceCounts sourceCounts;
  sourceCounts.source_sets.resize(sourceList.size());
  std::for_each(sourceList.begin(), sourceList.end(),
    [idx = 0, &sourceCounts](const auto& source) mutable {
      sourceCounts.addSources(idx, source.usedSources.sources);
      sourceCounts.addSources(idx, source.legacySources);
      idx++;
    });
  return sourceCounts;
}

  inline auto makeSourceFreqMap(const SourceCounts& /*sourceCounts*/) {
    SourceFreqMap freqMap;
    return freqMap;
  }

  inline auto makeSortedIndices(const SourceCounts& sourceCounts,
                         const SourceFreqMap& /*sourceFreqMap*/)
  {
    std::vector<int> indices(sourceCounts.source_sets.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
      [&sourceCounts/*, &sourceFreqMap*/](int i, int j) {
        return sourceCounts.source_sets[i].size() <
          sourceCounts.source_sets[j].size();
      });
    return indices;
  }

template<typename T>
auto getSortedSourceIndices(const T& sourceList, bool sort = true)
  -> std::vector<int>
{
  if (sort) {
    auto sourceCounts = makeSourceCounts(sourceList);
    auto sourceFreqMap = makeSourceFreqMap(sourceCounts);
    return makeSortedIndices(sourceCounts, sourceFreqMap);
  }
  return std::vector<int>{};
}

} // namespace cm

#endif // INCLUDE_SOURCECOUNTS_H

