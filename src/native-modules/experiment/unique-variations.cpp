#include <iostream>
#include <vector>
#include "unique-variations.h"
#include "cuda-types.h"
#include "log.h"
#include "util.h"

namespace cm {

namespace {

void show_all_sources(const MergeData::Host& host, fat_index_t flat_idx) {
  using namespace std::literals;
  Variations v = {-1, -1, -1, -1, -1, -1, -1, -1, -1};
  //  auto orig_flat_idx = flat_idx;
  std::cerr << "all sources for flat_idx: " << flat_idx << std::endl;
  util::for_each_source_index(flat_idx, host.compat_idx_lists,  //
      [&](index_t list_idx, index_t src_idx) {
        /*
        printf("list_idx: %d, list_size: %u"
               ", remain_flat_idx: %lu, idx: %d, src_idx: %u into:\n",
            list_idx, idx_list_size, flat_idx, int(idx), src_idx);
        */
        fprintf(stderr, "list_idx: %u, src_idx: %u:\n", list_idx, src_idx);

        const auto& src = host.src_lists.at(list_idx).at(src_idx);
        auto success =
            UsedSources::merge_variations(v, src.usedSources.variations);
        std::cerr << NameCount::listToString(src.primaryNameSrcList)
                  << std::endl
                  << "  merged as: " << util::join(v, ","s)
                  << " - success: " << std::boolalpha << success << std::endl;
      });
}

struct VariationsIndex {
  Variations variations{-1, -1, -1, -1, -1, -1, -1, -1, -1};
  index_t index{};
};

auto get_variations_index_list(const MergeData::Host& host) {
  using namespace std::literals;
  std::vector<VariationsIndex> variations_idx_list;
  for (index_t idx{}; auto combo_idx : host.compat_indices) {
    VariationsIndex vi;
    util::for_each_source_index(combo_idx, host.compat_idx_lists,
        [&host, &v = vi.variations, combo_idx](
            index_t list_idx, index_t src_idx) {
          const auto& src = host.src_lists.at(list_idx).at(src_idx);
          if (UsedSources::merge_variations(v, src.usedSources.variations)) {
            return true;
          }
          std::cerr << "failed merging variations of " << combo_idx << std::endl
                    << util::join(src.usedSources.variations, ","s) << " of "
                    << NameCount::listToString(src.primaryNameSrcList)
                    << " to:\n"
                    << util::join(v, ","s) << std::endl;
          //show_all_sources(host, combo_idx);
          assert(false);
        });
    vi.index = idx++;
    variations_idx_list.push_back(std::move(vi));
  }
  return variations_idx_list;
}

auto make_sorted_compat_indices(const FatIndexList& compat_indices,
    const std::vector<VariationsIndex>& sorted_variations_idx_list) {
  FatIndexList sorted_indices;
  for (const auto& vi : sorted_variations_idx_list) {
    sorted_indices.push_back(compat_indices.at(vi.index));
  }
  return sorted_indices;
}

auto make_unique_variations(const std::vector<VariationsIndex>& sorted_vi_list) {
  std::vector<UniqueVariations> unique_variations;
  index_t sum_of_indices{};
  for (auto it = sorted_vi_list.cbegin(); it != sorted_vi_list.cend();) {
    auto range_end = std::find_if_not(it, sorted_vi_list.end(),  //
        [&first_vi = *it](const VariationsIndex& vi) {
          return std::equal_to{}(first_vi.variations, vi.variations);
        });
    const auto num_indices = std::distance(it, range_end);
    unique_variations.emplace_back(it->variations, sum_of_indices,
        std::distance(sorted_vi_list.begin(), it), num_indices);
    sum_of_indices += num_indices;
    it = range_end;
  }
  return unique_variations;
}

// for generated sources

struct SourceCompatCRefSrcIndex {
  std::reference_wrapper<const SourceCompatibilityData> src_compat_cref;
  SourceIndex src_idx{};
};

auto make_src_compat_cref_src_idx_list(const CandidateList& candidates) {
  std::vector<SourceCompatCRefSrcIndex> result;
  for (size_t list_idx{}; list_idx < candidates.size(); ++list_idx) {
    const auto& src_compat_list = candidates.at(list_idx).src_list_cref.get();
    for (size_t src_idx{}; src_idx < src_compat_list.size(); ++src_idx) {
      result.emplace_back(std::cref(src_compat_list.at(src_idx)),
          SourceIndex{index_t(list_idx), index_t(src_idx)});
    }
  }
  return result;
}

auto make_idx_lists(const CandidateList& candidates,
    const std::vector<SourceCompatCRefSrcIndex>& src_compat_cref_src_idx_list) {
  std::vector<IndexList> idx_lists(candidates.size());
  for (size_t list_idx{} ; list_idx < idx_lists.size(); ++list_idx) {
    idx_lists.at(list_idx).resize(
        candidates.at(list_idx).src_list_cref.get().size());
  }
  for (size_t idx{}; idx < src_compat_cref_src_idx_list.size(); ++idx) {
    const auto src_idx = src_compat_cref_src_idx_list.at(idx).src_idx;
    idx_lists.at(src_idx.listIndex).at(src_idx.index) = idx;
  }
  return idx_lists;
}

}  // namespace

// For XOR and OR compat_indices lists
void build_unique_variations(
    FilterData::HostCommon& host, std::string_view name) {
  auto variations_idx_list = get_variations_index_list(host);
  std::ranges::sort(variations_idx_list,
      [](const VariationsIndex& a, const VariationsIndex& b) {
        return std::less{}(a.variations, b.variations);
      });
  host.compat_indices = std::move(
      make_sorted_compat_indices(host.compat_indices, variations_idx_list));
  host.unique_variations =
      std::move(make_unique_variations(variations_idx_list));
  if (log_level(Verbose)) {
    std::cerr << name << " unique variations: " << host.unique_variations.size()
              << std::endl;
  }
}

auto make_variations_sorted_idx_lists(
    const CandidateList& candidates) -> std::vector<IndexList> {
  auto src_compat_cref_src_idx_list = make_src_compat_cref_src_idx_list(candidates);
  std::ranges::sort(src_compat_cref_src_idx_list,
      [](const SourceCompatCRefSrcIndex& a, const SourceCompatCRefSrcIndex& b) {
        return std::less{}(a.src_compat_cref.get().usedSources.variations,
            b.src_compat_cref.get().usedSources.variations);
      });
  return make_idx_lists(candidates, src_compat_cref_src_idx_list);
}

auto make_src_compat_list(const CandidateList& candidates,
    const std::vector<IndexList>& idx_lists) -> SourceCompatibilityList {
  SourceCompatibilityList src_compat_list(util::sum_sizes(idx_lists));
  for (size_t list_idx{}; list_idx < idx_lists.size(); ++list_idx) {
    const auto& idx_list = idx_lists.at(list_idx);
    for (size_t idx{}; idx < idx_list.size(); ++idx) {
      src_compat_list.at(idx_list.at(idx)) =
          candidates.at(list_idx).src_list_cref.get().at(idx);
    }
  }
  return src_compat_list;
}

// NB: src_compat_list must be sorted by variations
auto make_unique_variations(const SourceCompatibilityList& sorted_src_compat_list)
    -> std::vector<UniqueVariations> {
  std::vector<UniqueVariations> unique_variations;
  index_t sum_of_indices{};
  for (auto it = sorted_src_compat_list.cbegin();
      it != sorted_src_compat_list.cend();) {
    auto range_end = std::find_if_not(it, sorted_src_compat_list.end(),  //
        [&first_src = *it](const SourceCompatibilityData& src) {
          return std::equal_to{}(
              first_src.usedSources.variations, src.usedSources.variations);
        });
    const auto num_indices = std::distance(it, range_end);
    unique_variations.emplace_back(it->usedSources.variations, sum_of_indices,
        std::distance(sorted_src_compat_list.begin(), it), num_indices);
    sum_of_indices += num_indices;
    it = range_end;
  }
  return unique_variations;
}

}  // namespace cm
