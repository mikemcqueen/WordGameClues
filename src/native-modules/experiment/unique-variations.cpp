#include <iostream>
#include <vector>
#include "combo-maker.h"
#include "cuda-types.h"
#include "unique-variations.h"
#include "log.h"
#include "util.h"

namespace cm {

namespace {

void show_all_sources(const MergeData::Host& host, fat_index_t flat_idx) {
  using namespace std::literals;
  UsedSources::Variations v = {-1, -1, -1, -1, -1, -1, -1, -1, -1};
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
  UsedSources::Variations variations{-1, -1, -1, -1, -1, -1, -1, -1, -1};
  index_t index{};
};

auto get_variations_index_list(const MergeData::Host& host) {  // host_or
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
                    //<< util::join(copy, ","s) << ", after merge:\n"
                    << util::join(v, ","s) << std::endl;
          //show_all_sources(host, combo_idx);
          assert(false);
        });
    vi.index = idx++;
    variations_idx_list.push_back(std::move(vi));
  }
  return variations_idx_list;
}

auto get_sorted_compat_indices(const FatIndexList& compat_indices,
    const std::vector<VariationsIndex>& sorted_variations_idx_list) {
  FatIndexList sorted_indices;
  for (const auto& vi : sorted_variations_idx_list) {
    sorted_indices.push_back(compat_indices.at(vi.index));
  }
  return sorted_indices;
}

auto get_unique_variations(const std::vector<VariationsIndex>& sorted_vi_list) {
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

}  // namespace

void build_unique_variations(
    FilterData::HostCommon& host, std::string_view name) {
  auto variations_idx_list = get_variations_index_list(host);
  std::ranges::sort(variations_idx_list,
      [](const VariationsIndex& a, const VariationsIndex& b) {
        return std::less{}(a.variations, b.variations);
      });
  host.compat_indices = std::move(
      get_sorted_compat_indices(host.compat_indices, variations_idx_list));
  host.unique_variations =
      std::move(get_unique_variations(variations_idx_list));
  if (log_level(Verbose)) {
    std::cerr << name << " unique variations: " << host.unique_variations.size()
              << std::endl;
  }
}

}  // namespace cm
