// validator.cpp

#include <array>
#include <iostream>
#include <vector>
#include "clue-manager.h"
#include "combo-maker.h"
#include "peco.h"
#include "util.h"
#include "validator.h"

using namespace cm;

namespace validator {

namespace {

auto buildNcSourceIndexLists(const NameCountList& nc_list) {
  Peco::IndexListVector idx_lists;
  for (const auto& nc : nc_list) {
    if (nc.count == 1) {
      idx_lists.emplace_back(
        Peco::make_index_list(clue_manager::getPrimaryClueSrcIndices(nc.name)));
    } else {
      idx_lists.emplace_back(
        Peco::make_index_list(clue_manager::get_num_nc_sources(nc)));
    }
  }
  return idx_lists;
}

void display_addends(int sum, const std::vector<std::vector<int>>& addends) {
  std::cout << "sum: " << sum << std::endl;
  for (const auto& combination : addends) {
    std::cout << "[ ";
    for (const auto& num : combination)
      std::cout << num << ' ';
    std::cout << "]" << std::endl;
  }
}

int num_merge_all = 0;;
int num_merge_combo = 0;
int num_full_merges = 0;
long merge_ns = 0;

#if 0
auto get_variation(const NameCount& nc, int s) {
  if (nc.count > 1) {
    const auto& src = clue_manager::get_nc_src_list(nc)[s];
    return Source::getVariation(src.source);
  } else {
    return Source::getVariation(s);
  }
}
#endif

using SrcCRefArray = std::array<SourceCRef, 32>;
static SourceData dummy;
static auto d = std::cref(dummy);
static SrcCRefArray src_cref_array = {d, d, d, d, d, d, d, d, d, d, d, d, d, d,
    d, d, d, d, d, d, d, d, d, d, d, d, d, d, d, d, d, d};

auto same_variations(const NameCountList& nc_list, const IndexList& idx_list) {
  UsedSources::Variations variations =
      make_array<UsedSources::variation_index_t, kNumSentences>(-1);
  for (size_t i{}; i < idx_list.size(); ++i) {
    const auto& nc = nc_list[i];
    if (nc.count > 1) {
      const auto& nc_src = clue_manager::get_nc_src_list(nc)[idx_list[i]];
      if (!UsedSources::merge_variations(
              variations, nc_src.usedSources.variations)) {
        return false;
      }
      src_cref_array[i] = std::cref(nc_src);
    } else {
      if (!UsedSources::merge_one_variation(variations, idx_list[i])) {
        return false;
      }
    }
  }
  return true;
}

// hot inner loop - called millions of times on startup. lots of otherwise
// seemingly unnecessary optimizations to shave off milliseconds
auto mergeNcListCombo(
    const NameCountList& nc_list, const IndexList& idx_list, SourceData& src) {
  // TODO: do we really even *need* primarynameSrcList? precompute is using
  // it for something but it's not clear to me what.

  NameCountList primaryNameSrcList;
  size_t num_sources{};

  ++num_merge_combo;
  src.usedSources.reset();
  using enum SourceData::AddLists;
  // 1st pass, update src compatibility bits only
  for (size_t i{}; i < idx_list.size(); ++i) {
    const auto& nc = nc_list[i];
    if (nc.count > 1) {
      // const auto& nc_src = clue_manager::get_nc_src_list(nc)[idx_list[i]];
      const auto& nc_src = src_cref_array[i].get();
      // NOTE: pass false ONLY if we call same_variations first
      if (!src.addCompoundSource(nc_src, No, false)) {
        num_sources = 0;
        break;
      }
      // src_cref_array[i] = std::cref(nc_src);
      num_sources += nc_src.primaryNameSrcList.size();
    } else {
      if (!src.addPrimaryNameSrc(nc, idx_list[i], No)) {
        num_sources = 0;
        break;
      }
      ++num_sources;
    }
  }
  if (num_sources) {
    // 2nd pass: populate primaryNameSrcList
    primaryNameSrcList.reserve(num_sources);
    for (size_t i{}; i < idx_list.size(); ++i) {
      const auto& nc = nc_list[i];
      if (nc.count > 1) {
        const auto& nc_src = src_cref_array[i].get();
        primaryNameSrcList.insert(primaryNameSrcList.end(),
            nc_src.primaryNameSrcList.begin(), nc_src.primaryNameSrcList.end());
      } else {
        primaryNameSrcList.emplace_back(nc.name, idx_list[i]);
      }
    }
    ++num_full_merges;
    if (0 && (num_full_merges < 1'000)) {
      std::cerr << NameCount::listToString(nc_list) << " - "
                << NameCount::listToString(primaryNameSrcList) << std::endl;
    }
  }
  return primaryNameSrcList;
}

auto mergeAllNcListCombinations(const NameCountList& nc_list,
    Peco::IndexListVector&& idx_lists, const std::string& clue_name) {
  ++num_merge_all;
  auto count = std::accumulate(nc_list.begin(), nc_list.end(), 0,
      [](int sum, const NameCount& nc) { return sum + nc.count; });
  NameCount nc{clue_name, count};
  NameCountList ncl = {nc};
  SourceList src_list;
  SourceData src;
  Peco peco(std::move(idx_lists));
  for (auto idx_list = peco.first_combination(); idx_list;
      idx_list = peco.next_combination()) {
    // NOTE: tried early-emplacing the empty SourceData into the src_list
    // and merging directly into it, to avoid 2.5M usedSources moves and
    // no improvement (possible pessimization?!)
    if (same_variations(nc_list, *idx_list)) {
      auto t = util::Timer::start_timer();
      auto pnsl = mergeNcListCombo(nc_list, *idx_list, src);
      t.stop();
      merge_ns += t.nanoseconds();
      if (!pnsl.empty()) {
        src_list.emplace_back(src.usedSources, std::move(pnsl), ncl);
      }
    }
  }
  return src_list;
}

auto mergeNcListResults(
    const NameCountList& nc_list, const std::string& clue_name) {
  auto idx_lists = buildNcSourceIndexLists(nc_list);
  return mergeAllNcListCombinations(nc_list, std::move(idx_lists), clue_name);
}

NameCountList copyNcListAddNc(
    const NameCountList& nc_list, const std::string& name, int count) {
  // for non-primary (count > 1) check for duplicate name:count entry
  // technically this is allowable if the there are multiple entries
  // of this clue name in the clueList[count] (at least as many entries
  // as there are copies of name in ncList)
  // TODO: make knownSourceMapArray store a count instead of boolean
  // TODO: linear search here yuck. since we have to linear copy
  //       anyway, maybe we could check as we copy if count > 1?
  NameCountList result;
  if ((count == 1) || !NameCount::listContains(nc_list, name, count)) {
    result = nc_list;  // copy
    result.emplace_back(name, count);
  }
  return result;
}

template <typename T>
// requires T = string | int
auto chop_copy(const std::vector<T>& list, const T& chop_value) {
  std::vector<T> result;
  bool chopped = false;
  for (const auto& value : list) {
    if (!chopped && (value == chop_value)) {
      chopped = true;
    } else {
      result.emplace_back(value);
    }
  }
  return result;
}

struct VSForNameAndCountListsArgs {
  NameCountList& nc_list;
  bool validate_all;
};

auto validateSourcesForNameAndCountLists(const std::string& clue_name,
    const std::vector<std::string>& name_list, std::vector<int> count_list,
    NameCountList& nc_list) -> SourceList;

struct VSForNameCountArgs {
  NameCountList& nc_list;
  const std::vector<std::string>& name_list;
  const std::vector<int>& count_list;
};

// TODO: broken NRVO
auto validateSourcesForNameCount(const std::string& clue_name,
    const std::string& name, int count,
    const VSForNameCountArgs& args) -> SourceList {
  auto nc_list = copyNcListAddNc(args.nc_list, name, count);
  if (nc_list.empty()) {
    // TODO:
    // duplicate name:count entry. technically this is allowable for
    // count > 1 if the there are multiple entries of this clue name
    // in the clueList[count]. (at least as many entries as there are
    // copies of name in ncList). SEE ALSO: copyAddNcList()
    // NOTE: this should be fixable with some effort if it ever fires.
    std::cerr << " duplicate nc, " << name << ":" << count << std::endl;
    return {};
  }
  // If only one name & count remain, we're done.
  // (name & count lists are equal length, just test one)
  if (args.name_list.size() == 1u) {
    // NOTE leave this here and at entry point of validateSources
    // assert(args.validate_all && "!validateAll not implemented");
    SourceList src_list = mergeNcListResults(nc_list, clue_name);
    if (!src_list.empty()) { args.nc_list.emplace_back(name, count); }
    return src_list;
  }
  // yeah this is kinda insane recursive logic. do better.
  // name_list.length > 1, remove current name & count, and validate
  // remaining
  auto src_list = validateSourcesForNameAndCountLists(clue_name,
      chop_copy(args.name_list, name), chop_copy(args.count_list, count),
      nc_list);
  if (!src_list.empty()) { args.nc_list = std::move(nc_list); }
  return src_list;
}

auto validateSourcesForNameAndCountLists(const std::string& clue_name,
    const std::vector<std::string>& name_list, std::vector<int> count_list,
    NameCountList& nc_list) -> SourceList {
  // optimization: could have a map of count:boolean entries here
  // on a per-name basis (new map for each outer loop; once a
  // count is checked for a name, no need to check it again
  SourceList src_list;
  const auto& name = name_list.at(0);
  // TODO: could do this test earlier, in calling function, check entire
  // name list.
  if (name != clue_name) {
    // So it seems to me like what I am doing here is testing only if the
    // first name, with all counts, is valid, and if so, pass the rest
    // of the names along blindly, without first verifying that they also
    // have matching valid counts from the remaining counts in count_list.
    //
    // And although I do a bunch of chop_copys in the process, which is
    // unnecessary, at least I don't call merge until all name:counts are
    // matched, which is good.
    //
    // So, bottom line, bullshit unnecessary complex logic that would
    // probably benefit from leveraging Peco. Probably would not really
    // put a big dent in performance by itself. However, doing it the
    // right way may make it somewhat more sane to introduce some kind
    // of parallelism.
    //
    // For parallelism, basically I just need to buffer or pass-off to
    // another thread the data passed to mergeAllNcListResults, or
    // perhaps mergeNcListCombo.
    //
    // One other thing that has me a bit worried here, is that once I
    // find a valid match for all name:counts, I bail. So for example
    // if jagoff:8 = jag:2,off:6, I take that as the answer and run.
    // What if there's also a jag:6,off:2? It's not clear to me that I
    // catch that. Peco could potentially help with that too (and as
    // a test I could just move_append to a local src_list and only
    // return once I finish the entire loop.)
    //
    std::vector<int> orig_count_list;
    //bool orig_displayed{};
    for (auto count : count_list) {
      if (clue_manager::is_known_name_count(name, count)) {
        auto sl = validateSourcesForNameCount(
            clue_name, name, count, {nc_list, name_list, count_list});
        if (!sl.empty()) {
          // if (!src_list.empty()) { return src_list; }
          // this was an attempt, as commented above, to find alternate
          // nc_lists. it found none, and it takes a lot longer at
          // max_sources = 19,20. also, it has presents a problem when
          //count_list is [1,1].
          if (src_list.empty()) {
            src_list = std::move(sl);
            break;
            orig_count_list = count_list;
          } else {
            std::cerr << "another count_list found for " << name << ":"
                      << util::sum(count_list) << std::endl;
            std::cerr << " orig: [" << util::join(orig_count_list, ",") << "]\n";
            std::cerr << " new:  [" << util::join(count_list, ",") << "]\n";
          }
        }
      }
    }
  }
  return src_list;
}

}  // anonymous namespace

auto validateSources(const std::string& clue_name,
    const std::vector<std::string>& src_names, int sum,
    bool validate_all) -> SourceList {
  SourceList results;
  const auto addends = Peco::make_addends(sum, src_names.size());
  // display_addends(sum, addends);
  for (const auto& count_list : addends) {
    NameCountList nc_list;
    auto src_list = validateSourcesForNameAndCountLists(
        clue_name, src_names, count_list, nc_list);
    if (!src_list.empty()) {
      util::move_append(results, std::move(src_list));
      if (!validate_all) {
        break;
      }
    }
  }
  return results;
};

void show_validator_durations() {
  std::cerr << " validatorMerge - " << int(merge_ns / 1e6) << "ms\n"
            << "  merge_all: " << num_merge_all
            << ", full_merges: " << num_full_merges << " of " << num_merge_combo
            << " attempts\n";
  //  << "  sources added, primary: " << num_primary
  //  << ", compound: " << num_compound << std::endl;
}

}  // namespace validator
