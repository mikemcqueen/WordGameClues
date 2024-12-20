// filter-types.h

#pragma once
#include <algorithm>
#include <array>
#include <chrono>
#include <forward_list>
#include <iostream>
#include <numeric>
#include <optional>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>
#include <cuda_runtime.h>
#include "candidates.h"
#include "cm-hash.h"
#include "cuda-types.h"
#include "filter-stream.h"
#include "log.h"
#include "merge-filter-common.h"
#include "source-index.h"
//#include "stream-data.h"

namespace cm {

// aliases

using filter_result_t = std::unordered_set<std::string>;

// types

struct FilterParams {
  int sum;
  int threads_per_block;
  int num_streams;
  int stride;
  int num_iters;
  bool synchronous;
  bool copy_all_prior_sources;
};

class IndexStates {
public:
  enum class Status {
    ready,
    compatible,
    done
  };

  struct Data {
    constexpr auto is_ready() const {
      return status == Status::ready;
    }

    constexpr auto is_compatible() const {
      return status == Status::compatible;
    }

    SourceIndex sourceIndex;
    Status status = Status::ready;
  };

  IndexStates() = delete;

  IndexStates(std::vector<IndexList>&& idx_lists)
      : idx_lists_(std::move(idx_lists)) {
    list_.resize(idx_lists_.size());
    for (index_t idx{}; auto& data : list_) {
      data.sourceIndex.listIndex = idx++;
    }
    /*
    // this can all go i think
    for (index_t list_start_idx{}; const auto& idx_list : idx_lists_) {
      auto list_size = (index_t)idx_list.size();
      list_sizes_.push_back(list_size);
      list_start_indices_.push_back(list_start_idx);
      list_start_idx += list_size;
    }
    */
    fill_indices_.resize(idx_lists_.size());
    std::ranges::iota(fill_indices_, 0);
    fill_iter_ = fill_indices_.before_begin();
  }

  /*
  auto list_size(index_t list_index) const {
    return list_sizes_.at(list_index);
  }
  */

  // Slow. For logging/debugging.
  auto num_in_state(int first, int count, Status status) const {
    // hack
    auto max = std::min(first + count, int(num_lists()));
    int total{};
    for (int i{first}; i < max; ++i) {
      if (list_.at(i).status == status) {
        ++total;
      }
    }
    return total;
  }

  auto num_ready(int first, int count) const {
    return num_in_state(first, count, Status::ready);
  }

  auto num_done(int first, int count) const {
    return num_in_state(first, count, Status::done);
  }

  auto num_compatible(int first, int count) const {
    return num_in_state(first, count, Status::compatible);
  }

  auto update(const std::vector<SourceIndex>& src_indices,
      const std::vector<result_t>& results, int stream_idx) {
    int num_compat{};
    int num_done{};
    int num_not_ready{};
    for (size_t i{}; i < src_indices.size(); ++i) {
      const auto src_idx = src_indices.at(i);
      auto& idx_state = list_.at(src_idx.listIndex);
      if (!idx_state.is_ready()) {
        ++num_not_ready;
        continue;
      }
      if (results.at(src_idx.listIndex)) {
        idx_state.status = Status::compatible;
        ++num_compat;
      } else if (src_idx.index == idx_lists_.at(src_idx.listIndex).back()) {
        // if this is the result for the last source in a sourcelist,
        // mark the list (indexState) as done.
        idx_state.status = Status::done;
        ++num_done;
      }
    }
    if (log_level(ExtraVerbose)) {
      std::cerr << "stream " << stream_idx << " update"
                << ", total: " << src_indices.size()
                << ", not ready: " << num_not_ready  //
                << ", compat: " << num_compat
                << ", done: " << num_done << std::endl;
    }
    return num_compat;
  }

  auto update(FilterStream& stream, const std::vector<result_t>& results) {
    return update(stream.host.src_idx_list, results, stream.stream_idx);
  }

  auto get(index_t list_index) const {
    return list_.at(list_index);
  }

  bool has_fill_indices() const {
    return !fill_indices_.empty();
  }

  /*
  const auto& list_start_indices() const {
    return list_start_indices_;
  }

  auto alloc_copy_start_indices(cudaStream_t stream) {
    return cuda_alloc_copy_start_indices(
        list_start_indices_, stream, "idx_states.list_start_indices");
  }
  */

  std::optional<SourceIndex> get_next_fill_idx() {
    while (has_fill_indices()) {
      auto next_fill_iter = std::next(fill_iter_);
      if (next_fill_iter == fill_indices_.end()) {
        fill_iter_ = fill_indices_.before_begin();
        continue;
      }
      auto opt_src_idx = get_and_increment_index(*next_fill_iter);
      if (opt_src_idx.has_value()) {
        fill_iter_ = next_fill_iter;
        return opt_src_idx.value();
      }
      fill_indices_.erase_after(fill_iter_);
    }
    return std::nullopt;
  }

  size_t num_lists() const {
    return list_.size();
  }

  auto get_incompatible_sources(const CandidateList& candidates,
      int* num_incompat_sources = nullptr) {
    std::unordered_set<CompatSourceIndices> src_indices_set;
    int num_incompat{};
    for (const auto& data : list_) {
      if (!data.is_compatible()) {
        const auto& indices_list = candidates.at(data.sourceIndex.listIndex)
                                       .compat_src_indices_cref.get();
        src_indices_set.insert(indices_list.begin(), indices_list.end());
        num_incompat += int(indices_list.size());
      }
    }
    if (num_incompat_sources) {
      *num_incompat_sources = num_incompat;
    }
    return src_indices_set;
  }

  void mark_compatible(const IndexList& list_indices) {
    for (const auto list_idx : list_indices) {
      list_.at(list_idx).status = Status::compatible;
    }
  }

  void dump_compatible() {
    int count{};
    for (size_t i{}; i < list_.size(); ++i) {
      if (list_.at(i).status == Status::compatible) {
        std::cout << i << ",";
        if (!(++count % 10)) {
          std::cout << std::endl;
        }
      }
    }
    std::cout << std::endl << "total: " << count << std::endl;
  }

private:
  auto get_and_increment_index(index_t list_idx)
      -> std::optional<SourceIndex> {
    auto& data = list_.at(list_idx);
    auto& idx_list = idx_lists_.at(list_idx);
    if (data.is_ready() && (data.sourceIndex.index < idx_list.size())) {
      // capture value before increment
      auto capture = std::make_optional(SourceIndex{
          data.sourceIndex.listIndex, idx_list.at(data.sourceIndex.index)});
      ++data.sourceIndex.index;
      return capture;
    }
    return std::nullopt;
  }

  std::vector<Data> list_;
  std::vector<IndexList> idx_lists_;

  /*  // can go
  IndexList list_start_indices_;
  IndexList list_sizes_;
  */
  
  std::forward_list<index_t> fill_indices_;
  std::forward_list<index_t>::iterator fill_iter_;
};  // class IndexStates

}  // namespace cm
