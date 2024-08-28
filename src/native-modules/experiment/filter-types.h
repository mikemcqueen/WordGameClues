// filter-types.h

#ifndef INCLUDE_FILTER_TYPES_H
#define INCLUDE_FILTER_TYPES_H

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
#include "cm-hash.h"
#include "cuda-types.h"
#include "candidates.h"
#include "merge-filter-common.h"
#include "stream-data.h"
#include "log.h"

namespace cm {

// aliases

using filter_result_t = std::unordered_set<std::string>;

using SentenceVariationIndices =
    std::array<std::vector<IndexList>, kNumSentences>;

// types

struct FilterParams {
  index_t sum;
  index_t threads_per_block;
  index_t num_streams;
  index_t stride;
  index_t num_iters;
  bool synchronous;
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

    /*
    void reset() {
      sourceIndex.index = 0;
      status = Status::ready;
    }
    */

    SourceIndex sourceIndex;
    Status status = Status::ready;
  };

  IndexStates() = delete;

  /*
  IndexStates(const CandidateList& candidates) {
    list_.resize(candidates.size());
    for (index_t idx{}; auto& data : list_) {
      data.sourceIndex.listIndex = idx++;
    }
    for (index_t list_start_index{}; const auto& candidate : candidates) {
      auto num_sources{(index_t)candidate.src_list_cref.get().size()};
      list_sizes_.push_back(num_sources);
      list_start_indices_.push_back(list_start_index);
      list_start_index += num_sources;
    }
    fill_indices_.resize(candidates.size());
    std::iota(fill_indices_.begin(), fill_indices_.end(), 0);
    fill_iter_ = fill_indices_.before_begin();
  }
  */

  IndexStates(std::vector<IndexList>&& idx_lists)
      : idx_lists_(std::move(idx_lists)) {
    list_.resize(idx_lists_.size());
    for (index_t idx{}; auto& data : list_) {
      data.sourceIndex.listIndex = idx++;
    }
    // this can all go i think
    for (index_t list_start_idx{}; const auto& idx_list : idx_lists_) {
      auto list_size = (index_t)idx_list.size();
      list_sizes_.push_back(list_size);
      list_start_indices_.push_back(list_start_idx);
      list_start_idx += list_size;
    }

    fill_indices_.resize(idx_lists_.size());
    std::ranges::iota(fill_indices_, 0);
    fill_iter_ = fill_indices_.before_begin();
  }

  /*
  void reset() {
    for (auto& data : list_) {
      data.reset();
    }
    todo:
    fill_indices next_fill_idx = 0;
    done = false;
  }
  */

  /*
  index_t flat_index(SourceIndex src_index) const {
    return list_start_indices_.at(src_index.listIndex) + src_index.index;
  }
  */

  auto list_size(index_t list_index) const {
    return list_sizes_.at(list_index);
  }

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
      } else if (src_idx.index == idx_lists_.at(src_idx.listIndex).back()) { // list_sizes_.at(src_idx.listIndex) - 1) {
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

  auto update(StreamBase& stream, const std::vector<result_t>& results) {
    return update(stream.src_indices, results, stream.stream_idx);
  }

  auto get(index_t list_index) const {
    return list_.at(list_index);
  }

  bool has_fill_indices() const {
    return !fill_indices_.empty();
  }

  const auto& list_start_indices() const {
    return list_start_indices_;
  }

  auto alloc_copy_start_indices(cudaStream_t stream) {
    return cuda_alloc_copy_start_indices(
        list_start_indices_, stream, "idx_states.list_start_indices");
  }

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
      index_t* num_incompat_sources = nullptr) {
    SourceCompatibilitySet src_set;
    index_t num_incompat{};
    for (const auto& data : list_) {
      if (!data.is_compatible()) {
        const auto& src_list =
            candidates.at(data.sourceIndex.listIndex).src_list_cref.get();
        src_set.insert(src_list.begin(), src_list.end());
        num_incompat += src_list.size();
      }
    }
    if (num_incompat_sources) {
      *num_incompat_sources = num_incompat;
    }
    return src_set;
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
    if (data.is_ready() && (data.sourceIndex.index < idx_list.size())) {      // list_sizes_.at(list_index))) {
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

  // can go
  IndexList list_start_indices_;
  IndexList list_sizes_;

  std::forward_list<index_t> fill_indices_;
  std::forward_list<index_t>::iterator fill_iter_;
};  // class IndexStates

class StreamSwarm {
  inline static std::vector<cudaStream_t> cuda_streams;

public:
  StreamSwarm() = delete;
  StreamSwarm(index_t num_streams, index_t stride) {
    init(num_streams, stride);
  }

  void init(index_t num_streams, index_t stride) {
    streams_.reserve(num_streams);
    for (index_t idx{}; idx < num_streams; ++idx) {
      if (idx == cuda_streams.size()) {
        cudaStream_t cuda_stream;
        auto err = cudaStreamCreate(&cuda_stream);
        assert_cuda_success(err, "cudaStreamCreate");
        cuda_streams.push_back(cuda_stream);
      }
      streams_.emplace_back(idx, stride, cuda_streams.at(idx));
    }
    reset();
  }

  void reset() {
    for (auto& stream : streams_) {
      stream.src_indices.resize(stream.stride);
      stream.is_running = false;
      stream.has_run = false;
    }
  }

  auto anyWithWorkRemaining() -> std::optional<int> {
    for (size_t i{}; i < streams_.size(); ++i) {
      const auto& stream = streams_.at(i);
      if (stream.hasWorkRemaining()) {
        return std::make_optional(i);
      }
    }
    return std::nullopt;
  }

  bool anyIdleWithWorkRemaining(int& index) {
    for (size_t i{}; i < streams_.size(); ++i) {
      const auto& stream = streams_.at(i);
      if (!stream.is_running && stream.hasWorkRemaining()) {
        index = i;
        return true;
      }
    }
    return false;
  }

  struct ValueIndex {
    int value{};
    int index{-1};
  };

  // TODO: std::optional, and above here
  bool anyRunningComplete(int& index) {
    ValueIndex lowest = {std::numeric_limits<int>::max()};
    for (size_t i{}; i < streams_.size(); ++i) {
      const auto& stream = streams_.at(i);
      if (stream.is_running) {
        cudaError_t err = cudaStreamQuery(stream.cuda_stream);
        if (err == cudaSuccess) {
          if (stream.sequence_num < lowest.value) {
            lowest.value = stream.sequence_num;
            lowest.index = i;
          }
        } else if (err != cudaErrorNotReady) {
          assert_cuda_success(err, "cudaStreamQuery");
        }
      }
    }
    if (lowest.index > -1) {
      index = lowest.index;
      return true;
    }
    return false;
  }

  bool get_next_available(int& current) {
    using namespace std::chrono_literals;

    // First: ensure all primary streams have started at least once
    if (++current >= (int)streams_.size()) {
      current = 0;
    } else {
      const auto& stream = streams_.at(current);
      if (!stream.is_running && !stream.has_run && stream.hasWorkRemaining()) {
        return true;
      }
    }

    // Second: process results for any "running" stream that has completed
    if (anyRunningComplete(current)) {
      return true;
    }

    // Third: run any idle (non-running) stream with work remaining
    if (anyIdleWithWorkRemaining(current)) {
      return true;
    }

    // There is no idle stream, and no attachable running stream that has work
    // remaining. Is there any stream with work remaining? If not, we're done.
    if (!anyWithWorkRemaining().has_value()) {
      return false;
    }

    // Wait for one to complete.
    while (!anyRunningComplete(current)) {
      std::this_thread::sleep_for(1ms);
    }
    return true;
  }

  auto& at(int idx) {
    return streams_.at(idx);
  }

private:
  std::vector<StreamData> streams_;
}; // class StreamSwarm

}  // namespace cm

#endif // INCLUDE_FILTER_TYPES_H
