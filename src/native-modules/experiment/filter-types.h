#ifndef INCLUDE_FILTER_TYPES_H
#define INCLUDE_FILTER_TYPES_H

#include <algorithm>
#include <chrono>
#include <iostream>
#include <optional>
#include <vector>
#include <cuda_runtime.h>
#include "candidates.h"

namespace cm {

using result_t = uint8_t;
using index_t = uint32_t;

struct SourceIndex {
  index_t listIndex{};
  index_t index{};

  bool operator<(const SourceIndex& rhs) const {
    // TODO: should be && ?
    return (listIndex < rhs.listIndex) || (index < rhs.index);
  }

  constexpr const char* as_string(char* buf) const {
    sprintf(buf, "%d:%d", listIndex, index);
    return buf;
  }
};

struct IndexStates {
  enum class State
  {
    ready,
    compatible,
    done
  };

  struct Data {
    constexpr auto ready_state() const {
      return state == State::ready;
    }

    void reset() {
      sourceIndex.index = 0;
      state = State::ready;
    }

    SourceIndex sourceIndex;
    State state = State::ready;
  };

  IndexStates() = delete;
  IndexStates(const SourceCompatibilityLists& sources) {
    list.resize(sources.size());  // i.e. "num_sourcelists"
    std::for_each(list.begin(), list.end(),
      [idx = 0](Data& data) mutable { data.sourceIndex.listIndex = idx++; });
    for (index_t list_start_index{}; const auto& sourceList : sources) {
      list_sizes.push_back(sourceList.size());
      list_start_indices.push_back(list_start_index);
      list_start_index += (index_t)sourceList.size();
    }
  }

  void reset() {
    std::for_each(
      list.begin(), list.end(), [](Data& data) mutable { data.reset(); });
    next_fill_idx = 0;
    done = false;
  }

  index_t flat_index(SourceIndex src_index) const {
    return list_start_indices.at(src_index.listIndex) + src_index.index;
  }

  auto list_size(index_t list_index) const {
    return list_sizes.at(list_index);
  }

  auto num_in_state(int first, int count, State state) const {
    int total{};
    for (int i{}; i < count; ++i) {
      if (list.at(first + i).state == state) {
        ++total;
      }
    }
    return total;
  }

  auto num_ready(int first, int count) const {
    return num_in_state(first, count, State::ready);
  }

  auto num_done(int first, int count) const {
    return num_in_state(first, count, State::done);
  }

  auto num_compatible(int first, int count) const {
    return num_in_state(first, count, State::compatible);
  }

  auto update(const std::vector<SourceIndex>& src_indices,
    const std::vector<result_t>& results,
    [[maybe_unused]] int stream_idx)  // for logging
  {
    constexpr static const bool logging = false;
    int num_compatible{};
    int num_done{};
    for (size_t i{}; i < src_indices.size(); ++i) {
      const auto src_idx = src_indices.at(i);
      auto& idx_state = list.at(src_idx.listIndex);
      const auto result = results.at(src_idx.listIndex);
      if (!idx_state.ready_state()) {
        continue;
      }
      if (result > 0) {
        idx_state.state = State::compatible;
        ++num_compatible;
      } else if (src_idx.index == list_sizes.at(src_idx.listIndex) - 1) {
        // if this is the result for the last source in a sourcelist,
        // mark the list (indexState) as done.
        idx_state.state = State::done;
        ++num_done;
      }
    }
    if (logging) {
      std::cerr << "stream " << stream_idx
                << " update, total: " << src_indices.size()
                << ", compat: " << num_compatible
                << ", done: " << num_done << std::endl;
    }
    return num_compatible;
  }

  auto get(index_t list_index) const {
    return list.at(list_index);
  }

  auto get_and_increment_index(index_t list_index)
    -> std::optional<SourceIndex> {
    auto& data = list.at(list_index);
    if (data.ready_state()
        && (data.sourceIndex.index < list_sizes.at(list_index))) {
      // capture and return value before increment
      auto capture = std::make_optional(data.sourceIndex);
      ++data.sourceIndex.index;
      return capture;
    }
    return std::nullopt;
  }

  size_t num_lists() const {
    return list.size();
  }

  auto get_next_fill_idx() {
    auto fill_idx = next_fill_idx;
    if (++next_fill_idx >= num_lists())
      next_fill_idx = 0;
    return fill_idx;
  }

  bool done{false};
  index_t next_fill_idx{0};
  std::vector<Data> list;
  std::vector<uint32_t> list_start_indices;
  std::vector<uint32_t> list_sizes;
};  // struct IndexStates

//////////

// the pointers in this are allocated in device memory
struct StreamData {
private:
  using hr_time_point_t = decltype(std::chrono::high_resolution_clock::now());
  static const auto num_cores = 1280;
  static const auto max_chunks = 20ul;

public:
  static int next_sequence_num() {
    static int sequence_num{};
    return sequence_num++;
  }

  //

  int num_ready(const IndexStates& indexStates) const {
    return indexStates.num_ready(0, num_list_indices);
  }

  int num_done(const IndexStates& indexStates) const {
    return indexStates.num_done(0, num_list_indices);
  }

  int num_compatible(const IndexStates& indexStates) const {
    return indexStates.num_compatible(0, num_list_indices);
  }

  auto fillSourceIndices(IndexStates& idx_states, int max_idx) {
    source_indices.resize(idx_states.done ? 0 : max_idx);
    for (int idx{}; !idx_states.done && (idx < max_idx);) {
      size_t num_skipped_idx{};  // how many idx were skipped in a row
      // this loop logic is funky and brittle, but intentional
      for (auto list_idx = idx_states.get_next_fill_idx(); /* empty */;
           list_idx = idx_states.get_next_fill_idx()) {
        const auto opt_src_idx = idx_states.get_and_increment_index(list_idx);
        if (opt_src_idx.has_value()) {
          const auto src_idx = opt_src_idx.value();
          assert(src_idx.listIndex == list_idx);
          source_indices.at(idx++) = src_idx;
          if (idx >= max_idx)
            break;
          num_skipped_idx = 0;
        } else if (++num_skipped_idx >= idx_states.num_lists()) {
          // we've skipped over the entire list (with index overlap)
          // and haven't consumed any indices. nothing left to do.
          idx_states.done = true;
          source_indices.resize(idx);
          break;
        }
      }
    }
#if 0
    std::cerr << "ending next_fill_idx: " << idx_states.next_fill_idx << std::endl;
    std::cerr << "stream " << stream_idx
              << " filled " << source_indices.size()
              << " of " << max_idx
              << ", first = " << (source_indices.empty() ? -1 : (int)source_indices.front().listIndex)
              << ", last = " << (source_indices.empty() ? -1 : (int)source_indices.back().listIndex)
              << ", done: " << std::boolalpha << idx_states.done
              << std::endl;
#endif
    return !source_indices.empty();
  }

  bool fillSourceIndices(IndexStates& idx_states) {
    return fillSourceIndices(idx_states, num_list_indices);
  }

  void allocCopy([[maybe_unused]] const IndexStates& idx_states) {
    cudaError_t err = cudaSuccess;
    auto indices_bytes = source_indices.size() * sizeof(SourceIndex);
    // alloc source indices
    if (!device_source_indices) {
      err =
        cudaMallocAsync((void**)&device_source_indices, indices_bytes, cuda_stream);
      assert((err == cudaSuccess) && "allocate source indices");
    }

    /*
    std::vector<index_t> flat_indices;
    flat_indices.reserve(source_indices.size());
    for (const auto& src_idx: source_indices) {
      flat_indices.push_back(idx_states.flat_index(src_idx));
    }
    */

    // copy source indices
    err = cudaMemcpyAsync(device_source_indices, source_indices.data(),
      indices_bytes, cudaMemcpyHostToDevice, cuda_stream);
    assert((err == cudaSuccess) && "copy source indices");
  }

  auto hasWorkRemaining() const {
    return !source_indices.empty();
  }

  void dump() const {
    std::cerr << "kernel " << stream_idx << ", is_running: " << std::boolalpha
              << is_running << ", source_indices: " << source_indices.size()
              << ", num_list_indices: " << num_list_indices
              << std::endl;
  }

  //  int num_src_lists;  // total # of sourcelists (== # of device_results)
                      // TODO: doesn't belong here
  int num_list_indices;
  int stream_idx{-1};
  int sequence_num{};
  bool is_running{false};  // is running (true until results retrieved)
  bool has_run{false};     // has run at least once
  SourceIndex* device_source_indices{nullptr};  // in
  cudaStream_t cuda_stream{nullptr};
  std::vector<SourceIndex> source_indices;  // hasWorkRemaining = (size() > 0)
  hr_time_point_t start_time;
};  // struct StreamData

class StreamSwarm {
  inline static std::vector<cudaStream_t> cuda_streams;

public:
  StreamSwarm() = delete;
  StreamSwarm(unsigned num_streams, unsigned stride) {
    init(num_streams, stride);
  }

  void init(unsigned num_streams, unsigned stride) {
    // TODO: reserve() + emplace_back()
    streams_.resize(num_streams);
    for (unsigned i{}; i < num_streams; ++i) {
      auto& stream = streams_.at(i);
      stream.num_list_indices = stride;
      if (i >= cuda_streams.size()) {
        cudaStream_t cuda_stream;
        cudaError_t err = cudaStreamCreate(&cuda_stream);
        assert((err == cudaSuccess) && "failed to create stream");
        cuda_streams.push_back(cuda_stream);
      }
      stream.stream_idx = i;
      stream.cuda_stream = cuda_streams.at(i);
    }
    reset();
  }

  void reset() {
    for (auto& stream : streams_) {
      stream.source_indices.resize(stream.num_list_indices);
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
      if (stream.is_running
          && (cudaSuccess == cudaStreamQuery(stream.cuda_stream))) {
        if (stream.sequence_num < lowest.value) {
          lowest.value = stream.sequence_num;
          lowest.index = i;
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
      std::this_thread::sleep_for(5ms);
    }
    return true;
  }

  auto& at(int idx) {
    return streams_.at(idx);
  }

  auto& hack_get_streams() {
    return streams_;
  }

private:
  std::vector<StreamData> streams_;
};

}  // namespace cm

#endif // INCLUDE_FILTER_TYPES_H
