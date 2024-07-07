// filter-types.h

#ifndef INCLUDE_FILTER_TYPES_H
#define INCLUDE_FILTER_TYPES_H

#include <algorithm>
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
#include "cuda-types.h"
#include "candidates.h"
#include "util.h"

namespace cm {

// types

using filter_result_t = std::unordered_set<std::string>;
//using hr_time_point_t = decltype(std::chrono::high_resolution_clock::now());

struct SourceIndex {
  index_t listIndex{};
  index_t index{};

  constexpr bool operator<(const SourceIndex& rhs) const {
    return (listIndex < rhs.listIndex)
           || ((listIndex == rhs.listIndex) && (index < rhs.index));
  }

  constexpr const char* as_string(char* buf) const {
    sprintf(buf, "%d:%d", listIndex, index);
    return buf;
  }
};

class IndexStates {
public:
  enum class State {
    ready,
    compatible,
    done
  };

  struct Data {
    constexpr auto ready_state() const {
      return state == State::ready;
    }

    constexpr auto is_compatible() const {
      return state == State::compatible;
    }

    void reset() {
      sourceIndex.index = 0;
      state = State::ready;
    }

    SourceIndex sourceIndex;
    State state = State::ready;
  };

  IndexStates() = delete;
  IndexStates(const CandidateList& candidates) {
    list.resize(candidates.size());
    for (index_t idx{}; auto& data : list) {
      data.sourceIndex.listIndex = idx++;
    }
    for (index_t list_start_index{}; const auto& candidate : candidates) {
      auto num_sources{(index_t)candidate.src_list_cref.get().size()};
      list_sizes.push_back(num_sources);
      list_start_indices.push_back(list_start_index);
      list_start_index += num_sources;
    }
    fill_indices.resize(candidates.size());
    std::iota(fill_indices.begin(), fill_indices.end(), 0);
    fill_iter = fill_indices.before_begin();
  }

      /*
  void reset() {
    for (auto& data: list) {
      data.reset();
    }
    todo: fill_indices
    next_fill_idx = 0;
    done = false;
  }
      */

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
    int num_compatible{};
    int num_done{};
    for (size_t i{}; i < src_indices.size(); ++i) {
      const auto src_idx = src_indices.at(i);
      auto& idx_state = list.at(src_idx.listIndex);
      if (!idx_state.ready_state()) {
        continue;
      }
      if (results.at(src_idx.listIndex)) {
        idx_state.state = State::compatible;
        ++num_compatible;
      } else if (src_idx.index == list_sizes.at(src_idx.listIndex) - 1) {
        // if this is the result for the last source in a sourcelist,
        // mark the list (indexState) as done.
        idx_state.state = State::done;
        ++num_done;
      }
    }
    if constexpr (0) {
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

  std::optional<SourceIndex> get_next_fill_idx() {
    while (!fill_indices.empty()) {
      auto next_fill_iter = std::next(fill_iter);
      if (next_fill_iter == fill_indices.end()) {
        fill_iter = fill_indices.before_begin();
        continue;
      }
      auto opt_src_idx = get_and_increment_index(*next_fill_iter);
      if (opt_src_idx.has_value()) {
        fill_iter = next_fill_iter;
        return opt_src_idx.value();
      }
      fill_indices.erase_after(fill_iter);
    }
    return std::nullopt;
  }

  size_t num_lists() const {
    return list.size();
  }

  void mark_compatible(const IndexList& list_indices) {
    for (auto idx : list_indices) {
      list.at(idx).state = State::compatible;
    }
  }

  void dump_compatible() {
    int count{};
    for (size_t i{}; i < list.size(); ++i) {
      if (list.at(i).state == State::compatible) {
        std::cout << i << ",";
        if (!(++count % 10)) {
          std::cout << std::endl;
        }
      }
    }
    std::cout << std::endl << "total: " << count << std::endl;
  }

  std::vector<Data> list;
  std::vector<index_t> list_start_indices;
  std::vector<index_t> list_sizes;
  std::forward_list<index_t> fill_indices;
  std::forward_list<index_t>::iterator fill_iter;
};  // class IndexStates

//////////

// the pointers in this are allocated in device memory
struct StreamData {
private:
  static const auto num_cores = 1280;
  static const auto max_chunks = 20ul;

public:
  static int next_sequence_num() {
    static int sequence_num{};
    return sequence_num++;
  }

  //

  StreamData(int idx, cudaStream_t stream, int stride)
      : stream_idx(idx),
        cuda_stream(stream),
        kernel_start(stream, false),
        kernel_stop(stream, false),
        num_list_indices(stride) {
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

  auto fill_source_indices(IndexStates& idx_states, int max_idx) {
    source_indices.resize(idx_states.fill_indices.empty() ? 0 : max_idx); // iters hackery
    for (size_t idx{}; idx < source_indices.size(); ++idx) {
      auto opt_src_idx = idx_states.get_next_fill_idx();
      if (!opt_src_idx.has_value()) {
        source_indices.resize(idx);
        break;
      }
      source_indices.at(idx) = opt_src_idx.value();
    }
    if constexpr (0) {
      // std::cerr << "ending next_fill_idx: " << idx_states.next_fill_idx
      //           << std::endl;
      std::cerr
        << "stream " << stream_idx << " filled " << source_indices.size()
        << " of " << max_idx << ", first = "
        << (source_indices.empty() ? -1 : (int)source_indices.front().listIndex)
        << ", last = "
        << (source_indices.empty() ? -1 : (int)source_indices.back().listIndex)
        << std::endl;
      //<< ", done: " << std::boolalpha << idx_states.done
    }
    return !source_indices.empty();
  }

  bool fill_source_indices(IndexStates& idx_states) {
    //, const std::vector<result_t>& compat_src_results) {
    return fill_source_indices(idx_states, num_list_indices);
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

  int stream_idx{-1};
  cudaStream_t cuda_stream{};
  CudaEvent kernel_start;
  CudaEvent kernel_stop;
  std::chrono::microseconds::rep fill_duration{};

  int sequence_num{};
  bool is_running{false};  // is running (true until results retrieved)
  bool has_run{false};     // has run at least once
  SourceIndex* device_source_indices{};
  std::vector<SourceIndex> source_indices;  // hasWorkRemaining = (size() > 0)

  int num_list_indices;    // TODO: this doesn't belong here
};  // struct StreamData

class StreamSwarm {
  inline static std::vector<cudaStream_t> cuda_streams;

public:
  StreamSwarm() = delete;
  StreamSwarm(unsigned num_streams, unsigned stride) {
    init(num_streams, stride);
  }

  void init(unsigned num_streams, unsigned stride) {
    streams_.reserve(num_streams);
    for (unsigned idx{}; idx < num_streams; ++idx) {
      if (idx == cuda_streams.size()) {
        cudaStream_t cuda_stream;
        auto err = cudaStreamCreate(&cuda_stream);
        assert_cuda_success(err, "cudaStreamCreate");
        cuda_streams.push_back(cuda_stream);
      }
      streams_.emplace_back(idx, cuda_streams.at(idx), stride);
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
