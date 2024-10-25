#pragma once

#include <cassert>
#include <mutex>
#include <semaphore>
#include <vector>
#include <cuda_runtime.h>
#include "cuda-types.h"
#include "filter-stream.h"

namespace cm {

class StreamSwarm {
public:
  StreamSwarm(int pool_idx = 0) : pool_idx(pool_idx), in_use(false) {}

  void ensure_streams(index_t num_streams) {
    for (auto idx = index_t(streams_.size()); idx < num_streams; ++idx) {
      cudaStream_t cuda_stream;
      auto err = cudaStreamCreate(&cuda_stream);
      assert_cuda_success(err, "cudaStreamCreate");
      streams_.emplace_back(idx, 0u, cuda_stream);
      //streams_.back().init();
    }
  }

  // bad name.  call it "set_stride"
  void init(int num_streams, int stride) {
    ensure_streams(num_streams);
    // not a strict requirement but matches existing use and simplifies logic.
    assert(num_streams == int(streams_.size()));
    for (auto& stream : streams_) {
      stream.stride = stride;
    }
    reset();
  }

  void reset() {
    for (auto& stream : streams_) {
      // TODO move this
      stream.host.src_idx_list.resize(stream.stride);
      stream.is_running = false;
      stream.has_run = false;
    }
  }

  int num_streams() const { return int(streams_.size()); }

  void destroy_all_streams() {
    for (auto& stream: streams_) {
      auto err = cudaStreamDestroy(stream.cuda_stream);
      assert_cuda_success(err, "cudaStreamDestroy");
    }
    streams_.clear();
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
        index = int(i);
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
            lowest.index = index_t(i);
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
  std::vector<FilterStream> streams_;

  // hacky
  friend class StreamSwarmPool;
  int pool_idx;
  bool in_use;
}; // class StreamSwarm

class StreamSwarmPool {
public:
  StreamSwarmPool(int initial_count) : semaphore_(initial_count) {
    for (int i{}; i < initial_count; ++i) {
      swarms_.emplace_back(i);
    }
  }

  auto& acquire() {
    semaphore_.acquire();
    return get_available();
  }

  void release(StreamSwarm& swarm) {
    make_available(swarm.pool_idx);
    semaphore_.release();
  }

  void destroy_all_streams() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& swarm : swarms_) {
      swarm.destroy_all_streams();
    }
  }

private:
  StreamSwarm& get_available() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& swarm : swarms_) {
      if (!swarm.in_use) {
        swarm.in_use = true;
        return swarm;
      }
    }
    assert(0 && "no swarm available!");
  }

  void make_available(int idx) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& swarm = swarms_.at(idx);
    assert(swarm.in_use && "swarm not in use!");
    swarm.in_use = false;
  }

  std::counting_semaphore<> semaphore_;
  std::mutex mutex_;
  std::vector<StreamSwarm> swarms_;
};

}  // namespace cm
