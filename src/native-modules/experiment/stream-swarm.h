#pragma once

#include <cassert>
#include <mutex>
#include <optional>
#include <semaphore>
#include <vector>
#include <cuda_runtime.h>
#include "cuda-types.h"
#include "stream-data.h"

namespace cm {

template <typename Swarm> class StreamSwarmPool;

template <typename Stream, typename HostData, typename DeviceData>
class StreamSwarm {
  struct ValueIndex {
    int value{};
    int index{-1};
  };

public:
  StreamSwarm(int pool_idx = 0) : pool_idx(pool_idx), in_use(false) {
    device.init();
  }

  void ensure_streams(index_t num_streams) {
    for (auto idx = index_t(streams_.size()); idx < num_streams; ++idx) {
      cudaStream_t cuda_stream;
      auto err = cudaStreamCreate(&cuda_stream);
      assert_cuda_success(err, "cudaStreamCreate");
      streams_.emplace_back(idx, 0u, cuda_stream);
    }
  }

  // bad name. call it "set_stride"?
  void init(int stride) {
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

  HostData host;
  DeviceData device;

private:
  std::vector<Stream> streams_;

  // hacky
  friend class StreamSwarmPool<StreamSwarm<Stream, HostData, DeviceData>>;
  int pool_idx;
  bool in_use;
};  // class StreamSwarm

template <typename Swarm> class StreamSwarmPool {
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

  void release(Swarm& swarm) {
    make_available(swarm.pool_idx);
    semaphore_.release();
  }

  void destroy_all_streams() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& swarm : swarms_) {
      swarm.destroy_all_streams();
    }
  }

  // lil hacky
  void for_each(auto& fn) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& swarm : swarms_) {
      assert(!swarm.in_use);
      fn(swarm);
    }
  }

private:
  Swarm& get_available() {
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
  std::vector<Swarm> swarms_;
};

}  // namespace cm
