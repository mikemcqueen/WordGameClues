#pragma once

#include <atomic>
#include "cuda-types.h"

namespace cm {

// template <typename Stream, typename HostData, typename DeviceData>
// class StreamSwarm;

struct KernelContext {
  void record(const CudaEvent& event) { event.record(cuda_stream); }

  cudaStream_t cuda_stream;
  CudaEvent copy_start{};
  CudaEvent copy_stop{};
  CudaEvent kernel_start{};
  CudaEvent kernel_stop{};
};  // struct KernelContext

struct HasStreamDeviceData {
  explicit HasStreamDeviceData() : stream_idx_{increment_stream_idx()} {}

  auto stream_idx() const { return stream_idx_; }

private:
  static index_t increment_stream_idx() {
    static std::atomic<index_t> stream_idx = 0;
    return stream_idx++;
  }

  index_t stream_idx_;
};

struct HasSwarmDeviceData {
  explicit HasSwarmDeviceData() : swarm_idx_{increment_swarm_idx()} {}

  auto swarm_idx() const { return swarm_idx_; }

private:
  static index_t increment_swarm_idx() {
    static std::atomic<index_t> swarm_idx = 0;
    return swarm_idx++;
  }

  index_t swarm_idx_;
};

template <typename HostData, typename DeviceData>
struct StreamData : KernelContext {
private:
  static int next_sequence_num() {
    static std::atomic<int> sequence_num{};
    return sequence_num++;
  }

public:
  StreamData() = delete;

  StreamData(index_t stream_idx, index_t stride, cudaStream_t stream)
      : KernelContext{stream}, stream_idx(stream_idx), stride(stride) {
    device.init();
  }

  auto increment_sequence_num() {
    sequence_num = next_sequence_num();
    return sequence_num;
  }

  index_t stream_idx;
  index_t stride;
  int sequence_num{};
  bool is_running{false};  // true until results retrieved
  bool has_run{false};     // has run at least once
  HostData host;
  DeviceData device;
};  // struct StreamData

}  // namespace cm
