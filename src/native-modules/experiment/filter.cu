// filter.cu

#include <algorithm>
#include <chrono>
#include <exception>
#include <future>
#include <limits>
#include <numeric>
#include <optional>
#include <span>
#include <thread>
#include <tuple>
#include <utility> // pair
#include <cuda_runtime.h>
#include "candidates.h"
#include "filter.h"

#define LOGGING

namespace {

using namespace cm;

template <typename T> __device__ __forceinline__ T load(const T* addr) {
  return *(const volatile T*)addr;
}

template <typename T> __device__ __forceinline__ void store(T* addr, T val) {
  *(volatile T*)addr = val;
}

__device__ __host__ auto isSourceXORCompatibleWithAnyXorSource(
  const SourceCompatibilityData& source, const XorSource* xorSources,
  size_t numXorSources) {
  bool compatible = true;
  for (size_t i{}; i < numXorSources; ++i) {
    compatible = source.isXorCompatibleWith(xorSources[i]);
    if (compatible) {
      break;
    }
  }
  return compatible;
}

#if 0
__device__ bool first = true;

__device__ void check_source(
  const SourceCompatibilityData& source, const device::OrSourceData* or_sources) {
  //
  if (source.usedSources.hasVariation(3)
      && (source.usedSources.getVariation(3) == 0)
      && source.usedSources.getBits().test(UsedSources::getFirstBitIndex(3))
      && source.legacySourceBits.test(1)) {
    printf("---match---\n");
    source.dump(nullptr, true);
    const auto& or_src = or_args[0].or_sources[0].source;
    bool xor_compat = source.isXorCompatibleWith(or_src);
    bool and_compat = source.isAndCompatibleWith(or_src);
    printf("xor: %d, and: %d\n", xor_compat, and_compat);
  }
}
#endif

__device__ bool is_source_or_compatibile(const SourceCompatibilityData& source,
  const unsigned num_or_args,
  const device::OrSourceData* __restrict__ or_sources,
  const unsigned num_or_sources) {
  //
  extern __shared__ volatile result_t or_arg_results[];

  // ASSUMPTION: # of --or args will always be smaller than block size.
  if (threadIdx.x < num_or_args) {
    // don't think store required here, even without volatile, because __sync
    or_arg_results[threadIdx.x] = (result_t)0;
  }
  __syncthreads();
  for (unsigned or_chunk{}; or_chunk * blockDim.x < num_or_sources;
       ++or_chunk) {
    //__syncthreads();
    const auto or_idx = or_chunk * blockDim.x + threadIdx.x;
    // TODO: if this works without sync in loop, i can possibly move this
    // conditional to loop definition 
    if (or_idx < num_or_sources) {
      const auto& or_src = or_sources[or_idx];
      if (source.isOrCompatibleWith(or_src.source)) {
        //store(&or_arg_results[or_src.or_arg_idx], (result_t)1);
        // don't think store required here, even without volatile, because __sync
        or_arg_results[or_src.or_arg_idx] = (result_t)1;
      }
    }
  }
  // i could safely initialize reduce_idx to 16 I think (max 32 --or args)
  for (int reduce_idx = blockDim.x / 2; reduce_idx > 0; reduce_idx /= 2) {
    __syncthreads();
    if ((threadIdx.x < reduce_idx) && (reduce_idx + threadIdx.x < num_or_args)) {
      // g++ has deprecated += on volatile destination;
      or_arg_results[threadIdx.x] =
        or_arg_results[threadIdx.x] + or_arg_results[reduce_idx + threadIdx.x];
    }
  }
  if (!threadIdx.x) {
    const auto compat_with_all = or_arg_results[threadIdx.x] == num_or_args;
    return compat_with_all;
  }
  return false;
}

__device__ index_t get_variation_index(
  index_t flat_idx, const IndexSpanPair& idx_spans) {
  //
  if (flat_idx < idx_spans.first.size()) {
    return idx_spans.first[flat_idx];
  }
  flat_idx -= idx_spans.first.size();
  assert(flat_idx < idx_spans.second.size());
  return idx_spans.second[flat_idx];
}

// Test if a source is XOR -and- OR compatible with supplied xor/or sources.
__device__ bool is_source_xor_or_compatible(
  const SourceCompatibilityData& source,
  const SourceCompatibilityData* __restrict__ xor_sources,
  const IndexSpanPair& idx_spans,
  const unsigned num_or_args,
  const device::OrSourceData* __restrict__ or_sources,
  const unsigned num_or_sources) {
  //
  __shared__ bool is_xor_compat;
  __shared__ bool is_or_compat;

  if (!threadIdx.x) {
    store(&is_xor_compat, false);
    store(&is_or_compat, false);
  }
  // NOTE: chunk-indexing in the manner used here is necessary for syncthreads()
  //   to work, at least on SM_6 hardware (GTX1060), where *all threads* in the
  //   block must execute the synchthreads() call. In later architectures, those
  //   restrictions may be relaxed, but possibly only for "completely exited
  //   (the kernel)" threads, which wouldn't be relevant here anyway (because
  //   we're in a function called from within a loop in a parent kernel).
  // Therefore, the following is not an equivalent replacement:
  // for (unsigned xor_idx{threadIdx.x}; xor_idx < num_xor_sources;
  //   xor_idx += blockDim.x) {
  //
  // for each xor_source (one thread per xor_source)
  const auto num_indices = idx_spans.first.size() + idx_spans.second.size();
  for (unsigned idx_chunk{}; idx_chunk * blockDim.x < num_indices;
       ++idx_chunk) {
    __syncthreads();
    // "flat" index, i.e. not yet indexed into appropriate idx_spans array
    const auto flat_idx = idx_chunk * blockDim.x + threadIdx.x;
    if (flat_idx < num_indices) {
      // "actual" index
      const auto xor_src_idx = get_variation_index(flat_idx, idx_spans);
#if 0
      if (!threadIdx.x && !blockIdx.x) {
        printf("flat_idx: %d, xor_idx: %d\n", flat_idx, xor_src_idx);
      }
#endif
      if (source.isXorCompatibleWith(xor_sources[xor_src_idx])) {
        is_xor_compat = true;
      }
    }
    __syncthreads();
    // if source is not XOR compatible with any --xor sources
    if (!is_xor_compat) {
      continue;
    }
    if (num_or_args > 0) {
      // source must also be OR compatible with at least one source
      // of each or_arg
      if (is_source_or_compatibile(
            source, num_or_args, or_sources, num_or_sources)) {
        is_or_compat = true;
      }
      __syncthreads();
      if (!is_or_compat) {
        if (!threadIdx.x) {
          // reset is_xor_compat. sync will happen at loop entrance.
          is_xor_compat = false;
        }
        continue;
      }
    }
    return true;
  }
  return false;
}

__device__ bool variation_indices_shown = false;

__device__ void print_variation_indices(
  const device::VariationIndices* __restrict__ variation_indices) {
  //
  if (variation_indices_shown) return;
  printf("device:\n");
  for (int s{}; s < kNumSentences; ++s) {
    const auto& vi = variation_indices[s];
    for (index_t v{}; v < vi.num_variations; ++v) {
      const auto n = vi.num_src_indices[v];
      printf("sentence %d, variation %d, indices: %d\n", s, v, n);
    }
  }
  variation_indices_shown = true;
}

__device__ bool get_smallest_src_index_spans(
  const SourceCompatibilityData& source,
  const device::VariationIndices* __restrict__ variation_indices,
  IndexSpanPair& idx_spans) {
  //
  int fewest_indices{std::numeric_limits<int>::max()};
  int sentence_with_fewest{-1};
  for (int s{}; s < kNumSentences; ++s) {
    const auto& vi = variation_indices[s];
    // skip sentences for which there are no xor_sources with a primary clue
    // (or it could be a legacy clue)
    // TODO: it is concievable that there may be variation count of "1"
    // for xor_sources that contain no primary clues from a particular sentence.
    // I should be sure to eliminate that possibility, as it is unnecessary
    // memory usage and will double search time.
    if (!vi.num_variations) {
      continue;
    }
    const auto variation = source.usedSources.variations[s] + 1;
    // skip sources that have no primary clue from this sentence
    if (!variation) {
      continue;
    }
    const auto num_indices =
      vi.num_src_indices[0] + vi.num_src_indices[variation];
    if (num_indices < fewest_indices) {
      fewest_indices = num_indices;
      sentence_with_fewest = s;
      if (!num_indices) {
        break;
      }
    }
  }
  if (!fewest_indices) {
    return false;
  }
  if (sentence_with_fewest > -1) {
    const auto variation =
      source.usedSources.variations[sentence_with_fewest] + 1;
    const auto& vi = variation_indices[sentence_with_fewest];
    idx_spans = std::make_pair(
      vi.get_src_index_span(0), vi.get_src_index_span(variation));
  } else {
    // TODO: this conditional is only required until all clues converted to
    // sentences. same with the xor_src_indices in other overload.
    idx_spans = std::make_pair(IndexSpan{}, IndexSpan{});
  }
  return true;
}

__device__ IndexSpanPair get_smallest_src_index_spans(
  const SourceCompatibilityData& source,
  const device::VariationIndices* __restrict__ variation_indices,
  // TODO: the following only required until all clues converted to sentences
  const index_t* __restrict__ xor_src_indices,
  unsigned num_xor_src_indices) {
  //
  // The logic here is a bit weird, because we must differentiate between
  // "this source contains no variations" (source contains only legacy clues:
  // fallback to using all xor_src_indices), and "the number of matching
  // xor_sources for this source's smallest sentence variation is zero" (skip
  // the is_xor_compatible loop for this source).
  // TODO: After all clues are converted to sentences, this can be simplified.
  IndexSpanPair isp;
  const auto has_variation_match =
    get_smallest_src_index_spans(source, variation_indices, isp);
  if (!has_variation_match) {
    isp = std::make_pair(IndexSpan{}, IndexSpan{});
  } else if (!isp.first.size() && !isp.second.size()) {
    isp = std::make_pair(
      std::span{xor_src_indices, num_xor_src_indices}, IndexSpan{});
  }
  return isp;
}

__global__ void xor_kernel_new(
  const SourceCompatibilityData* __restrict__ sources,
  const unsigned num_sources,
  const SourceCompatibilityData* __restrict__ xor_sources,
  const unsigned num_xor_sources,
  // TODO: only required until all clues converted to sentences
  const index_t* all_xor_src_indices,
  const device::VariationIndices* __restrict__ variation_indices,
  const unsigned num_or_args,
  const device::OrSourceData* __restrict__ or_sources,
  const unsigned num_or_sources, const SourceIndex* __restrict__ source_indices,
  const index_t* __restrict__ list_start_indices,
  result_t* __restrict__ results, int stream_idx) {

#if 0
  if (!threadIdx.x && !blockIdx.x && !stream_idx) {
    print_variation_indices(variation_indices);
  }
#endif

  // for each source (one block per source)
  for (unsigned idx{blockIdx.x}; idx < num_sources; idx += gridDim.x) {
    const auto src_idx = source_indices[idx];
    const auto flat_index =
      list_start_indices[src_idx.listIndex] + src_idx.index;
    auto& source = sources[flat_index];
    auto& result = results[src_idx.listIndex];
    auto idx_spans = get_smallest_src_index_spans(
      source, variation_indices, all_xor_src_indices, num_xor_sources);
#if 0
    if (!threadIdx.x && !stream_idx) {
      printf("block %d, first: %d, second: %d, total: %d\n", blockIdx.x,
        (int)idx_spans.first.size(), (int)idx_spans.second.size(),
        (int)(idx_spans.first.size() + idx_spans.second.size()));
    }
#endif
    __syncthreads();
    if (!(idx_spans.first.size() + idx_spans.second.size())) {
      continue;
    }
    if (is_source_xor_or_compatible(source, xor_sources, idx_spans, num_or_args,
          or_sources, num_or_sources)) {
      if (!threadIdx.x) {
        // TODO: store probably not necessary
        store(&result, (result_t)1);
      }
    }
  }
}

// TODO: stream.get_results()? just to call out to free function?  maybe
/*
auto getKernelResults(StreamData& stream, result_t* device_results) {
  return copy_device_results(
    device_results, stream.num_src_lists, stream.stream);
}
*/

auto flat_index(
  const SourceCompatibilityLists& sources, const SourceIndex src_idx) {
  uint32_t flat{};
  for (size_t i{}; i < src_idx.listIndex; ++i) {
    flat += sources.at(i).size();
  }
  return flat + src_idx.index;
}

void check(
  const SourceCompatibilityLists& sources, index_t list_index, index_t index) {
  constexpr const auto logging = true;
  if constexpr (logging) {
    SourceIndex src_idx{list_index, index};
    char idx_buf[32];
    char buf[64];
    snprintf(buf, sizeof(buf), "%s, flat: %d", src_idx.as_string(idx_buf),
      flat_index(sources, src_idx));
    auto& source = sources.at(list_index).at(index);
    source.dump(buf);
    //int compat_index{-1};
    auto compat = isSourceXORCompatibleWithAnyXorSource(
      source, PCD.xorSourceList.data(), PCD.xorSourceList.size());
    std::cerr << "compat: " << compat
              << std::endl;  //<< " (" << compat_index << ")"
  }
}

void dump_xor(int index) {
  const XorSourceList& xorSources = PCD.xorSourceList;
  auto host_index = index;
  const auto& src = xorSources.at(host_index);
  char buf[32];
  snprintf(buf, sizeof(buf), "xor: device(%d) host(%d)", index, host_index);
  src.dump(buf);
}

}  // namespace

namespace cm {

void run_xor_kernel(StreamData& stream, int threads_per_block,
  const SourceCompatibilityData* device_sources, result_t* device_results,
  const index_t* device_list_start_indices) {
  //
  int num_sm;
  cudaDeviceGetAttribute(&num_sm, cudaDevAttrMultiProcessorCount, 0);
  auto threads_per_sm = 2048;
  auto block_size = threads_per_block ? threads_per_block : 1024;
  auto blocks_per_sm = threads_per_sm / block_size;
  // assert(blocks_per_sm * block_size == threads_per_sm);
  auto grid_size = num_sm * blocks_per_sm;  // aka blocks per grid
  auto shared_bytes = PCD.orArgList.size() * sizeof(result_t);
  // enforce assumption in is_source_or_compatible()
  assert(PCD.orArgList.size() < block_size);

  stream.is_running = true;
  stream.sequence_num = StreamData::next_sequence_num();
  stream.start_time = std::chrono::high_resolution_clock::now();
  dim3 grid_dim(grid_size);
  dim3 block_dim(block_size);
  // TODO: probably this donesn't belong here. we should call it once in
  // run_task (with cudaStreamPerThread), and maybe here with
  // stream.cuda_stream. probably maybe.
  cudaError_t err = cudaStreamSynchronize(cudaStreamPerThread);
  if (err != cudaSuccess) {
    fprintf(stdout, "sync before kernel, error: %s", cudaGetErrorString(err));
    assert((err == cudaSuccess) && "sync before kernel");
  }
  xor_kernel_new<<<grid_dim, block_dim, shared_bytes, stream.cuda_stream>>>(
    device_sources, stream.source_indices.size(), PCD.device_xorSources,
    PCD.xorSourceList.size(), PCD.device_xor_src_indices,
    PCD.device_sentenceVariationIndices, PCD.orArgList.size(),
    PCD.device_or_sources, PCD.num_or_sources, stream.device_source_indices,
    device_list_start_indices, device_results, stream.stream_idx);

  //  err = cudaStreamSynchronize(stream.cuda_stream);
  //  assert(err == cudaSuccess);

#if 1 || defined(LOGGING)
  std::cerr << "stream " << stream.stream_idx
            << " started with " << grid_size << " blocks"
            << " of " << block_size << " threads"
          //<< " starting, sequence: " << stream.sequence_num
            << std::endl;
#endif
}

}
