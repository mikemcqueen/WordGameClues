#include <chrono>
#include <iostream>
#include <cuda_runtime.h>
#include "merge.h"
#include "merge-filter-common.h"
#include "merge-filter-data.h"
#include "filter-types.h" // StreamSwarm, should generalize header
#include "peco.h"
#include "util.h"

namespace {

using namespace cm;

auto anyCompatibleXorSources(const SourceData& source,
  const Peco::IndexList& indexList, const SourceList& sourceList) {
  //
  for (auto it = indexList.begin(); it != indexList.end(); ++it) {
    if (source.isXorCompatibleWith(sourceList[*it])) {
      return true;
    }
  }
  return false;
}

int getNumEmptySublists(const std::vector<SourceList>& src_lists) {
  auto count = 0;
  for (const auto& sl : src_lists) {
    if (sl.empty()) count++;
  }
  return count;
}

// TODO: comment this. code is tricky
auto filterXorIncompatibleIndices(Peco::IndexListVector& index_lists,
  int first_list_idx, int second_list_idx,
  const std::vector<SourceList>& src_lists) {
  //
  Peco::IndexList& first_list = index_lists[first_list_idx];
  for (auto it_first = first_list.before_begin();
    std::next(it_first) != first_list.end(); /* nothing */)
  {
    const auto& first_src = src_lists[first_list_idx][*std::next(it_first)];
    bool any_compat = anyCompatibleXorSources(
      first_src, index_lists[second_list_idx], src_lists[second_list_idx]);
    if (!any_compat) {
      first_list.erase_after(it_first);
    } else {
      ++it_first;
    }
  }
  return !first_list.empty();
}
  
bool filterAllXorIncompatibleIndices(Peco::IndexListVector& idx_lists,
  const std::vector<SourceList>& src_lists)
{
  using namespace std::chrono;
  auto t0 = high_resolution_clock::now();

  if (idx_lists.size() < 2u) return true;
  for (size_t first{}; first < idx_lists.size(); ++first) {
    for (size_t second{}; second < idx_lists.size(); ++second) {
      if (first == second) continue;
      if (!filterXorIncompatibleIndices(
            idx_lists, first, second, src_lists)) {
        return false;
      }
    }
  }
  auto t1 = high_resolution_clock::now();
  auto t_dur = duration_cast<milliseconds>(t1 - t0).count();
  std::cerr << "  filter incompatible - " << t_dur << "ms" << std::endl;

  return true;
}

auto get_flat_indices(Peco& peco, int num_combinations, bool first = false) {
  auto indexList = first ? peco.first_combination() : peco.next_combination();
  const auto row_size = indexList->size();
  std::vector<index_t> flat_indices(row_size * num_combinations);
  int idx{};
  for (; indexList && (idx < num_combinations);
       indexList = peco.next_combination(), ++idx) {
    std::copy(indexList->begin(), indexList->end(),
      flat_indices.begin() + idx * row_size);
  }
  flat_indices.resize(idx * row_size);
  return flat_indices;
}

void dump_flat_indices(const std::vector<index_t>& flat_indices, size_t row_size) {
  for (size_t i{}; i < flat_indices.size(); i += row_size) {
    for (size_t j{}; j < row_size; ++j) {
      if (j) std::cerr << ", ";
      std::cerr << flat_indices[i + j];
    }
    std::cerr << std::endl;
  }
}

void zero_results(merge_result_t* results, size_t num_results, cudaStream_t stream) {
  cudaError_t err = cudaSuccess;
  auto results_bytes = num_results * sizeof(merge_result_t);
  err = cudaMemsetAsync(results, 0, results_bytes, stream);
  assert((err == cudaSuccess) && "merge zero results");
}

auto alloc_results(size_t num_results) {  // TODO cudaStream_t stream) {
  cudaStream_t stream = cudaStreamPerThread;
  cudaError_t err = cudaSuccess;
  // alloc results
  auto results_bytes = num_results * sizeof(result_t);
  result_t* device_results;
  err = cudaMallocAsync((void**)&device_results, results_bytes, stream);
  assert((err == cudaSuccess) && "merge alloc results");
  return device_results;
}

void sync_copy_results(std::vector<result_t>& results, unsigned num_results,
  result_t* device_results, cudaStream_t stream) {
  //
  cudaError_t err = cudaStreamSynchronize(stream);
  assert((err == cudaSuccess) && "merge copy results pre-sync");

  auto results_bytes = num_results * sizeof(result_t);
  err = cudaMemcpyAsync(results.data(), device_results, results_bytes,
    cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    fprintf(stdout, "merge copy results, error: %s", cudaGetErrorString(err));
    assert((err != cudaSuccess) && "merge copy results");
  }
  err = cudaStreamSynchronize(stream);
  assert((err == cudaSuccess) && "merge copy results post-sync");
}

void add_compat_sources(
  std::vector<SourceCompatibilityData>& compat_sources,
  const SourceList& src_list) {
  //
  for (const auto& src : src_list) {
    compat_sources.push_back(src);
  }
}

auto alloc_copy_src_lists(
  const std::vector<SourceList>& src_lists, size_t* num_bytes = nullptr) {
  // alloc sources
  const cudaStream_t stream = cudaStreamPerThread;
  cudaError_t err = cudaSuccess;
  const auto num_sources = sum_sizes(src_lists);
  const auto sources_bytes = num_sources * sizeof(SourceCompatibilityData);
  std::cerr << "  allocating device_sources (" << sources_bytes << " bytes)"
            << std::endl;
  SourceCompatibilityData* device_sources;
  err = cudaMallocAsync((void**)&device_sources, sources_bytes, stream);
  std::cerr << "  allocate complete" << std::endl;
  assert((err == cudaSuccess) && "merge alloc sources");
  // copy sources
  std::vector<SourceCompatibilityData> src_compat_list;
  src_compat_list.reserve(num_sources);
  std::cerr << "  building src_compat_lists..." << std::endl;
  for (const auto& src_list : src_lists) {
    // this is somewhat braindead.perhaps I could only marshal the
    // SourceCompatibiltyData when passed from JavaScript? since I
    // no longer need to round-trip it back to JavaScript?
    // Alternatively, build one gimungous array and blast it over
    // in one memcopy.
    // TODO: comments in todo file or in notebook about how to split
    // pnsl/ncList from SourceCompatibilityData.
    add_compat_sources(src_compat_list, src_list);
  }
  std::cerr << "  copying src_compat_lists..." << std::endl;
  err = cudaMemcpyAsync(device_sources, src_compat_list.data(), sources_bytes,
    cudaMemcpyHostToDevice, stream);
  if (err != cudaSuccess) {
    fprintf(stdout, "merge copy sources, error: %s", cudaGetErrorString(err));
    throw std::runtime_error("merge copy sources");
  }
  if (num_bytes) {
    *num_bytes = sources_bytes;
  }
  return device_sources;
}

auto alloc_copy_idx_lists(
  const std::vector<IndexList>& idx_lists, size_t* num_bytes = nullptr) {
  // alloc indices
  const cudaStream_t stream = cudaStreamPerThread;
  cudaError_t err = cudaSuccess;
  auto indices_bytes = sum_sizes(idx_lists) * sizeof(index_t);
  index_t* device_indices;
  err = cudaMallocAsync((void**)&device_indices, indices_bytes, stream);
  assert((err == cudaSuccess) && "merge alloc idx lists");

  // copy indices
  size_t index{};
  for (const auto& idx_list : idx_lists) {
    err = cudaMemcpyAsync(&device_indices[index], idx_list.data(),
      idx_list.size() * sizeof(index_t), cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
      fprintf(
        stdout, "merge copy idx lists, error: %s", cudaGetErrorString(err));
      throw std::runtime_error("merge copy idx lists");
    }
    index += idx_list.size();
  }
  if (num_bytes) {
    *num_bytes = indices_bytes;
  }
  return device_indices;
}

// nC2 formula (# of combinations: pick 2 from n)
auto calc_num_list_pairs(const std::vector<IndexList>& idx_lists) {
  return idx_lists.size() * (idx_lists.size() - 1) / 2;
}

void for_each_list_pair(
  const std::vector<IndexList>& idx_lists, const auto& fn) {
  //
  for (size_t i{}, n{}; i < idx_lists.size() - 1; ++i) {
    for (size_t j{i + 1}; j < idx_lists.size(); ++j, ++n) {
      fn(i, j, n);
    }
  }
}

auto alloc_compat_matrices(
  const std::vector<IndexList>& idx_lists, size_t* num_bytes = nullptr) {
  size_t num_iterated_list_pairs{};
  uint64_t num_results{};
  for_each_list_pair(idx_lists, [&](size_t i, size_t j, size_t n) {
    num_results += idx_lists.at(i).size() * idx_lists.at(j).size();
    ++num_iterated_list_pairs;
  });
  // because we use 32 bit start-indices.
  assert((num_results < std::numeric_limits<index_t>::max())
         && "numeric limit exceeded");
  // sanity check iteration count vs. nC2 formula
  assert((num_iterated_list_pairs == calc_num_list_pairs(idx_lists))
         && "num_list_pairs mismatch");

  cudaStream_t stream = cudaStreamPerThread;
  cudaError_t err = cudaSuccess;
  auto matrix_bytes = num_results * sizeof(result_t);
  result_t* device_compat_matrices;
  err = cudaMallocAsync((void**)&device_compat_matrices, matrix_bytes, stream);
  assert((err == cudaSuccess) && "merge alloc compat matrices");
  if (num_bytes) {
    *num_bytes = matrix_bytes;
  }
  return device_compat_matrices;
}

auto make_compat_matrix_start_indices(const std::vector<IndexList>& idx_lists) {
  IndexList start_indices{};
  index_t index{};
  for_each_list_pair(idx_lists, [&](size_t i, size_t j, size_t n) {
    start_indices.push_back(index);
    index += idx_lists.at(i).size() * idx_lists.at(j).size();
  });
  return start_indices;
}

auto make_list_sizes(const std::vector<IndexList>& idx_lists) {
  IndexList sizes;
  sizes.reserve(idx_lists.size());
  for (const auto& idx_list : idx_lists) {
    sizes.push_back(idx_list.size());
  }
  return sizes;
}

auto alloc_copy_list_sizes(
  const std::vector<index_t>& list_sizes, size_t* num_bytes = nullptr) {
  //
  cudaStream_t stream = cudaStreamPerThread;
  cudaError_t err = cudaSuccess;
  // alloc list sizes
  auto list_sizes_bytes = list_sizes.size() * sizeof(index_t);
  index_t* device_list_sizes;
  err = cudaMallocAsync((void**)&device_list_sizes, list_sizes_bytes, stream);
  assert((err == cudaSuccess) && "merge alloc list sizes");
  // copy list sizes
  err = cudaMemcpyAsync(device_list_sizes, list_sizes.data(), list_sizes_bytes,
    cudaMemcpyHostToDevice, stream);
  assert((err == cudaSuccess) && "merge copy list sizes");
  if (num_bytes) {
    *num_bytes = list_sizes_bytes;
  }
  return device_list_sizes;
}

// for debugging
void debug_copy_show_compat_matrix_hit_count(
  const result_t* device_compat_matrices, size_t num_bytes) {
  //
  cudaStream_t stream = cudaStreamPerThread;
  cudaError_t err = cudaSuccess;
  std::vector<result_t> results(num_bytes);
  err = cudaMemcpyAsync(results.data(), device_compat_matrices, num_bytes,
    cudaMemcpyDeviceToHost, stream);
  assert((err == cudaSuccess) && "merge copy device compat matrices");
  cudaStreamSynchronize(stream);
  uint64_t num_compat{};
  for (auto result: results) {
    if (result) {
      ++num_compat;
    }
  }
  std::cerr << " device_compat_matrices, num_compat: " << num_compat
            << std::endl;
}

// Get compat matrices for each list-pair.
// TODO: std::vector<bool> would save memory. would it be faster?
auto get_compat_matrices(
  const std::vector<SourceList>& src_lists, std::vector<IndexList> idx_lists) {
  //
  using namespace std::chrono;
  std::vector<std::vector<result_t>> host_compat_matrices;
  auto t0 = high_resolution_clock::now();

  int total_compat{};
  int pair_idx{};
  for (size_t i{}; i < idx_lists.size() - 1; ++i) {
    const auto src_list1 = src_lists.at(i);
    const auto idx_list1 = idx_lists.at(i);
    for (size_t j{i + 1}; j < idx_lists.size(); ++j, ++pair_idx) {
      const auto src_list2 = src_lists.at(j);
      const auto idx_list2 = idx_lists.at(j);
      std::vector<result_t> compat_matrix(idx_list1.size() * idx_list2.size());
      int num_compat{};
      int matrix_idx{};
      for (size_t k{}; k < idx_list1.size(); ++k) {
        const auto& src1 = src_list1.at(idx_list1.at(k));
        for (size_t l{}; l < idx_list2.size(); ++l, ++matrix_idx) {
          const auto& src2 = src_list2.at(idx_list2.at(l));
          auto compat = src1.isXorCompatibleWith(src2);
          if (compat) {
            ++num_compat;
          }
          compat_matrix.at(matrix_idx) = compat ? 1 : 0;
        }
      }
      host_compat_matrices.emplace_back(std::move(compat_matrix));
      total_compat += num_compat;
      std::cerr << "  host list_pair " << pair_idx
                << ", num_compat: " << num_compat
                << ", total_compat: " << total_compat << std::endl;
    }
  }
  auto t1 = high_resolution_clock::now();
  auto t_dur = duration_cast<milliseconds>(t1 - t0).count();
  std::cerr << " get_compat_matrices complete - " << t_dur
    << "ms" << std::endl;
  return host_compat_matrices;
}

//
void host_show_num_compat_combos(const uint64_t first_combo,
  const uint64_t num_combos,
  const std::vector<std::vector<result_t>>& compat_matrices,
  const IndexList& list_sizes) {
  //
  assert(list_sizes.size() == compat_matrices.size());
  using namespace std::chrono;
  auto t0 = high_resolution_clock::now();
  IndexList row_indices(10);
  int num_compat{};
  for (auto idx{first_combo}; idx < num_combos; ++idx) {
    auto tmp_idx{idx};
    for (int i{(int)list_sizes.size() - 1}; i >= 0; --i) {
      auto list_size = list_sizes.at(i);
      row_indices.at(i) = tmp_idx % list_size;
      tmp_idx /= list_size;
    }
    // not using for_each_list_pair here because I may need to translate
    // this to a cuda kernel, this is more likely what it will look like.
    bool compat = true;
    for (size_t i{}, n{}; compat && (i < compat_matrices.size() - 1); ++i) {
      for (size_t j{i + 1}; j < compat_matrices.size(); ++j, ++n) {
        auto offset = row_indices.at(i) * list_sizes.at(j) + row_indices.at(j);
        if (!compat_matrices.at(n).at(offset)) {
          compat = false;
          break;
        }
      }
    }
    if (compat) {
      ++num_compat;
    }
  }

  auto t1 = high_resolution_clock::now();
  auto t_dur = duration_cast<milliseconds>(t1 - t0).count();
  std::cerr << " host_show_num_compat_combos "
            << " for [" << first_combo << ", " << num_combos
            << "]: " << num_compat << " - " << t_dur << "ms" << std::endl;
}

//
auto process_results(const std::vector<result_t>& results, uint64_t start_idx,
  uint64_t num_results, std::vector<uint64_t>& result_indices) {
  uint64_t hits{};
  for (uint64_t i{}; i < num_results; ++i) {
    if (results[i]) {
      result_indices.push_back(start_idx + i);
      ++hits;
    }
  }
  return hits;
}

//
auto run_get_compat_combos_task(const result_t* device_compat_matrices,
  const index_t* device_compat_matrix_start_indices,
  const std::vector<index_t>& idx_list_sizes) {
  //
  constexpr auto logging = false;

  using namespace std::chrono;
  uint64_t first_combo{};
  uint64_t num_combos{600'000'000}; // chunk size
  const uint64_t max_combos{multiply_with_overflow_check(idx_list_sizes)};
  // TODO: free
  auto device_results = alloc_results(num_combos);
  std::vector<result_t> host_results(num_combos);
  std::vector<uint64_t> result_indices;
  for (int n{};; ++n, first_combo += num_combos) {
    num_combos = std::min(num_combos, max_combos - first_combo);
    if (!num_combos) {
      break;
    }
    if constexpr (logging) {
      std::cerr << " launching get_compat_combos_kernel " << n << " for ["
                << first_combo << ", " << first_combo + num_combos << "]"
                << std::endl;
    }
    auto gcc0 = high_resolution_clock::now();

    run_get_compat_combos_kernel(first_combo, num_combos,
      device_compat_matrices, device_compat_matrix_start_indices,
      idx_list_sizes.size(), MFD.device_idx_list_sizes, device_results);

    // sync is only necessary for accurate timing
    cudaStreamSynchronize(cudaStreamPerThread);
    auto gcc1 = high_resolution_clock::now();
    auto gcc_dur = duration_cast<milliseconds>(gcc1 - gcc0).count();
    if constexpr (logging) {
      std::cerr << " completed get_compat_combos_kernel " << n << " - "
                << gcc_dur << "ms" << std::endl;
    }
    auto cpr0 = high_resolution_clock::now();

    sync_copy_results(host_results, num_combos, device_results, cudaStreamPerThread);
    auto num_hits =
      process_results(host_results, first_combo, num_combos, result_indices);

    auto cpr1 = high_resolution_clock::now();
    auto cpr_dur = duration_cast<milliseconds>(cpr1 - cpr0).count();
    if constexpr (logging) {
      std::cerr << " copy/process results, hits: " << num_hits << " - "
                << cpr_dur << "ms" << std::endl;
    }
  }
  return result_indices;
}

auto get_src_indices(
  uint64_t combo_idx, const std::vector<IndexList>& idx_lists) {
  std::vector<index_t> src_indices(idx_lists.size());
  for (int i{(int)idx_lists.size() - 1}; i >= 0; --i) {
    const auto& idx_list = idx_lists.at(i);
    src_indices.at(i) = idx_list.at(combo_idx % idx_list.size());
    combo_idx /= idx_list.size();
  }
  return src_indices;
}

XorSource merge_sources(const std::vector<index_t>& src_indices,
  const std::vector<SourceList>& src_lists) {
  // this code is taken from cm-precompute.cpp.
  // TODO: this is kind of a weird way of doing things that requires a
  // XorSource (SourceData) multi-type move constructor. couldn't I just
  // start with an XorSource here initialized to sourceList[0] values
  // and merge-in-place all the subsequent elements? could even be a
  // SourceData member function.
  NameCountList primaryNameSrcList{};
  NameCountList ncList{};
  UsedSources usedSources{};
  for (size_t i{}; i < src_indices.size(); ++i) {
    const auto& src = src_lists.at(i).at(src_indices.at(i));
    const auto& pnsl = src.primaryNameSrcList;
    primaryNameSrcList.insert(
      primaryNameSrcList.end(), pnsl.begin(), pnsl.end());  // copy
    const auto& ncl = src.ncList;
    ncList.insert(ncList.end(), ncl.begin(), ncl.end()); // copy
    usedSources.mergeInPlace(src.usedSources);
  }
  assert(!primaryNameSrcList.empty() && !ncList.empty() && "empty ncList");
  return XorSource{
    std::move(primaryNameSrcList), std::move(ncList), std::move(usedSources)};
}

}  // namespace

namespace cm {

auto cuda_get_compat_xor_src_indices(const std::vector<SourceList>& src_lists,
  const std::vector<IndexList>& idx_lists) -> std::vector<uint64_t> {
  //
  using namespace std::chrono;
  assert(!src_lists.empty() && !idx_lists.empty()
         && !getNumEmptySublists(src_lists) && "cuda_merge: invalid param");

  auto t0 = high_resolution_clock::now();
  size_t total_bytes_allocated{};
  size_t total_bytes_copied{};
  size_t num_bytes{};
  // std::cerr << " copying src_lists to device..." << std::endl;
  // alloc/copy source lists and generate start indices
  // TODO: free
  MFD.device_src_lists = alloc_copy_src_lists(src_lists, &num_bytes);
  total_bytes_allocated += num_bytes;
  total_bytes_copied += num_bytes;
  auto src_list_start_indices = make_start_indices(src_lists);

  // std::cerr << " copying idx_lists to device..." << std::endl;
  // alloc/copy index lists and generate start indices
  // TODO: free
  MFD.device_idx_lists = alloc_copy_idx_lists(idx_lists, &num_bytes);
  total_bytes_allocated += num_bytes;
  total_bytes_copied += num_bytes;
  auto idx_list_start_indices = make_start_indices(idx_lists);

  // alloc compatibility result matrices and generate start indices
  std::cerr << " allocating compat_matrics on device..." << std::endl;
  size_t num_compat_matrix_bytes{};
  // TODO: free
  auto device_compat_matrices =
    alloc_compat_matrices(idx_lists, &num_compat_matrix_bytes);
  total_bytes_allocated += num_compat_matrix_bytes;
  auto compat_matrix_start_indices =
    make_compat_matrix_start_indices(idx_lists);
  if constexpr (0) {
    // sync if we want accurate timing
    cudaError_t err = cudaStreamSynchronize(cudaStreamPerThread);
    assert(err == cudaSuccess);
    auto t1 = high_resolution_clock::now();
    auto t_dur = duration_cast<milliseconds>(t1 - t0).count();
    std::cerr << "  copy complete - " << t_dur << "ms" << std::endl;
    std::cerr << "  allocated: " << total_bytes_allocated
              << ", copied: " << total_bytes_copied << std::endl;
  }

  auto lp0 = high_resolution_clock::now();
  //  run list_pair_compat kernels
  //  the only unnecessary capture here is src_lists
  for_each_list_pair(idx_lists, [&](size_t i, size_t j, size_t n) {
    if constexpr (0) {
      std::cerr << "  launching list_pair_compat_kernel " << n << " for [" << i
                << ", " << j << "]" << std::endl;
    }
    run_list_pair_compat_kernel(
      &MFD.device_src_lists[src_list_start_indices.at(i)],
      &MFD.device_src_lists[src_list_start_indices.at(j)],
      &MFD.device_idx_lists[idx_list_start_indices.at(i)],
      idx_lists.at(i).size(),
      &MFD.device_idx_lists[idx_list_start_indices.at(j)],
      idx_lists.at(j).size(),
      &device_compat_matrices[compat_matrix_start_indices.at(n)]);
  });
  if constexpr (0) {
    // sync if we want accurate timing
    cudaError_t err = cudaStreamSynchronize(cudaStreamPerThread);
    assert(err == cudaSuccess);
    auto lp1 = high_resolution_clock::now();
    auto lp_dur = duration_cast<milliseconds>(lp1 - lp0).count();
    std::cerr << "  list_pair_compat_kernels complete - " << lp_dur << "ms"
              << std::endl;
  }

  // debugging
  // debug_copy_show_compat_matrix_hit_count(device_compat_matrices,
  // num_compat_matrix_bytes);

  // alloc/copy start indices
  // TODO: free
  auto device_compat_matrix_start_indices =
    alloc_copy_start_indices(compat_matrix_start_indices);
  const auto idx_list_sizes = make_list_sizes(idx_lists);
  MFD.device_idx_list_sizes = alloc_copy_list_sizes(idx_list_sizes);

  auto gcc0 = high_resolution_clock::now();

  // run get_compat_combos kernels
  auto combo_indices = run_get_compat_combos_task(
    device_compat_matrices, device_compat_matrix_start_indices, idx_list_sizes);

  auto gcc1 = high_resolution_clock::now();
  auto gcc_dur = duration_cast<milliseconds>(gcc1 - gcc0).count();
  std::cerr << "  get_compat_combos kernels complete (" << combo_indices.size()
            << ") - " << gcc_dur << "ms" << std::endl;

  auto t1 = high_resolution_clock::now();
  auto t_dur = duration_cast<milliseconds>(t1 - t0).count();
  std::cerr << " cuda_get_compat_xor_src_indices - " << t_dur << "ms" << std::endl;

#if defined(HOST_SIDE_COMPARISON)
  // host-side compat combo counter for debugging/comparison. slow.
  host_show_num_compat_combos(0, multiply_with_overflow_check(list_sizes),
    get_compat_matrices(src_lists, idx_lists), list_sizes);
#endif
  return combo_indices;
}

// Given a vector of source lists, generate a vector of index lists: one for
// each source list, representing every source in every list. Then filter out
// incompatible indices from each index list, returning a vector of compatible
// index lists.
auto get_compatible_indices(const std::vector<SourceList>& src_lists)
  -> std::vector<IndexList> {
  //
  std::vector<size_t> lengths{};
  lengths.reserve(src_lists.size());
  for (const auto& sl : src_lists) {
    lengths.push_back(sl.size());
  }
  std::cerr << "  initial lengths: " << vec_to_string(lengths)
            << ", product: " << multiply_with_overflow_check(lengths)
            << std::endl;
  auto idx_lists = Peco::initial_indices(lengths);
  bool valid = filterAllXorIncompatibleIndices(idx_lists, src_lists);
  lengths.clear();
  for (const auto& il : idx_lists) {
    lengths.push_back(std::distance(il.begin(), il.end()));
  }
  std::cerr << "  filtered lengths: " << vec_to_string(lengths)
            << ", product: " << multiply_with_overflow_check(lengths)
            << ", valid: " << std::boolalpha << valid << std::endl;
  // "valid" means all resulting index lists have non-zero length
  if (!valid) {
    return {};
  }
  return Peco::to_vectors(idx_lists);
}

auto merge_xor_sources(const std::vector<SourceList>& src_lists,
  const std::vector<IndexList>& idx_lists,
  const std::vector<uint64_t>& combo_indices) -> XorSourceList {
  //
  using namespace std::chrono;
  XorSourceList xorSourceList;

  std::cerr << "  starting merge_xor_sources..." << std::endl;
  auto hm0 = high_resolution_clock::now();
  for (auto combo_idx : combo_indices) {
    auto src_indices = get_src_indices(combo_idx, idx_lists);
    xorSourceList.emplace_back(merge_sources(src_indices, src_lists));
  }
  auto hm1 = high_resolution_clock::now();
  auto hm_dur = duration_cast<milliseconds>(hm1 - hm0).count();
  std::cerr << "  merge_xor_sources complete (" << xorSourceList.size()
            << ") - " << hm_dur << "ms" << std::endl;

  return xorSourceList;
}

}  // namespace cm
