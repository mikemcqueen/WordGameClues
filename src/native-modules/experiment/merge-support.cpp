#include <experimental/scope>
#include <chrono>
#include <iostream>
#include <cuda_runtime.h>
#include "merge.h"
#include "merge-filter-common.h"
#include "merge-filter-data.h"
#include "peco.h"
#include "util.h"
#include "cuda-types.h"
#include "log.h"

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

// TODO: comment this. code is tricky, but fast.
auto filterXorIncompatibleIndices(Peco::IndexListVector& index_lists,
  int first_list_idx, int second_list_idx,
  const std::vector<SourceList>& src_lists) {
  //
  Peco::IndexList& first_list = index_lists[first_list_idx];
  for (auto it_first = first_list.before_begin();
       std::next(it_first) != first_list.end();) {
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
  const auto t0 = high_resolution_clock::now();

  if (idx_lists.size() < 2u)
    return true;
  for (size_t first{}; first < idx_lists.size(); ++first) {
    for (size_t second{}; second < idx_lists.size(); ++second) {
      if (first == second)
        continue;
      if (!filterXorIncompatibleIndices(idx_lists, first, second, src_lists)) {
        return false;
      }
    }
  }
  if (log_level(Verbose)) {
    const auto t1 = high_resolution_clock::now();
    const auto t_dur = duration_cast<milliseconds>(t1 - t0).count();
    std::cerr << "  filter incompatible - " << t_dur << "ms" << std::endl;
  }
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

void dump_flat_indices(
  const std::vector<index_t>& flat_indices, size_t row_size) {
  //
  for (size_t i{}; i < flat_indices.size(); i += row_size) {
    for (size_t j{}; j < row_size; ++j) {
      if (j)
        std::cerr << ", ";
      std::cerr << flat_indices[i + j];
    }
    std::cerr << std::endl;
  }
}

void sync_copy_results(std::vector<result_t>& results, unsigned num_results,
  result_t* device_results, cudaStream_t stream) {
  // TODO: I don't remember what my logic was here in using events to synchronize.
  CudaEvent temp1;
  temp1.synchronize();
  const auto results_bytes = num_results * sizeof(result_t);
  cudaError_t err = cudaMemcpyAsync(results.data(), device_results, results_bytes,
    cudaMemcpyDeviceToHost, stream);
  assert_cuda_success(err, "merge copy results");
  CudaEvent temp2;
  temp2.synchronize();
}

void add_compat_sources(std::vector<SourceCompatibilityData>& compat_sources,
    const SourceList& src_list) {
  for (const auto& src : src_list) {
    compat_sources.push_back(src);
  }
}

// nC2 formula (# of combinations: pick 2 from n)
auto calc_num_list_pairs(const std::vector<IndexList>& idx_lists) {
  return idx_lists.size() * (idx_lists.size() - 1) / 2;
}

void for_each_list_pair(
    const std::vector<IndexList>& idx_lists, const auto& fn) {
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

  const auto matrix_bytes = num_results * sizeof(result_t);
  auto stream = cudaStreamPerThread;
  result_t* device_compat_matrices;
  //auto err = cudaMallocAsync((void**)&device_compat_matrices, matrix_bytes, stream);
  //assert_cuda_success(err, "alloc compat matrices");
  cuda_malloc_async((void**)&device_compat_matrices, matrix_bytes, stream,
      "compat_matrices");  // cl-format
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

// for debugging
void debug_copy_show_compat_matrix_hit_count(
    const result_t* device_compat_matrices, size_t num_bytes) {
  cudaStream_t stream = cudaStreamPerThread;
  std::vector<result_t> results(num_bytes);
  auto err = cudaMemcpyAsync(results.data(), device_compat_matrices, num_bytes,
    cudaMemcpyDeviceToHost, stream);
  assert_cuda_success(err, "copy compat matrices");
  CudaEvent temp;
  temp.synchronize();
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
auto get_compat_matrices(const std::vector<SourceList>& src_lists,
    std::vector<IndexList> idx_lists) {
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
          const auto compat = src1.isXorCompatibleWith(src2);
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
    unsigned num_compat_matrices,
    const index_t* device_compat_matrix_start_indices,
    const index_t* device_idx_list_sizes, uint64_t max_combos) {
  using namespace std::chrono;
  using namespace std::experimental::fundamentals_v3;
  constexpr auto logging = false;
  const auto stream = cudaStreamPerThread;
  uint64_t first_combo{};
  uint64_t num_combos{600'000'000}; // chunk size
  auto device_results =
      cuda_alloc_results(num_combos, stream, "get_compat_combos results");
  scope_exit free_results{[device_results]() { cuda_free(device_results); }};
  std::vector<result_t> host_results(num_combos);
  std::vector<uint64_t> result_indices;
  for (int n{};; ++n, first_combo += num_combos) {
    num_combos = std::min(num_combos, max_combos - first_combo);
    if (!num_combos)
      break;
    if constexpr (logging) {
      std::cerr << " launching get_compat_combos_kernel " << n << " for ["
                << first_combo << ", " << first_combo + num_combos << "]"
                << std::endl;
    }

    CudaEvent gcc_start;
    run_get_compat_combos_kernel(first_combo, num_combos,
      device_compat_matrices, num_compat_matrices,
      device_compat_matrix_start_indices, device_idx_list_sizes,
      device_results);

    if (log_level(Verbose)) {
      CudaEvent gcc_stop;
      auto gcc_elapsed = gcc_stop.synchronize(gcc_start);
        std::cerr << " completed get_compat_combos_kernel " << n << " - "
                  << gcc_elapsed << "ms" << std::endl;
    }

    auto cpr0 = high_resolution_clock::now();
    sync_copy_results(host_results, num_combos, device_results, stream);
    auto num_hits =
      process_results(host_results, first_combo, num_combos, result_indices);

    if constexpr (logging) {
      auto cpr1 = high_resolution_clock::now();
      auto cpr_dur = duration_cast<milliseconds>(cpr1 - cpr0).count();
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
    ncList.insert(ncList.end(), ncl.begin(), ncl.end());  // copy
    usedSources.mergeInPlace(src.usedSources);
  }
  assert(!primaryNameSrcList.empty() && !ncList.empty() && "empty ncList");
  return {
    std::move(primaryNameSrcList), std::move(ncList), std::move(usedSources)};
}

}  // anonymous namespace

namespace cm {

auto cuda_get_compat_xor_src_indices(const std::vector<SourceList>& src_lists,
    const SourceCompatibilityData* device_src_lists,
    const std::vector<IndexList>& idx_lists, const index_t* device_idx_lists,
    const index_t* device_idx_list_sizes) -> std::vector<uint64_t> {
  assert(!src_lists.empty() && !idx_lists.empty()
         && !getNumEmptySublists(src_lists) && "cuda_merge: invalid param");
  using namespace std::chrono;
  using namespace std::experimental::fundamentals_v3;
  const auto stream = cudaStreamPerThread;
  auto t0 = high_resolution_clock::now();
  auto src_list_start_indices = make_start_indices(src_lists);
  auto idx_list_start_indices = make_start_indices(idx_lists);
  size_t num_bytes{};
  auto device_compat_matrices = alloc_compat_matrices(idx_lists, &num_bytes);
  scope_exit free_compat_matrices(
    [device_compat_matrices]() { cuda_free(device_compat_matrices); });
  auto compat_matrix_start_indices =
    make_compat_matrix_start_indices(idx_lists);

  if constexpr (0) {
    // sync if we want accurate timing
    auto err = cudaStreamSynchronize(stream);
    assert_cuda_success(err, "cudaStreamSynchronize");
    auto t1 = high_resolution_clock::now();
    auto t_dur = duration_cast<milliseconds>(t1 - t0).count();
    std::cerr << "  alloc/copy complete, allocated: " << num_bytes << " - "
              << t_dur << "ms" << std::endl;
  }

  CudaEvent lp_start;
  // run list_pair_compat kernels
  // the only unnecessary capture here is src_lists
  for_each_list_pair(idx_lists, [&](size_t i, size_t j, size_t n) {
    if constexpr (0) {
      std::cerr << "  launching list_pair_compat_kernel " << n << " for [" << i
                << ", " << j << "]" << std::endl;
    }
    run_list_pair_compat_kernel(
      &device_src_lists[src_list_start_indices.at(i)],
      &device_src_lists[src_list_start_indices.at(j)],
      &device_idx_lists[idx_list_start_indices.at(i)],
      idx_lists.at(i).size(),
      &device_idx_lists[idx_list_start_indices.at(j)],
      idx_lists.at(j).size(),
      &device_compat_matrices[compat_matrix_start_indices.at(n)]);
  });
  if constexpr (0) {
    CudaEvent lp_stop;
    auto lp_dur = lp_stop.synchronize(lp_start);
    std::cerr << "  list_pair_compat_kernels complete - " << lp_dur << "ms"
              << std::endl;
  }

  // debugging
  // debug_copy_show_compat_matrix_hit_count(device_compat_matrices,
  // num_compat_matrix_bytes);

  // alloc/copy start indices
  auto device_compat_matrix_start_indices = alloc_copy_start_indices(
      compat_matrix_start_indices, stream, "compat_matrix_start_indices");
  scope_exit free_start_indices([device_compat_matrix_start_indices]() {
    cuda_free(device_compat_matrix_start_indices);
  });
  const uint64_t max_combos{
    util::multiply_with_overflow_check(util::make_list_sizes(idx_lists))};
  // run get_compat_combos kernels
  CudaEvent gcc_start;
  auto combo_indices = run_get_compat_combos_task(device_compat_matrices,
    idx_lists.size(), device_compat_matrix_start_indices, device_idx_list_sizes,
    max_combos);  // clang-format

  if (log_level(ExtraVerbose)) {
    CudaEvent gcc_stop;
    gcc_stop.synchronize(gcc_start);
    auto t1 = high_resolution_clock::now();
    auto t_dur = duration_cast<milliseconds>(t1 - t0).count();
    std::cerr << " cuda_get_compat_xor_src_indices - " << t_dur << "ms"
              << std::endl;
  }

#if defined(HOST_SIDE_COMPARISON)
  // host-side compat combo counter for debugging/comparison. slow.
  // TODO: if i ever enable this again, consider not passing get_compat_matrices
  // by value. might require we initialize it to local var that is passed by ref.
  host_show_num_compat_combos(0, multiply_with_overflow_check(list_sizes),
    get_compat_matrices(src_lists, idx_lists), list_sizes);
#endif
  return combo_indices;
}

// Given a vector of source lists, generate a vector of index lists: one for
// each source list, together containing indices for every source in every
// list. Then filter out all incompatible, and return a vector of compatible
// index lists.
auto get_compatible_indices(const std::vector<SourceList>& src_lists)
    -> std::vector<IndexList> {
  std::vector<size_t> lengths{};
  lengths.reserve(src_lists.size());
  for (const auto& sl : src_lists) {
    lengths.push_back(sl.size());
  }
  if (log_level(Verbose)) {
    std::cerr << "  initial lengths: " << util::vec_to_string(lengths)
              << ", product: " << util::multiply_with_overflow_check(lengths)
              << std::endl;
  }
  auto idx_lists = Peco::initial_indices(lengths);
  bool valid = filterAllXorIncompatibleIndices(idx_lists, src_lists);
  if (log_level(Verbose)) {
    lengths.clear();
    for (const auto& il : idx_lists) {
      lengths.push_back(std::distance(il.begin(), il.end()));
    }
    std::cerr << "  filtered lengths: " << util::vec_to_string(lengths)
              << ", product: " << util::multiply_with_overflow_check(lengths)
              << ", valid: " << std::boolalpha << valid << std::endl;
  }
  // "valid" means all resulting index lists have non-zero length
  if (!valid) {
    return {};
  }
  return Peco::to_vectors(idx_lists);
}

auto merge_xor_sources(const std::vector<SourceList>& src_lists,
    const std::vector<IndexList>& idx_lists,
    const std::vector<uint64_t>& combo_indices) -> XorSourceList {
  using namespace std::chrono;
  XorSourceList xorSourceList;

  if (log_level(Verbose)) {
    std::cerr << "  starting merge_xor_sources..." << std::endl;
  }
  auto t = util::Timer::start_timer();
  for (auto combo_idx : combo_indices) {
    auto src_indices = get_src_indices(combo_idx, idx_lists);
    xorSourceList.emplace_back(merge_sources(src_indices, src_lists));
  }
  if (log_level(Verbose)) {
    t.stop();
    std::cerr << "  merge_xor_sources complete (" << xorSourceList.size()
              << ") - " << t.count() << "ms" << std::endl;
  }
  return xorSourceList;
}

SourceCompatibilityData* alloc_copy_src_lists(
  const std::vector<SourceList>& src_lists, size_t* num_bytes /* = nullptr*/) {
  // alloc sources
  const auto stream = cudaStreamPerThread;
  const auto num_sources = util::sum_sizes(src_lists);
  const auto sources_bytes = num_sources * sizeof(SourceCompatibilityData);
  cudaError_t err{};
  SourceCompatibilityData* device_sources;
  // err = cudaMallocAsync((void**)&device_sources, sources_bytes, stream);
  // assert_cuda_success(err, "merge alloc sources");
  cuda_malloc_async((void**)&device_sources, sources_bytes, stream,
      "merge src_lists");  // cl-format
  // copy sources
  std::vector<SourceCompatibilityData> src_compat_list;
  src_compat_list.reserve(num_sources);
  //std::cerr << "  building src_compat_lists..." << std::endl;
  for (const auto& src_list : src_lists) {
    // this is somewhat braindead. i'm basically "slicing" each SourceData
    // into a subset of itself (SourceCompatibilityData) and shoving it into
    // a 2nd list.
    // perhaps I could only marshal the SourceCompatibiltyData when passed
    // from JavaScript instead? since I no longer need to round-trip it back
    // to JavaScript.
    // alternatively, build one gimungous array and blast it over
    // in one memcpy. (uh, aren't I doing that now?)
    // TODO: comments in todo file or in notebook about how to split
    // pnsl/ncList from SourceCompatibilityData.
    add_compat_sources(src_compat_list, src_list);
  }
  //std::cerr << "  copying src_compat_lists..." << std::endl;
  err = cudaMemcpyAsync(device_sources, src_compat_list.data(), sources_bytes,
    cudaMemcpyHostToDevice, stream);
  assert_cuda_success(err, "merge copy sources");
  CudaEvent temp;
  temp.synchronize();
  if (num_bytes) {
    *num_bytes = sources_bytes;
  }
  return device_sources;
}

index_t* alloc_copy_idx_lists(
    const std::vector<IndexList>& idx_lists, size_t* num_bytes /* = nullptr*/) {
  // alloc indices
  const auto stream = cudaStreamPerThread;
  const auto indices_bytes = util::sum_sizes(idx_lists) * sizeof(index_t);
  cudaError_t err{};
  index_t* device_indices;
  // err = cudaMallocAsync((void**)&device_indices, indices_bytes, stream);
  // assert_cuda_success(err, "merge alloc idx lists");
  cuda_malloc_async((void**)&device_indices, indices_bytes, stream,
      "merge idx_lists");  // cl-format

  // copy indices
  size_t index{};
  for (const auto& idx_list : idx_lists) {
    err = cudaMemcpyAsync(&device_indices[index], idx_list.data(),
      idx_list.size() * sizeof(index_t), cudaMemcpyHostToDevice, stream);
    assert_cuda_success(err, "merge copy idx_lists");
    index += idx_list.size();
  }
  CudaEvent temp;
  temp.synchronize();
  if (num_bytes) {
    *num_bytes = indices_bytes;
  }
  return device_indices;
}

index_t* alloc_copy_list_sizes(
    const std::vector<index_t>& list_sizes, size_t* num_bytes /* = nullptr*/) {
  const auto stream = cudaStreamPerThread;
  // alloc list sizes
  const auto list_sizes_bytes = list_sizes.size() * sizeof(index_t);
  cudaError_t err{};
  index_t* device_list_sizes;
  // err = cudaMallocAsync((void**)&device_list_sizes, list_sizes_bytes, stream);
  // assert_cuda_success(err, "alloc list sizes");
  cuda_malloc_async((void**)&device_list_sizes, list_sizes_bytes, stream,
      "merge list_sizes");  // cl-format

  // copy list sizes
  err = cudaMemcpyAsync(device_list_sizes, list_sizes.data(), list_sizes_bytes,
      cudaMemcpyHostToDevice, stream);
  assert_cuda_success(err, "copy list sizes");
  CudaEvent temp;
  temp.synchronize();
  if (num_bytes) {
    *num_bytes = list_sizes_bytes;
  }
  return device_list_sizes;
}

// TODO: const src_lists input, return merged
void merge_xor_src_lists(MergeFilterData& mfd,  //
    std::vector<SourceList>& src_lists, bool merge_only) {
  if (!merge_only || (src_lists.size() > 1)) {
    // TODO: support for single-list compat indices
    auto compat_idx_lists = get_compatible_indices(src_lists);
    if (!merge_only || log_level(Verbose)) {
      std::cerr << " compat_idx_lists(" << compat_idx_lists.size() << ")"
                << std::endl;
    }
    if (!compat_idx_lists.empty()) {
      // TODO: free if already set. set_src_lists?
      mfd.device.src_lists = alloc_copy_src_lists(src_lists);
      mfd.device.idx_lists = alloc_copy_idx_lists(compat_idx_lists);
      const auto idx_list_sizes = util::make_list_sizes(compat_idx_lists);
      mfd.device.idx_list_sizes = alloc_copy_list_sizes(idx_list_sizes);
      auto combo_indices = cuda_get_compat_xor_src_indices(
          src_lists, mfd.device.src_lists, compat_idx_lists,
          mfd.device.idx_lists, mfd.device.idx_list_sizes);
      if (merge_only) {
        // compat_idx_lists and combo_indices are used in the merge_only case
        // for the sole purpose of generating merged_xor_src_list, so there is
        // no need to save them for later use
        mfd.host.merged_xor_src_list = merge_xor_sources(
            src_lists, compat_idx_lists, combo_indices);
      } else {
        // filter on the other hand will need them both later
        mfd.host.compat_idx_lists = std::move(compat_idx_lists);
        mfd.host.combo_indices = std::move(combo_indices);
      }
    }
  } else if (src_lists.size() == 1) {
    assert(merge_only);
    mfd.host.merged_xor_src_list = std::move(src_lists.back());
  }
}

}  // namespace cm
