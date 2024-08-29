#include <experimental/scope>
#include <chrono>
#include <iostream>
#include "merge.cuh"
#include "merge.h"
#include "merge-filter-common.h"
#include "merge-filter-data.h"
#include "peco.h"
#include "util.h"
#include "log.h"

namespace cm {

namespace {

auto getNumEmptySublists(const std::vector<SourceList>& src_lists) {
  int count{};
  for (const auto& sl : src_lists) {
    if (sl.empty()) count++;
  }
  return count;
}

auto any_compatible_sources(const SourceData& source,
    const Peco::IndexList& idx_list, const SourceList& src_list,
    MergeType merge_type) {
  for (auto idx : idx_list) {
    if (merge_type == MergeType::XOR) {
      if (source.isXorCompatibleWith(src_list.at(idx))) return true;
    } else { // MergeType::OR
      if (source.hasCompatibleVariationsWith(src_list.at(idx))) return true;
    }
  }
  return false;
}

// TODO: comment this. code is tricky, but fast.
auto filter_indices(Peco::IndexListVector& index_lists, size_t first_list_idx,
    size_t second_list_idx, const std::vector<SourceList>& src_lists,
    MergeType merge_type) {
  Peco::IndexList& first_list = index_lists[first_list_idx];
  for (auto it_first = first_list.before_begin();
       std::next(it_first) != first_list.end();) {
    const auto& first_src = src_lists[first_list_idx][*std::next(it_first)];
    bool any_compat = any_compatible_sources(first_src,
        index_lists[second_list_idx], src_lists[second_list_idx], merge_type);
    if (!any_compat) {
      first_list.erase_after(it_first);
    } else {
      ++it_first;
    }
  }
  return !first_list.empty();
}

auto filter_all_indices(Peco::IndexListVector& idx_lists,
    const std::vector<SourceList>& src_lists, MergeType merge_type) {
  // TODO: or, assert? is this the right place for this check?
  if (idx_lists.size() < 2u) return true;
  util::LogDuration ld("  filter incompatible", Verbose);
  for (size_t first{}; first < idx_lists.size(); ++first) {
    for (size_t second{}; second < idx_lists.size(); ++second) {
      if (first == second) continue;
      if (!filter_indices(idx_lists, first, second, src_lists, merge_type)) {
        return false;
      }
    }
  }
  return true;
}

/*
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
  for (size_t i{}; i < flat_indices.size(); i += row_size) {
    for (size_t j{}; j < row_size; ++j) {
      if (j)
        std::cerr << ", ";
      std::cerr << flat_indices[i + j];
    }
    std::cerr << std::endl;
  }
}
*/

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

// The back-to-front order of pair processing is intentional and important here,
// as other code depends on it. E.g. the order for 4 lists is:
//
// [2,3],[1,3],[0,3],[1,2],[0,2],[0,1]
//
void for_each_list_pair(
    const std::vector<IndexList>& idx_lists, const auto& fn) {
  for (int i{int(idx_lists.size()) - 1}, n{}; i > 0; --i) {
    for (auto j{i - 1}; j >= 0; --j, ++n) {
      fn(j, i, n);
    }
  }
}

auto alloc_compat_matrices(const std::vector<IndexList>& idx_lists,
    cudaStream_t stream, size_t* num_bytes = nullptr) {
  size_t num_iterated_list_pairs{};
  fat_index_t num_results{};
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
  result_t* device_compat_matrices;
  cuda_malloc_async((void**)&device_compat_matrices, matrix_bytes, stream,
      "compat_matrices");  // cl-format
  if (num_bytes) {
    *num_bytes = matrix_bytes;
  }
  return device_compat_matrices;
}

auto make_compat_matrix_start_indices(const std::vector<IndexList>& idx_lists) {
  IndexList start_indices{};
  size_t index{};
  for_each_list_pair(idx_lists, [&](size_t i, size_t j, size_t n) {
    start_indices.push_back(index_t(index));
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
  CudaEvent temp(stream);
  temp.synchronize();
  fat_index_t num_compat{};
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

void host_show_num_compat_combos(const int first_combo,
    const int num_combos,
    const std::vector<std::vector<result_t>>& compat_matrices,
    const IndexList& list_sizes) {
  assert(list_sizes.size() == compat_matrices.size());
  auto t = util::Timer::start_timer();
  IndexList row_indices(10);
  int num_compat{};
  for (auto idx{first_combo}; idx < num_combos; ++idx) {
    auto tmp_idx{idx};
    // TODO: this is wrong/backwards
    assert(false && "bad code");
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
  t.stop();
  std::cerr << " host_show_num_compat_combos "
            << " for [" << first_combo << ", " << num_combos
            << "]: " << num_compat << " - " << t.count() << "ms" << std::endl;
}

auto log_compat_combos_kernel(int idx, CudaEvent& start, LogLevel level = Verbose) {
  long elapsed{};
  if (log_level(level)) {
    CudaEvent stop(start.stream());
    elapsed = stop.synchronize(start);
    if (log_level(ExtraVerbose)) {
      std::cerr << "  completed get_compat_combos_kernel " << idx << " - "
                << elapsed << "ms" << std::endl;
    }
  }
  return elapsed;
}

auto log_compat_copy_process(util::Timer& t_copy, util::Timer& t_proc,
    fat_index_t num_hits, LogLevel level = Verbose) {
  long elapsed{};
  if (log_level(level)) {
    t_proc.stop();
    elapsed = t_proc.count();
    if (log_level(ExtraVerbose)) {
      std::cerr << "  copy/process results, hits: " << num_hits
                << ", copy: " << t_copy.count() << "ms"
                << ", process: " << t_proc.count() << "ms" << std::endl;
    }
  }
  return elapsed;
}

auto sync_copy_compat_results(std::vector<result_t>& results,
    fat_index_t num_results, result_t* device_results, cudaStream_t stream) {
  const auto num_bytes = num_results * sizeof(result_t);
  CudaEvent start(stream);
  auto err = cudaMemcpyAsync(results.data(), device_results, num_bytes,
      cudaMemcpyDeviceToHost, stream);
  assert_cuda_success(err, "merge copy results");
  CudaEvent stop(start.stream());
  return stop.synchronize(start);
}

auto process_results(const std::vector<result_t>& results, fat_index_t start_idx,
    fat_index_t num_results, std::vector<fat_index_t>& result_indices) {
  fat_index_t hits{};
  for (fat_index_t i{}; i < num_results; ++i) {
    if (results[i]) {
      result_indices.push_back(start_idx + i);
      ++hits;
    }
  }
  if (result_indices.size() > 500'000'000) {
    // 500M * 8-byte combo_index_t = 4GB. that's an acceptable limit for
    // choking at this time.
    std::cerr << "too many combo_indices: " << result_indices.size()
              << ", refine your parameters, terminating\n";
    std::terminate();
  }
  return hits;
}

auto run_get_compat_combos_task(const result_t* device_compat_matrices,
    const index_t* device_compat_matrix_start_indices,
    const index_t* device_idx_list_sizes, int num_idx_lists,
    fat_index_t max_indices, MergeType merge_type, cudaStream_t stream) {
  using namespace std::chrono;
  using namespace std::experimental::fundamentals_v3;
  constexpr auto logging = false;

  const fat_index_t chunk_size = 600'000'000;
  fat_index_t num_indices{chunk_size};
  auto device_results =
      cuda_alloc_results(num_indices, stream, "get_compat_combos results");
  scope_exit free_results{[device_results]() { cuda_free(device_results); }};

  std::vector<result_t> host_results(num_indices);
  std::vector<fat_index_t> result_indices;
  CudaEvent gcc_start(stream);
  long gcc_elapsed{};
  long copy_elapsed{};
  long proc_elapsed{};
  fat_index_t total_hits{};
  int kernel_idx{};
  for (fat_index_t first_idx{};; ++kernel_idx, first_idx += num_indices) {
    num_indices = std::min(num_indices, max_indices - first_idx);
    if (!num_indices) break;
    if constexpr (logging) {
      std::cerr << "  launching get_compat_combos_kernel " << kernel_idx
                << " for [" << first_idx << ", " << first_idx + num_indices
                << "]" << std::endl;
    }
    if (log_level(Verbose)) gcc_start.record();
    run_get_compat_combos_kernel(first_idx, num_indices, device_compat_matrices,
        device_compat_matrix_start_indices, device_idx_list_sizes,
        num_idx_lists, device_results, stream, merge_type == MergeType::OR);
    gcc_elapsed += log_compat_combos_kernel(kernel_idx, gcc_start);

    // TODO: sync_copy_results returns event.elapsed but precision is low
    // does it matter? could return a timer instead
    auto t_copy = util::Timer::start_timer();
    sync_copy_compat_results(host_results, num_indices, device_results, stream);
    t_copy.stop();
    copy_elapsed += t_copy.count();

    auto t_proc = util::Timer::start_timer();
    auto num_hits =
        process_results(host_results, first_idx, num_indices, result_indices);
    proc_elapsed += log_compat_copy_process(t_copy, t_proc, num_hits);
    total_hits += num_hits;
  }
  if (log_level(Verbose)) {
    std::cerr << " get_compat_combos_kernel total " << kernel_idx << " runs of "
              << chunk_size / 1'000'000 << "M - " << gcc_elapsed << "ms\n";
    std::cerr << " copy/process results total hits: " << total_hits
              << ", copy: " << copy_elapsed << "ms"
              << ", process: " << proc_elapsed << "ms" << std::endl;
  }
  return result_indices;
}

auto get_src_indices(
    fat_index_t flat_idx, const std::vector<IndexList>& idx_lists) {
  std::vector<index_t> src_indices(idx_lists.size());
  util::for_each_source_index(flat_idx, idx_lists,  //
      [&src_indices](index_t list_idx, index_t src_idx) {
        src_indices.at(list_idx) = src_idx;
      });
  return src_indices;
}

// Given a vector of source lists, generate a vector of index lists: one for
// each source list, each containing indices for every source in each list.
// Then filter out all incompatible, and return a vector of compatible index
// lists.
auto get_compatible_indices(const std::vector<SourceList>& src_lists,
    MergeType merge_type) -> std::vector<IndexList> {
  std::vector<size_t> lengths;
  for (const auto& sl : src_lists) {
    lengths.push_back(sl.size());
  }
  if (log_level(Verbose)) {
    std::cerr << "  initial lengths: " << util::join(lengths, ",")
              << " product: " << util::multiply_with_overflow_check(lengths)
              << std::endl;
  }
  auto idx_lists = Peco::initial_indices(lengths);
  bool valid = filter_all_indices(idx_lists, src_lists, merge_type);
  // "valid" means all resulting index lists have non-zero length
  if (!valid) return {};
  return Peco::to_vectors(idx_lists);
}

SourceCompatibilityData* cuda_alloc_copy_src_lists(
    const std::vector<SourceList>& src_lists, cudaStream_t stream) {
  // alloc sources
  const auto num_sources = util::sum_sizes(src_lists);
  const auto sources_bytes = num_sources * sizeof(SourceCompatibilityData);
  SourceCompatibilityData* device_sources;
  cuda_malloc_async((void**)&device_sources, sources_bytes, stream,  //
      "merge src_lists");
  // copy sources
  std::vector<SourceCompatibilityData> src_compat_list;
  src_compat_list.reserve(num_sources);
  // std::cerr << "  building src_compat_lists..." << std::endl;
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
  auto err = cudaMemcpyAsync(device_sources, src_compat_list.data(),
      sources_bytes, cudaMemcpyHostToDevice, stream);
  assert_cuda_success(err, "merge copy sources");
  CudaEvent temp(stream);
  temp.synchronize();
  return device_sources;
}

index_t* cuda_alloc_copy_idx_lists(
    const std::vector<IndexList>& idx_lists, cudaStream_t stream) {
  // alloc indices
  const auto indices_bytes = util::sum_sizes(idx_lists) * sizeof(index_t);
  cudaError_t err{};
  index_t* device_indices;
  cuda_malloc_async((void**)&device_indices, indices_bytes, stream,  //
      "const index/size data");  // "merge idx_lists"

  // copy indices
  size_t index{};
  for (const auto& idx_list : idx_lists) {
    err = cudaMemcpyAsync(&device_indices[index], idx_list.data(),
        idx_list.size() * sizeof(index_t), cudaMemcpyHostToDevice, stream);
    assert_cuda_success(err, "merge copy idx_lists");
    index += idx_list.size();
  }
  CudaEvent temp(stream);
  temp.synchronize();
  return device_indices;
}

index_t* cuda_alloc_copy_list_sizes(
    const std::vector<index_t>& list_sizes, cudaStream_t stream) {
  // , size_t* num_bytes = nullptr) {
  // alloc list sizes
  const auto list_sizes_bytes = list_sizes.size() * sizeof(index_t);
  cudaError_t err{};
  index_t* device_list_sizes;
  cuda_malloc_async((void**)&device_list_sizes, list_sizes_bytes, stream,  //
      "const index/size data");  // "merge list_sizes"
  // copy list sizes
  err = cudaMemcpyAsync(device_list_sizes, list_sizes.data(), list_sizes_bytes,
      cudaMemcpyHostToDevice, stream);
  assert_cuda_success(err, "copy list sizes");
  CudaEvent temp(stream);
  temp.synchronize();
  //  if (num_bytes) { *num_bytes = list_sizes_bytes; }
  return device_list_sizes;
}

/* debugging
void check_list_pair_results(combo_index_t combo_idx, result_t* device_results,
    size_t num_bytes, const std::vector<IndexList>& idx_lists,
    const IndexList& start_indices, cudaStream_t stream) {
  std::cerr << "check_list_pairs: " << start_indices.size()
            << ", num_bytes: " << num_bytes << std::endl;
  std::vector<result_t> results(num_bytes);
  sync_copy_compat_results(results, num_bytes, device_results, stream);
#if 0
  for_each_list_pair(idx_lists, [&](size_t i, size_t j, size_t n) {
    std::cerr << "list_pair " << n << " is [" << i << ", " << j << "]"
              << std::endl;
  });
#endif
  // HAND SET THESE VALUES
  // 1068
  auto list1 = 0;
  auto list2 = 1;
  auto matrix = 0; // [0,1]

  index_t row{};
  index_t col{};
  auto val = combo_idx;
  for (int i{}; i <= list2; ++i) {
    auto size = idx_lists.at(i).size();
    auto idx = val % size;
    if (i == list1) {
      row = idx;
    } else if (i == list2) {
      col = idx;
    }
    val /= size;
  }
  auto start = start_indices.at(matrix);
  auto width = idx_lists.at(list2).size();
  auto offset = row * width + col;
  std::cerr << "idx: " << combo_idx << ", matrix " << matrix
            << ", rowcol: " << row << "," << col << "  start: " << start
            << ", width: " << width << ", offset: " << offset
            << ", result: " << (int)results.at(offset) << std::endl;
}
*/
  
auto cuda_get_compatible_indices(const std::vector<SourceList>& src_lists,
    const SourceCompatibilityData* device_src_lists,
    const std::vector<IndexList>& idx_lists, const index_t* device_idx_lists,
    const index_t* device_idx_list_sizes, MergeType merge_type,
    cudaStream_t stream) -> std::vector<fat_index_t> {
  assert(!src_lists.empty() && !idx_lists.empty()
         && !getNumEmptySublists(src_lists) && "cuda_merge: invalid param");
  using namespace std::experimental::fundamentals_v3;
  auto t = util::Timer::start_timer();
  auto src_list_start_indices = make_start_indices(src_lists);
  auto idx_list_start_indices = make_start_indices(idx_lists);
  size_t num_bytes{};
  auto device_compat_matrices =
      alloc_compat_matrices(idx_lists, stream, &num_bytes);
  scope_exit free_compat_matrices(
      [device_compat_matrices]() { cuda_free(device_compat_matrices); });
  auto compat_matrix_start_indices =
      make_compat_matrix_start_indices(idx_lists);

  if constexpr (0) {
    auto err = cudaStreamSynchronize(stream);
    t.stop();
    assert_cuda_success(err, "cudaStreamSynchronize");
    std::cerr << "  compat matrix alloc/copy complete, allocated: " << num_bytes
              << " - " << t.count() << "ms" << std::endl;
  }
  // CudaEvent lp_start;
  // run list_pair_compat kernels
  // the only unnecessary capture here is src_lists
  for_each_list_pair(idx_lists, [&](size_t i, size_t j, size_t n) {
    if constexpr (0) {
      std::cerr << "  launching list_pair_compat_kernel " << n << " for [" << i
                << ", " << j << "]" << std::endl;
    }
    bool flag = (merge_type == MergeType::OR) && (n == 3);

    run_list_pair_compat_kernel(&device_src_lists[src_list_start_indices.at(i)],
        &device_src_lists[src_list_start_indices.at(j)],
        &device_idx_lists[idx_list_start_indices.at(i)],
        unsigned(idx_lists.at(i).size()),
        &device_idx_lists[idx_list_start_indices.at(j)],
        unsigned(idx_lists.at(j).size()),
        &device_compat_matrices[compat_matrix_start_indices.at(n)], merge_type,
        stream, flag);
  });
  // debugging
  /*
  if constexpr (0) {
    CudaEvent lp_stop;
    auto lp_dur = lp_stop.synchronize(lp_start);
    std::cerr << "  list_pair_compat_kernels complete - " << lp_dur <<
      "ms\n";
  }
  debug_copy_show_compat_matrix_hit_count(device_compat_matrices,
    num_compat_matrix_bytes);
  */

  // alloc/copy start indices
  auto device_compat_matrix_start_indices = cuda_alloc_copy_start_indices(
      compat_matrix_start_indices, stream, "compat_matrix_start_indices");
  scope_exit free_start_indices([device_compat_matrix_start_indices]() {
    cuda_free(device_compat_matrix_start_indices);
  });
  const auto max_indices =
      util::multiply_with_overflow_check(util::make_list_sizes(idx_lists));
  auto compat_indices = run_get_compat_combos_task(device_compat_matrices,
      device_compat_matrix_start_indices, device_idx_list_sizes,
      int(idx_lists.size()), max_indices, merge_type, stream);

#if defined(HOST_SIDE_COMPARISON)
  // host-side compat combo counter for debugging/comparison. slow.
  // TODO: if i ever enable this again, consider not passing get_compat_matrices
  // by value. might require we initialize it to local var that is passed by ref.
  host_show_num_compat_combos(0, multiply_with_overflow_check(list_sizes),
    get_compat_matrices(src_lists, idx_lists), list_sizes);
#endif
  return compat_indices;
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
    // TODO: std::ranges::copy
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

auto xor_merge_sources(const std::vector<SourceList>& src_lists,
    const std::vector<IndexList>& idx_lists,
    const std::vector<fat_index_t>& compat_flat_indices) -> XorSourceList {
  XorSourceList xorSourceList;
  // TODO:
  //LogDuration ld("xor_merge_sources", Verbose);
  if (log_level(Verbose)) {
    std::cerr << "  starting xor_merge_sources..." << std::endl;
  }
  auto t = util::Timer::start_timer();
  for (auto flat_idx : compat_flat_indices) {
    auto src_indices = get_src_indices(flat_idx, idx_lists);
    xorSourceList.emplace_back(merge_sources(src_indices, src_lists));
  }
  if (log_level(Verbose)) {
    t.stop();
    std::cerr << "  xor_merge_sources complete (" << xorSourceList.size()
              << ") - " << t.count() << "ms" << std::endl;
  }
  return xorSourceList;
}

void log_compat_indices(const std::vector<IndexList>& idx_lists,
    MergeType merge_type, bool merge_only) {
  std::vector<size_t> lengths;
  for (const auto& idx_list : idx_lists) {
    lengths.push_back(idx_list.size());
  }
  auto product = util::multiply_with_overflow_check(lengths);
  if (log_level(Verbose)) {
    std::cerr << "  filtered lengths: " << util::join(lengths, ",")
              << " sum: " << util::sum_sizes(idx_lists)
              << " product: " << product << std::endl;
  }
  if (!merge_only || log_level(Verbose)) {
    std::cerr << (merge_type == MergeType::XOR ? "XOR" : "OR")
              << " compat_idx_lists(" << idx_lists.size() << ")" << std::endl;
  }
#if 1
  for (size_t i{}; i < idx_lists.size(); ++i) {
    const auto& list = idx_lists.at(i);
    std::cerr << "list[" << i << "][0] = "<<  list.at(0) << " ";
  }
  std::cerr << std::endl;
#endif
}

}  // anonymous namespace

///////////////////////////////////////////////////////////////////////////////

auto get_merge_data(const std::vector<SourceList>& src_lists,
    MergeData::Host& host, MergeData::Device& device, MergeType merge_type,
    cudaStream_t stream, bool merge_only /* = false */) -> bool {
  // TODO: support for single-list compat indices (??)
  auto compat_idx_lists =
    get_compatible_indices(src_lists, merge_type);
  log_compat_indices(compat_idx_lists, merge_type, merge_only);
  if (compat_idx_lists.empty()) return false;
  device.src_lists = cuda_alloc_copy_src_lists(src_lists, stream);
  device.idx_lists = cuda_alloc_copy_idx_lists(compat_idx_lists, stream);
  const auto idx_list_sizes = util::make_list_sizes(compat_idx_lists);
  device.idx_list_sizes = cuda_alloc_copy_list_sizes(idx_list_sizes, stream);
  device.sum_idx_list_sizes = util::sum(idx_list_sizes);
  device.num_idx_lists = int(compat_idx_lists.size());
  host.compat_idx_lists = std::move(compat_idx_lists);
  const auto level = merge_only ? ExtraVerbose : Normal;
  util::LogDuration ld("cuda_get_compatible_indices", level);
  host.compat_indices = cuda_get_compatible_indices(src_lists, device.src_lists,
      host.compat_idx_lists, device.idx_lists, device.idx_list_sizes,
      merge_type, stream);
  return true;
}

auto merge_xor_compatible_src_lists(
    const std::vector<SourceList>& src_lists) -> SourceList {
  assert(src_lists.size() > 1);
  MergeData md;
  if (!get_merge_data(src_lists, md.host, md.device, MergeType::XOR,
          cudaStreamPerThread, true)) {
    return {};
  }
  return xor_merge_sources(
      src_lists, md.host.compat_idx_lists, md.host.compat_indices);
}

}  // namespace cm
