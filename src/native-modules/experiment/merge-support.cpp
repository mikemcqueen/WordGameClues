#include <chrono>
#include <iostream>
#include <cuda_runtime.h>
#include "filter-types.h" // index_t, StreamSwarm, should generalize header
#include "merge.h"
#include "peco.h"
#include "combo-maker.h"

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

// Given a vector of source lists, generate a vector of index lists: one for
// each source list, representing every source in every list. Then filter out
// incompatible indices from each index list, returning a vector of compatible
// index lists.
std::vector<IndexList> get_compatible_indices(
  const std::vector<SourceList>& src_lists) {
  //
  std::vector<size_t> lengths{};
  lengths.reserve(src_lists.size());
  for (const auto& sl : src_lists) {
    lengths.push_back(sl.size());
  }
  std::cerr << "  initial lengths: " << vec_to_string(lengths)
            << ", product: " << vec_product(lengths) << std::endl;
  auto idx_lists = Peco::initial_indices(lengths);
  bool valid = filterAllXorIncompatibleIndices(idx_lists, src_lists);
  lengths.clear();
  for (const auto& il : idx_lists) {
    lengths.push_back(list_size(il));
  }
  std::cerr << "  filtered lengths: " << vec_to_string(lengths)
            << ", product: " << vec_product(lengths)
            << ", valid: " << std::boolalpha << valid << std::endl;
  // "valid" means all resulting index lists have non-zero length
  if (!valid) {
    return {};
  }
  return Peco::to_vectors(idx_lists);
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

void copy_results(std::vector<result_t>& results,
  result_t* device_results, cudaStream_t stream) {
  //
  cudaError_t err = cudaStreamSynchronize(stream);
  assert((err == cudaSuccess) && "merge copy sychronize1");

  auto results_bytes = results.size() * sizeof(result_t);
  err = cudaMemcpyAsync(results.data(), device_results, results_bytes,
    cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    fprintf(stdout, "merge copy results, error: %s", cudaGetErrorString(err));
    assert((err != cudaSuccess) && "merge copy device results");
  }
  err = cudaStreamSynchronize(stream);
  assert((err == cudaSuccess) && "merge copy cudaStreamSynchronize2");
}

template <typename T> auto sum_sizes(const std::vector<T>& vecs) {
  size_t sum{};
  for (const auto& v : vecs) {
    sum += v.size();
  }
  return sum;
}

auto alloc_copy_src_lists(
  const std::vector<SourceList>& src_lists, size_t* num_bytes = nullptr) {
  // alloc sources
  const cudaStream_t stream = cudaStreamPerThread;
  cudaError_t err = cudaSuccess;
  auto sources_bytes = sum_sizes(src_lists) * sizeof(SourceCompatibilityData);
  SourceCompatibilityData* device_sources;
  err = cudaMallocAsync((void**)&device_sources, sources_bytes, stream);
  assert((err == cudaSuccess) && "merge alloc sources");

  // copy sources
  size_t index{};
  for (const auto& src_list : src_lists) {
    auto compat_sources = makeCompatibleSources(src_list);
    err = cudaMemcpyAsync(&device_sources[index], compat_sources.data(),
      compat_sources.size() * sizeof(SourceCompatibilityData),
      cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
      fprintf(stdout, "merge copy sources, error: %s", cudaGetErrorString(err));
      throw std::runtime_error("merge copy sources");
    }
    index += src_list.size();
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

template <typename T>
auto make_start_indices(const std::vector<T>& vecs) {
  IndexList start_indices{};
  index_t index{};
  for (const auto& v : vecs) {
    start_indices.push_back(index);
    index += v.size();
  }
  return start_indices;
}

auto alloc_copy_start_indices(
  const IndexList& start_indices, size_t* num_bytes = nullptr) {
  cudaStream_t stream = cudaStreamPerThread;
  cudaError_t err = cudaSuccess;
  // alloc indices
  auto indices_bytes = start_indices.size() * sizeof(index_t);
  index_t* device_indices{};
  err = cudaMallocAsync((void**)&device_indices, indices_bytes, stream);
  assert((err == cudaSuccess) && "merge alloc start indices");
  // copy indices
  err = cudaMemcpyAsync(device_indices, start_indices.data(),
    indices_bytes, cudaMemcpyHostToDevice, stream);
  assert((err == cudaSuccess) && "merge copy start indices");
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
  /*
  for (size_t i{}; i < idx_lists.size() - 1; ++i) {
    for (size_t j{i + 1}; j < idx_lists.size(); ++j) {
      num_results += idx_lists.at(i).size() * idx_lists.at(j).size();
      ++num_iterated_list_pairs;
    }
  }
  */
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
  /*
  for (size_t i{}; i < idx_lists.size() - 1; ++i) {
    for (size_t j{i + 1}; j < idx_lists.size(); ++j) {
      start_indices.push_back(index);
      index += idx_lists.at(i).size() * idx_lists.at(j).size();
    }
  }
  */
  return start_indices;
}

auto make_matrix_dims(const std::vector<IndexList>& idx_lists) {
  std::vector<MatrixDim> matrix_dims;
  for_each_list_pair(idx_lists, [&](size_t i, size_t j, size_t n) {
    matrix_dims.push_back(
      {(index_t)idx_lists.at(i).size(), (index_t)idx_lists.at(j).size()});
  });
  return matrix_dims;
}

auto alloc_copy_compat_matrix_dims(
  const std::vector<IndexList>& idx_lists, size_t* num_bytes = nullptr) {
  //
  auto matrix_dims = make_matrix_dims(idx_lists);
  cudaStream_t stream = cudaStreamPerThread;
  cudaError_t err = cudaSuccess;
  // alloc matrix data
  auto matrix_dims_bytes = matrix_dims.size() * sizeof(MatrixDim);
  MatrixDim* device_matrix_dims;
  err = cudaMallocAsync((void**)&device_matrix_dims, matrix_dims_bytes, stream);
  assert((err == cudaSuccess) && "merge alloc compat matrix data");
  // copy matrix data
  err = cudaMemcpyAsync(device_matrix_dims, matrix_dims.data(), matrix_dims_bytes,
    cudaMemcpyHostToDevice, stream);
  assert((err == cudaSuccess) && "merge copy compat matrix data");
  if (num_bytes) {
    *num_bytes = matrix_dims_bytes;
  }
  return device_matrix_dims;
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
  std::vector<std::vector<result_t>> host_compat_matrices;

  using namespace std::chrono;
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

void host_show_num_compat_combos(unsigned first_combo, unsigned max_combos,
  const std::vector<std::vector<result_t>>& compat_matrices) {
  //
  using namespace std::chrono;
  auto t0 = high_resolution_clock::now();

  IndexList offset_indices(10);
  int num_compat{};
  for (auto idx{first_combo}; idx < max_combos; ++idx) {
    auto tmp_idx{idx};
    for (int m{(int)compat_matrices.size() - 1}; m >= 0; --m) {
      auto matrix_size = compat_matrices.at(m).size();
      offset_indices.at(m) = tmp_idx % matrix_size;
      tmp_idx /= matrix_size;
    }
    bool compat = true;
    for (unsigned m{}; m < compat_matrices.size(); ++m) {
      if (!compat_matrices.at(m).at(offset_indices.at(m))) {
        compat = false;
        break;
      }
    }
    if (compat) {
      ++num_compat;
    }
  }

  auto t1 = high_resolution_clock::now();
  auto t_dur = duration_cast<milliseconds>(t1 - t0).count();
  std::cerr << " host_show_num_compat_combos "
            << " for [" << first_combo << ", " << max_combos
            << "]: " << num_compat << " - " << t_dur << "ms" << std::endl;
}

auto make_list_sizes(const std::vector<IndexList>& idx_lists) {
  std::vector<index_t> sizes;
  sizes.reserve(idx_lists.size());
  for (const auto& idx_list : idx_lists) {
    sizes.push_back(idx_list.size());
  }
  return sizes;
}

//
void host_show_num_compat_combos(const uint64_t first_combo,
  const uint64_t num_combos,
  const std::vector<std::vector<result_t>>& compat_matrices,
  const std::vector<MatrixDim>& compat_matrix_dims,
  const IndexList& list_sizes) {
  //
  assert(list_sizes.size() == compat_matrices.size());
  assert(compat_matrix_dims.size() == compat_matrices.size());
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
        assert(((!i && (j == 1)) || n) && "yeah no");
        if (!compat_matrices.at(n).at(offset)) {
          compat = false;
          break;
        }
      }
    }

    /*
    for (unsigned m{}; m < compat_matrices.size(); ++m) {
      if (!compat_matrices.at(m).at(offset_indices.at(m))) {
        compat = false;
        break;
      }
    }
    */
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

/*
auto alloc_copy_flat_indices(const IndexList& flat_indices) {
  cudaStream_t stream = cudaStreamPerThread;
  cudaError_t err = cudaSuccess;
  // alloc indices
  auto indices_bytes = flat_indices.size() * sizeof(index_t);
  index_t* device_indices;
  err = cudaMallocAsync((void**)&device_indices, indices_bytes, stream);
  assert((err == cudaSuccess) && "merge alloc flat indices");
  // copy indices
  err = cudaMemcpyAsync(device_indices, flat_indices.data(), indices_bytes,
    cudaMemcpyHostToDevice, stream);
  assert((err == cudaSuccess) && "merge copy flat indices");
  return device_indices;
}
*/

auto run_merge_task(const SourceCompatibilityData* device_sources,
  const index_t* device_list_start_indices, const index_t* device_flat_indices,
  unsigned row_size, unsigned num_rows, merge_result_t* device_results) {
  //
  using namespace std::chrono;

  cudaStream_t stream = cudaStreamPerThread;
  const auto threads_per_block = 256;
  std::vector<merge_result_t> results(num_rows);

  auto k0 = high_resolution_clock::now();

  run_merge_kernel(stream, threads_per_block, device_sources,
    device_list_start_indices, device_flat_indices, row_size, num_rows,
    device_results);

  cudaStreamSynchronize(stream);
  // copy_results(results, device_results, stream);

  auto k1 = high_resolution_clock::now();
  auto d_kernel = duration_cast<milliseconds>(k1 - k0).count();
  std::cerr << "kernel finished in " << d_kernel << "ms" << std::endl;
  return d_kernel;
}

}  // namespace

namespace cm {
  
int getNumEmptySublists(const std::vector<SourceList>& src_lists) {
  auto count = 0;
  for (const auto& sl : src_lists) {
    if (sl.empty()) count++;
  }
  return count;
}

bool filterAllXorIncompatibleIndices(Peco::IndexListVector& idx_lists,
  const std::vector<SourceList>& src_lists)
{
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
  return true;
}

int list_size(const Peco::IndexList& indexList) {
  // TODO: WTF am I doing here. forward_list doesn't have a size()?
  // iterator arithmetic doesn't work? std::accumulate doesn't work?
  int size = 0;
  std::for_each(indexList.cbegin(), indexList.cend(),
                [&size](int i){ ++size; });
  return size;
}

int64_t vec_product(const std::vector<size_t>& v) {
  int64_t result{1};
  for (auto i : v) {
    result *= i;
  }
  return result;
}

std::string vec_to_string(const std::vector<size_t>& v) {
  std::string result{};
  for (auto i : v) {
    result.append(std::to_string(i));
    result.append(" ");
  }
  return result;
}

auto cuda_mergeCompatibleXorSourceCombinations(
  const std::vector<SourceList>& src_lists) -> XorSourceList {
  using namespace std::chrono;

  if (src_lists.empty()) {
    return {};
  }
  assert(!getNumEmptySublists(src_lists) && "cuda_merge: empty sublist");
  auto idx_lists = get_compatible_indices(src_lists);
  if (idx_lists.empty()) {
    return {};
  }

  std::cerr << " copying data to device...";

  auto cdd0 = high_resolution_clock::now();

  size_t total_bytes_allocated{};
  size_t total_bytes_copied{};
  size_t num_bytes{};
  // alloc/copy source lists and generate start indices
  auto device_src_lists = alloc_copy_src_lists(src_lists, &num_bytes);
  total_bytes_allocated += num_bytes;
  total_bytes_copied += num_bytes;
  auto src_list_start_indices = make_start_indices(src_lists);

  // alloc/copy index lists and generate start indices
  auto device_idx_lists = alloc_copy_idx_lists(idx_lists, &num_bytes);
  total_bytes_allocated += num_bytes;
  total_bytes_copied += num_bytes;
  auto idx_list_start_indices = make_start_indices(idx_lists);

  // alloc compatibility result matrices and generate start indices
  auto device_compat_matrices = alloc_compat_matrices(idx_lists, &num_bytes);
  total_bytes_allocated += num_bytes;
  auto compat_matrix_start_indices =
    make_compat_matrix_start_indices(idx_lists);

  // sync if we want accurate timing
  cudaStreamSynchronize(cudaStreamPerThread);
  auto cdd1 = high_resolution_clock::now();
  auto cdd_dur = duration_cast<milliseconds>(cdd1 - cdd0).count();
  std::cerr << " complete"  << " - " << cdd_dur << "ms"<< std::endl;
  std::cerr << "  allocated: " << total_bytes_allocated
            << ", copied: " << total_bytes_copied << std::endl;

  auto lp0 = high_resolution_clock::now();

  // the only unnecessary capture here is src_lists
  for_each_list_pair(idx_lists, [&](size_t i, size_t j, size_t n) {
    std::cerr << " launching list_pair_compat_kernel " << n << " for [" << i << ", "
              << j << "]" << std::endl;
    run_list_pair_compat_kernel(&device_src_lists[src_list_start_indices.at(i)],
      &device_src_lists[src_list_start_indices.at(j)],
      &device_idx_lists[idx_list_start_indices.at(i)], idx_lists.at(i).size(),
      &device_idx_lists[idx_list_start_indices.at(j)], idx_lists.at(j).size(),
      &device_compat_matrices[compat_matrix_start_indices.at(n)]);
  });

  // temp
  cudaStreamSynchronize(cudaStreamPerThread);
  auto lp1 = high_resolution_clock::now();
  auto lp_dur = duration_cast<milliseconds>(lp1 - lp0).count();
  std::cerr << " list_pair_compat_kernels complete - " << lp_dur << "ms" << std::endl;

  // debugging
  debug_copy_show_compat_matrix_hit_count(device_compat_matrices, num_bytes);

  auto host_compat_matrices = get_compat_matrices(src_lists, idx_lists);

  // alloc/copy start indices
  /*
  auto device_src_list_start_indices =
    alloc_copy_start_indices(src_list_start_indices);
  auto device_idx_list_start_indices =
    alloc_copy_start_indices(idx_list_start_indices);
  */
  auto device_compat_matrix_start_indices =
    alloc_copy_start_indices(compat_matrix_start_indices);
  auto device_compat_matrix_dims = alloc_copy_compat_matrix_dims(idx_lists);

  //  const int row_size = src_lists.size();
  uint64_t first_combo{};
  const uint64_t num_combos{100'000'000u};
  auto device_results = alloc_results(num_combos);

  int n{};
  std::cerr << " launching get_compat_combos_kernel " << n << " for ["
            << first_combo << ", " << num_combos << "]" << std::endl;

  auto gcc0 = high_resolution_clock::now();
  run_get_compat_combos_kernel(first_combo, num_combos, device_compat_matrices,
    device_compat_matrix_start_indices, device_compat_matrix_dims,
    compat_matrix_start_indices.size(), device_results);

  // temp
  cudaStreamSynchronize(cudaStreamPerThread);

  auto gcc1 = high_resolution_clock::now();
  auto gcc_dur = duration_cast<milliseconds>(gcc1 - gcc0).count();

  std::cerr << " completed "
            // << num_iter << " iters "
            // << " of " << num_rows << " rows, total " << total_rows  << " rows
            << " - " << gcc_dur << "ms" << std::endl;

  std::vector<result_t> results(num_combos);
  auto cr0 = high_resolution_clock::now();
  copy_results(results, device_results, cudaStreamPerThread);
  auto hits = std::accumulate(
    results.begin(), results.end(), 0, [](int total, result_t result) {
      if (result) {
        ++total;
      }
      return total;
    });
  auto cr1 = high_resolution_clock::now();
  auto cr_dur = duration_cast<milliseconds>(cr1 - cr0).count();
  std::cerr << " copied results, hits(" << hits << ")"
            << " - " << cr_dur << "ms" << std::endl;

  const auto max_combos = std::accumulate(idx_lists.begin(),
    idx_lists.end(), (uint64_t)1, [](uint64_t total, const IndexList& idx_list) {
      // TODO: multiply_with_overflow_check
      total *= idx_list.size();
      return total;
    });
  auto num_actual_combos = std::min(num_combos * 40, max_combos - first_combo);

  // debugging, for now at least
  host_show_num_compat_combos(first_combo, num_actual_combos,
    host_compat_matrices, make_matrix_dims(idx_lists),
    make_list_sizes(idx_lists));

  XorSourceList xorSourceList{};

  /*
    SourceCRefList sourceCRefList =
      getCompatibleXorSources(*indexList, src_lists);
    if (sourceCRefList.empty()) continue;
    XorSourceList mergedSources = mergeCompatibleXorSources(sourceCRefList);
    if (mergedSources.empty()) continue;
    xorSourceList.emplace_back(std::move(mergedSources.back()));
  }
  auto peco1 = high_resolution_clock::now();
  [[maybe_unused]] auto d_peco =
    duration_cast<milliseconds>(peco1 - peco0).count();
#if 1 || defined(PRECOMPUTE_LOGGING)
  std::cerr << " native peco loop: " << d_peco << "ms"
            << ", combos: " << combos << ", compatible: " << compatible
            << ", get_compat_merge: " << get_compat_merge
            << ", merged: " << merged
            << ", XorSources: " << xorSourceList.size() << std::endl;
#endif
  */

  return xorSourceList;
}

}  // namespace cm
