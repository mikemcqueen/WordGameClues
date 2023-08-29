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
  const Peco::IndexList& indexList, const SourceList& sourceList)
{
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
  assert((err == cudaSuccess) && "zero results");
}

auto alloc_results(size_t num_results) {  // TODO cudaStream_t stream) {
  cudaStream_t stream = cudaStreamPerThread;
  cudaError_t err = cudaSuccess;
  // alloc results
  auto results_bytes = num_results * sizeof(merge_result_t);
  merge_result_t* device_results;
  err = cudaMallocAsync((void**)&device_results, results_bytes, stream);
  assert((err == cudaSuccess) && "alloc results");
  return device_results;
}

void copy_results(std::vector<merge_result_t>& results,
  merge_result_t* device_results, cudaStream_t stream) {
  //
  cudaError_t err = cudaStreamSynchronize(stream);
  assert((err == cudaSuccess) && "sychronize");

  auto results_bytes = results.size() * sizeof(merge_result_t);
  err = cudaMemcpyAsync(results.data(), device_results, results_bytes,
    cudaMemcpyDeviceToHost, stream);
  if (err != cudaSuccess) {
    fprintf(stdout, "merge copy results, error: %s", cudaGetErrorString(err));
    assert((err != cudaSuccess) && "merge copy device results");
  }
  err = cudaStreamSynchronize(stream);
  assert((err == cudaSuccess) && "cudaStreamSynchronize");
}

auto count(const std::vector<SourceList>& src_lists) {
  size_t sum{};
  for (const auto& src_list : src_lists) {
    sum += src_list.size();
  }
  return sum;
}

auto alloc_copy_src_lists(const std::vector<SourceList>& src_lists) {
  // alloc sources
  const cudaStream_t stream = cudaStreamPerThread;
  cudaError_t err = cudaSuccess;
  auto sources_bytes = count(src_lists) * sizeof(SourceCompatibilityData);
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
  return device_sources;
}

auto make_start_indices(const std::vector<SourceList>& src_lists) {
  std::vector<index_t> start_indices{};
  index_t index{};
  for (const auto& src_list : src_lists) {
    start_indices.push_back(index);
    index += src_list.size();
  }
  return start_indices;
}

auto alloc_copy_list_start_indices(const std::vector<index_t>& start_indices) {
  cudaStream_t stream = cudaStreamPerThread;
  cudaError_t err = cudaSuccess;
  // alloc indices
  auto indices_bytes = start_indices.size() * sizeof(index_t);
  index_t* device_indices{};
  err = cudaMallocAsync((void**)&device_indices, indices_bytes, stream);
  assert((err == cudaSuccess) && "merge alloc src_list start indices");
  // copy indices
  err = cudaMemcpyAsync(device_indices, start_indices.data(),
    indices_bytes, cudaMemcpyHostToDevice, stream);
  assert((err == cudaSuccess) && "merge copy src_list start indices");
  return device_indices;
}

auto alloc_copy_flat_indices(const std::vector<index_t>& flat_indices) {
  cudaStream_t stream = cudaStreamPerThread;
  cudaError_t err = cudaSuccess;
  // alloc indices
  auto indices_bytes = flat_indices.size() * sizeof(index_t);
  index_t* device_indices;
  err = cudaMallocAsync((void**)&device_indices, indices_bytes, stream);
  assert((err == cudaSuccess) && "merge alloc flat indices");
  // copy indices
  err = cudaMemcpyAsync(device_indices, flat_indices.data(),
    indices_bytes, cudaMemcpyHostToDevice, stream);
  assert((err == cudaSuccess) && "merge copy flat indices");
  return device_indices;
}

auto run_merge_task(
  const SourceCompatibilityData* device_sources,
  const index_t* device_list_start_indices, const index_t* device_flat_indices,
  unsigned row_size, unsigned num_rows, merge_result_t* device_results) {
  //
  using namespace std::chrono;

  cudaStream_t stream = cudaStreamPerThread;
  const auto threads_per_block = 256;
  std::vector<merge_result_t> results(num_rows);

  auto k0 = high_resolution_clock::now();

  cm::run_merge_kernel(stream, threads_per_block, device_sources,
    device_list_start_indices, device_flat_indices, row_size, num_rows,
    device_results);

  cudaStreamSynchronize(stream);
  //copy_results(results, device_results, stream);

  auto k1 = high_resolution_clock::now();
  auto d_kernel = duration_cast<milliseconds>(k1 - k0).count();
  std::cerr << "kernel finished in " << d_kernel << "ms" << std::endl;
  return d_kernel;
}

}  // namespace

namespace cm {
  
int getNumEmptySublists(const std::vector<SourceList>& sourceLists) {
  auto count = 0;
  for (const auto& sl : sourceLists) {
    if (sl.empty()) count++;
  }
  return count;
}

bool filterAllXorIncompatibleIndices(Peco::IndexListVector& indexLists,
  const std::vector<SourceList>& sourceLists)
{
  if (indexLists.size() < 2u) return true;
  for (size_t first{}; first < indexLists.size(); ++first) {
    for (size_t second{}; second < indexLists.size(); ++second) {
      if (first == second) continue;
      if (!filterXorIncompatibleIndices(
            indexLists, first, second, sourceLists)) {
        return false;
      }
    }
  }
  return true;
}

int list_size(const Peco::IndexList& indexList) {
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
  const std::vector<SourceList>& sourceLists) -> XorSourceList {
  using namespace std::chrono;

  if (sourceLists.empty()) {
    return {};
  }
  assert(!getNumEmptySublists(sourceLists) && "cuda_merge: empty sublist");
  std::vector<size_t> lengths{};
  lengths.reserve(sourceLists.size());
  for (const auto& sl : sourceLists) {
    lengths.push_back(sl.size());
  }
  std::cerr << "  initial lengths: " << vec_to_string(lengths)
            << ", product: " << vec_product(lengths) << std::endl;
  auto indexLists = Peco::initial_indices(lengths);
  bool valid = filterAllXorIncompatibleIndices(indexLists, sourceLists);
  lengths.clear();
  for (const auto& il : indexLists) {
    lengths.push_back(list_size(il));
  }
  std::cerr << "  filtered lengths: " << vec_to_string(lengths)
            << ", product: " << vec_product(lengths)
            << ", valid: " << std::boolalpha << valid << std::endl;
  if (!valid) return {};

  auto device_src_lists = alloc_copy_src_lists(sourceLists);
  auto start_indices = make_start_indices(sourceLists);
  auto device_list_start_indices = alloc_copy_list_start_indices(start_indices);
  const int row_size = sourceLists.size();
  const int num_rows = 100'000'000;
  auto device_results = alloc_results(num_rows);

  Peco peco(std::move(indexLists));

  auto t0 = high_resolution_clock::now();

  int num_iter{};
  int64_t total_rows{};
  for (auto first{true};;first = false, ++num_iter) {
    // TODO: member function of Peco
    auto flat_indices = get_flat_indices(peco, num_rows, first);
    if (flat_indices.empty()) {
      break;
    }
    assert(!(flat_indices.size() % row_size));
    total_rows += flat_indices.size() / row_size;

    // temp to avoid segfault
    if (flat_indices.size() / row_size < num_rows) break;

    std::cerr << "iter " << num_iter << ", " << flat_indices.size() / row_size
              << " rows, total: " << total_rows << std::endl;
    auto device_flat_indices = alloc_copy_flat_indices(flat_indices);

    run_merge_task(device_src_lists, device_list_start_indices,
      device_flat_indices, row_size, flat_indices.size() / row_size,
      device_results);

    cudaError_t err = cudaFree(device_flat_indices);
    assert(err == cudaSuccess);
  }

  auto t1 = high_resolution_clock::now();
  auto d_task = duration_cast<milliseconds>(t1 - t0).count();

  std::cerr << "completed " << num_iter << " iters "
            << " of " << num_rows << " rows, total " << total_rows
            << " rows - " << d_task << "ms" << std::endl;

  // auto peco0 = high_resolution_clock::now();

  /*
  int64_t combos = 0;
  int compatible = 0;
  int merged = 0;
  */
  XorSourceList xorSourceList{};

  /*
    SourceCRefList sourceCRefList =
      getCompatibleXorSources(*indexList, sourceLists);
    if (sourceCRefList.empty()) continue;
    XorSourceList mergedSources = mergeCompatibleXorSources(sourceCRefList);
    if (mergedSources.empty()) continue;
    xorSourceList.emplace_back(std::move(mergedSources.back()));
  }
  */
  /*
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
