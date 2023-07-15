#include <algorithm>
#include <exception>
#include <cuda_runtime.h>
//#include <helper_cuda.h>
#include "candidates.h"
#include "combo-maker.h"

namespace cm {

PreComputedData PCD;

auto isSourceORCompatibleWithAnyOrSource(const SourceCompatibilityData& compatData,
  const OrSourceList& orSourceList)
{
  auto compatible = false;
  for (const auto& orSource : orSourceList) {
    // skip any sources that were already determined to be XOR incompatible
    // or AND compatible with --xor sources.
    if (!orSource.xorCompatible || orSource.andCompatible) continue;
    compatible = compatData.isOrCompatibleWith(orSource.source);
    if (compatible) break;
  }
  return compatible;
};

auto isSourceCompatibleWithEveryOrArg(const SourceCompatibilityData& compatData,
  const OrArgDataList& orArgDataList)
{
  auto compatible = true; // if no --or sources specified, compatible == true
  for (const auto& orArgData : orArgDataList) {
    // TODO: skip calls to here if container.compatible = true  which may have
    // been determined in Precompute phase @ markAllANDCompatibleOrSources()
    // and skip the XOR check as well in this case.
    compatible = isSourceORCompatibleWithAnyOrSource(compatData,
      orArgData.orSourceList);
    if (!compatible) break;
  }
  return compatible;
}
 
auto isSourceXORCompatibleWithAnyXorSource(
  const SourceCompatibilityData& compatData, const XorSourceList& xorSourceList,
  const std::vector<int>& indices)
{
  bool compatible = true; // empty list == compatible
#if PERF
  isany_perf.range_calls++;
#endif
  for (auto index : indices) {
    const auto& xorSource = xorSourceList[index];
#if PERF
    isany_perf.comps++;
#endif
    compatible = compatData.isXorCompatibleWith(xorSource);
    if (compatible) break;
  }
#if PERF
  if (compatible) isany_perf.compat++;
#endif
  return compatible;
};

auto isSourceXORCompatibleWithAnyXorSource(
  const SourceCompatibilityData& compatData, const XorSourceList& xorSourceList,
  const std::array<VariationIndicesMap, kNumSentences>& variationIndicesMaps)
{
#if PERF
  isany_perf.calls++;
#endif
  for (auto s = 0; s < kNumSentences; ++s) {
    auto variation = compatData.usedSources.variations[s];
    const auto& map = variationIndicesMaps[s];
    if ((variation < 0) || (map.size() == 1)) continue;
#if PERF
    isany_perf.ss_attempt++;
#endif
    if (auto it = map.find(variation); it != map.end()) {
      if (isSourceXORCompatibleWithAnyXorSource(compatData, xorSourceList,
        it->second))
      {
        return true;
      }
    }
    if (auto it = map.find(-1); it != map.end()) {
      if (isSourceXORCompatibleWithAnyXorSource(compatData, xorSourceList,
        it->second))
      {
        return true;
      }
    }
#if PERF
    isany_perf.ss_fail++;
#endif
    return false;
  }
#if PERF
  isany_perf.full++;
#endif
  return isSourceXORCompatibleWithAnyXorSource(compatData, xorSourceList,
    variationIndicesMaps[1].at(-1)); // hack: we know this is the full index list
}
  
bool isAnySourceCompatibleWithUseSources(
  const SourceCompatibilityList& sourceCompatList)
{
  if (sourceCompatList.empty()) return true;
  auto compatible = false;
  for (const auto& compatData : sourceCompatList) {
    compatible = isSourceXORCompatibleWithAnyXorSource(compatData,
      PCD.xorSourceList, PCD.variationIndicesMaps);
    // if there were --xor sources specified, and none are compatible with the
    // current source, no further compatibility checking is necessary; continue
    // to next source.
    if (!compatible) continue;
    compatible = isSourceCompatibleWithEveryOrArg(compatData, PCD.orArgDataList);
    if (compatible) break;
  }
  return compatible;
};

__global__ void kernel(const SourceCompatibilityList *compatLists, size_t count) {
  auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
}

__global__ void test_kernel(int *data, size_t count) {
  auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  data[index] = 1;
}

void test() {
  cudaError_t err = cudaSuccess;

  int numElements = 1'000'000;
  size_t size = numElements * sizeof(int);
  printf("[%d elements]\n", numElements);

  int *host_data = (int *)malloc(size);
  //  for (int i = 0; i < numElements; ++i) 
  //    host_data[i] = rand() / RAND_MAX;

  // Allocate the device input vector A
  int *device_data = nullptr;
  err = cudaMalloc((void **)&device_data, size);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device data (error code %s)!\n",
            cudaGetErrorString(err));
    throw std::runtime_error("failed to allocate device data");
  }

  int threadsPerBlock = 256;
  int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
  fprintf(stderr, "CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid,
         threadsPerBlock);
  test_kernel<<<blocksPerGrid, threadsPerBlock>>>(device_data, numElements);

  err = cudaMemcpy(host_data, device_data, size, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to copy device data to host (error code %s)!\n",
            cudaGetErrorString(err));
    throw std::runtime_error("failed to allocate device data");
  }

  int ones{};
  for (int i = 0; i < numElements; ++i) {
    if (host_data[i] == 1) ++ones;
  }
  fprintf(stderr, "ones: actual(%d), expected(%d)\n", ones, numElements);

  err = cudaFree(device_data);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device data (error code %s)!\n",
            cudaGetErrorString(err));
    throw std::runtime_error("failed to allocate device data");
  }
  free(host_data);
}

void filterCandidatesCuda(int sum) {
  std::cerr << "filerCandidatesCuda" << std::endl;
  test();
  //kernel<<<40, 32>>>(allSumsCandidateData[sum - 2].
}

} // namespace cm
