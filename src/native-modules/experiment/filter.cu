//#define HOST_DEVICE_ATTRIBUTES __host__ __device__

#include "candidates.h"
#include <algorithm>
#include <exception>
#include <numeric>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>

//#include <thrust/device_vector.h>
//#include <helper_cuda.h>
//#include "combo-maker.h"

namespace cm {

__device__ auto isSourceORCompatibleWithAnyOrSource(
  const SourceCompatibilityData& compatData, const OrSourceList& orSourceList)
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

__device__ auto isSourceCompatibleWithEveryOrArg(
  const SourceCompatibilityData& compatData, const OrArgDataList& orArgDataList)
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
 
// "temporarily" re-added this original version without indices, as part of 
// prototyping cuda integration
__host__ __device__ auto isSourceXORCompatibleWithAnyXorSource(
  const SourceCompatibilityData& compatData,
  const XorSource* xorSources, size_t numXorSources,
  int* outIndex = nullptr)
{
  bool compatible = true; // empty list == compatible
  //  for (const auto& xorSource: xorSourceList) {
  for (auto i = 0u; i < numXorSources; ++i) {
    compatible = compatData.isXorCompatibleWith(xorSources[i], false);
    if (compatible) {
      if (outIndex) *outIndex = i;
      break;
    }
  }
  return compatible;
};

//__device__
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
    compatible = compatData.isXorCompatibleWith(xorSource, false);
    if (compatible) break;
  }
#if PERF
  if (compatible) isany_perf.compat++;
#endif
  return compatible;
};

//__device__
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
    variationIndicesMaps[1].at(-1)); // hack: we know "sentence" 2 doesn't exist
}
  
__host__ __device__ bool isAnySourceCompatibleWithUseSources(
  const SourceCompatibilityData *compatData, int numCompatData,
  const XorSource* xorSources, size_t numXorSources)
{
  if (!numCompatData) return true;
  auto compatible = false;
  for (auto i = 0; i < numCompatData; ++i) {
    compatible = isSourceXORCompatibleWithAnyXorSource(compatData[i],
      xorSources, numXorSources); // , PCD.variationIndicesMaps);
    // if there were --xor sources specified, and none are compatible with the
    // current source, no further compatibility checking is necessary; continue
    // to next source.
    if (!compatible) continue;
    // TODO
    //compatible = isSourceCompatibleWithEveryOrArg(compatData[i],
    //  PCD.orArgDataList);
    if (compatible) break;
  }
  return compatible;
};

using result_t = uint8_t;
#define DEBUG_MAX 5

__host__ __device__
void strcat_char(char* buf, char c) {
  while (*buf) buf++;
  *(buf++) = c;
  *buf = 0;
}

__host__ __device__
void strcat_int(char* buf, int i) {
  while (*buf) buf++;
  int factor = 100;
  while (factor) {
    int val = i / factor;
    if (val > 0) {
      *(buf++) = val + '0';
      i %= factor;
    }
    factor /= 10;
  }
  *buf = 0;
}

__host__ __device__
char* buildLegacySourcesString(const SourceCompatibilityData& scd, char* buf){
  *buf = 0;
  for (int i{}; i < kMaxLegacySources; ++i) {
    if (scd.legacySources[i]) {
      strcat_char(buf, ' ');
      strcat_int(buf, i);
    }
  }
  return buf;
}

__host__ __device__
char* buildSourcesString(const SourceCompatibilityData& scd, char* buf){
  *buf = 0;
  for (int s{1}; s <= kNumSentences; ++s) {
    auto first = Source::getFirstIndex(s);
    for (int i{}; i < kMaxUsedSourcesPerSentence; ++i) {
      if (scd.usedSources.sources[first + i] == -1) break;
      strcat_char(buf, ' ');
      strcat_int(buf, s);
      strcat_char(buf, ':');
      strcat_int(buf, scd.usedSources.sources[first + i]);
    }
  }
  return buf;
}

__host__ __device__ void printSources(const SourceCompatibilityData& scd) {
  char buf[256];
  buildLegacySourcesString(scd, buf);
  printf("  legacySources %s\n", buf);
  buildSourcesString(scd, buf);
  printf("  sources %s\n", buf);
}

__global__ void kernel(const SourceCompatibilityData* compatData,
  size_t num_compatData, const XorSource* xorSources, size_t num_xorSources,
  result_t* results)
{
  auto index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= num_compatData) return;
  if (index == 1) { //< DEBUG_MAX) {
    const auto& cd = compatData[index];
    const auto& v = cd.usedSources.variations;
    const auto& b = cd.usedSources.bits;
    printf("idx: %d, variations %2d %2d, legacyBits %ld, bits %ld\n",
           index, v[0], v[2], cd.sourceBits.to_ulong(), b.to_ulong());
    printSources(cd);
    printf("----\n");
    printSources(xorSources[1392]);
  }
  bool compat = isSourceXORCompatibleWithAnyXorSource(compatData[index],
    xorSources, num_xorSources);
  if (compat) printf("COMPAT: %d\n", index);
  results[index] = compat ? 1 : 0;
}

auto count(const SourceCompatibilityLists& compatLists) {
  size_t num{};
  for (const auto& compatList: compatLists) {
    for (auto& compatData: compatList) {
      num++;
    }
  }
  return num;
}
 
void filterCandidatesCuda(int sum) {
  std::cerr << "++filterCandidatesCuda" << std::endl;

  cudaError_t err = cudaSuccess;
  const auto& compatLists = allSumsCandidateData[sum - 2].sourceCompatLists;
  auto num_compatData = count(compatLists);
  auto compatData_bytes = num_compatData * sizeof(SourceCompatibilityData);
  SourceCompatibilityData *device_compatList = nullptr;
  err = cudaMalloc((void **)&device_compatList, compatData_bytes);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device compatList, error: %s\n",
            cudaGetErrorString(err));
    throw std::runtime_error("failed to allocate device compatList");
  }
  int index{};
  int compatFound = false;
  for (const auto& compatList: compatLists) {
    err = cudaMemcpy(&device_compatList[index], compatList.data(),
      compatList.size() * sizeof(SourceCompatibilityData),
      cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      fprintf(stderr, "Failed to copy compatList host -> device, error: %s\n",
        cudaGetErrorString(err));
      throw std::runtime_error("failed to copy compatLists host -> device");
    }


    if (/*(DEBUG_MAX && (index < DEBUG_MAX)) ||*/ !compatFound) {
      int source{ index };
      for (const auto& compatData: compatList) {
        int xorSource = -1;
        if (isSourceXORCompatibleWithAnyXorSource(compatData,
          PCD.xorSourceList.data(), PCD.xorSourceList.size(), &xorSource)
            ) // || (DEBUG_MAX && (source < DEBUG_MAX)))
        {
          const auto& v = compatData.usedSources.variations;
          const auto& b = compatData.usedSources.bits;
          printf("compat source: %d, variations %2d %2d"
            ", legacyBits %ld, bits %ld"
            ", xorSourceIndex = %d\n", source, v[0], v[2],
                 compatData.sourceBits.to_ulong(), b.to_ulong(), xorSource);
          printSources(compatData);

          const auto& x = PCD.xorSourceList[xorSource];
          const auto& xv = compatData.usedSources.variations;
          const auto& xb = compatData.usedSources.bits;
          printf("  xorSource: variations %2d %2d"
            ", legacyBits %ld, bits %ld\n", xv[0], xv[2],
            x.sourceBits.count(), xb.count());
          printSources(x);
          compatFound = true;
          break;
          xorSource = -1;
        }
        source++;
      }
    }


    index += compatList.size();
  }
  std::cerr << "  done copying " << index << " sources" << std::endl;

  // TODO: we only need to do this once! across all invocations.
  auto num_xorSources = PCD.xorSourceList.size();
  auto xorSources_bytes = num_xorSources * sizeof(XorSource);
  XorSource *device_xorSources = nullptr;
  err = cudaMalloc((void **)&device_xorSources, xorSources_bytes);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device xorSources, errror: %s\n",
      cudaGetErrorString(err));
    throw std::runtime_error("failed to allocate device xorSources");
  }
  err = cudaMemcpy(device_xorSources, PCD.xorSourceList.data(),
    xorSources_bytes, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to copy xorSources host -> device, error: %s\n",
      cudaGetErrorString(err));
    throw std::runtime_error("failed to copy xorSources host -> device");
  }

  //
  auto results_bytes = num_compatData * sizeof(result_t);
  result_t *device_results = nullptr;
  err = cudaMalloc((void **)&device_results, results_bytes);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device result (error code %s)!\n",
            cudaGetErrorString(err));
    throw std::runtime_error("failed to allocate device result");
  }

  std::cerr << "  alloc/copy compatLists(" << num_compatData << ")"
            << ", xorSources(" << num_xorSources << ") " 
            << " done" << std::endl;

  int threadsPerBlock = 32;
  int blocksPerGrid = (num_compatData + threadsPerBlock - 1) / threadsPerBlock;
  fprintf(stderr, "  kernel launch with %d blocks of %d threads...\n",
    blocksPerGrid, threadsPerBlock);

  threadsPerBlock = DEBUG_MAX;
  blocksPerGrid = 1; 
  #if 0

  index = 0;
  int show = threadsPerBlock * blocksPerGrid;
  for (const auto& compatList: compatLists) {
    for (const auto& compatData: compatList) {
      const auto& v = compatData.usedSources.variations;
      printf("idx: %d, variations %2d %2d %2d %2d %2d\n", index++,
             v[0], v[1], v[2], v[3], v[4]);
      if (!--show) break;
    }
    if (!show) break;
  }
  printf("------------------------------------------\n");
  #endif

  kernel<<<blocksPerGrid, threadsPerBlock>>>(device_compatList, num_compatData,
    device_xorSources, num_xorSources, device_results);
  std::cerr << "  kernel done" << std::endl;

  std::vector<result_t> results;
  results.resize(num_compatData);
  err = cudaMemcpy(results.data(), device_results, results_bytes,
    cudaMemcpyDeviceToHost);
  int numCompatible = std::accumulate(results.cbegin(), results.cend(), 0,
    [](int numCompatible, result_t result) {
      if (result) numCompatible++;
      return numCompatible;
    });
  /*  for (auto r: results) {
    if (r == 1)
    }*/
  std::cerr << "  compatible: " << numCompatible << std::endl;

  err = cudaFree(device_results);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device results (error code %s)!\n",
      cudaGetErrorString(err));
    throw std::runtime_error("failed to free device results");
  }

  err = cudaFree(device_xorSources);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device xorSources (error code %s)!\n",
      cudaGetErrorString(err));
    throw std::runtime_error("failed to free device xorSources");
  }

  err = cudaFree(device_compatList);
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to free device compatList (error code %s)!\n",
      cudaGetErrorString(err));
    throw std::runtime_error("failed to free device compatList");
  }
  std::cerr << "--filterCandidatesCuda" << std::endl;
}

} // namespace cm
