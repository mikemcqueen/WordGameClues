// cuda-common.cpp

#include <format>
#include <iostream>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include "cuda-types.h"
#include "log.h"

namespace cm {

namespace {

using PtrMap = std::unordered_map<void*, std::pair<std::string, size_t>>;
using TagMap = std::unordered_map<std::string, size_t>;

PtrMap ptr_map_;
TagMap tag_map_;
std::mutex pm_mutex_;
std::mutex tm_mutex_;

void add_error(std::string_view msg, void* ptr, std::string_view tag_sv,
    size_t size) {
  std::scoped_lock lk(pm_mutex_, tm_mutex_);
  auto it_pm = ptr_map_.find(ptr);
  // TODO: could make key a string_view, and avoid copy
  const auto& existing_tag = it_pm->second.first;
  auto existing_size = it_pm->second.second;
  auto it_tm = tag_map_.find(std::string{tag_sv});
  auto tag_size = it_tm == tag_map_.end() ? size_t(0) : it_tm->second;
  std::cerr << "ERROR: " << msg << ": " << ptr << ", attempted size: " << size
            << std::endl
            << " attempted tag: " << tag_sv
            << ", size: " << tag_size << std::endl
            << " existing tag: " << existing_tag
            << ", size: " << existing_size << std::endl;
  std::terminate();
}

void add_ptr(void* ptr, std::string_view tag_sv, size_t size) {
  if (ptr_map_.contains(ptr)) {
    add_error("add_ptr: ptr_map already contains ptr", ptr, tag_sv, size);
  }
  bool inserted = false;
  std::string tag{tag_sv};
  {
    std::scoped_lock lk(pm_mutex_);
    auto it_bool = ptr_map_.insert(
        std::make_pair(ptr, std::make_pair(tag, size)));  //
    inserted = it_bool.second;
  }
  assert(inserted && "add_ptr: ptr_map insertion failed");
  auto it_tm = tag_map_.find(tag);
  if (it_tm == tag_map_.end()) {
    inserted = false;
  }
  {
    std::scoped_lock lk(tm_mutex_);
    if (!inserted) {
      inserted = tag_map_.insert(std::make_pair(tag, size)).second;
    } else {
      it_tm->second += size;
    }
  }
  assert(inserted && "add_ptr: tag_map insertion failed");
  if (1 || log_level(Verbose)) {
    std::cerr << "allocated " << size << " " << tag_sv << " at " << ptr
              << std::endl;
  }
}

void remove_ptr(void* ptr) {
  auto it_pm = ptr_map_.find(ptr);
  assert(it_pm != ptr_map_.end() && "remove_ptr: ptr not found in ptr_map");
  // TODO: could make key a string_view, and avoid copy
  auto tag = it_pm->second.first;
  auto size = it_pm->second.second;
  {
    std::scoped_lock lk(pm_mutex_);
    ptr_map_.erase(it_pm);
  }
  auto it_tm = tag_map_.find(tag);
  assert(it_tm != tag_map_.end()
         && "remove_ptr: tag not found in tag_map");
  auto& tag_size = it_tm->second;
  assert(size <= tag_size
         && "remove_ptr: size to remove is more than tag size");
  {
    std::scoped_lock lk(tm_mutex_);
    tag_size -= size;
  }
  if (1 || log_level(Verbose)) {
    std::cerr << "freed " << size << " " << tag << " at " << ptr
              << ", remaining: " << tag_size << std::endl;
  }
}

}  // anonymous namespace

void cuda_malloc_async(void** ptr, size_t num_bytes,
    cudaStream_t stream = cudaStreamPerThread,
    std::string_view tag = "unspecified") {
  if (!num_bytes) {
    *ptr = 0;
    return;
  }
  auto err = cudaMallocAsync(ptr, num_bytes, stream);
  assert_cuda_success(err, "cuda_malloc_async");
  add_ptr(*ptr, tag, num_bytes);
}

void cuda_free(void* ptr) {
  if (!ptr)
    return;
  auto err = cudaFree(ptr);
  assert_cuda_success(err, "cuda_free");
  remove_ptr(ptr);
}

void cuda_memory_dump() {
  // TODO: holding a lock while doing io is dumb.
  std::scoped_lock lk(tm_mutex_);
  std::cerr << "cuda_memory_dump:" << std::endl;
  for (const auto& it : tag_map_) {
    std::cerr << " " << it.first << ": " << it.second << std::endl;
  }
}

size_t cuda_get_free_mem() {
  size_t free;
  size_t total;
  auto err = cudaMemGetInfo(&free, &total);
  assert_cuda_success(err, "cudaMemGetInfo");
  return free;
}

}  // namespace cm
