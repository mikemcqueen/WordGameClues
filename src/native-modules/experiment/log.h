#ifndef INCLUDE_LOG_H
#define INCLUDE_LOG_H

#pragma once

namespace cm {

enum class LogLevel : int {
  OrVariations = -3,
  MemoryAllocations = -2,
  MemoryDumps = -1,
  //
  Normal = 0,
  Verbose = 1,
  ExtraVerbose = 2,
  Ludicrous = 3
};
using enum LogLevel;

struct LogOptions {
  bool quiet{};
  bool or_variations{};
  bool mem_dumps{};
  bool mem_allocs{};
  LogLevel level{};
};

inline LogOptions the_log_args_;

inline bool log_level(LogLevel level) {
  switch (level) {
  case OrVariations:
    return the_log_args_.or_variations;
  case MemoryAllocations:
    return the_log_args_.mem_allocs;
  case MemoryDumps:
    return the_log_args_.mem_dumps;
  default:
    if (!the_log_args_.quiet) {
      return the_log_args_.level >= level;
    }
    return false;
  }
}

inline void set_log_options(const LogOptions& args) {
  the_log_args_ = args;
}

}  // namespace cm

#endif // INCLUDE_LOG_H
