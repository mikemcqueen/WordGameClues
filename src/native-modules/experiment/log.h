#ifndef INCLUDE_LOG_H
#define INCLUDE_LOG_H

namespace cm {

enum class LogLevel : int {
  MemoryAllocations = -2,
  MemoryDumps = -1,
  //
  Normal = 0,
  Verbose = 1,
  ExtraVerbose = 2,
  Ludicrous = 3
};
using enum LogLevel;


struct LogArgs {
  bool quiet{};
  bool mem_dumps{};
  bool mem_allocs{};
  LogLevel level{};
};

inline LogArgs  the_log_args_;

/*
inline const auto Normal = 1;
inline const auto Verbose = 2;
inline const auto ExtraVerbose = 3;
inline const auto Ludicrous = 4;
*/

inline bool log_level(LogLevel level) {
  switch (level) {
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

inline void set_log_args(const LogArgs& args) {
  the_log_args_ = args;
  /*
  if (args.quiet) {
    the_log_level = 0;
  } else if (!args.verbose) {
    the_log_level = Normal;
  } else {
    the_log_level = args.verbose;
  }
  */
}

}  // namespace cm

#endif // INCLUDE_LOG_H
