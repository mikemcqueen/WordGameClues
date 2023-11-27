#ifndef INCLUDE_LOG_H
#define INCLUDE_LOG_H

namespace cm {

// TODO: log.h
struct LogArgs {
  bool quiet{};
  int verbose{};
};

inline const auto Normal = 1;
inline const auto Verbose = 2;
inline const auto ExtraVerbose = 3;
inline const auto Ludicrous = 4;

inline int the_log_level = Normal;

inline bool log_level(int level) {
  return the_log_level >= level;
}

inline void set_log_args(const LogArgs& args) {
  if (args.quiet) {
    the_log_level = 0;
  } else if (!args.verbose) {
    the_log_level = Normal;
  } else {
    the_log_level = args.verbose;
  }
}

}  // namespace cm

#endif // INCLUDE_LOG_H
