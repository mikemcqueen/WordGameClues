#ifndef INCLUDE_LOG_H
#define INCLUDE_LOG_H

namespace cm {

// TODO: log.h
struct LogArgs {
  bool quiet{};
  bool verbose{};
};

inline int log_level = 1;

inline void set_log_args(const LogArgs& args) {
  if (args.quiet) {
    log_level = 0;
  } else if (args.verbose) {
    log_level = 2;
  } else {
    log_level = 1;
  }
}

}  // namespace cm

#endif // INCLUDE_LOG_H
