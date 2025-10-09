#ifndef PROCESS_HPP
#define PROCESS_HPP

#include <stdexcept>
#include <string>
#include <vector>

namespace crossproc {

class ProcessError : public std::runtime_error {
public:
  explicit ProcessError(const std::string &msg) : std::runtime_error(msg) {}
};

struct ProcessResult {
  int exit_code;
  bool success;
};

ProcessResult run_process(const std::string &path,
                          const std::vector<std::string> &args, bool wait);
} // namespace crossproc
#endif
