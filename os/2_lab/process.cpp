#include "process.hpp"
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <cstring>
#include <errno.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace crossproc {

ProcessResult run_process(const std::string &path,
                          const std::vector<std::string> &args, bool wait) {
  if (path.empty())
    throw ProcessError("Path is empty");

#ifdef _WIN32
#else
  pid_t pid = fork();

  if (pid < 0)
    throw ProcessError(std::string("fork(); failed: ") + strerror(errno));

  if (pid == 0) {
    std::vector<char *> argv;
    argv.push_back(const_cast<char *>(path.c_str()));

    for (const auto &arg : args) {
      argv.push_back(const_cast<char *>(arg.c_str()));
    }

    argv.push_back(nullptr);

    execv(path.c_str(), argv.data());
    _exit(127);
  }

  int exit_code = 0;

  if (wait) {
    int status;
    if (waitpid(pid, &status, 0) < 0) {
      throw ProcessError(std::string("waitpid() failed: ") + strerror(errno));
    }

    if (WIFEXITED(status))
      exit_code = WEXITSTATUS(status);
    else
      exit_code = -1;
  }

  return {exit_code, true};
#endif // _WIN32
}
} // namespace crossproc
