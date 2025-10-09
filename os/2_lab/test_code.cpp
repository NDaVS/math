#include "process.hpp"
#include <iostream>
#include <string>
#include <vector>

using namespace crossproc;

int main() {
  try {
#ifdef _WIN32
    std::vector<std::string> args = {"/c", "echo Hello from Windows"};
    auto result = run_proc("cmd", args, true);
#else
    std::vector<std::string> args = {"hello from Unix"};
    auto result = run_process("/bin/echo", args, true);

#endif // _WIN32
    std::cout << "Process finished successfully. Exit code: "
              << result.exit_code << std::endl;

  } catch (const ProcessError &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
