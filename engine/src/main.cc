#include <string>

#include "args.h"
#include "grpc_service.h"

using namespace fq;

int main(int argc, char** argv) {
  Args a;
  if (int rc = ParseArgs(argc, argv, a); rc >= 0)
    return rc;

  std::string id = a.model_dir;
  if (auto p = id.find_last_of('/'); p != std::string::npos)
    id = id.substr(p + 1);
  if (id.empty()) id = "flashqwen";

  return RunEngine(a, id);
}
