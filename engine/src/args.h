#pragma once
#include <string>

// CLI args (serve mode only), passed to ModelRuntime / SchedulerConfig in
// RunEngine, where model-arch support is also checked via ModelSpec.
struct Args {
  std::string model_dir;
  int max_ctx = 4096;
  float gpu_mem_fraction = 0.9f;
  unsigned seed = 1234;
  std::string address = "127.0.0.1:50051";
  int slots = 16;
  int max_waiting = 0;
  int token_budget = 1024;
  int prefill_chunk = 512;
  bool use_prefix_cache = true;
};

// argv -> out. Returns <0 to keep running, else an exit code (--help / error).
int ParseArgs(int argc, char** argv, Args& out);
