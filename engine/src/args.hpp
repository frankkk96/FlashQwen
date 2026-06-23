// CLI args. Model-support check lives in main.cpp; only mode is serve.
#pragma once
#include <string>

struct Args {
  std::string model_dir;
  int max_ctx = 4096;
  float gpu_mem_fraction = 0.9f;  // VRAM cap; KV pool gets what's left under it
  unsigned seed = 1234;           // sampling RNG seed
  std::string address = "127.0.0.1:50051";  // gRPC listen addr
  int slots = 16;               // max concurrent seqs (max_num_seqs)
  int max_queue = 0;            // waiting-request cap (<=0 => 4*slots)
  int max_batch_tokens = 1024;  // tokens/step (max_num_batched_tokens)
  int max_prefill_tokens =
      512;  // prefill chunk cap/req (long_prefill_token_threshold)
};

// argv -> out. Returns <0 to keep running, else exit code (--help or parse
// error).
int ParseArgs(int argc, char** argv, Args& out);
