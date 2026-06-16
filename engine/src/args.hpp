// Command-line layer for the token engine. Pure CLI syntax — no model knowledge; whether the
// model is supported is checked in main.cpp via ModelSpec. The engine has a single mode (serve).
#pragma once
#include <string>

struct Args {
    std::string model_dir;
    int   max_ctx          = 4096;
    float gpu_mem_fraction = 0.9f;                // VRAM cap; the KV pool gets whatever is left under it
    unsigned seed          = 1234;                // RNG seed for sampling
    std::string address    = "127.0.0.1:50051";   // gRPC listen address
    int   slots            = 16;                   // max concurrent sequences
    int   max_queue        = 0;                    // admission cap on waiting requests (<=0 => 4*slots)
};

// Parse argv into `out`. Returns <0 to continue running; returns >=0 when the program should exit
// now (--help printed or a parse error), the value being the exit code.
int parse_args(int argc, char** argv, Args& out);
