// Command-line layer: turn argv into a validated Args (argument parsing + model-arch check).
// The actual run lives in main.cpp, which just dispatches on Args::mode.
#pragma once
#include <string>

struct Args {
    enum class Mode { Chat, Benchmark, Serve };
    std::string model_dir;
    int   max_ctx          = 4096;
    float temp             = 0.0f, top_p = 0.95f;
    unsigned seed          = 1234;
    bool  think            = false;
    float gpu_mem_fraction = 0.9f;            // VRAM cap; the KV pool gets whatever is left under it
    std::string address    = "127.0.0.1:50051";  // serve: gRPC listen address
    int   slots            = 16;              // serve: max concurrent sequences
    Mode  mode             = Mode::Chat;
};

// Parse argv into `out`. Returns <0 to continue running; returns >=0 when the program should exit
// now (--help printed, a parse error, or an unsupported model), the value being the exit code.
int parse_args(int argc, char** argv, Args& out);
