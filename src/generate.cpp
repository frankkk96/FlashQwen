#include "generate.hpp"
#include <chrono>
#include <cstdio>
#include <string>

using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

static const int EOS1 = 151645, EOS2 = 151643;   // <|im_end|>, <|endoftext|>

GenStats generate(Model& model, const Tokenizer& tok, const std::vector<int>& chunk,
                  int& past, int max_gen, const SampleParams& sp, std::mt19937& rng,
                  bool stream, bool stop_on_eos) {
    GenStats st;
    Tokenizer::Stream detok;

    // prefill (TTFT)
    auto t0 = Clock::now();
    const std::vector<float>& l0 = model.forward(chunk, past);
    past += (int)chunk.size();
    int next = sample(l0, sp, rng);
    st.ttft_ms = ms_since(t0);

    // decode: emit the current token, then feed it back to get the next one
    while (st.n_out < max_gen) {
        if (stop_on_eos && (next == EOS1 || next == EOS2)) break;
        if (past + 1 > model.max_ctx()) break;

        if (stream) {
            std::string piece = tok.stream_decode(detok, next);
            if (!piece.empty()) { std::fputs(piece.c_str(), stdout); std::fflush(stdout); }
        }
        st.n_out++;

        auto ts = Clock::now();
        const std::vector<float>& lg = model.forward({next}, past);
        past += 1;
        next = sample(lg, sp, rng);
        st.step_ms.push_back(ms_since(ts));
    }
    return st;
}
