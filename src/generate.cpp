#include "generate.hpp"
#include <chrono>
#include <cstdio>
#include <string>

using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

static const int EOS1 = 151645, EOS2 = 151643;   // <|im_end|>, <|endoftext|>

// Single-sequence streaming generation (interactive chat). Runs as a batch of one through the
// same prefill + batched-decode path the benchmark uses for many sequences; slot is always 0.
GenStats generate(Model& model, const Tokenizer& tok, const std::vector<int>& chunk,
                  int& past, int max_gen, const SampleParams& sp, std::mt19937& rng,
                  bool stream, bool stop_on_eos) {
    GenStats st;
    Tokenizer::Stream detok;
    std::vector<int> out;

    // prefill (TTFT): extend slot 0's sequence by `chunk` at position `past`.
    auto t0 = Clock::now();
    model.prefill(chunk, /*slot=*/0, past);
    past += (int)chunk.size();
    int next = sp.temp <= 0.0f ? model.argmax_last() : sample(model.copy_logits(), sp, rng);
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
        model.decode({next}, {past}, {0}, out);   // greedy result in out[0]; logits row 0 too
        past += 1;
        next = sp.temp <= 0.0f ? out[0] : sample(model.copy_logits(), sp, rng);
        st.step_ms.push_back(ms_since(ts));
    }
    return st;
}
