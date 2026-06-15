#include "generate.hpp"
#include "special_tokens.hpp"
#include <chrono>
#include <cstdio>
#include <string>

using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

// Single-sequence streaming generation (interactive chat). Runs as a batch of one through the
// same paged prefill + batched-decode path the scheduler uses. Chat is the only sequence, so it
// gets an identity block table [0,1,2,...] over the whole pool (a contiguous KV mapping).
GenStats generate(Model& model, const Tokenizer& tok, const std::vector<int>& chunk,
                  int& past, int max_gen, const SampleParams& sp, std::mt19937& rng,
                  bool stream, bool stop_on_eos) {
    GenStats st;
    Tokenizer::Stream detok;
    std::vector<int> out;
    std::vector<int> bt(model.max_blocks_per_seq());
    for (int i = 0; i < (int)bt.size(); ++i) bt[i] = i;

    // prefill (TTFT): extend the sequence by `chunk` at position `past`.
    auto t0 = Clock::now();
    model.prefill(chunk, bt, past);
    past += (int)chunk.size();
    int next = sp.temp <= 0.0f ? model.argmax_last() : sample(model.copy_logits(), sp, rng);
    st.ttft_ms = ms_since(t0);

    // decode: emit the current token, then feed it back to get the next one
    while (st.n_out < max_gen) {
        if (stop_on_eos && special::is_eos(next)) break;
        if (past + 1 > model.max_ctx()) break;

        if (stream) {
            std::string piece = tok.stream_decode(detok, next);
            if (!piece.empty()) { std::fputs(piece.c_str(), stdout); std::fflush(stdout); }
        }
        st.n_out++;

        auto ts = Clock::now();
        model.decode({next}, {past}, {bt}, out);   // greedy result in out[0]; logits row 0 too
        past += 1;
        next = sp.temp <= 0.0f ? out[0] : sample(model.copy_logits(), sp, rng);
        st.step_ms.push_back(ms_since(ts));
    }
    return st;
}
