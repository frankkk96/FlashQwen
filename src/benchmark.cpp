#include "benchmark.hpp"
#include "generate.hpp"
#include "sampler.hpp"
#include <cstdio>
#include <algorithm>
#include <vector>
#include <random>

// Fixed sweep configuration.
static const std::vector<int> INPUT_LENS = {16, 128, 512, 1024};  // prompt lengths (tokens)
static const int OUTPUT_LEN = 128;                                // tokens generated per run
static const int WARMUP     = 1;                                  // untimed runs
static const int REPEAT     = 3;                                  // measured runs -> median

static double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    return (n % 2) ? v[n/2] : 0.5 * (v[n/2 - 1] + v[n/2]);
}

struct RunMetrics { double ttft, tpot, decode_tps, output_tps, peak_tps; };

static RunMetrics metrics_of(const GenStats& st) {
    double decode_ms = 0, min_step = 1e30;
    for (double m : st.step_ms) { decode_ms += m; min_step = std::min(min_step, m); }
    int n = (int)st.step_ms.size();
    RunMetrics r{};
    r.ttft       = st.ttft_ms;
    r.tpot       = n > 0 ? decode_ms / n : 0.0;
    r.decode_tps = n > 0 ? n / (decode_ms / 1000.0) : 0.0;
    double total_ms = st.ttft_ms + decode_ms;
    r.output_tps = st.n_out / (total_ms / 1000.0);
    r.peak_tps   = (min_step < 1e30 && min_step > 0) ? 1000.0 / min_step : 0.0;
    return r;
}

// A synthetic prompt of `n` arbitrary (but valid) token ids. Content is irrelevant to
// timing; only the length matters.
static std::vector<int> make_prompt(int n, int vocab, std::mt19937& rng) {
    std::uniform_int_distribution<int> d(100, vocab - 1000);
    std::vector<int> ids(n);
    for (int& x : ids) x = d(rng);
    return ids;
}

int run_benchmark(Model& model, const Tokenizer& tok, int max_ctx) {
    SampleParams sp{0.0f, 1.0f, 0};   // greedy: timing shouldn't depend on sampling
    std::mt19937 rng(1234);
    int vocab = model.config().vocab_size;

    std::fprintf(stderr, "[benchmark: output %d tok, warmup %d, repeat %d (median), batch 1]\n\n",
                 OUTPUT_LEN, WARMUP, REPEAT);
    std::fprintf(stderr, "  input    TTFT      TPOT     decode     output      peak\n");
    std::fprintf(stderr, "  (tok)    (ms)    (ms/tok)  (tok/s)    (tok/s)    (tok/s)\n");

    for (int isl : INPUT_LENS) {
        if (isl + OUTPUT_LEN >= max_ctx) {
            std::fprintf(stderr, "  %5d   (skipped: input+output exceeds max-ctx %d)\n", isl, max_ctx);
            continue;
        }
        std::vector<int> ids = make_prompt(isl, vocab, rng);

        auto one_run = [&]() {
            int past = 0;
            return generate(model, tok, ids, past, OUTPUT_LEN, sp, rng,
                            /*stream=*/false, /*stop_on_eos=*/false);
        };
        for (int w = 0; w < WARMUP; ++w) one_run();

        std::vector<double> ttft, tpot, dtps, otps, ptps;
        for (int r = 0; r < REPEAT; ++r) {
            RunMetrics m = metrics_of(one_run());
            ttft.push_back(m.ttft); tpot.push_back(m.tpot);
            dtps.push_back(m.decode_tps); otps.push_back(m.output_tps); ptps.push_back(m.peak_tps);
        }
        std::fprintf(stderr, "  %5d  %8.1f  %7.2f  %8.1f  %9.1f  %9.1f\n",
                     isl, median(ttft), median(tpot), median(dtps), median(otps), median(ptps));
    }
    return 0;
}
