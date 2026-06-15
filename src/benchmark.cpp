#include "benchmark.hpp"
#include "generate.hpp"
#include "scheduler.hpp"
#include "sampler.hpp"
#include <cstdio>
#include <algorithm>
#include <vector>
#include <random>
#include <chrono>

// Fixed sweep configuration.
static const std::vector<int> INPUT_LENS = {16, 128, 512, 1024};  // prompt lengths (tokens)
static const std::vector<int> BATCH_INPUT_LENS = {128, 512, 1024};// 2D sweep prompt lengths
static const std::vector<int> BATCH_SIZES = {1, 8, 16};           // 2D sweep batch sizes
static const int OUTPUT_LEN = 128;                                // tokens generated per run
static const int WARMUP     = 1;                                  // untimed runs
static const int REPEAT     = 3;                                  // measured runs -> median

using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

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

// One static-batch run: prefill B sequences (each `input_len` tokens) into slots 0..B-1, then
// decode all B in lock-step for `steps` tokens. Returns total decode wall time (ms).
static double batch_decode_ms(Model& model, int input_len, int B, int steps,
                              int vocab, std::mt19937& rng) {
    std::vector<std::vector<int>> prompts(B);
    std::vector<int> cur(B), past(B), slots(B), out;
    for (int b = 0; b < B; ++b) {
        prompts[b] = make_prompt(input_len, vocab, rng);
        model.prefill(prompts[b], /*slot=*/b, /*past_len=*/0);
        cur[b]  = model.argmax_last();   // first generated token for sequence b
        past[b] = input_len;
        slots[b] = b;
    }
    auto t0 = Clock::now();
    for (int s = 0; s < steps; ++s) {
        model.decode(cur, past, slots, out);
        for (int b = 0; b < B; ++b) past[b] += 1;
        cur = out;
    }
    return ms_since(t0);
}

// Section 1: single-sequence latency sweep (batch 1). TTFT / TPOT / throughput vs prompt length.
static void single_seq_sweep(Model& model, const Tokenizer& tok, int max_ctx, int vocab,
                             SampleParams sp, std::mt19937& rng) {
    std::fprintf(stderr, "[1] single-sequence latency (batch 1, output %d tok, warmup %d, median of %d)\n\n",
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
}

// Section 2: static-batch decode throughput. For each (prompt length, batch) the per-sequence
// TPOT and the aggregate decode throughput (all sequences together) show the batching win.
static void batch_sweep(Model& model, int max_ctx, int vocab, std::mt19937& rng) {
    int max_b = model.max_batch();
    std::fprintf(stderr, "\n[2] static-batch decode throughput (output %d tok/seq, median of %d, "
                 "max batch %d)\n\n", OUTPUT_LEN, REPEAT, max_b);
    std::fprintf(stderr, "  input   batch    TPOT/seq    aggregate\n");
    std::fprintf(stderr, "  (tok)            (ms/tok)     (tok/s)\n");

    for (int isl : BATCH_INPUT_LENS) {
        if (isl + OUTPUT_LEN >= max_ctx) {
            std::fprintf(stderr, "  %5d   (skipped: input+output exceeds max-ctx %d)\n", isl, max_ctx);
            continue;
        }
        for (int B : BATCH_SIZES) {
            if (B > max_b) {
                std::fprintf(stderr, "  %5d  %5d   (skipped: exceeds max batch %d)\n", isl, B, max_b);
                continue;
            }
            batch_decode_ms(model, isl, B, OUTPUT_LEN, vocab, rng);   // warmup
            std::vector<double> tpot, agg;
            for (int r = 0; r < REPEAT; ++r) {
                double dec = batch_decode_ms(model, isl, B, OUTPUT_LEN, vocab, rng);
                tpot.push_back(dec / OUTPUT_LEN);
                agg.push_back((double)B * OUTPUT_LEN / (dec / 1000.0));
            }
            std::fprintf(stderr, "  %5d  %5d   %8.2f  %11.1f\n", isl, B, median(tpot), median(agg));
        }
    }
}

// Section 3: continuous-batching throughput on a workload with VARIED output lengths. Slots=1
// is sequential serving (one request at a time); more slots let the scheduler keep the GPU busy
// by admitting a new request the instant one finishes, so aggregate throughput climbs.
static void continuous_sweep(Model& model, int max_ctx, int vocab) {
    const int R = 32, INPUT = 128, LEN_MIN = 16, LEN_MAX = 128;
    if (INPUT + LEN_MAX >= max_ctx) {
        std::fprintf(stderr, "\n[3] continuous batching (skipped: input+output exceeds max-ctx)\n");
        return;
    }

    // Build the workload once: R requests, fixed-length prompts, output lengths spread over
    // [LEN_MIN, LEN_MAX] so sequences finish at very different times.
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> len(LEN_MIN, LEN_MAX);
    std::vector<Request> base(R);
    long total_tok = 0;
    for (auto& r : base) { r.prompt = make_prompt(INPUT, vocab, rng); r.max_new = len(rng); total_tok += r.max_new; }

    auto time_slots = [&](int ns) {
        std::vector<Request> w = base;                       // fresh copy (base.output stays empty)
        auto t0 = Clock::now();
        run_continuous(model, w, ns, /*stop_on_eos=*/false);
        return ms_since(t0);
    };

    std::fprintf(stderr, "\n[3] continuous-batching throughput (%d requests, input %d, output %d-%d "
                 "random)\n\n", R, INPUT, LEN_MIN, LEN_MAX);
    std::fprintf(stderr, "  slots    wall (s)   aggregate tok/s\n");
    for (int ns : {1, 8, 16}) {
        if (ns > model.max_batch()) continue;
        double ms = time_slots(ns);
        std::fprintf(stderr, "  %5d   %7.2f      %9.1f\n", ns, ms / 1000.0, total_tok / (ms / 1000.0));
    }
}

int run_benchmark(Model& model, const Tokenizer& tok, int max_ctx) {
    SampleParams sp{0.0f, 1.0f, 0};   // greedy: timing shouldn't depend on sampling
    std::mt19937 rng(1234);
    int vocab = model.config().vocab_size;

    single_seq_sweep(model, tok, max_ctx, vocab, sp, rng);
    batch_sweep(model, max_ctx, vocab, rng);
    continuous_sweep(model, max_ctx, vocab);
    return 0;
}
