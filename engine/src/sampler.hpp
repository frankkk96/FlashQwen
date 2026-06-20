// Per-request sampling knobs. Greedy (temp<=0), or temperature + top-p (nucleus). These are the two
// knobs the OpenAI API exposes; top-k is intentionally not supported. The sampling itself runs on the
// GPU (sample_batch_kernel, see kernels.cu / ModelRuntime::forward).
#pragma once

struct SampleParams {
    float temp;    // <= 0 means greedy (argmax)
    float top_p;   // nucleus cutoff (1.0 = no truncation)
};
