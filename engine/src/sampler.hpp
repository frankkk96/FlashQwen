// Per-request sampling knobs (temp + top-p; no top-k). Sampling runs on the GPU
// (SampleBatchKernel, see kernels.cu / ModelRuntime::Forward).
#pragma once

struct SampleParams {
  float temp;   // <= 0 means greedy (argmax)
  float top_p;  // nucleus cutoff (1.0 = no truncation)
};
