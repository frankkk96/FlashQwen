#pragma once

namespace fq {

// Per-request sampling knobs, applied on the GPU in SampleBatchKernel:
// temperature (<= 0 means greedy argmax) and top-p nucleus (1.0 = no cutoff).
struct SampleParams {
  float temp;
  float top_p;
};

}
