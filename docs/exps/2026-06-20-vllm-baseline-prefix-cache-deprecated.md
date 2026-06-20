# Deprecated vLLM baseline — prefix caching ON (record only)

Through most of the journey the vLLM reference was run with its **default prefix caching ON**. On the
`vllm bench serve --dataset-name random` workload every request shares the same chat-template prefix
(the system/user boilerplate), which prefix caching reuses — giving vLLM ~7% it does not get on a
feature-matched comparison. FlashQwen has no prefix sharing, so this was **not an apples-to-apples
baseline**.

It is superseded by the feature-matched baseline (`--no-enable-prefix-caching`) used in
`2026-06-20-journey-replay.md` / `2026-06-20-journey.csv`, which is the canonical reference in
`optimization.md`. Kept here only as a record of the earlier numbers.

vLLM default (prefix cache ON), 1024/128, conc=32, same machine:
| input | vLLM (prefix cache ON) | vLLM (no prefix cache — canonical) |
|---|---|---|
| 512  | 1031 | 944 |
| 1024 | 698  | 652 |

The historical `optimization.md` progress percentages (e.g. S3 50.1%, S12 87.2%) and the inline
"→698" targets in older step entries were computed against this prefix-cache baseline; they are
contemporaneous and have been superseded by the canonical Final-results table (vs no-cache vLLM).
