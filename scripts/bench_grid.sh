#!/usr/bin/env bash
# Orthogonal serving benchmark grid: FlashQwen vs vLLM, default-vs-default, bf16 parity.
#
#   dimensions:  concurrency x input-length x dataset
#     random:    CONCS x INLENS, out=128; cells where conc*(in+128) exceeds the FQ KV pool are skipped
#     sharegpt:  CONCS only (real data has no input-length knob), out=256, num-prompts scaled per conc
#
#   resumable:   appends to $OUT_DIR/grid.csv and SKIPS cells already present, so an interrupted run
#                (or a later re-run after a code change: delete the CSV first) just continues.
#
#   usage:       scripts/bench_grid.sh                # defaults below
#                MODEL=/path/to/model VLLM_BIN=... OUT_DIR=... scripts/bench_grid.sh
#
set -uo pipefail
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY

REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
MODEL=${MODEL:-/autodl-fs/data/models/qwen3-8b}
VLLM_BIN=${VLLM_BIN:-/root/autodl-tmp/envs/vllm/bin/vllm}
SHAREGPT_JSON=${SHAREGPT_JSON:-/root/prof/sharegpt.json}
OUT_DIR=${OUT_DIR:-/root/prof/grid}
CONCS=${CONCS:-"1 8 16 32"}
INLENS=${INLENS:-"128 512 1024 2048"}
KV_POOL_TOKENS=${KV_POOL_TOKENS:-39000}   # FQ KV pool at --gpu-mem-fraction 0.9 on 24GB

mkdir -p "$OUT_DIR"; CSV="$OUT_DIR/grid.csv"
[ -f "$CSV" ] || echo "engine,dataset,inlen,conc,np,otps,tpot_ms,ttft_ms,gen_tok" > "$CSV"

have(){ grep -q "^$1,$2,$3,$4," "$CSV"; }   # engine,dataset,inlen,conc already recorded?

drain(){ pkill -9 -f flashqwen-engine 2>/dev/null; pkill -9 -f "flashqwen serve" 2>/dev/null
  pkill -9 -f "vllm serve" 2>/dev/null; pkill -9 -f EngineCore 2>/dev/null
  for i in $(seq 1 120); do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null||echo 0)
    [ "${u:-0}" -lt 500 ] && break; sleep 1; done; sleep 2; }

metrics(){ local o; o=$(cat)
  echo "$(echo "$o"|grep -aE 'Output token throughput'|grep -oE '[0-9.]+'|tail -1) \
$(echo "$o"|grep -aE 'Mean TPOT'|grep -oE '[0-9.]+'|tail -1) \
$(echo "$o"|grep -aE 'Mean TTFT'|grep -oE '[0-9.]+'|tail -1) \
$(echo "$o"|grep -aE 'Total generated tokens'|grep -oE '[0-9]+'|tail -1)"; }

rnd(){ local eng=$1 base=$2 inl=$3 conc=$4 np=$5 r
  have "$eng" random "$inl" "$conc" && { echo "  [$eng] random in=$inl c=$conc: cached, skip"; return; }
  r=$($VLLM_BIN bench serve --backend openai-chat --endpoint /v1/chat/completions --base-url "$base" \
    --model qwen3-8b --tokenizer "$MODEL" --dataset-name random --seed 1234 --temperature 0 \
    --random-input-len "$inl" --random-output-len 128 --num-prompts "$np" --max-concurrency "$conc" \
    --request-rate inf --extra-body '{"max_tokens":128,"chat_template_kwargs":{"enable_thinking":false}}' 2>&1 | metrics)
  read t p f g <<<"$r"; echo "${eng},random,${inl},${conc},${np},${t},${p},${f},${g}" >>"$CSV"
  echo "  [$eng] random in=$inl c=$conc np=$np: ${t} tok/s TPOT ${p} TTFT ${f}"; }

sg(){ local eng=$1 base=$2 conc=$3 np=$4 r
  have "$eng" sharegpt - "$conc" && { echo "  [$eng] sharegpt c=$conc: cached, skip"; return; }
  r=$($VLLM_BIN bench serve --backend openai-chat --endpoint /v1/chat/completions --base-url "$base" \
    --model qwen3-8b --tokenizer "$MODEL" --seed 1234 --temperature 0 --num-prompts "$np" --max-concurrency "$conc" \
    --request-rate inf --dataset-name sharegpt --dataset-path "$SHAREGPT_JSON" --sharegpt-output-len 256 \
    --extra-body '{"max_tokens":256,"chat_template_kwargs":{"enable_thinking":false}}' 2>&1 | metrics)
  read t p f g <<<"$r"; echo "${eng},sharegpt,-,${conc},${np},${t},${p},${f},${g}" >>"$CSV"
  echo "  [$eng] sharegpt c=$conc np=$np: ${t} tok/s TPOT ${p} TTFT ${f} (gen=${g})"; }

run_all(){ local eng=$1 base=$2 conc inl np
  for conc in $CONCS; do
    if [ "$conc" -le 1 ]; then np=32; else np=$((conc*4)); fi
    for inl in $INLENS; do
      [ $((conc*(inl+128))) -gt "$KV_POOL_TOKENS" ] && { echo "  [$eng] random in=$inl c=$conc: exceeds KV pool, skip"; continue; }
      rnd "$eng" "$base" "$inl" "$conc" "$np" || true
    done
  done
  for conc in $CONCS; do
    case $conc in 1) np=64;; 8) np=256;; 16) np=512;; *) np=1000;; esac
    sg "$eng" "$base" "$conc" "$np" || true
  done; }

echo "=== BUILD ==="; ( cd "$REPO" && make ) >"$OUT_DIR/build.log" 2>&1 || { echo BUILD_FAILED; tail -20 "$OUT_DIR/build.log"; exit 1; }
HEAD=$(cd "$REPO" && git rev-parse --short HEAD); echo "build ok @ $HEAD"

echo "=== FlashQwen @ $HEAD ==="; drain
( cd "$REPO" && ./flashqwen serve --model "$MODEL" --max-ctx 4096 --addr :8000 --slots 32 >"$OUT_DIR/fq.log" 2>&1 ) &
for i in $(seq 1 180); do curl -sf http://127.0.0.1:8000/v1/models>/dev/null 2>&1 && break; sleep 1; done
grep -aq "out of memory" "$OUT_DIR/fq.log" && { echo "FQ STARTUP OOM"; exit 1; }
echo ">>> FQ up ($(nvidia-smi --query-gpu=memory.used --format=csv,noheader))"
run_all flashqwen http://127.0.0.1:8000

echo "=== vLLM ($($VLLM_BIN --version 2>/dev/null|tail -1)) ==="; drain
$VLLM_BIN serve "$MODEL" --served-model-name qwen3-8b --max-model-len 4096 --max-num-seqs 32 \
  --gpu-memory-utilization 0.9 --port 8001 >"$OUT_DIR/vllm.log" 2>&1 &
for i in $(seq 1 400); do curl -sf http://127.0.0.1:8001/v1/models>/dev/null 2>&1 && break; sleep 1; done
grep -aiq "out of memory" "$OUT_DIR/vllm.log" && echo "vLLM STARTUP OOM"
echo ">>> vLLM up ($(nvidia-smi --query-gpu=memory.used --format=csv,noheader))"
run_all vllm http://127.0.0.1:8001

drain
echo "=== RESULTS CSV ==="; cat "$CSV"; echo "ALL_DONE"
