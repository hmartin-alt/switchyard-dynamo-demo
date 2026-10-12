#!/usr/bin/env bash
# Run one immutable AIPerf point. Run once per role/rate; never overwrite a point.
set -euo pipefail

if (( $# != 5 )); then
  echo "Usage: $0 <e4b_only|capable_only|switchyard> <served-model> <base-url> <rps> <work-dir>" >&2
  exit 2
fi
role=$1 model=$2 url=$3 rate=$4 work_dir=$5
case "$role" in e4b_only|capable_only|switchyard) ;; *) echo "Invalid role" >&2; exit 2;; esac
[[ "$rate" =~ ^[1-9][0-9]*$ ]] || { echo "RPS must be a positive integer" >&2; exit 2; }
[[ -s "$work_dir/shopping_mmlu_pool.jsonl" && -s "$work_dir/system.txt" ]] || {
  echo "Run prepare-inputs.py and stage its files in the work directory" >&2; exit 2;
}
output="$work_dir/artifacts/$role/rps-$rate"
if [[ -e "$output" ]]; then
  echo "Point already exists: $output (refusing to overwrite or duplicate it)" >&2
  exit 2
fi

duration=300 warmup=60
conversations=$((rate * (duration + warmup + 40)))
if (( conversations < 10000 )); then conversations=10000; fi
aiperf profile --model "$model" --url "$url" \
  --endpoint-type chat --endpoint /v1/chat/completions \
  --input-file "$work_dir/shopping_mmlu_pool.jsonl" --custom-dataset-type single_turn \
  --num-dataset-entries 500 --dataset-sampling-strategy random \
  --num-conversations "$conversations" --random-seed 42 \
  --system-prompt-file "$work_dir/system.txt" \
  --cache-bust first_turn_prefix \
  --extra-inputs '{"temperature":0,"seed":42,"max_tokens":16,"chat_template_kwargs":{"enable_thinking":false}}' \
  --tokenizer builtin --use-server-token-count \
  --request-rate "$rate" --request-rate-mode constant --concurrency 256 \
  --request-timeout-seconds 15 --benchmark-grace-period 20 \
  --warmup-duration "$warmup" --benchmark-duration "$duration" \
  --goodput request_latency:1000 --export-level raw --ui none \
  --artifact-dir "$output"
