#!/usr/bin/env bash
# Run one frozen role/rate schedule on an already deployed topology.
set -euo pipefail

if (( $# != 5 )); then
  echo "Usage: $0 <e4b_only|capable_only|switchyard> <served-model> <base-url> <work-dir> <protocol.json>" >&2
  exit 2
fi
role=$1 model=$2 url=$3 work_dir=$4 protocol=$5
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

case "$role:$model" in
  e4b_only:efficient|capable_only:capable|switchyard:shopping) ;;
  *) echo "Role/model mismatch" >&2; exit 2 ;;
esac

rates=$(python3 -c 'import json,sys; p=json.load(open(sys.argv[1])); print(" ".join(map(str,p["rates"][sys.argv[2]])))' "$protocol" "$role")
hourly_usd=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["g7e_hourly_usd_for_frozen_comparison"])' "$protocol")
expected_hash=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["test_sha256"])' "$protocol")
actual_hash=$(python3 -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$work_dir/test.jsonl")
[[ "$actual_hash" == "$expected_hash" ]] || { echo "Frozen test hash mismatch" >&2; exit 1; }

for rate in $rates; do
  point="$work_dir/artifacts/$role/rps-$rate"
  if [[ -e "$point" ]]; then
    echo "Validating existing $role/$rate (never rerunning an existing point)"
  else
    bash "$script_dir/run-point.sh" "$role" "$model" "$url" "$rate" "$work_dir"
  fi
  result=$(python3 "$script_dir/validate-point.py" "$work_dir" "$role" "$rate" \
    --g7e-hourly-usd "$hourly_usd")
  printf '%s\n' "$result"
  slo_pass=$(printf '%s\n' "$result" | python3 -c 'import json,sys; print(int(json.load(sys.stdin)["slo_pass"]))')
  if [[ "$slo_pass" == 0 ]]; then
    echo "First valid tested SLO failure for $role at $rate RPS; stopping this arm."
    break
  fi
done
