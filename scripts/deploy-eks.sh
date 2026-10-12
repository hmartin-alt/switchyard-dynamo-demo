#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || -z "$1" ]]; then
  echo "Usage: bash scripts/deploy-eks.sh KUBE_CONTEXT" >&2
  exit 2
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
context=$1

kubectl --context "$context" get namespace dynamo >/dev/null
nodes=$(kubectl --context "$context" get nodes -l workload=model-serving -o name)
node_count=$(printf '%s\n' "$nodes" | awk 'NF {count++} END {print count+0}')
if [[ "$node_count" -ne 1 ]]; then
  echo "Expected exactly one node labeled workload=model-serving in context $context; found $node_count." >&2
  exit 1
fi

echo "Deploying to context $context on $nodes"
kubectl --context "$context" apply -f "$repo_root/manifests/dynamo/dgd-efficient-colocated.yaml"
kubectl --context "$context" apply -f "$repo_root/manifests/dynamo/dgd-capable.yaml"
kubectl --context "$context" apply -f "$repo_root/manifests/router/colocated-service.yaml"

kubectl --context "$context" -n dynamo get dgd efficient capable
kubectl --context "$context" -n dynamo get pods -l app=switchyard-production-rr -o wide
kubectl --context "$context" -n dynamo get endpointslices -l kubernetes.io/service-name=switchyard-production-rr
