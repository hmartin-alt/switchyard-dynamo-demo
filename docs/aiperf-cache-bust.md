# Reproduce the Shopping MMLU comparison

This recipe covers the **clean 500-question accuracy evaluation** and a separate **three-arm AIPerf capacity comparison**. It does not retrain the router. The committed [protocol](../benchmark/protocol.json) freezes the dataset/checkpoint hashes, offered-rate schedule, 60-second warmup, 300-second measurement, one-second latency cutoff, and 95% required attainment. Run each arm on the **same complete two-GPU `g7e.12xlarge`**, one arm at a time, with a dedicated `c7i.2xlarge` CPU load generator. Model-only arms use two workers of the same model; Switchyard uses one worker of each model plus one colocated router encoder per GPU.

Prefix caching is enabled for all three AIPerf arms. AIPerf 0.13.0 uses `--cache-bust first_turn_prefix` to place a unique `[rid:...]` marker before each replayed user question. The Switchyard encoder ignores that marker, but the original tagged request reaches Dynamo/vLLM. The shared system prefix can be cached; repeated complete questions cannot. `validate-point.py` requires all 500 questions to appear, unique markers, and no more than 32 cached prompt tokens per observed response.

The [frozen inputs](../benchmark/inputs/SOURCE.md) and [original selected-point evidence](../benchmark/evidence/README.md) are included. The model weights, EKS cluster, and full intermediate-rate raw exports are not. The 40% estimated G7e cost reduction compares separate 31B-only and Switchyard benchmark runs using the same cache-busted workload, with request IDs hidden only from Switchyard’s router encoder; it is **not** a simultaneous or repeated production-cost measurement.

## 1. Prerequisites and inputs

Use the Dynamo operator and runtime image `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.0`, the pinned model snapshots and patched Switchyard runtime from the [deployment guide](shopping-demo.md), a `dynamo` namespace, and `hf-token-secret`. Provide one node labeled `workload=model-serving` with two allocatable RTX PRO 6000 GPUs and one `c7i.2xlarge` node labeled `workload=aiperf`. The optional `benchmark=aiperf:NoSchedule` taint is tolerated by the load-generator Pod. Verify instance types, GPU allocation, and readiness before each arm; the repo does not create or pay for EKS nodegroups.

```bash
python3 scripts/aiperf/check-nodes.py --context YOUR_KUBE_CONTEXT
```

From the repo root, create an ignored local work directory and verify the exact input hash:

```bash
python3 scripts/aiperf/prepare-inputs.py benchmark/inputs/test.jsonl output/benchmark-work \
  --expected-sha256 07882e772a3aa492e97747aa88a83a61ca5975810842e88539ee9b07e2427c64
shasum -a 256 benchmark/inputs/prefill_router.pt
```

The checkpoint hash must be `5c3e0704368f248a469d6762252bb3a1257bfd7016c885417de1707488247bdf`. Stage **this** checkpoint in the builder Pod using the deployment guide. Do not use another local checkpoint or question split.

## 2. Measure clean-prompt accuracy

Deploy the standard Switchyard DGDs and Service from the deployment guide (the defaults have prefix caching disabled). They expose both direct Dynamo frontends and the router Service on one G7e instance. In three separate terminals, keep these port-forwards running:

```bash
kubectl -n dynamo port-forward svc/efficient-frontend 18000:8000
kubectl -n dynamo port-forward svc/capable-frontend 18001:8000
kubectl -n dynamo port-forward svc/switchyard-production-rr 14000:4000
```

Then run each of the 500 questions once through E4B, 31B, and Switchyard with the same system prompt and decoding settings. This command verifies the frozen policy and input hashes and refuses to overwrite results:

```bash
python3 scripts/shopping-mmlu/tiered-heldout-live.py \
  --input benchmark/inputs/test.jsonl \
  --router-dir benchmark/inputs \
  --checkpoint benchmark/inputs/prefill_router.pt \
  --policy benchmark/inputs/policy.json \
  --costs configs/prefill-router/shopping-costs-token-proxy.json \
  --output-rows output/clean-accuracy-rows.jsonl \
  --output-report output/clean-accuracy-report.json
```

Install the Python `openai` package if it is not already available. Accuracy uses **clean** questions; do not substitute the tagged AIPerf responses or claim that accuracy and cost came from one run.

## 3. Prepare AIPerf on the CPU node

Stop the clean-accuracy port-forwards and remove the standard Switchyard DGDs and Service before deploying a baseline. Apply the [load-generator Pod](../manifests/aiperf/loadgen.yaml), wait for `/work/installed`, and copy the prepared input files, runner scripts, and protocol into it. The Pod installs AIPerf **0.13.0** and Python 3.12.

```bash
kubectl -n dynamo delete dgd efficient capable
kubectl -n dynamo delete service switchyard-production-rr
kubectl apply -f manifests/aiperf/loadgen.yaml
kubectl -n dynamo wait --for=condition=Ready pod/aiperf-loadgen --timeout=10m
until kubectl -n dynamo exec aiperf-loadgen -- test -f /work/installed; do sleep 5; done
kubectl -n dynamo cp output/benchmark-work/. aiperf-loadgen:/work
kubectl -n dynamo cp scripts/aiperf aiperf-loadgen:/work/scripts-aiperf
kubectl -n dynamo cp benchmark/protocol.json aiperf-loadgen:/work/protocol.json
kubectl -n dynamo exec aiperf-loadgen -- python3 /work/scripts-aiperf/patch-export.py
```

The version-guarded export patch avoids duplicate/slow artifact export without changing requests or metrics. Confirm `aiperf profile --help` offers `--cache-bust first_turn_prefix` before running. Keep only **one AIPerf process** alive. Preserve incomplete attempts separately; the runner refuses to overwrite an existing rate directory.

## 4. Run the three topologies, one at a time

For each arm, wait until **two worker Pods** are Ready on the same G7e node and verify the expected frontend is reachable from the load-generator Pod. Do not run two arms concurrently. `run-sweep.sh` reads the exact rates from `protocol.json`, resumes only valid existing points, and stops after the first valid SLO failure. Each rate is one 60-second warmup plus 300-second measurement, with 500 questions sampled randomly with replacement, fixed decoding, unique request IDs, and a 1-second goodput cutoff. A valid SLO failure is a result; an incomplete or cancelled export is not.

E4B-only (two E4B workers):

```bash
kubectl apply -f manifests/benchmark/dgd-e4b-only.yaml
kubectl -n dynamo wait --for=condition=Ready pod -l benchmark-role=e4b-only --timeout=20m
kubectl -n dynamo exec aiperf-loadgen -- bash /work/scripts-aiperf/run-sweep.sh \
  e4b_only efficient http://efficient-frontend:8000 /work /work/protocol.json
kubectl -n dynamo delete dgd efficient
kubectl -n dynamo wait --for=delete pod -l benchmark-role=e4b-only --timeout=10m
```

31B-only (two 31B workers):

```bash
kubectl apply -f manifests/benchmark/dgd-31b-only.yaml
kubectl -n dynamo wait --for=condition=Ready pod -l benchmark-role=capable-only --timeout=20m
kubectl -n dynamo exec aiperf-loadgen -- bash /work/scripts-aiperf/run-sweep.sh \
  capable_only capable http://capable-frontend:8000 /work /work/protocol.json
kubectl -n dynamo delete dgd capable
kubectl -n dynamo wait --for=delete pod -l benchmark-role=capable-only --timeout=10m
```

Switchyard (E4B + 31B, with two colocated encoders):

```bash
kubectl kustomize --load-restrictor=LoadRestrictionsNone \
  manifests/benchmark/switchyard-cache-bust | kubectl apply -f -
kubectl -n dynamo wait --for=condition=Ready pod -l app=switchyard-production-rr --timeout=20m
kubectl -n dynamo get endpointslices -l kubernetes.io/service-name=switchyard-production-rr
kubectl -n dynamo exec aiperf-loadgen -- bash /work/scripts-aiperf/run-sweep.sh \
  switchyard shopping http://switchyard-production-rr:4000 /work /work/protocol.json
```

The Kustomize overlay changes **both** workers to ordinary prefix caching and removes request IDs only from the router encoder copy. Check router logs to confirm both replicas received traffic; the Service balances TCP connections, not individual requests. `kubectl kustomize` needs the shown load-restrictor setting because this small overlay references manifests in a sibling directory.

## 5. Copy results and calculate the comparison

Copy the new artifacts back to the prepared local work directory, then validate **every** planned point and calculate the selected highest-tested passing goodput and G7e instance cost:

```bash
kubectl -n dynamo cp aiperf-loadgen:/work/artifacts output/benchmark-work/artifacts
python3 scripts/aiperf/summarize-comparison.py output/benchmark-work \
  --output output/benchmark-report.json
```

`protocol.json` freezes the historical `$14.40172/hour` G7e rate used for the reported cost estimate; replace it and record the source if using a different rate. Cost per 1,000 compliant requests is `hourly G7e cost × 1,000 / (good RPS × 3,600)`. The report explicitly identifies passing and failing tested points and whether a higher tested failure bounds each arm. The E4B-only sweep ended at 768 RPS without a failure, so do not call that an exact limit. One trial per rate does not establish a stable production capacity boundary; repeat near-boundary points if precision matters.

This estimate excludes the CPU load generator, EKS control plane, storage, network, and idle time. It is **not** an AWS bill or evidence of savings on production traffic. After backing up artifacts, delete the benchmark DGDs/Service and scale down both nodegroups according to your organization's approved procedure; verify EC2 instances are terminated.

To audit the original quoted points without a cluster, follow the [evidence extraction steps](../benchmark/evidence/README.md). The published evidence includes the selected 96/160 RPS points and the next tested 112/192 RPS failures, but not all intermediate-rate raw exports.
