# Reproduce the cache-busted AIPerf comparison

This is an **opt-in benchmark mode**, not the default serving configuration. AIPerf adds a unique `[rid:<12 hex>]` marker at the start of each replayed user message. Switchyard removes that marker **only from the copy of the question scored by the prefill encoder**. The unchanged tagged request goes to the selected Dynamo Graph Deployment (DGD). Ordinary vLLM prefix caching stays enabled, so the shared system prefix can be reused but complete repeated questions cannot. The original request object is not edited.

The comparison uses three complete, separate `g7e.12xlarge` configurations in turn: two E4B workers, two 31B workers, or one E4B plus one 31B with two colocated router encoders. The same 500 held-out Shopping MMLU questions, system prompt, checkpoint, tolerance, decoding, and instance size are used. Accuracy is evaluated separately on **clean** questions; do not treat tagged AIPerf responses as the clean accuracy measurement.

## 1. Prepare the exact inputs

The held-out data and checkpoint are not committed. Obtain the frozen `test.jsonl` and `prefill_router.pt` through your approved channel. Verify their SHA-256 values against the frozen policy file before deploying. The demo checkpoint hash is in the [deployment guide](shopping-demo.md). Then prepare a local work directory:

```bash
python3 scripts/aiperf/prepare-inputs.py /path/to/test.jsonl /path/to/benchmark-work \
  --expected-sha256 YOUR_FROZEN_TEST_SHA256
```

This produces the original system message, unchanged question pool, and a checksum record. Do **not** commit the work directory or question file.

## 2. Deploy the benchmark serving mode

Build Switchyard from this repo's current `patches/batching/algorithm.rs` overlay, following the [deployment guide](shopping-demo.md). The compiled binary must include `prefill_prompt_without_aiperf_rid`; editing only AIPerf cannot hide tags from the encoder. For the Switchyard setup, make these two changes in **both** local DGD manifest copies before applying them:

```yaml
- name: SWITCHYARD_PREFIX_CACHE_MODE
  value: enabled
- name: SWITCHYARD_STRIP_AIPERF_RID_FOR_ROUTING
  value: "1"
```

The first setting launches each `dynamo.vllm` worker with `--enable-prefix-caching`. The second affects only the router encoder. The committed DGD manifests keep both benchmark options off by default. Confirm both routers are ready and receiving traffic. For model-only baselines, deploy two workers of the same model on the same full instance, with prefix caching enabled; they do not need a router. **Run one configuration at a time**, and use the same model revisions and inference settings. This repository does not provision AWS resources or model-only DGDs automatically.

## 3. Run AIPerf on a dedicated CPU load generator

Use AIPerf **0.13.0**. The optional [load-generator Pod](../manifests/aiperf/loadgen.yaml) expects a prepared `workload=aiperf` CPU node. Stage the contents of the prepared work directory and `scripts/aiperf/` into `/work` on that Pod. Before high-rate runs, execute `python3 /work/patch-export.py`; its version-guarded changes prevent duplicate/slow artifact export, without changing requests or metrics.

For each offered rate, run **one** point (example URLs are cluster-local):

```bash
bash /work/run-point.sh capable_only capable http://capable-frontend:8000 96 /work
bash /work/run-point.sh switchyard shopping http://switchyard-production-rr:4000 160 /work
bash /work/run-point.sh e4b_only efficient http://efficient-frontend:8000 768 /work
```

These are examples of previously tested *passing* points, not a substitute for testing the SLO boundary. The script fixes: 500 questions sampled randomly with replacement; `--cache-bust first_turn_prefix`; temperature 0, seed 42, max 16 tokens, thinking disabled; 60-second warmup; 300-second measurement; 1-second good-request cutoff; and raw export. It refuses to overwrite an existing role/rate directory. Run additional offered rates until the next valid point fails the criterion below. Never count an incomplete or cancelled point as a result.

## 4. Validate each point and report only scoped results

```bash
python3 /work/validate-point.py /work switchyard 160
```

The validator checks completed/non-cancelled export, full measurement duration, 500-question coverage, unique request IDs, unchanged prompts/decoding, actual issued records, and at most **32 cached shared-prefix tokens** per observed response. Check pod restarts, load-generator memory events, and both router replicas' traffic separately. An SLO failure is a valid capacity result; an incomplete export or operational failure is not.

A point passes when at least **95%** of requests meet the **1-second** latency cutoff. For each configuration, select the highest tested passing goodput and compute estimated G7e cost per 1,000 compliant requests as `G7e hourly price × 1000 / (good RPS × 3600)`. Because each setup uses the same full instance, relative cost per compliant request is the inverse goodput ratio. This excludes the CPU load generator, EKS overhead, storage, network, and idle time; it is **not** a measured AWS bill or production cost claim. Keep the clean-prompt accuracy result visibly separate from the tagged performance result.

The recorded comparison behind the current demo selected 95.56 good RPS for 31B-only and 159.63 for Switchyard, a **1.67× throughput ratio** and approximately **40% lower estimated G7e instance cost per compliant request**. On a separate clean 500-question accuracy pass, Switchyard scored 93.6% and 31B-only 95.2%. Those figures are examples to reproduce, not guaranteed outcomes on another cluster.
