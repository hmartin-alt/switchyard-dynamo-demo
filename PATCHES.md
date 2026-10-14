# Patch inventory

This demo runs a pinned Switchyard build plus local changes to Switchyard and the model-router toolkit. This file lists each change, what it is for, how it works today, and what it would take to upstream. Nothing here has been proposed upstream yet.

| Upstream | Pinned commit |
| --- | --- |
| [Switchyard](https://github.com/NVIDIA-NeMo/Switchyard) | `a601a9a3f9db149a1ad430fa43b1463a170c8a82` |
| [llm-router](https://github.com/NVIDIA-AI-Blueprints/llm-router) (model-router toolkit), branch `v3` | `8a9d3509fbde879d9795258081bab9553b458e04` |
| [AIPerf](https://pypi.org/project/aiperf/) | 0.13.0 |

`restore-colocated-runtime.sh` applies them in this order: the Switchyard patch, the batching overlays (which replace three Switchyard files with versions that already contain the patch), then the two toolkit patches.

| # | Change | Where | Type | Upstream issue |
| --- | --- | --- | --- | --- |
| 1 | Toolkit checkpoint adapter | Switchyard | functionality | 1, 6 |
| 2 | Tolerance routing | Switchyard | routing policy | 2 |
| 3 | Microbatching | Switchyard | performance | 3, 6 |
| 4 | Skip unneeded logits and per-request cache flush | toolkit | performance | 4 |
| 5 | Early exit after the deepest feature layer | toolkit | performance | 5 |
| 6 | Strip AIPerf request ID before routing | Switchyard | benchmark only | stays local |
| 7 | AIPerf export fix | AIPerf | benchmark only | check release first |

## About the evidence

The benchmark evidence committed in `benchmark/evidence/` compares whole setups (31B-only against Switchyard). It does not isolate any single change below. The per-change numbers cited here come from earlier experiments run on 2026-10-04 and 2026-10-05 that are **not committed**. They used an older checkpoint (SHA-256 `49d5ed78…`, tolerance 0.195), a four-GPU fleet, and 100 to 500 validation prompts, so they show direction and size, not the final results. Before opening upstream issues, rerun the comparisons for items 3 to 5 on the final checkpoint and commit the output.

## 1. Toolkit checkpoint adapter

- **Target:** Switchyard `crates/prefill-router/python/transformers_forward.py`, in `switchyard-prefill-local.patch`.
- **Problem:** Switchyard's prefill router expects its own checkpoint layout and loads it with `weights_only=True`. Checkpoints trained with the toolkit have different keys (`model_names`, `transforms`, `shared_trunk`).
- **Workaround:** `TransformersForward.__new__` loads the file, and if it has those keys, returns a `ToolkitForward` that calls the toolkit's `load_scorer` and sets the scorer's private `_device` field. The load uses `weights_only=False`, which runs arbitrary pickle code, so the checkpoint must be trusted. The adapter's scoring loop calls `scorer.score` once per prompt and accepts `batch_size` and `max_length` without using them.
- **Changes:** functionality only. Scores come from the toolkit's own scorer, so they are the toolkit's scores.
- **Evidence:** none needed beyond the loader. By construction the unmodified loader rejects these checkpoints. I have not run unmodified Switchyard against this checkpoint to confirm that.
- **Production version:** a documented checkpoint contract shared by both projects, or a supported adapter, that does not touch private scorer fields. Prefer a tensor-only format such as safetensors over pickle.
- **Tests for upstreaming:** load a toolkit checkpoint through Switchyard; scores equal the toolkit's `score` output; malformed or foreign checkpoints fail with a clear error; loading never executes pickle code.

## 2. Tolerance routing

- **Target:** Switchyard `crates/prefill-router/src/algorithm.rs` and `crates/switchyard-runner/src/algorithm.rs`, in `switchyard-prefill-local.patch`.
- **Problem:** Switchyard picks the target with the highest predicted success. This demo wants the cheaper model whenever its prediction is close enough.
- **Workaround:** a `tolerance` field on the `prefill_router` route (default 0). The router computes `threshold = best − tolerance` and picks the first target in the route's `targets` list whose predicted success is at or above it. List order is therefore the cost order (`targets = ["weak", "strong"]` in `colocated-routes.toml`). The demo uses `0.1760250329971314`.
- **Changes:** routing policy. At tolerance 0 the rule matches the old argmax except on exact ties: the new rule takes the first target, while Rust's `max_by` returns the last.
- **Evidence:** the tolerance was chosen on validation data. At 0.176 it sent 78.2% of validation questions to E4B at 91.2% accuracy (`benchmark/inputs/policy.json`). On the 500 held-out questions it sent 77.2% to E4B at 93.6% accuracy.
- **Production version:** make priority or cost explicit per target instead of relying on list order; validate the range 0 to 1; document tie behavior.
- **Tests for upstreaming:** tolerance 0 equals argmax away from ties; tolerance picks the earlier target inside the band and the later one outside it; ties; NaN scores; config parsing and defaults; target order is respected.

## 3. Microbatching

- **Target:** Switchyard, files `patches/batching/algorithm.rs`, `microbatch.rs` (new), `transformers_forward.py`, plus `switchyard_batch.py` on the Python path. Tests are in `microbatch.rs` and `algorithm-tests.rs` (see "Existing tests here").
- **Problem:** with one request scored at a time, encoder work is serialized and the router becomes the throughput limit.
- **Workaround:** opt in with `SWITCHYARD_PREFILL_MICROBATCH=1`. A bounded queue (128 jobs) feeds one worker that collects up to `SWITCHYARD_PREFILL_BATCH_SIZE` prompts (default 16, max 64) for up to `SWITCHYARD_PREFILL_BATCH_WAIT_MS` (default 2, max 100) and scores them together. `switchyard_batch.py` reaches into private toolkit fields (`_extractor`, `_ckpt`, `_trunk_nets`, `_device`, `_needed_layers`, `_needed_pooling_modes`), sets the tokenizer's `padding_side` to right, and temporarily wraps `_forward_for_features` to time the encoder. Because batch shape changes BF16 rounding, any request whose score is within `SWITCHYARD_PREFILL_NUMERIC_GUARD` of the routing threshold is rescored one at a time. The server refuses to start if `SWITCHYARD_PREFILL_TOLERANCE` differs from the route's tolerance. Queue wait, batch collection and prediction times are logged, not exported as metrics.
- **Changes:** performance, by intent. A request near the threshold could still route differently if the guard window is too narrow.
- **Evidence (earlier experiment, not committed):** with 16 concurrent clients and 500 requests, end-to-end Switchyard throughput (router plus both models) went from 29.9 to 64.8 requests/s over three trials. 31B-only reached 91.7 in the same run, so in that setup Switchyard was still slower than 31B-only. Offline scoring of 100 prompts ran at about 47, 73, 104 and 128 prompts/s at batch sizes 2, 4, 8 and 16. Without the guard, batched scoring flipped 1 of 100 routing decisions at batch sizes 4, 8 and 16 (the flipped request sat 0.0045 from the threshold) and none at batch size 2. With the guard enabled there were no flips; the result file does not record the guard value (the code default is 0.06). Batched and single scores differed by up to 0.026.
- **Parity result on the final checkpoint (2026-10-12, g7e.12xlarge, RTX PRO 6000 Blackwell, torch 2.11.0+cu130):** the launchers set the guard to `0.0024222002754211427`. That number is not from upstream; it came from a 2026-10-05 parity run on an earlier router (checkpoint `abfed7c2…`, tolerance 0.0446, 75 prompts) and was carried over unchanged. A one-off parity script (not kept in this repo) run on the 500 held-out questions (tolerance 0.176, batch sizes 2, 4, 8 and 16, three shuffles each) found that batching changes routes: 40 flips across 12 runs (2 to 6 per run), from only 8 distinct questions, each within 0.013 of the cutoff. With the current guard 25 of those flips remained, so it is too narrow. Single-request scoring was repeatable (difference 0.0), and batched scores differed from it by up to 0.039. A guard of 0.0125, picked from the first three shuffles, left 2 flips on six fresh shuffles (they needed 0.0175). A guard of 0.04 left no flips in 24 runs (6 shuffles × 4 batch sizes) and rescored about 8% of requests, against 0.2% to 0.8% at the current guard. On the held-out set the 8 flippable questions have the same correctness under both models (7 right for both, 1 wrong for both), so no answer changes there. The effect is limited to near-ties, but the launchers' guard does not do what it is meant to do. Widening it is a launcher setting, not a code change; it has not been applied here, and the 24-run result above is not a benchmark of the widened setting. The reports are not committed.
- **Production version:** a public batched-scoring API in the toolkit; in Switchyard, batch size, wait time and queue depth as route config, with queue-depth and encoder-latency metrics. Padding handling belongs inside the toolkit.
- **Tests for upstreaming:** scalar and batched routing agree across batch sizes, padding lengths and near-threshold cases; queue-full behavior; cancelled requests are dropped; shutdown; metrics emitted.
- **Existing tests here:** three Python tests for the guard (`tests/test_prefill_batch.py`). `microbatch.rs` also contains Rust tests for batching, request mapping, error recovery and flushing an incomplete batch, and `algorithm-tests.rs` holds tolerance and routing tests. Neither is run by the documented demo build: `restore-colocated-runtime.sh` copies `microbatch.rs` (so its tests compile in only under `cargo test`, which the script does not run) but does not copy `algorithm-tests.rs` into the Switchyard test tree. Treat both as unverified until run upstream.

## 4. Skip unneeded logits and per-request cache flush

- **Target:** toolkit `src/model_router_toolkit/prefill/extract.py`, in `patches/prefill-inference-overhead.patch`.
- **Problem:** the extractor runs the language-model head over every token, though only hidden states are used, and calls `torch.cuda.empty_cache()` after every request.
- **Workaround:** two opt-in switches. `PREFILL_LAST_LOGIT_ONLY=1` passes `logits_to_keep=1` and raises if the model does not expose that argument. `PREFILL_RETAIN_CUDA_CACHE=1` skips the flush.
- **Changes:** performance only. Hidden states are unchanged.
- **Evidence (earlier experiment, not committed):** over 100 prompts and two repeats, scores differed by 0.0 and no routing decision changed. Peak allocated GPU memory fell from 1.592 to 1.531 GB with last-logit-only. End-to-end Switchyard throughput (router plus both models) with the overhead patches went from 8.37 to 8.99 requests/s with one client (+7%), 21.5 to 23.7 with four (+10%), and 23.4 to 28.8 with sixteen (+23%). That run included the unbatched router only, and I cannot separate this change from item 5 in it.
- **Production version:** never compute logits when only hidden states are needed (use the base model or an explicit hidden-states path), and do not flush the cache per request.
- **Tests for upstreaming:** hidden states bit-identical with and without the change; clear failure for models lacking `logits_to_keep`; memory behavior over many requests.

## 5. Early exit after the deepest feature layer

- **Target:** toolkit `extract.py`, in `patches/prefill-unused-layers.patch`.
- **Problem:** the encoder computes every layer even when the checkpoint only needs features from earlier ones.
- **Workaround:** with `PREFILL_SKIP_UNUSED_LAYERS=1`, `_forward_for_features` temporarily sets `config.num_hidden_layers` to the deepest needed layer plus one, runs the forward pass, and restores it in a `finally` block. It only allows `Qwen3_5ForCausalLM`. It is not safe for concurrent calls on one model instance; the Switchyard side holds a mutex around scoring.
- **Changes:** performance only.
- **Evidence (earlier experiment, not committed):** scores differed by 0.0 with no routing changes over 100 prompts and two repeats. Peak memory was 1.530 GB against 1.592 GB for the baseline. I do not have a committed latency measurement for this change alone.
- **Production version:** an explicit early-exit API that is safe under concurrency and does not mutate shared config, for example by running only the needed layers on a view of the model.
- **Tests for upstreaming:** selected hidden states identical to a full forward pass; concurrent calls; unsupported architectures rejected; config restored after errors.

## 6. Strip the AIPerf request ID before routing

- **Target:** Switchyard `algorithm.rs` (batching overlay), with a Rust unit test in `algorithm-tests.rs` (not copied or run by the demo build).
- **Problem:** AIPerf's cache-bust mode prepends `[rid:<12 hex>]\n\n` to each user message. The router should classify the clean question while the model still receives the tagged one.
- **Workaround:** with `SWITCHYARD_STRIP_AIPERF_RID_FOR_ROUTING=1`, the router removes that exact prefix from its own copy of the prompt. The request sent to Dynamo is unchanged.
- **Changes:** the routing input, for benchmarking only. Off by default.
- **Evidence:** the 2026-10-11 Switchyard runs in `benchmark/evidence/` used it. Routing share under load (77.8% to E4B) matched the clean run (77.2%).
- **Production version:** none. This is specific to the demo and stays local.

## 7. AIPerf export fix

- **Target:** AIPerf 0.13.0, applied at run time by `scripts/aiperf/patch-export.py`.
- **Problem:** at high request rates the export writes `inputs.json` redundantly and copies the raw record file, which is slow.
- **Workaround:** the script skips the `inputs.json` export and moves the single raw record file instead of copying it. It checks the AIPerf version and the exact source text first, and refuses to run otherwise.
- **Changes:** artifact export only, not requests or metrics.
- **Evidence:** none committed.
- **Next step:** check whether a newer AIPerf release already fixes this. If not, open an issue with the version-guarded change.

## Environment variables the launchers set

| Variable | Purpose | Item |
| --- | --- | --- |
| `PREFILL_LAST_LOGIT_ONLY`, `PREFILL_RETAIN_CUDA_CACHE` | toolkit overhead switches | 4 |
| `PREFILL_SKIP_UNUSED_LAYERS` | early exit | 5 |
| `SWITCHYARD_PREFILL_MICROBATCH`, `_BATCH_SIZE`, `_BATCH_WAIT_MS` | batching on and its limits | 3 |
| `SWITCHYARD_PREFILL_NUMERIC_GUARD` | rescore window around the threshold | 3 |
| `SWITCHYARD_PREFILL_TOLERANCE` | must equal the route's `tolerance` when batching is on | 2, 3 |
| `SWITCHYARD_STRIP_AIPERF_RID_FOR_ROUTING` | benchmark only | 6 |
| `SWITCHYARD_PREFIX_CACHE_MODE` | `disabled` or `enabled`, picks the vLLM prefix-cache flag | demo launchers |

The tolerance appears in two places, `colocated-routes.toml` and the launcher environment. The server checks they match only when batching is enabled.

## Proposed upstream tracking issues

- Checkpoint compatibility: patch item 1
- Tolerance routing: item 2
- Microbatching: item 3
- Inference overhead: item 4
- Early exit: item 5
- Public batched-scoring API: items 1 and 3
- AIPerf export: item 7

These should go upstream as separate issues and PRs. Checkpoint compatibility, routing policy, batching and inference optimization are independent changes.
