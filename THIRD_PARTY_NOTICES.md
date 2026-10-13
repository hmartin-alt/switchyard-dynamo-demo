# Third-party notices

This repository contains modified copies of, and patches against, upstream source files, plus data drawn from a public dataset. The upstream projects keep their own licenses. A copy of the Apache License 2.0 is in [`licenses/Apache-2.0.txt`](licenses/Apache-2.0.txt).

## Included in this repository

### Switchyard — Apache-2.0

Source: https://github.com/NVIDIA-NeMo/Switchyard, commit `a601a9a3f9db149a1ad430fa43b1463a170c8a82`.
Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

| File here | Relationship to upstream |
| --- | --- |
| `patches/switchyard-prefill-local.patch` | Changes to `crates/prefill-router/python/transformers_forward.py`, `crates/prefill-router/src/algorithm.rs` and `crates/switchyard-runner/src/algorithm.rs` |
| `patches/batching/algorithm.rs` | Modified copy of `crates/prefill-router/src/algorithm.rs` (includes the patch above) |
| `patches/batching/transformers_forward.py` | Modified copy of `crates/prefill-router/python/transformers_forward.py` |
| `patches/batching/algorithm-tests.rs` | Modified copy of `crates/prefill-router/tests/unit/algorithm.rs` |

These files keep the upstream SPDX headers. The changes are listed in [PATCHES.md](PATCHES.md). `patches/batching/microbatch.rs` and `patches/batching/switchyard_batch.py` are new files written for this repository.

### Shopping MMLU — Apache-2.0

Source: https://github.com/KL4805/ShoppingMMLU.
`benchmark/inputs/test.jsonl` holds 500 questions, choices and answer keys from the public dataset, unmodified. The benchmark archives in `benchmark/evidence/` contain those questions together with model outputs.

### Router checkpoint

`benchmark/inputs/prefill_router.pt` is a trained routing head. It was built with the llm-router toolkit (below) on top of features from the Qwen3.5-0.8B encoder, using correctness labels from Gemma 4 E4B and Gemma 4 31B outputs. It does not contain the Gemma or Qwen model weights.

## Used but not included

| Project | License | Use |
| --- | --- | --- |
| [llm-router](https://github.com/NVIDIA-AI-Blueprints/llm-router) (model-router toolkit), branch `v3`, commit `8a9d3509fbde879d9795258081bab9553b458e04` | Apache-2.0 | `patches/prefill-inference-overhead.patch` and `patches/prefill-unused-layers.patch` modify its `extract.py`. The toolkit is fetched separately. |
| [AIPerf](https://pypi.org/project/aiperf/) 0.13.0 | Apache-2.0 | Load generator, installed from PyPI. `scripts/aiperf/patch-export.py` contains short excerpts of two of its source files and edits them at run time. |
| Qwen/Qwen3.5-0.8B | Apache-2.0 (per its model card) | Router encoder, downloaded from Hugging Face. |
| google/gemma-4-E4B-it, google/gemma-4-31B-it | See each model card | Served models, downloaded from Hugging Face with your own access token. |
| NVIDIA Dynamo runtime image `nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.0` and vLLM | Per the image | Serving runtime. |
