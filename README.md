# Switchyard + Dynamo shopping router

This repository is a serving reference for the shopping demo. Switchyard's trained prefill router chooses between Gemma 4 E4B (`efficient`) and Gemma 4 31B (`capable`) for each Shopping MMLU question. Both models have separate Dynamo Graph Deployments (DGDs) on one two-GPU `g7e.12xlarge`. Each GPU runs one model worker and one colocated copy of the Qwen3.5-0.8B prefill encoder/router; a Kubernetes Service exposes the two router replicas.

| GPU allocation | Dynamo Graph Deployment | Colocated processes |
| --- | --- | --- |
| One GPU | `efficient` | Gemma 4 E4B worker + Switchyard prefill router |
| Other GPU | `capable` | Gemma 4 31B worker + Switchyard prefill router |

The manifests allocate one GPU to each worker but do not pin numeric GPU IDs. The two router replicas share the same frozen checkpoint, targets, tolerance, and request settings. 

## Reproduce the serving setup

Start with the [deployment guide](docs/shopping-demo.md). It covers model access, pinned model snapshots, building the patched Switchyard runtime, staging the trained checkpoint, applying both DGDs and the Service, and checking both router replicas. The default manifests disable vLLM prefix caching. An **opt-in, version-guarded experimental shared-prefix-only cache policy** is retained for the earlier controlled cache study; it is not a general production recommendation. The [AIPerf cache-bust recipe](docs/aiperf-cache-bust.md) reproduces the newer benchmark with ordinary vLLM prefix caching and request IDs hidden only from the router encoder.

The checkpoint, model weights, credentials, source checkouts, and Kubernetes cluster are **not** committed. The repo therefore provides serving code and a deployment recipe, not a turnkey cluster. Training/tuning helpers under `scripts/shopping-mmlu/` document how the routing policy was produced. AIPerf runner and validation scripts are included; the held-out dataset and benchmark artifacts are not.

## Scope of the measured results

Switchyard scored **93.6%** on 500 clean held-out questions versus **95.2%** for 31B-only. A *separate* AIPerf capacity test with first-turn request IDs sent unchanged to Dynamo found **159.63** versus **95.56** SLO-compliant requests/s on the same full G7e size. This implies about **40% lower estimated G7e instance cost per compliant request**. The router alone ignored the synthetic ID during prediction. These are separate quality and performance measurements, not a single production-cost or tagged-prompt accuracy result; the tagged prompts changed five net answers in the diagnostic accuracy run. See the [benchmark recipe](docs/aiperf-cache-bust.md) for controls and limits.

Upstream: [Switchyard](https://github.com/NVIDIA-NeMo/Switchyard), [Dynamo](https://github.com/ai-dynamo/dynamo), [Shopping MMLU](https://github.com/KL4805/ShoppingMMLU).
