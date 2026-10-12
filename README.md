# Switchyard + Dynamo shopping router

This repository is a serving reference for the shopping demo. Switchyard's trained prefill router chooses between Gemma 4 E4B (`efficient`) and Gemma 4 31B (`capable`) for each Shopping MMLU question. Both models have separate Dynamo Graph Deployments (DGDs) on one two-GPU `g7e.12xlarge`. Each GPU runs one model worker and one colocated copy of the Qwen3.5-0.8B prefill encoder/router; a Kubernetes Service exposes the two router replicas.

| GPU allocation | Dynamo Graph Deployment | Colocated processes |
| --- | --- | --- |
| One GPU | `efficient` | Gemma 4 E4B worker + Switchyard prefill router |
| Other GPU | `capable` | Gemma 4 31B worker + Switchyard prefill router |

The manifests allocate one GPU to each worker but do not pin numeric GPU IDs. The two router replicas share the same frozen checkpoint, targets, tolerance, and request settings.

## Deploy on EKS

1. Prepare one `g7e.12xlarge` node labeled `workload=model-serving`, a `dynamo` namespace with the Dynamo operator, and the Hugging Face token Secret. Confirm your `kubectl` context points to that cluster.
2. Prefetch the pinned model snapshots and build the patched Switchyard runtime using the commands in the [deployment guide](docs/shopping-demo.md). Stage the frozen router checkpoint, routes, and both launchers on the same node. Verify the checkpoint hash before deployment.
3. Apply the two DGDs and shared router Service with `bash scripts/deploy-eks.sh YOUR_KUBE_CONTEXT`. The script refuses to deploy unless exactly one node has the serving label.
4. Check that both worker Pods and both Service endpoints are ready. The [deployment guide](docs/shopping-demo.md) gives the exact checks and a test-request path.

The default manifests disable vLLM prefix caching. The [benchmark recipe](docs/aiperf-cache-bust.md) includes the frozen 500-question split, router checkpoint and policy, clean-accuracy command, two-worker E4B and 31B baseline DGDs, cache-busted Switchyard overlay, AIPerf rate schedule, validation/cost calculation, and original selected-point evidence.

The repo can rerun the three-arm benchmark on a prepared EKS cluster and audit the quoted selected points offline. It uses the **already-trained, frozen router checkpoint**; reproducing the benchmark does not require training a new router. The original training labels and intermediate features are not included, so the repo cannot reproduce *this checkpoint's* training from scratch. It is also **not** a turnkey cluster: gated model weights, credentials, infrastructure provisioning, the original clean-accuracy output, and intermediate-rate raw exports are not committed. `patches/` contains the Switchyard runtime changes; `scripts/shopping-mmlu/` documents parts of the offline policy workflow. To train a new prefill router on your own labeled data, see the [NVIDIA model-router toolkit training guide](https://github.com/NVIDIA-AI-Blueprints/llm-router/blob/v3/docs/guide-training-and-evaluation.md); the [Switchyard prefill-router reference](https://nvidia-nemo.github.io/Switchyard/reference/toml_schema/#prefill_router) covers checkpoint configuration, not end-to-end training.

## Scope of the measured results

Switchyard scored **93.6%** on 500 clean held-out questions versus **95.2%** for 31B-only. A *separate* AIPerf capacity test with first-turn request IDs sent unchanged to Dynamo found **159.63** versus **95.56** SLO-compliant requests/s on the same full G7e size. This implies about **40% lower estimated G7e instance cost per compliant request**. The router alone ignored the synthetic ID during prediction. These are separate quality and performance measurements, not a single production-cost or tagged-prompt accuracy result; the tagged prompts changed five net answers in the diagnostic accuracy run. See the [benchmark recipe](docs/aiperf-cache-bust.md) for controls and limits.

Upstream: [Switchyard](https://github.com/NVIDIA-NeMo/Switchyard), [Dynamo](https://github.com/ai-dynamo/dynamo), [Shopping MMLU](https://github.com/KL4805/ShoppingMMLU).
