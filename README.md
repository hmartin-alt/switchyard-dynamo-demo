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

The default manifests disable vLLM prefix caching. For the opt-in cache-busted AIPerf comparison, follow the [benchmark recipe](docs/aiperf-cache-bust.md): enable ordinary prefix caching and hide each synthetic request ID from the router encoder only.

The checkpoint, model weights, credentials, source checkouts, and Kubernetes cluster are **not** committed. This is a reproducible serving recipe, not a turnkey cluster. `patches/` and `scripts/shopping-mmlu/` preserve the Switchyard runtime changes and offline policy workflow; `scripts/aiperf/` contains the benchmark runner and checks. The held-out dataset and benchmark artifacts are not committed.

## Scope of the measured results

Switchyard scored **93.6%** on 500 clean held-out questions versus **95.2%** for 31B-only. A *separate* AIPerf capacity test with first-turn request IDs sent unchanged to Dynamo found **159.63** versus **95.56** SLO-compliant requests/s on the same full G7e size. This implies about **40% lower estimated G7e instance cost per compliant request**. The router alone ignored the synthetic ID during prediction. These are separate quality and performance measurements, not a single production-cost or tagged-prompt accuracy result; the tagged prompts changed five net answers in the diagnostic accuracy run. See the [benchmark recipe](docs/aiperf-cache-bust.md) for controls and limits.

Upstream: [Switchyard](https://github.com/NVIDIA-NeMo/Switchyard), [Dynamo](https://github.com/ai-dynamo/dynamo), [Shopping MMLU](https://github.com/KL4805/ShoppingMMLU).
