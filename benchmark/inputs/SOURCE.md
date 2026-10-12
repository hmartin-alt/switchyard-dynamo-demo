# Frozen benchmark inputs

`test.jsonl` is the exact 500-question held-out split used for the demo. Its questions were selected from the public [Shopping MMLU dataset](https://github.com/KL4805/ShoppingMMLU), which is distributed in an [Apache-2.0 licensed repository](https://github.com/KL4805/ShoppingMMLU/blob/main/LICENSE). The split is deliberately easy-heavy (400 easy and 100 hard questions); it is not representative of production traffic. See `benchmark/protocol.json` for its SHA-256.

`prefill_router.pt` is the frozen, trained routing-head checkpoint used by the demo. It is **not** the Gemma or Qwen model weights; those gated/base-model weights are fetched separately. Verify its SHA-256 before loading it. Like other PyTorch pickle checkpoints, it should only be loaded by a trusted runtime.

`policy.json` records the validation-selected tolerance and the hashes that freeze the checkpoint and held-out split. The training labels and intermediate features are not included, so this package reproduces serving and evaluation, **not router retraining**.
