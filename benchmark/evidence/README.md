# Original AIPerf evidence

These archives contain the original AIPerf 0.13.0 summary and raw request export for each quoted selected point, plus the next tested SLO failures for 31B-only and Switchyard. They contain public Shopping MMLU questions and generated answers; request headers were checked for credential fields before packaging. The five archives total about 80 MB. They do **not** include every intermediate rate, Kubernetes logs, or an AWS bill.

| Archive | Source run | Offered RPS | Result | SHA-256 of archive |
| --- | --- | ---: | --- | --- |
| `e4b-rps-768.tar.zst` | 2026-10-09 cache-bust | 768 | Pass; no higher rate measured | `d82648dc9479cc845435874743f363651bae697ddf30c033e0babbe596448222` |
| `capable-rps-96.tar.zst` | 2026-10-09 cache-bust | 96 | Pass | `ba178b58eec9019a5658acc28afa21a209ab3fc9ec729834cfb86153e97b2e19` |
| `capable-rps-112.tar.zst` | 2026-10-09 cache-bust | 112 | Fail | `abd1f6439af146bd7a90b9e40c99bebded705e85ab9eefb593224f7e29a66435` |
| `switchyard-rps-160.tar.zst` | 2026-10-11 split-view router | 160 | Pass | `03a1d2c62d19cb698e367960f0736b3da6a5e2136160d29bdbf2cdedad7ddf7d` |
| `switchyard-rps-192.tar.zst` | 2026-10-11 split-view router | 192 | Fail | `bab5ceba801c31de0e1fbd6efbb193fa12f6a4ad70c3603d4d71d611aefa441f` |

To audit the published points, extract each archive into a fresh prepared work directory (requires `zstd`), then run the summarizer with `benchmark/evidence-protocol.json`:

```bash
python3 scripts/aiperf/prepare-inputs.py benchmark/inputs/test.jsonl /path/to/work \
  --expected-sha256 07882e772a3aa492e97747aa88a83a61ca5975810842e88539ee9b07e2427c64
for archive in benchmark/evidence/*.tar.zst; do
  zstd -dc "$archive" | tar -xf - -C /path/to/work
done
python3 scripts/aiperf/summarize-comparison.py /path/to/work \
  --protocol benchmark/evidence-protocol.json
```

The Switchyard point and model-only reference were collected on different days with the same frozen workload, instance size, cache-busting method, and model settings. Do not call these archives a simultaneous, repeated production-cost measurement. The output reports **highest tested passing goodput**, not an exact capacity limit.
