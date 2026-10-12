"""Reject incomplete or improperly cache-busted AIPerf points."""

import argparse
import json
import re
from pathlib import Path


MARKER = re.compile(r"^\[rid:[0-9a-fA-F]{12}\]\n\n")
SLO_MS = 1000
REQUIRED_ATTAINMENT = 0.95


def validate(work_dir: Path, role: str, rate: int, *, point: Path | None = None,
             test_jsonl: Path | None = None, system_file: Path | None = None,
             hourly_usd: float | None = None) -> dict:
    point = point or work_dir / "artifacts" / role / f"rps-{rate}"
    report = json.loads((point / "profile_export_aiperf.json").read_text())
    if report.get("is_complete") is not True or report.get("was_cancelled") is not False:
        raise ValueError("AIPerf export is incomplete or cancelled")
    duration = report["benchmark_duration"]["avg"]
    if not 295 <= duration <= 310:
        raise ValueError(f"Expected approximately 300 measurement seconds, got {duration}")

    system = (system_file or work_dir / "system.txt").read_text().strip()
    with (test_jsonl or work_dir / "test.jsonl").open() as input_file:
        questions = {json.loads(line)["question"] for line in input_file}
    if len(questions) != 500:
        raise ValueError("Expected 500 distinct held-out questions")
    markers: set[str] = set()
    observed_questions: set[str] = set()
    cached = []
    rows = 0
    with (point / "profile_export_raw.jsonl").open() as raw_file:
        for line in raw_file:
            record = json.loads(line)
            if record.get("metadata", {}).get("benchmark_phase") != "profiling":
                continue
            rows += 1
            payload = record.get("payload", {})
            messages = payload.get("messages", [])
            if len(messages) != 2 or messages[0] != {"role": "system", "content": system}:
                raise ValueError(f"Request {rows}: system prompt changed")
            if messages[1].get("role") != "user" or not isinstance(messages[1].get("content"), str):
                raise ValueError(f"Request {rows}: user message missing")
            user = messages[1]["content"]
            marker = MARKER.match(user)
            if marker is None or user[marker.end():] not in questions:
                raise ValueError(f"Request {rows}: tag missing or question changed")
            markers.add(marker.group())
            observed_questions.add(user[marker.end():])
            if (payload.get("temperature"), payload.get("seed"), payload.get("max_tokens"),
                    payload.get("chat_template_kwargs")) != (0, 42, 16, {"enable_thinking": False}):
                raise ValueError(f"Request {rows}: decoding settings changed")
            if record.get("status") == 200 and record.get("responses"):
                response = json.loads(record["responses"][0]["text"])
                count = response.get("usage", {}).get("prompt_tokens_details", {}).get("cached_tokens")
                if count is not None:
                    cached.append(count)
                    if count > 32:
                        raise ValueError(f"Request {rows}: {count} cached tokens (>32 shared prefix)")

    issued = report["request_count"]["avg"]
    if rows < 100 or len(markers) != rows or len(observed_questions) != 500:
        raise ValueError("Profiling records, unique markers, or question coverage are incomplete")
    if abs(issued - rows) > 1 or len(cached) < 100:
        raise ValueError("Issued traffic or cache observations missing from export")
    good_rps = report["goodput"]["avg"]
    attainment = report["good_request_fraction"]["avg"]
    if not 0 <= attainment <= 1 or good_rps <= 0:
        raise ValueError("Invalid goodput or SLO attainment in AIPerf export")
    if hourly_usd is not None and hourly_usd <= 0:
        raise ValueError("Hourly G7e cost must be positive")
    return {"role": role, "offered_rps": rate, "issued_requests": rows,
            "good_rps": good_rps,
            "slo_attainment": attainment,
            "slo_pass": attainment >= REQUIRED_ATTAINMENT,
            "slo_ms": SLO_MS,
            "required_attainment": REQUIRED_ATTAINMENT,
            "p95_ms": report["request_latency"]["p95"],
            "max_cached_tokens": max(cached), "question_coverage": len(observed_questions),
            "g7e_cost_per_1000_good_requests_usd":
                hourly_usd * 1000 / (good_rps * 3600) if hourly_usd is not None else None}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("work_dir", type=Path)
    parser.add_argument("role", choices=("e4b_only", "capable_only", "switchyard"))
    parser.add_argument("rate", type=int)
    parser.add_argument("--artifact-dir", type=Path, help="Read an existing point from this directory")
    parser.add_argument("--test-jsonl", type=Path)
    parser.add_argument("--system-file", type=Path)
    parser.add_argument("--g7e-hourly-usd", type=float)
    parser.add_argument("--require-slo-pass", action="store_true",
                        help="Exit nonzero on a valid point that misses the 95%% / 1s SLO")
    args = parser.parse_args()
    result = validate(args.work_dir, args.role, args.rate,
                      point=args.artifact_dir, test_jsonl=args.test_jsonl,
                      system_file=args.system_file, hourly_usd=args.g7e_hourly_usd)
    print(json.dumps(result, indent=2))
    if args.require_slo_pass and not result["slo_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
