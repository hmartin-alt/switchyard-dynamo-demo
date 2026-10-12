"""Prepare the frozen Shopping MMLU pool without changing question text."""

import argparse
import hashlib
import json
from pathlib import Path


SYSTEM_PROMPT = (
    "Answer the shopping multiple-choice question using the question and all four choices. "
    "Return only the letter A, B, C, or D for the best answer. Do not explain."
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("test_jsonl", type=Path, help="Frozen 500-question test.jsonl")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--expected-sha256", help="Optional frozen test file checksum")
    args = parser.parse_args()

    data = args.test_jsonl.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if args.expected_sha256 and digest != args.expected_sha256:
        raise SystemExit("Test file SHA-256 does not match the frozen value")
    rows = [json.loads(line) for line in data.splitlines()]
    if len(rows) != 500 or any(row.get("split") != "test" for row in rows):
        raise SystemExit("Expected exactly 500 held-out test questions")
    if any(not isinstance(row.get("question"), str) or not row["question"] for row in rows):
        raise SystemExit("A test row has no question")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "test.jsonl").write_bytes(data)
    (args.output_dir / "system.txt").write_text(SYSTEM_PROMPT)
    with (args.output_dir / "shopping_mmlu_pool.jsonl").open("w") as target:
        for row in rows:
            target.write(json.dumps({"text": row["question"], "output_length": 16}) + "\n")
    (args.output_dir / "frozen-inputs.json").write_text(
        json.dumps({"test_sha256": digest, "question_count": len(rows),
                    "system_prompt": SYSTEM_PROMPT}, indent=2) + "\n"
    )
    print(f"Prepared 500 questions; test SHA-256: {digest}")


if __name__ == "__main__":
    main()
