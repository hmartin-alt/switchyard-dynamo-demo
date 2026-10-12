"""Offline checks for SLO decisions and instance-cost calculations."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "aiperf_validator", ROOT / "scripts/aiperf/validate-point.py"
)
VALIDATOR = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(VALIDATOR)
NODE_SPEC = importlib.util.spec_from_file_location(
    "benchmark_nodes", ROOT / "scripts/aiperf/check-nodes.py"
)
NODES = importlib.util.module_from_spec(NODE_SPEC)
assert NODE_SPEC.loader is not None
NODE_SPEC.loader.exec_module(NODES)


class AiperfReportingTests(unittest.TestCase):
    def make_point(self, work_dir: Path, attainment: float) -> Path:
        questions = [f"Question {index}" for index in range(500)]
        (work_dir / "system.txt").write_text("Fixed system prompt")
        (work_dir / "test.jsonl").write_text("".join(
            json.dumps({"question": question}) + "\n" for question in questions
        ))
        point = work_dir / "artifacts/switchyard/rps-1"
        point.mkdir(parents=True)
        (point / "profile_export_aiperf.json").write_text(json.dumps({
            "is_complete": True,
            "was_cancelled": False,
            "benchmark_duration": {"avg": 300},
            "request_count": {"avg": 500},
            "goodput": {"avg": 2.0},
            "good_request_fraction": {"avg": attainment},
            "request_latency": {"p95": 900},
        }))
        with (point / "profile_export_raw.jsonl").open("w") as output:
            for index, question in enumerate(questions):
                output.write(json.dumps({
                    "metadata": {"benchmark_phase": "profiling"},
                    "payload": {
                        "messages": [
                            {"role": "system", "content": "Fixed system prompt"},
                            {"role": "user", "content": f"[rid:{index:012x}]\n\n{question}"},
                        ],
                        "temperature": 0,
                        "seed": 42,
                        "max_tokens": 16,
                        "chat_template_kwargs": {"enable_thinking": False},
                    },
                    "status": 200,
                    "responses": [{"text": json.dumps({"usage": {
                        "prompt_tokens_details": {"cached_tokens": 32}
                    }})}],
                }) + "\n")
        return point

    def test_passing_point_and_cost(self):
        with tempfile.TemporaryDirectory() as directory:
            work_dir = Path(directory)
            self.make_point(work_dir, 0.95)
            result = VALIDATOR.validate(work_dir, "switchyard", 1, hourly_usd=14.4)
            self.assertTrue(result["slo_pass"])
            self.assertEqual(result["question_coverage"], 500)
            self.assertAlmostEqual(result["g7e_cost_per_1000_good_requests_usd"], 2.0)

    def test_valid_slo_failure_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            work_dir = Path(directory)
            self.make_point(work_dir, 0.949)
            result = VALIDATOR.validate(work_dir, "switchyard", 1)
            self.assertFalse(result["slo_pass"])
            self.assertIsNone(result["g7e_cost_per_1000_good_requests_usd"])

    def test_frozen_node_types(self):
        def node(name, workload, instance_type, gpus):
            return {
                "metadata": {"name": name, "labels": {
                    "workload": workload,
                    "node.kubernetes.io/instance-type": instance_type,
                }},
                "spec": {},
                "status": {"allocatable": {"nvidia.com/gpu": str(gpus)},
                           "conditions": [{"type": "Ready", "status": "True"}]},
            }
        nodes = [node("gpu", "model-serving", "g7e.12xlarge", 2),
                 node("cpu", "aiperf", "c7i.2xlarge", 0)]
        self.assertEqual(NODES.check(nodes)["model-serving"]["node"], "gpu")
        nodes[0]["status"]["allocatable"]["nvidia.com/gpu"] = "1"
        with self.assertRaises(ValueError):
            NODES.check(nodes)


if __name__ == "__main__":
    unittest.main()
