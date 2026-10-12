"""Validate a complete three-arm cache-busted sweep and calculate G7e cost."""

import argparse
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "aiperf_validate_point", Path(__file__).with_name("validate-point.py")
)
VALIDATOR = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(VALIDATOR)


def summarize(work_dir: Path, protocol: dict) -> dict:
    hourly_usd = protocol["g7e_hourly_usd_for_frozen_comparison"]
    arms = {}
    for role, rates in protocol["rates"].items():
        points = []
        for rate in rates:
            point_dir = work_dir / "artifacts" / role / f"rps-{rate}"
            if not point_dir.is_dir():
                if points and not points[-1]["slo_pass"]:
                    break  # The runner stops after the first valid SLO failure.
                raise ValueError(f"Missing planned point: {point_dir}")
            points.append(VALIDATOR.validate(work_dir, role, rate,
                                             hourly_usd=hourly_usd))
        passing = [point for point in points if point["slo_pass"]]
        if not passing:
            raise ValueError(f"No SLO-passing point for {role}")
        selected = max(passing, key=lambda point: point["good_rps"])
        higher_failures = [point for point in points if not point["slo_pass"]
                           and point["offered_rps"] > selected["offered_rps"]]
        lower_failures = [point for point in points if not point["slo_pass"]
                          and point["offered_rps"] < selected["offered_rps"]]
        if lower_failures:
            raise ValueError(f"Non-monotonic SLO results for {role}; repeat boundary points")
        arms[role] = {
            "points": points,
            "selected_highest_tested_passing_goodput": selected,
            "first_higher_tested_failure": min(
                higher_failures, key=lambda point: point["offered_rps"]
            ) if higher_failures else None,
            "capacity_bounded_by_tested_failure": bool(higher_failures),
        }
    capable = arms["capable_only"]["selected_highest_tested_passing_goodput"]
    switchyard = arms["switchyard"]["selected_highest_tested_passing_goodput"]
    return {
        "protocol": protocol["version"],
        "test_sha256": protocol["test_sha256"],
        "checkpoint_sha256": protocol["checkpoint_sha256"],
        "g7e_hourly_usd": hourly_usd,
        "arms": arms,
        "switchyard_vs_capable_only": {
            "goodput_ratio": switchyard["good_rps"] / capable["good_rps"],
            "estimated_g7e_cost_reduction_fraction": 1 - (
                switchyard["g7e_cost_per_1000_good_requests_usd"]
                / capable["g7e_cost_per_1000_good_requests_usd"]
            ),
        },
        "scope": "Highest tested passing points, not exact capacity limits. "
                 "G7e instance cost only; excludes CPU load generator, EKS, "
                 "storage, network, and idle time. Accuracy is measured separately.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work_dir", type=Path)
    parser.add_argument("--protocol", type=Path, default=ROOT / "benchmark/protocol.json")
    parser.add_argument("--output", type=Path, help="Save report to a new JSON file")
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise SystemExit("Refusing to overwrite an existing report")
    protocol = json.loads(args.protocol.read_text())
    result = summarize(args.work_dir, protocol)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
