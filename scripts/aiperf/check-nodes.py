"""Read-only check for the frozen GPU and CPU benchmark node types."""

import argparse
import json
import subprocess


def check(nodes: list[dict]) -> dict:
    expected = {
        "model-serving": ("g7e.12xlarge", 2),
        "aiperf": ("c7i.2xlarge", 0),
    }
    result = {}
    for workload, (instance_type, gpu_count) in expected.items():
        matches = [node for node in nodes if node["metadata"]["labels"].get("workload") == workload]
        if len(matches) != 1:
            raise ValueError(f"Expected one {workload} node, found {len(matches)}")
        node = matches[0]
        actual_type = node["metadata"]["labels"].get("node.kubernetes.io/instance-type")
        actual_gpus = int(node["status"].get("allocatable", {}).get("nvidia.com/gpu", "0"))
        ready = any(condition["type"] == "Ready" and condition["status"] == "True"
                    for condition in node["status"].get("conditions", []))
        if (actual_type, actual_gpus, ready, node["spec"].get("unschedulable", False)) != (
            instance_type, gpu_count, True, False
        ):
            raise ValueError(f"{workload} node has wrong type/GPU/readiness: "
                             f"{actual_type}, {actual_gpus}, ready={ready}")
        result[workload] = {"node": node["metadata"]["name"],
                            "instance_type": actual_type, "allocatable_gpus": actual_gpus}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", required=True, help="Explicit kubectl context to inspect")
    args = parser.parse_args()
    output = subprocess.run(
        ["kubectl", "--context", args.context, "get", "nodes", "-o", "json"],
        check=True, capture_output=True, text=True,
    )
    print(json.dumps(check(json.loads(output.stdout)["items"]), indent=2))


if __name__ == "__main__":
    main()
