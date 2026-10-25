"""Standalone Reproducible Benchmark Runner for CodeGhost (Milestone 8)."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.orchestration.config import OrchestrationConfig
from backend.orchestration.executor import LocalDirectExecutor


def main():
    parser = argparse.ArgumentParser(description="CodeGhost Reproducible Benchmark Runner")
    parser.add_argument("--output", default="reports/validation/run_latest", help="Directory for run artifacts")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--skip-tests", action="store_true", help="Skip full test suite execution")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = OrchestrationConfig()
    executor = LocalDirectExecutor(config)

    manifest = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python_version": sys.version,
        "random_seed": args.seed,
        "git_commit": executor.run_command("git rev-parse HEAD").stdout.strip(),
        "tests_status": "SKIPPED" if args.skip_tests else "PENDING",
    }

    if not args.skip_tests:
        print("Running full test suite...")
        test_res = executor.run_command(".venv\\Scripts\\python -m pytest tests/ -v")
        manifest["tests_status"] = "PASSED" if test_res.success else "FAILED"
        manifest["test_exit_code"] = test_res.exit_code

    manifest_file = out_dir / "run_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Benchmark run complete. Manifest saved to: {manifest_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
