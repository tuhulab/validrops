"""Command-line entry point for the Package-1 benchmark harness.

Example::

    rtk uv run python -m benchmarks.run --benchmark cluster-means --implementation baseline \
        --profile S --output-dir benchmark-results/package1/baseline

The ``if __name__ == "__main__"`` guard is mandatory: every timed repetition
spawns a fresh child, and the child must not re-run the CLI.
"""

from __future__ import annotations

import argparse
import json
import sys

from benchmarks import _measure
from benchmarks._datasets import PROFILES


def main(argv: list[str] | None = None) -> int:
    """Run one benchmark family from the command line and write artifacts."""
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.run",
        description="Spawn-isolated benchmark harness for valiDrops validation (data CPO).",
    )
    parser.add_argument("--benchmark", choices=sorted(_measure.KERNEL_NAMES), required=True)
    parser.add_argument("--implementation", choices=["baseline", "current"], required=True)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="S")
    parser.add_argument("--repetitions", type=int, default=None, help="default per benchmark family")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="benchmark-results")
    parser.add_argument("--timeout", type=float, default=300.0, help="per-repetition child timeout (seconds)")
    parser.add_argument("--pbmc-override", choices=["auto", "yes", "no"], default="auto")
    parser.add_argument("--pbmc-path", default=None)
    args = parser.parse_args(argv)

    artifact = _measure.run_benchmark(
        benchmark=args.benchmark,
        implementation=args.implementation,
        profile=args.profile,
        seed=args.seed,
        repetitions=args.repetitions,
        timeout=args.timeout,
        output_dir=args.output_dir,
        pbmc_override=args.pbmc_override,
        pbmc_path=args.pbmc_path,
    )
    summary = {
        "benchmark": artifact["benchmark"],
        "implementation": artifact["implementation"]["label"],
        "profile": artifact["input"]["profile"],
        "revision": artifact["revision"],
        "median_seconds": artifact["timing_seconds"]["median"],
        "worker_max_rss_bytes": artifact["peak_rss_bytes"]["worker_max"],
        "correctness": artifact["correctness"]["status"],
    }
    print(json.dumps(summary, indent=2, default=str))
    for warning in artifact["warnings"]:
        print(f"WARNING: {warning}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
