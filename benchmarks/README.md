# valiDrops Performance Package 1 — benchmark harness

Deterministic, spawn-isolated micro-benchmarks for the exact-CPU optimization
work. Everything here lives **outside the installed package** and writes
artifacts only under the gitignored `benchmark-results/`.

## Quick start

```bash
# Schema/smoke check (one repetition, allowed only for --profile smoke)
rtk uv run python -m benchmarks.run --benchmark cluster-means --implementation baseline \
    --profile smoke --output-dir benchmark-results/package1/smoke --repetitions 1

# Tier-S baseline (this is the "before" artifact)
rtk uv run python -m benchmarks.run --benchmark cluster-means --implementation baseline \
    --profile S --output-dir benchmark-results/package1/baseline
rtk uv run python -m benchmarks.run --benchmark graph-conversion --implementation baseline \
    --profile S --output-dir benchmark-results/package1/baseline
rtk uv run python -m benchmarks.run --benchmark tier-s --implementation baseline \
    --profile S --output-dir benchmark-results/package1/baseline --repetitions 3
```

Each command writes `<benchmark>-<implementation>.json` (schema version 1) plus a
Markdown summary to `--output-dir`. Repetitions default to 7 for the two
kernels and 3 for the pipeline tiers; `--repetitions 1` is only allowed for
`smoke` (schema checks).

## Families

| Family | What is timed | Baseline implementation | Current implementation |
|---|---|---|---|
| `cluster-means` | `expm1` column means of a sparse matrix | dense float64 oracle (cast to float64 before `expm1`, float64 mean) | `_sparse_expm1_col_mean` (sparse reduction, no `todense()`) |
| `graph-conversion` | deep-clustering resolution search (`louvain` per resolution) | `louvain(adjacency, res)` — igraph built per call | `prepare_louvain_graph` once + `louvain_prepared` per call |
| `tier-s` | full Stage-3 `expression_metrics` on deterministic synthetic data | current production code | same entry point, run after the optimization |
| `tier-m-pbmc` | full Stage-3 on a bounded real PBMC4K subset | current production code | same, with a configured run |

`tier-m-pbmc` detects the fixture at `tests/data/pbmc4k/raw.h5` (rooted at the
repository, never the CWD). With `--pbmc-override no`, or when the fixture is
absent, the artifact reports `available: false`, the expected path, and
correctness `skipped` — it never fabricates a run.

Honest naming: `tier-s` and `tier-m-pbmc` are **Stage-3 workloads** (barcode QC
stages excluded); thread experiments on them say nothing about Stage-4 ridge
oversubscription.

## Measurement protocol

- Every measured repetition runs in a fresh `multiprocessing("spawn")` child.
- Warm-up runs inside the child, untimed, before the measured run.
- Timing: `time.perf_counter()`; RSS: `resource.getrusage(RUSAGE_SELF)` high
  water read inside the child (bytes on macOS, KiB normalized to bytes on
  Linux), so native numpy/scipy/igraph allocations are included.
- Report median, minimum, median-absolute deviation, raw samples, and
  coefficient of variation. A CV above 10% is flagged inconclusive, never
  hidden.
- Inputs are regenerated deterministically from `--seed` in each child; a
  warm-up/timed fingerprint equality check doubles as the determinism gate.
- Artifacts record git revision/dirty state, OS/architecture, logical and
  physical CPU counts (when discoverable), available memory, Python/package
  versions, NumPy BLAS config, effective native-thread environment variables,
  input fingerprints and exact parameters, source and harness file hashes.
- Incompatible comparisons (different benchmark, profile, or parameters) are
  rejected by `compare_artifacts`.

## Thread-control experiments (Package 1)

Decision gate: do **not** add `threadpoolctl` or alter package runtime behavior
in Package 1. These experiments only **document** the effect of native thread
limits on the Stage-3 workload so a future Stage-4 plan can decide whether a
runtime dependency is justified.

First start a fresh interpreter with the native thread env vars set **before**
NumPy/SciPy import (artifact `environment.thread_environment` records what the
child inherited):

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  rtk uv run python -m benchmarks.run --benchmark tier-s --implementation current \
  --profile S --output-dir benchmark-results/package1/threads/1 --repetitions 3
```

Repeat with limits 2 and 4 (`threads/2`, `threads/4`). If Tier-S is too slow on
the machine, run `smoke` first and label `S` as unavailable rather than
extrapolating. Conclusions from these Stage-3 measurements must not be
extrapolated to Stage-4 ridge oversubscription.
