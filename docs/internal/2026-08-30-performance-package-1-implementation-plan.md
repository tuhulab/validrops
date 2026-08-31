# Performance Package 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Follow superpowers:test-driven-development for every production-code change and superpowers:verification-before-completion before reporting success. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish reproducible performance measurement and deliver two low-risk, exact CPU optimizations: sparse cluster means and one-time igraph conversion during Stage-3 resolution search.

**Architecture:** Keep performance tooling outside the installed package in `benchmarks/`. Each timed repetition runs in a fresh spawned process so elapsed time and process peak RSS include native NumPy/SciPy/igraph allocations without lifetime-high-water contamination. Production changes remain narrow internal helpers: sparse `expm1` column means in `expression_metrics.py`, and explicit sparse-to-igraph preparation plus prepared-graph Louvain in `_snn.py`.

**Tech Stack:** Python 3.11+, NumPy, SciPy sparse, AnnData, python-igraph, scikit-learn, pytest, standard-library `argparse`, `multiprocessing`, `resource`, `statistics`, `json`, and `platform`. Do not add a runtime dependency in this package.

**Spec:** `docs/internal/performance-optimization-design.md` and `docs/internal/performance-validation-contract.md`.

## Global Constraints

- Treat the R source and current reference fixtures as authoritative.
- Preserve public signatures, defaults, return values, AnnData keys, categories, warnings, exceptions, fixed-seed behavior, float64 reductions, positional gene indexing, constants, and documented R quirks.
- Do not change fast `Sn`, Wilcoxon/AUC, dense SVD, Stage-4 ridge code, neighbor search, public backend parameters, RAPIDS, or any approximate behavior.
- Do not stage, commit, or otherwise add `docs/internal/**` to Git. The three design/plan documents are untracked implementation inputs.
- Do not stage or commit any file during this execution. Generated benchmark artifacts belong under the gitignored `benchmark-results/` directory.
- Use `rtk` before every shell command, as required by the project instructions.
- Run every production change test-first: add a focused test, run it and confirm the expected failure, make the smallest implementation change, then rerun it to green.
- Do not claim a performance improvement without baseline and candidate artifacts produced on the same machine and profile. Report missing PBMC4K as unavailable, never as zero or pass.

---

## Task 1: Add the deterministic benchmark and artifact foundation

**Files:**

- Create: `benchmarks/__init__.py`
- Create: `benchmarks/_datasets.py`
- Create: `benchmarks/_measure.py`
- Create: `benchmarks/run.py`
- Create: `benchmarks/README.md`
- Create: `tests/test_benchmarks.py`
- Modify: `.gitignore`

- [ ] Add `/benchmark-results/` to `.gitignore`. Do not add `docs/internal/` to a tracked file; simply leave it untracked and never stage it.

- [ ] Write failing tests in `tests/test_benchmarks.py` for deterministic direct-sparse generation:

```python
def test_tier_s_generator_is_deterministic_and_sparse():
    first, first_meta = make_sparse_counts("S", seed=0)
    second, second_meta = make_sparse_counts("S", seed=0)
    assert sp.isspmatrix_csr(first)
    assert first.shape == (500, 2_000)
    assert first.dtype == np.float32
    assert (first != second).nnz == 0
    assert first_meta == second_meta
```

Also test the small CI-only `smoke` profile, fixed-seed clustered embeddings, and the fact that generators never return a dense matrix. Generate CSR data directly, row by row, with fixed non-zero counts and no dense `cells x genes` intermediate. Include metadata for cells, genes, realized density, dtype, seed, simulated clusters, and duplicate-name pattern.

- [ ] Run `rtk uv run pytest tests/test_benchmarks.py -v` and confirm the import/collection failure before implementation.

- [ ] Implement `benchmarks/_datasets.py` with these profiles:

```python
PROFILES = {
    "smoke": {"cells": 60, "genes": 120, "density": 0.05},
    "S": {"cells": 500, "genes": 2_000, "density": 0.02},
    "M-synthetic": {"cells": 4_000, "genes": 20_000, "density": 0.01},
    "L": {"cells": 10_000, "genes": 25_000, "density": 0.005},
    "XL-50": {"cells": 50_000, "genes": 30_000, "density": 0.002},
    "XL-100": {"cells": 100_000, "genes": 30_000, "density": 0.001},
}
```

Only `smoke` and `S` are exercised automatically. Larger profiles are explicit manual choices. Provide a deterministic `make_clustered_embedding` for graph benchmarks. Any duplicate gene names are metadata/AnnData labels only; matrix operations remain positional.

- [ ] Write failing schema and measurement tests. A one-repetition smoke run must produce a dict with this exact top-level contract:

```python
{
    "schema_version": 1,
    "benchmark": "cluster_means",
    "revision": "<sha or unavailable>",
    "dirty": True,
    "environment": {},
    "input": {},
    "implementation": {},
    "timing_seconds": {
        "median": 0.0,
        "minimum": 0.0,
        "mad": 0.0,
        "coefficient_of_variation": 0.0,
        "repetitions": 1,
        "raw": [0.0],
    },
    "peak_rss_bytes": {
        "worker_max": 0,
        "raw": [0],
        "method": "resource.getrusage",
        "scope": "worker_self",
    },
    "correctness": {"status": "pass", "metrics": {}},
    "warnings": [],
}
```

Assert positive elapsed time and peak RSS, JSON round-trip, and a Markdown summary. Validate PBMC4K detection at `tests/data/pbmc4k/raw.h5`; absence must yield an explicit `available: false`, expected path, and skip warning.

- [ ] Run the focused tests and confirm they fail for missing measurement/report functionality.

- [ ] Implement `benchmarks/_measure.py` and `benchmarks/run.py`:

  - each measured sample executes in a fresh `multiprocessing.get_context("spawn")` child; inside that child, construct inputs and import the target outside the timer, warm the exact kernel once, restore mutable input state, and then time it;
  - worker targets are top-level picklable functions behind an `if __name__ == "__main__"` guard, with controller timeouts, exit-code checks, and child-traceback propagation;
  - measure with `time.perf_counter()`;
  - read `resource.getrusage(resource.RUSAGE_SELF).ru_maxrss` in the child and normalize macOS bytes versus Linux KiB to bytes;
  - report median, minimum, median absolute deviation, repetitions, coefficient of variation, and maximum child peak RSS;
  - record Git revision/dirty state, hashes of relevant source and harness files, generator/schema version, input fingerprints, exact parameters, OS/architecture, physical/logical CPU counts where discoverable, available memory, Python/package versions, NumPy BLAS configuration summary, effective thread environment variables, input metadata, implementation/component labels, and PBMC4K availability;
  - label Package-1 memory as `worker_self` and single-process; unsupported RSS platforms emit `unavailable`, never zero, and no process-tree claim is made;
  - use seven repetitions by default for kernels and three for Tier-S; `--repetitions 1` is allowed only for smoke/schema checks;
  - write `<benchmark>-<implementation>.json` and `.md` to `--output-dir`, defaulting to `benchmark-results/`;
  - expose `python -m benchmarks.run --benchmark {cluster-means,graph-conversion,stage3-s,tier-m-pbmc} --implementation {baseline,current} --profile {smoke,S,...}`;
  - compare artifacts only when schema/generator versions, workload parameters and fingerprints, hardware/thread configuration, RSS method/scope, and harness measurement code are compatible;
  - lazily import optional candidate helpers so the baseline harness runs before production helpers exist.

- [ ] Make `cluster-means/baseline` reproduce the current dense expression exactly. Make `graph-conversion/baseline` repeatedly call the current `louvain(adjacency, ...)`. Make `stage3-s` build an AnnData with deterministic positional gene labels, `qc_pass=True`, and `uns["validrops"]["gene_sets"]`, then call `expression_metrics` with bounded smoke/S parameters. Correctness metadata must record output shape/counts, not make an R-concordance or full-pipeline claim. Root PBMC4K detection at the repository directory and test present/absent override states; when safe and practical, `tier-m-pbmc` records the representative fixture workload, otherwise it emits an explicit deferred/unavailable artifact.

- [ ] Run `rtk uv run pytest tests/test_benchmarks.py -v` to green, then `rtk uvx ruff check benchmarks tests/test_benchmarks.py` and `rtk uvx ruff format --check benchmarks tests/test_benchmarks.py`.

- [ ] Capture pre-optimization baselines on the current implementation:

```bash
rtk uv run python -m benchmarks.run --benchmark cluster-means --implementation baseline --profile S --output-dir benchmark-results/package1/baseline
rtk uv run python -m benchmarks.run --benchmark graph-conversion --implementation baseline --profile S --output-dir benchmark-results/package1/baseline
rtk uv run python -m benchmarks.run --benchmark stage3-s --implementation baseline --profile S --output-dir benchmark-results/package1/baseline --repetitions 3
```

Confirm artifacts exist and record high timing variability as inconclusive rather than hiding it.

## Task 2: Replace dense cluster means with an exact sparse reduction

**Files:**

- Modify: `tests/test_expression_metrics.py`
- Modify: `src/validrops/tl/expression_metrics.py`
- Modify: `benchmarks/run.py`

- [ ] Add a focused test for a private helper named `_sparse_expm1_col_mean`. The matrix must include an all-zero gene, a single non-zero value, tied non-zero values, unequal zero fractions, float32 storage, and a sliced non-canonical CSR input. Compare to a float64 dense oracle:

```python
expected = np.expm1(matrix.toarray().astype(np.float64)).mean(axis=0, dtype=np.float64)
actual = _sparse_expm1_col_mean(matrix)
np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
assert actual.dtype == np.float64
```

Also construct a genuinely noncanonical CSR matrix with duplicate coordinates. Assert the input sparse matrix's `data`, `indices`, `indptr`, shape, format, sorted-index flag, and canonical-format flag are unchanged.

- [ ] Run only the new test and confirm it fails because the helper is absent.

- [ ] Implement the minimal helper in `expression_metrics.py`:

```python
def _sparse_expm1_col_mean(matrix: sp.spmatrix) -> np.ndarray:
    transformed = matrix.tocsr(copy=True).astype(np.float64, copy=False)
    transformed.sum_duplicates()
    transformed.data = np.expm1(transformed.data)
    return np.asarray(transformed.sum(axis=0, dtype=np.float64)).ravel() / transformed.shape[0]
```

Reject zero-row input with a clear `ValueError`; `_cluster_stats` should never pass it because it already skips empty target/rest groups.

- [ ] Replace only the two dense mean expressions in `_cluster_stats` with the helper. Do not change percentage, eligibility, Wilcoxon, fold-change, or output-table logic.

- [ ] Run the new unit test, all `tests/test_expression_metrics.py`, and applicable reference tests. If PBMC4K is unavailable, report the skip exactly.

- [ ] Extend `cluster-means/current` to call `_sparse_expm1_col_mean` and compare every output value with the dense baseline. Correctness status must fail on any mismatch beyond the test tolerance.

- [ ] Run matching baseline/current S benchmarks and produce artifacts under `benchmark-results/package1/{baseline,current}`. A merge-quality performance claim needs at least 1.25x median speedup or 25% peak-RSS reduction; otherwise retain the exact change only if the dense-allocation removal is clearly demonstrated and explain the measured crossover.

## Task 3: Convert the SNN adjacency to igraph once per resolution search

**Files:**

- Modify: `tests/test_snn.py`
- Modify: `tests/test_expression_metrics.py`
- Modify: `src/validrops/tl/_snn.py`
- Modify: `src/validrops/tl/expression_metrics.py`
- Modify: `benchmarks/run.py`

- [ ] Write failing tests for explicit internal functions `prepare_louvain_graph(adjacency)` and `louvain_prepared(graph, resolution, random_state=0)`:

  - wrapper `louvain(adjacency, ...)` and prepared path return identical labels for several resolutions and seeds;
  - a pre-refactor golden case using disconnected cliques of sizes two then four returns exactly `[1, 1, 0, 0, 0, 0]` at resolution 1 and seed 0, protecting descending-size label remapping from a tautological wrapper comparison;
  - graph vertex/edge counts and weights match the upper triangle of the sparse adjacency;
  - repeated prepared calls are deterministic and do not mutate edge weights;
  - monkeypatch `prepare_louvain_graph` in `expression_metrics` and confirm a Stage-3 run without supplied clusters converts exactly once while the resolution sweep invokes prepared Louvain multiple times; remove the `louvain` binding from `expression_metrics` and make any accidental wrapper call raise so hidden reconversions cannot escape the counter;
  - exercise the actual varying-resolution sweep order and assert after every call that edge order/weights, vertex and edge counts, undirectedness, and isolated vertices are unchanged.

- [ ] Run the focused tests and confirm expected missing-symbol/call-count failures.

- [ ] Refactor `_snn.py` without changing the public internal wrapper contract:

```python
def prepare_louvain_graph(adjacency: sp.spmatrix) -> ig.Graph:
    coo = sp.triu(adjacency.tocoo(), k=1).tocoo()
    graph = ig.Graph(n=adjacency.shape[0], edges=list(zip(coo.row.tolist(), coo.col.tolist(), strict=True)))
    graph.es["weight"] = coo.data.tolist()
    return graph


def louvain_prepared(graph: ig.Graph, resolution: float, random_state: int = 0) -> np.ndarray:
    random.seed(random_state)
    clustering = graph.community_multilevel(weights="weight", resolution=resolution)
    # Preserve the current descending-cluster-size label remap exactly.
    ...


def louvain(adjacency: sp.spmatrix, resolution: float, random_state: int = 0) -> np.ndarray:
    return louvain_prepared(prepare_louvain_graph(adjacency), resolution, random_state)
```

- [ ] In `expression_metrics`, call `prepare_louvain_graph(graph)` once after `snn_graph`; use `louvain_prepared` for shallow clustering and pass the prepared graph into `_deep_clustering`. Update `_deep_clustering` to accept the prepared graph and make all coarse/fine calls through `louvain_prepared`. Apply the same one-conversion rule when `clusters` is supplied.

- [ ] Run `tests/test_snn.py`, `tests/test_expression_metrics.py`, and the full fast test suite. Existing seed labels, shallow ARI threshold, and known deep-clustering ceiling are acceptance constraints, not targets to revise.

- [ ] Extend `graph-conversion/current` to prepare once and reuse the graph across the exact same resolution list as baseline. Record label equality/ARI=1.0 and unchanged graph edges/weights.

- [ ] Run matching baseline/current graph and Tier-S benchmarks under `benchmark-results/package1/{baseline,current}`.

## Task 4: Run thread-control experiments without changing runtime behavior

**Files:**

- Modify: `benchmarks/README.md`
- Generated only: `benchmark-results/package1/threads/**`

- [ ] Document exact controlled commands that start a fresh interpreter with native thread environment variables set before NumPy/SciPy import:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
rtk uv run python -m benchmarks.run --benchmark stage3-s --implementation current --profile S --output-dir benchmark-results/package1/threads/1 --repetitions 3
```

Repeat with limits 2 and 4. Record the effective environment in every artifact. Do not dynamically mutate these variables after NumPy import.

- [ ] Run the 1/2/4 experiments on the same machine. Compare runtime, variability, and peak RSS. If Tier-S is too slow, run `smoke` first and clearly label S as unavailable/incomplete; do not extrapolate.

- [ ] Decision gate: do not add `threadpoolctl` or alter package runtime behavior in Package 1. These experiments measure native Stage-3 threading only and cannot establish whether joblib plus BLAS oversubscription exists in Stage 4. A future runtime dependency is justified only if a separate Stage-4 plan measures and defines process/inner-thread semantics. Apply `CV > 10%` as inconclusive/rerun and record this limited conclusion in the generated Markdown report and `benchmarks/README.md`.

## Task 5: Complete Package-1 verification and scope audit

**Files:**

- Review all changed tracked files
- Generated only: `benchmark-results/package1/package1-summary.md`

- [ ] Review `rtk git diff -- . ':(exclude)docs/internal/**'` for accidental API, numerical, dependency, or scope changes. Confirm `docs/internal/**` remains untracked and nothing is staged.

- [ ] Run fresh targeted verification:

```bash
rtk uv run pytest tests/test_benchmarks.py tests/test_snn.py tests/test_expression_metrics.py -v
rtk uv run pytest tests/ -v
rtk uvx ruff check src/validrops benchmarks tests
rtk uvx ruff format --check src/validrops benchmarks tests
rtk uv run python -m benchmarks.run --benchmark cluster-means --implementation current --profile smoke --output-dir benchmark-results/package1/smoke --repetitions 1
rtk uv run python -m benchmarks.run --benchmark graph-conversion --implementation current --profile smoke --output-dir benchmark-results/package1/smoke --repetitions 1
```

- [ ] If the PBMC4K raw fixture exists, run the applicable slow/reference tests. If absent, list the exact missing path and skipped checks.

- [ ] Create the generated summary using the validation-contract reporting template. Include baseline/current runtime and RSS, correctness gates, variability, unavailable tiers, thread-control conclusion, and exact-versus-approximate classification. Keep it under ignored `benchmark-results/`, not `docs/internal/`.

- [ ] Final Git-state check:

```bash
rtk git status --short --untracked-files=all
rtk git diff --cached --name-only
```

The cached diff must be empty. `docs/internal/README.md`, both approved specs, and this plan must remain untracked. Do not commit.

## Completion Criteria

- Benchmark smoke/schema tests pass and generated results remain ignored.
- Sparse cluster means equal the dense float64 oracle, do not mutate inputs, and remove the `todense()` calls from that calculation.
- Prepared and wrapper Louvain paths return identical fixed-seed labels; Stage 3 constructs igraph once per run.
- All existing fast tests pass; available reference tests preserve current thresholds and known ceilings.
- No runtime dependency, public parameter, approximate behavior, or out-of-scope algorithm is introduced.
- Thread experiments are reported as measurements or explicitly unavailable, never inferred.
- Nothing is staged or committed, and every `docs/internal/**` file remains outside Git.
