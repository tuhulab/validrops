# Performance and concordance contract

**Status:** Proposed — awaiting maintainer review

**Date:** 2026-08-30

**Applies to:** All performance-oriented changes described in
[Performance optimization architecture](performance-optimization-design.md)

## Purpose

Performance changes are accepted only when they improve a measured bottleneck and
retain the intended scientific behavior. This document defines the evidence required
before implementation begins and the gates applied to each pull request or local
integration.

The contract separates four questions:

1. Is the optimized kernel mathematically equivalent to its reference?
2. Does the complete stage retain its established R-concordance behavior?
3. Is runtime or peak memory materially better on a controlled workload?
4. Did the optimization cause regressions elsewhere?

## Benchmark structure

Use two layers rather than one large end-to-end timer.

### Kernel benchmarks

Small, deterministic benchmarks isolate the code being changed:

| Kernel | Inputs | Primary measurements |
|---|---|---|
| `sn` | continuous, tied, odd/even, heavy-tailed vectors | runtime, peak RSS, exact value |
| scaled SVD | synthetic sparse matrices with fixed seed | runtime, peak RSS, singular values, subspace |
| sparse marker means | CSR matrices across densities | runtime, peak RSS, dense-oracle equality |
| `wilcoxauc` | sparse genes with implicit-zero ties | runtime, peak RSS, all result columns |
| SNN/graph conversion | fixed embeddings at increasing cell counts | runtime, peak RSS, edge equality |
| ridge path | fixed design, labels, weights, and folds | runtime, fit count, deviance path, selected penalties |

Kernel benchmarks must include degenerate and adversarial inputs represented in the
unit tests, not only random well-behaved matrices.

### Pipeline benchmarks

Pipeline benchmarks measure interactions and allocations that kernel tests miss:

| Tier | Workload | Required use |
|---|---|---|
| S | deterministic synthetic smoke data | every implementation iteration |
| M | PBMC4K reference fixture | correctness and representative end-to-end baseline |
| L | synthetic 10k-cell sparse matrix | routine scaling and memory evaluation |
| XL | 50k- and 100k-cell sparse matrices | scheduled/manual scalability validation |

The PBMC4K H5 fixture is intentionally not committed. Benchmark commands must detect
its absence and report a skip with the expected path, not download data or treat the
missing result as a pass.

XL benchmarks are not required on every developer machine. Their hardware profile
and dataset generator parameters must accompany the results.

## Dataset generation

Synthetic matrices must be deterministic and specified by:

- cell count;
- gene count;
- density or counts-per-cell distribution;
- matrix format and dtype;
- random seed;
- number and size of simulated clusters;
- mitochondrial/ribosomal/coding-gene annotations where applicable;
- expected duplicate-gene-name pattern when positional indexing is exercised.

Generators should create sparse matrices directly. They must not allocate a dense
`cells x genes` array as an intermediate.

Recommended initial shapes are:

| Profile | Cells | Genes | Density | Storage |
|---|---:|---:|---:|---|
| S | 500 | 2,000 | 2% | CSR float32 |
| M-synthetic | 4,000 | 20,000 | 1% | CSR float32 |
| L | 10,000 | 25,000 | 0.5% | CSR float32 |
| XL-50 | 50,000 | 30,000 | 0.2% | CSR float32 |
| XL-100 | 100,000 | 30,000 | 0.1% | CSR float32 |

These are benchmark profiles, not claims about a biological dataset. Adjustments are
allowed only when recorded with the result and compared against a baseline generated
with the same parameters.

## Measurement protocol

### Environment record

Every benchmark artifact records:

- commit SHA and dirty-tree state;
- operating system and architecture;
- physical/logical CPU counts and available RAM;
- Python version;
- NumPy, SciPy, scikit-learn, joblib, igraph, and optional-backend versions;
- BLAS/LAPACK implementation and effective thread limits;
- matrix shape, density, dtype, and seed;
- execution mode, component implementations, and `n_jobs`;
- whether the PBMC4K fixture was available.

### Timing

- Run one untimed warm-up for JIT-compiled or cache-sensitive kernels.
- Use at least seven measured repetitions for sub-second kernels and at least three
  for expensive pipeline runs.
- Report median, minimum, and median absolute deviation.
- Compare identical inputs in the same environment.
- Treat results with a coefficient of variation above 10% as inconclusive and rerun
  after investigating background load or insufficient duration.
- Report compilation time separately from steady-state Numba execution.

### Memory

Record process peak RSS with a tool that includes native NumPy/SciPy allocations.
Python-only allocators such as `tracemalloc` are insufficient for the primary memory
claim. Use the same measurement tool for baseline and candidate results.

For parallel runs, report both coordinator peak RSS and aggregate worker peak RSS
when the platform permits. At minimum, report total process-tree peak RSS.

### Fit and iteration counts

Runtime alone is insufficient for Stage 4. Instrument benchmark-only counters for:

- ridge-path invocations;
- individual estimator fits;
- skipped correlation candidates;
- threshold-range retries;
- degenerate resamples;
- solver iterations and convergence warnings;
- Louvain calls and graph conversions;
- SVD matrix-vector and transpose-matrix-vector products.

Instrumentation must not alter the default scientific output or be stored in AnnData
unless explicitly requested for diagnostics.

## Correctness gates for exact implementations

### Package-wide gates

- All existing fast tests pass.
- All applicable slow/reference tests pass when fixtures are available.
- Public signatures, defaults, return values, AnnData keys, categories, warnings, and
  exceptions remain unchanged unless separately approved.
- Fixed seeds remain reproducible under the same environment.
- Count reductions continue to accumulate in float64.
- Duplicate gene symbols continue to use positional indexing.
- No constant or preserved R quirk changes as part of performance work.

Existing measured concordance floors remain the baseline; an optimization must not
quietly reset them to aspirational values from the original port design.

### `Sn`

- Compare against the current implementation for tractable inputs and R
  `robustbase::Sn` fixtures.
- Preserve odd/even finite-sample corrections and samples of fewer than two values.
- Use `np.testing.assert_allclose(rtol=1e-5)` or the tighter existing fixture
  tolerance when present.
- Include ties, constant vectors, sorted/reversed inputs, infinities/NaNs according
  to the current behavior, and sizes surrounding algorithmic branch boundaries.

### Sparse marker means and Wilcoxon/AUC

- Compare every output column against a dense oracle on small matrices.
- Include all-zero genes, one non-zero value, tied non-zero values, groups with
  different zero fractions, excluded labels, and empty-group errors.
- Preserve average ranks, zero ties, continuity correction, p-value fallback,
  `pct_in`, and `pct_out` semantics.
- Compare Stage-3 cluster-statistics tables after sorting by their existing keys.

### Matrix-free SVD

- Validate forward and transpose products against an explicitly scaled dense matrix.
- Compare singular values and projection subspaces; do not require raw singular-vector
  signs to match.
- Compare pairwise distances or the resulting exact neighbor indices where ties do
  not make the ordering ambiguous.
- Run the downstream shallow/deep clustering and expression-filter tests.
- Preserve the distinct Stage-3 per-gene and Stage-4 per-cell scaling rules.

### Ridge path and Stage 4

- Compare the complete fold-by-penalty deviance matrix.
- Compare `C_min`, `C_1se`, final coefficients, probabilities, and ROC thresholds.
- Preserve fold construction, weights, resampling order, random seeds, and 1-SE rule.
- Evaluate the existing Stage-4 dead-count and consensus bounds over the documented
  seed set rather than one favorable seed.
- Treat any change in convergence warnings or failed/degenerate runs as a scientific
  difference requiring review.

### Graph optimization

- Require identical weighted edges for graph-construction-only changes.
- Require identical cluster memberships after the existing size-based relabeling for
  graph-reuse changes.
- Preserve classic Louvain and existing seed behavior.
- Do not compare cluster integer labels before the established relabeling step.

## Gates for approximate or alternative execution modes

Approximate behavior requires a separate approved specification and may not use the
exact label.

At minimum, its evaluation reports:

- neighbor recall against exact kNN;
- SNN edge overlap and weight error;
- shallow/deep clustering ARI against the exact Python path and R reference;
- per-stage and end-to-end barcode concordance;
- continuous-metric correlations;
- Stage-4 dead/uncertain counts across seeds;
- runtime and memory improvement;
- the dataset sizes at which it becomes beneficial.

No universal acceptance threshold is set here because the first approximate backend
has not been selected. Its design must define thresholds before implementation.

## Performance gates

A performance change is material when, on at least one workload it targets, it
achieves either:

- at least a 1.25x median speedup; or
- at least a 25% reduction in measured peak RSS; or
- removal of a demonstrated allocation failure at a larger tier while remaining
  within the same runtime order.

Additionally:

- unaffected kernel and Tier-S pipeline medians must not regress by more than 5%;
- a 5–10% apparent regression requires explanation and rerun;
- a regression above 10% blocks integration unless it is an explicitly approved
  trade-off with a larger measured benefit;
- shared CI timing does not enforce these thresholds automatically;
- results must distinguish cold-start and steady-state execution for JIT backends.

These gates prevent merging complexity for negligible benchmark noise. They are
project engineering thresholds, not user-facing performance guarantees.

## Benchmark artifacts

Each comparison writes a machine-readable JSON artifact and a short Markdown summary
outside the installed package. The JSON schema contains:

```json
{
  "schema_version": 1,
  "benchmark": "sn_scaling",
  "revision": "<git-sha>",
  "dirty": false,
  "environment": {},
  "input": {},
  "implementation": {},
  "timing_seconds": {
    "median": 0.0,
    "minimum": 0.0,
    "mad": 0.0,
    "repetitions": 7
  },
  "peak_rss_bytes": 0,
  "correctness": {
    "status": "pass",
    "metrics": {}
  },
  "warnings": []
}
```

The implementation plan will select the concrete benchmark directory and commands.
Generated result artifacts should be gitignored unless a maintainer explicitly
chooses a small canonical baseline for version control.

## CI policy

- Shared CI runs deterministic correctness tests and a small benchmark smoke test
  that validates commands and schemas, not speed thresholds.
- Slow PBMC4K tests remain conditional on the local fixture.
- Controlled performance runs may be manual or scheduled on named hardware.
- Optional-backend jobs verify import, fallback, and a small equivalence test on every
  supported Python version for which the extra is available.
- A missing optional backend is a skip only in jobs that do not promise that backend;
  it is a failure in a backend-specific job.

## Reporting template

Every optimization handoff answers:

1. What bottleneck was measured?
2. What algorithm or allocation changed?
3. Which workloads and hardware were used?
4. What were baseline and candidate runtime and peak RSS?
5. Which correctness and R-concordance gates passed?
6. Did warnings, fit counts, iterations, or stochastic distributions change?
7. What limitations or scale crossover remain?
8. Is the implementation exact, approximate, or backend-specific?

Do not describe unavailable measurements as zero and do not extrapolate a small
synthetic speedup to PBMC4K or 100k cells without running that tier.

## Initial implementation readiness

After maintainer approval, the Package-1 implementation plan must define:

- benchmark package choices and exact file locations;
- deterministic Tier-S and synthetic generators;
- environment and JSON-schema collection;
- PBMC4K fixture detection;
- baseline commands;
- sparse cluster-mean replacement tests;
- reusable graph conversion tests;
- thread-control experiments and the decision on whether any runtime dependency is
  justified.

The plan must not include fast `Sn`, matrix-free SVD, sparse Wilcoxon, Stage-4 ridge
changes, public backend parameters, or RAPIDS; those remain separate packages.
