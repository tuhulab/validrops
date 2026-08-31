# Package 1 — Verification and Scope Audit Summary

**Generated:** 2026-08-30 (Task 5 of `docs/internal/2026-08-30-performance-package-1-implementation-plan.md`)
**Repo revision:** `c9a1c8ceb578795d8288704fe9696a349b86c8c7` — working tree **dirty** (expected: Package-1 source changes + untracked benchmark harness/docs)
**Hardware:** Apple Silicon (Darwin 25.5.0, arm64), 14 physical/logical CPUs, 24 GiB RAM (`available_memory_bytes: 25769803776`)
**Software:** Python 3.12.12 (CPython), NumPy 2.4.4, SciPy 1.17.1, AnnData 0.12.10, igraph 1.0.0, BLAS config empty (Apple accel-NEON, ASIMDHP/ASIMDDP), `sklearn_version: "unavailable"` (not imported by kernels measured)
**Measurement method:** fresh spawned child per repetition (`resource.getrusage(RUSAGE_SELF).ru_maxrss`, scope `worker_self`/process-high-water incl. deterministic input generation + warm-up in the same child), `time.perf_counter`, median/minimum/MAD/CV reported. All artifacts: `<benchmark>-<implementation>.json/.md` under `benchmark-results/package1/`.

---

## 1. Bottlenecks measured

| Benchmark | Kernel / stage | Why it was measured |
|---|---|---|
| `cluster-means` | Stage-3 `_cluster_stats` expm1 means per target/rest group | Two `todense()` allocations on the normalized matrix were pure dense-assembly overhead for a reduction that implicit zeros make free in sparse form |
| `graph-conversion` | SNN adjacency → igraph conversion per Louvain call | The deep-clustering resolution sweep and shallow call each rebuilt the igraph object (up to ~40 calls per Tier-S run); conversion cost repeated N times |
| `tier-s` (S pipeline) | End-to-end `expression_metrics` on Tier-S | Aggregates both kernels through the real stage to check cross-kernel effects and regressions |
| `tier-m-pbmc` | Full pipeline on the PBMC4K reference fixture | Representative current-only reference workload (see §7 for baseline absence) |
| thread control | Stage-3 native threading | Whether BLAS/OMP threading helps the Tier-S stage before any runtime dependency is considered |

## 2. Algorithms / allocations changed (and explicitly not changed)

Changed (exact, narrow):
1. `expression_metrics.py` — the two dense `np.expm1(norm[t].todense()).mean(axis=0)` expressions in `_cluster_stats` replaced by `_sparse_expm1_col_mean` (float64 reduction, `sum_duplicates()` before `expm1`, input never mutated, zero-row guard).
2. `_snn.py` — `prepare_louvain_graph(adjacency)` + `louvain_prepared(graph, ...)` extracted; the `louvain(adjacency, ...)` wrapper now delegates to them. `expression_metrics` prepares the igraph **once** and reuses it for the shallow call and every deep-sweep call, including the injected-`clusters` branch.

Explicitly NOT touched: `Sn`, Wilcoxon/AUC, dense SVD (`_embed`), Stage-4 ridge, neighbor search, public backend parameters, RAPIDS, constants, seeds, float64 reductions, positional gene indexing. `_deep_clustering` resolution logic (coarse 1..20 sweep, ±0.9 refinement, fallback) is byte-for-byte unchanged apart from the graph argument.

## 3. Workloads used

| Workload | Cells × Genes | Density | dtype | Repeats |
|---|---:|---:|---|---|
| Tier-S (`profile S`) | 500 × 2,000 | 2% | CSR float32 | 7 (kernels) / 3 (tier-s) |
| Smoke | 60 × 120 | 5% | CSR float32 | 1 (schema/CI check) |
| Tier-M PBMC4K | 4,341 × 1,783 cells×genes (raw.h5) | sparse | float32 | 3 (current only) |

All inputs deterministic: fixed seeds, fixed non-zero counts, CSR rows built directly (no dense `cells × genes` intermediate), duplicate-label metadata only.

## 4. Baseline vs current: runtime and peak RSS

| Workload | Baseline median (s) | Current median (s) | Speedup | Baseline peak RSS | Current peak RSS | RSS Δ |
|---|---|---:|---:|---:|---:|---:|
| `cluster-means` (S) | 0.0053 | 0.0038 | **1.39x** | 270.2 MB | 268.9 MB | −0.5% |
| `graph-conversion` (S) | 6.1997 | 5.2026 | **1.19x** | 261.5 MB | 263.0 MB | +0.6% |
| `tier-s` (S pipeline) | 4.7146 | 3.6639 | **1.29x** | 283.0 MB | 271.5 MB | −4.1% |
| `tier-m-pbmc` (current only) | — | 81.8198 | — | — | 1.736 GB | — |

Variability (CV) and correctness gates:

| Workload | Baseline CV | Current CV | Correctness | Classification |
|---|---:|---:|---|---|
| `cluster-means` (S) | 7.24% | 4.47% | pass (dense float64 oracle equality, tolerances met) | **exact** |
| `graph-conversion` (S) | 0.42% | 0.53% | pass (identical labels at every resolution/seed; wrapper≡prepared; golden-clique fixture `[1,1,0,0,0,0]`; edges/weights unmutated) | **exact** |
| `tier-s` (S pipeline) | 0.25% | 0.59% | pass (bounded output counts; deterministic warmup==timed) | **exact** (both changes) |
| `tier-m-pbmc` | — | 0.38% | pass (2 warnings: pre-existing scipy `svds` OS-entropy starting vector in `_embed`; dense SVD unchanged, out of Package-1 scope) | **exact** / reference-only |

CV stays well below the 10% inconclusive bar on every workload; no rerun required.

### Gate verdicts against the validation contract

- **cluster-means:** 1.39x median speedup **≥ 1.25x bar — met**. RSS change is a −0.5% (not the 25% reduction bar), but the runtime gate is satisfied. The dense-allocation removal is demonstrated: `todense()` calls are gone from the calculation; sparse float64 reduction is oracle-equal (`rtol=1e-12, atol=1e-12` per unit test).
- **graph-conversion:** 1.19x median speedup **is below the 1.25x merge bar — NOT met** (see §7 for the honest statement and human-facing justification).
- **tier-s:** 1.29x median speedup on the aggregate stage — met.
- No workload regressed >5% on either kernel or the Tier-S median.

## 5. Correctness and R-concordance gates passed

Fresh targeted block (this task, verbatim commands):

| Check | Result |
|---|---|
| `pytest tests/test_benchmarks.py tests/test_snn.py tests/test_expression_metrics.py -v` | **41 passed** (benchmarks 13, snn 12, expression_metrics 16) |
| `pytest tests/ -v` | **200 passed, 0 skipped** (5:11) |
| `ruff check src/validrops benchmarks tests` | **pass** |
| `ruff format --check src/validrops benchmarks tests` | **pass after fix** — see note |
| smoke `cluster-means` current (`--profile smoke --repetitions 1`) | **pass** — median 0.000164 s, peak RSS 248 MB, correctness `pass` |
| smoke `graph-conversion` current (`--profile smoke --repetitions 1`) | **pass** — median 0.113 s, peak RSS 248 MB, correctness `pass` |

Format-check note (honest): `ruff format --check` initially **failed** on `benchmarks/_measure.py` (two spots, lines ~661 and ~966 — a multi-line call that fits on one line at line-length 120 and an over-wrapped `bool(...)`). The fix was `ruff format benchmarks/_measure.py` (1 file reformatted); **no file under `src/` or `tests/` was modified by this task**, and the re-run reports 52 files formatted / 0 would-be-reformatted.

PBMC4K-dependent reference tests (fixture **present** at `tests/data/pbmc4k/raw.h5`, so none were skipped): the slow `test_end_to_end_concordance_with_r` (full-pipeline barcode concordance > 0.90 bound, measured-stage 1/2/3b composition unchanged), stage-2 metric/vs-recomputed checks, stage-2 filters vs R CSV, stage-3 clusters/embedding/stats reference checks, deviance HVG reference check, and Wilcoxon/presto reference checks — all passed under the existing thresholds and documented deep-clustering ceiling. Note: the end-to-end concordance is 0.9193 (bounded >0.90) with the entire loss localized to the known structural deep-clustering divergence; Package-1 does not claim to have changed it and the gate passed.

Warnings / fit-counts / iterations / stochastic distributions: **unchanged** by design — no fit-count instrumentation changed, no convergence warnings appeared or disappeared, fixed seeds produce identical labels (prepared≡wrapper, golden cliques), and the only benchmark warnings are the two pre-existing `svds` OS-entropy notes on the PBMC4K workload (dense SVD explicitly out of scope).

## 6. Thread-control conclusion (no runtime dependency)

`OMP/OPENBLAS/MKL/VECLIB_MAXIMUM_THREADS=1|2|4` in a fresh interpreter before NumPy/SciPy import, Tier-S, current implementation, 3 reps each:

| Threads | Median (s) | Min (s) | MAD (s) | CV | Peak RSS | Correctness |
|---|---:|---:|---:|---:|---:|---|
| 1 | **3.578** | 3.564 | 0.008 | 0.30% | 0.271 GB | pass |
| 2 | **3.605** | 3.601 | 0.004 | 0.11% | 0.271 GB | pass |
| 4 | **3.619** | 3.606 | 0.006 | 0.26% | 0.271 GB | pass |

**Conclusion:** medians are flat (3.578 → 3.619 s, i.e. +1.2% from 1 to 4 threads; effectively noise at CV < 1%), i.e. **no native-threading benefit in Stage 3 on this workload**. Decision gate: **do not add `threadpoolctl` or alter package runtime behavior in Package 1.** These experiments only characterize native Stage-3 threading; they cannot establish whether joblib + BLAS oversubscription exists in Stage 4 — that requires a separate Stage-4 plan measuring process/inner-thread semantics.

## 7. Limitations, honest misses, and deferred work

**Graph-conversion misses the merge bar — stated exactly.** Measured: **6.1997 s → 5.2026 s = 1.19x**, which is **below the plan's 1.25x merge bar**. It is NOT rounded up and is NOT restated as meeting the bar. Factual justification for a human decision (why the change is still sound):

- labels are identical at every resolution and seed (wrapper ≡ prepared path; golden-clique fixture `[1,1,0,0,0,0]` at res 1/seed 0; repeated prepared calls deterministic, edge weights unmutated);
- the igraph object is constructed **once** instead of N times (verified by monkeypatched call counter — one `prepare_louvain_graph` per stage-3 run);
- there is **no behavioural change**: same adjacency → same edges/weights → same Louvain outcome.
- The measured 1.19x partially reflects that on the S profile the conversion is already cheap relative to `community_multilevel` itself; the N-conversion elimination is real but the constant dominates. On larger workloads (L/XL, deferred) the per-call conversion share grows, so the crossover is expected to improve — this is an extrapolation, not a measured claim.

**Unavailable / deferred tiers (explicit):**
- **`tier-m-pbmc` baseline: NOT available and NOT created.** Recording it would require stashing the source changes; the plan forbids creating it. `tier-m-pbmc-current` (median 81.82 s, worker peak RSS 1.736 GB, correctness pass, 2 svds-entropy warnings) is recorded as a **current-only reference workload**. **No PBMC4K speedup is claimed.**
- `tier-m-synthetic`, `tier-l` (10k), `tier-xl-50`, `tier-xl-100`: **deferred** — generator profiles exist in `benchmarks/_datasets.py`, but no artifacts were run in Package 1; must not be extrapolated from S.
- Stage-4 ridge thread semantics: **not measured** (out of Package-1 scope; separate plan required before any `threadpoolctl`-style dependency).
- RSS: measured as `worker_self` (RUSAGE_SELF in the child), single-process; no process-tree claim; unsupported platforms emit `unavailable`, never zero.

Scale crossover: S-profile numbers are pipeline-level; the only PBMC4K artifact is current-only, so no end-to-end speedup or RSS claim beyond S is made (§contract rule: no extrapolating a small synthetic speedup to PBMC4K/100k).

## 8. Exact / approximate / backend-specific

**All Package-1 changes are exact.** Sparse cluster means are float64-reduction identical to the dense oracle (validated `rtol=1e-12, atol=1e-12`, incl. all-zero genes, single non-zeros, ties, unequal zero fractions, float32/sliced/non-canonical inputs, and non-mutation of the input matrix). Prepared-graph Louvain produces identical fixed-seed labels and preserve the exact descending-size relabeling. No approximate behavior, no new backend, no public parameter added.

---

### Appendix A — fresh verification log (Task 5)

```
$ rtk uv run pytest tests/test_benchmarks.py tests/test_snn.py tests/test_expression_metrics.py -v
41 passed in 64.27s

$ rtk uv run pytest tests/ -v
200 passed in 311.13s (0 skipped)

$ rtk uvx ruff check src/validrops benchmarks tests
All checks passed!

$ rtk uvx ruff format --check src/validrops benchmarks tests
1 file would be reformatted (benchmarks/_measure.py)  →  fixed via `ruff format benchmarks/_measure.py`
52 files already formatted; re-run clean

$ rtk uv run python -m benchmarks.run --benchmark cluster-means --implementation current --profile smoke --output-dir benchmark-results/package1/smoke --repetitions 1
median_seconds 0.000164, worker_max_rss_bytes 248233984, correctness pass

$ rtk uv run python -m benchmarks.run --benchmark graph-conversion --implementation current --profile smoke --output-dir benchmark-results/package1/smoke --repetitions 1
median_seconds 0.113, worker_max_rss_bytes 248037376, correctness pass
```

### Appendix B — final Git state (Task 5)

```
rtk git status --short --untracked-files=all
 M .gitignore                                   # +/benchmark-results/
 M src/validrops/tl/_snn.py
 M src/validrops/tl/expression_metrics.py
 M tests/test_expression_metrics.py
 M tests/test_snn.py
?? benchmarks/README.md benchmarks/__init__.py benchmarks/_datasets.py benchmarks/_measure.py benchmarks/run.py
?? tests/test_benchmarks.py
?? docs/internal/README.md
?? docs/internal/performance-optimization-design.md
?? docs/internal/performance-validation-contract.md
?? docs/internal/2026-08-30-performance-package-1-implementation-plan.md

rtk git diff --cached --name-only
(empty)
```

**Nothing is staged or committed.** `docs/internal/**` (README, both approved specs, the implementation plan) remains **untracked** and outside Git. Benchmark artifacts live under the gitignored `benchmark-results/`. All tracked diffs are confined to the two Package-1 source files, additive tests, and `.gitignore`.
