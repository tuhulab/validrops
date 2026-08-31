# Performance optimization architecture

**Status:** Proposed — awaiting maintainer review

**Date:** 2026-08-30

**Scope:** CPU efficiency, memory scalability, optional accelerators, and execution provenance

**Related:** [Performance and concordance contract](performance-validation-contract.md)

## Decision summary

Optimize the exact CPU implementation before adding a general accelerator
abstraction. Preserve the existing public behavior and R-concordant defaults while
replacing avoidable quadratic algorithms, dense materializations, repeated model
fits, and repeated graph construction.

Use narrow internal seams around the computational kernels rather than a package-wide
backend protocol. Add public backend or execution-mode parameters only when the
package first supports behavior-changing alternatives such as approximate neighbors
or RAPIDS.

Implementation is decomposed into independently reviewable workstreams. The first
implementation plan will cover only the performance foundation and low-risk exact
CPU improvements; matrix-free SVD, Stage-4 model-path optimization, out-of-core
execution, and optional accelerators receive separate plans after their prerequisites
are measured.

## Context

The current port deliberately prioritizes R fidelity. Public functions accept an
`AnnData`, mutate it in place, and retain results in `adata.obs`, `adata.var`,
`adata.obsm`, or `adata.uns`. That contract does not change.

The current implementation has five structural hotspots:

1. `tl._stats.sn` constructs an `n x n` float64 pairwise-distance matrix.
2. Stage 3 and Stage 4 materialize centered/scaled expression matrices as dense
   float64 arrays before partial SVD.
3. Cluster marker means and `wilcoxauc` densify sparse expression data.
4. Stage 4 repeatedly fits the complete five-fold, 30-penalty logistic ridge path.
5. Deep-clustering resolution search reconstructs the same igraph graph for each
   Louvain call, while exact brute-force kNN eventually becomes expensive at scale.

Representative theoretical working sets illustrate the memory ceiling:

| Operation | Shape | Approximate float64 allocation |
|---|---:|---:|
| Pairwise `Sn` distances | 10,000 x 10,000 | 0.75 GiB |
| Pairwise `Sn` distances | 50,000 x 50,000 | 18.63 GiB |
| Stage-3 dense PCA input | 100,000 x 5,000 | 3.73 GiB |
| Stage-4 dense PCA input | 100,000 x 33,000 | 24.59 GiB |

These figures are shape-based estimates, not measured peak RSS. Measured baselines
must be recorded before implementation.

With the current defaults, one complete ridge path performs `5 * 30 + 1 = 151`
`LogisticRegression.fit` calls. If all correlation candidates are usable, Stage 4
can request approximately 7,001 paths with one correlation-range evaluation per
consensus run, or 22,001 paths when every threshold-search retry is used. This is
roughly 1.06–3.32 million individual fits before accounting for candidates that are
skipped or resamples that degenerate.

## Goals

- Reduce runtime and peak memory on CPU without weakening the default scientific
  contract.
- Keep the exact path deterministic for a fixed seed to the degree supported by the
  current implementation.
- Preserve float64 accumulation, positional gene indexing, constants, R quirks, and
  output locations.
- Support sparse input without unnecessary dense copies.
- Make concurrency predictable and prevent nested thread oversubscription.
- Establish measurement and provenance before exposing alternative backends.
- Keep optional accelerators optional; the base package remains usable without a GPU,
  JIT compiler, Rust toolchain, Dask, or vendor-specific runtime.

## Non-goals

- Changing the valiDrops statistical procedure to improve biological results.
- Replacing classic Louvain with Leiden on the exact path.
- Changing constants, the Stage-4 consensus rule, or the ridge 1-SE selection rule.
- Making float32 the default arithmetic for reductions or statistical kernels.
- Adding Dask or Ray around algorithms that still allocate dense or quadratic
  intermediates.
- Guaranteeing bitwise equality across BLAS libraries, operating systems, or sparse
  SVD solvers.
- Delivering RAPIDS in the first implementation work package.

## Invariants

Every workstream must preserve the following unless a later, separately approved
fast-mode specification explicitly relaxes one:

- R source and generated reference fixtures override plans or biological
  expectations.
- Count reductions use float64 accumulation even when sparse storage is float32.
- Duplicate gene symbols are resolved by original integer position, never by name.
- Existing AnnData keys, public defaults, warnings, exceptions, and inplace behavior
  remain unchanged.
- Exact mode never silently falls back to an approximate algorithm.
- Random seeds are propagated through all stochastic algorithms.
- Optimization-specific metadata must not overwrite scientific intermediates in
  `adata.uns["validrops"]`.

## Architecture

```text
AnnData input
    |
    +-- streaming/sparse reductions --------> Stage 1 and Stage 2 annotations
    |
    +-- sparse normalization
    |       |
    |       +-- matrix-free scaled operator -> partial SVD -> compact embedding
    |       |                                      |
    |       |                                      +-- exact or explicit fast kNN
    |       |                                      +-- reusable graph object
    |       |
    |       +-- sparse marker kernels --------> cluster metrics
    |
    +-- compact Stage-4 embedding
            |
            +-- cached folds/resamples
            +-- warm-started ridge paths
            +-- bounded outer parallelism
            +-- consensus labels
```

The initial code should not introduce a generic `Backend` base class. Instead, use
small internal functions with stable array-oriented contracts:

- robust scale kernel: one-dimensional float64 input to scalar output;
- scaled SVD operator: sparse matrix plus centering/scaling vectors to embedding;
- marker kernel: gene-by-cell sparse matrix plus labels to result columns;
- neighbor kernel: dense embedding to neighbor indices;
- ridge-path kernel: design matrix, labels, weights, folds, and penalty grid to
  selected fitted model.

This keeps the scientific stages independent of a future implementation choice
without forcing unrelated NumPy, Numba, native, or RAPIDS operations behind one
leaky abstraction.

## Workstreams

### A. Performance foundation

Create the benchmark structure and record the baseline described in the validation
contract. Add lightweight stage timing that is disabled by default or emitted only
through existing logging. Do not gate pull requests on noisy wall-clock thresholds
from shared CI runners.

This workstream is a prerequisite for every optimization below.

### B. Exact sparse and algorithmic improvements

#### Fast exact `Sn`

Replace the pairwise matrix with the Rousseeuw–Croux exact algorithm used by
`robustbase`, which runs in `O(n log n)` time and `O(n)` storage. Preserve the current
finite-sample correction, float64 input semantics, edge cases, and reference values.

Preferred implementation order:

1. pure NumPy/Python reference implementation for correctness if needed;
2. optional Numba kernel with `cache=True`, `nogil=True`, and no `fastmath`;
3. Cython or Rust/PyO3 only if Numba does not meet measured goals or packaging needs.

The base installation must retain a correct fallback. If an optional compiled path is
introduced, its output must be validated against both the current Python reference
and R fixtures.

#### Sparse cluster means

Avoid `todense()` when calculating means of `expm1(log-normalized)` values. Apply
`expm1` to a sparse copy's stored data and reduce columns sparsely. Preserve the
mathematical zero contribution of implicit entries.

#### Sparse Wilcoxon/AUC

Implement gene-wise rank sums using the non-zero values plus the implicit-zero tie
group. Preserve average ranks, continuity correction, tie correction, two-sided
p-values, AUC, and detection percentages. Process genes in bounded chunks if the
working set would otherwise grow with `genes * cells`.

#### Reusable graph representation

Build the igraph vertex/edge/weight representation once for a resolution search and
reuse it for each classic-Louvain call. Reset the same random seed at the same logical
points as the current implementation. Do not warm-start partitions on the exact path
unless validation demonstrates unchanged results.

### C. Matrix-free SVD

Represent centered/scaled sparse expression matrices with
`scipy.sparse.linalg.LinearOperator` rather than materializing dense arrays. Implement
both forward and transpose products and validate them against explicitly constructed
small dense matrices before using `svds`.

Stage 3 scales per gene; Stage 4 scales per cell and intentionally uses all non-zero
genes. These are separate operators or configurations and must not be unified in a
way that hides their different semantics.

Validation compares:

- forward and transpose products;
- singular values;
- embedding subspaces or pairwise distances, allowing sign indeterminacy;
- downstream neighbor graph and cluster labels;
- final stage outputs and existing concordance bounds.

ARPACK non-convergence must remain visible. An exact execution must not silently
switch to randomized SVD.

### D. Stage-4 ridge path and parallelism

Keep the current folds, penalty grid, deviance calculation, and 1-SE selection rule.
Reduce repeated work through:

- precomputed fold indices and class encoding;
- penalty traversal from stronger to weaker regularization;
- `warm_start=True` for solvers that support it;
- cached resampling indices and selected-PC masks where inputs are identical;
- reuse of allocated design buffers;
- outer joblib parallelism with one BLAS/OpenMP thread per worker.

Do not replace the routine with `LogisticRegressionCV` unless a separate proof shows
that its aggregation and selection semantics are identical. Solver changes require
comparison of selected `C_min`, `C_1se`, probabilities, ROC thresholds, consensus
labels, and Stage-4 count bounds.

Concurrency is constrained by memory. `n_jobs="auto"`, if later exposed, must use a
documented conservative policy rather than all logical CPUs. Explicit `n_jobs` keeps
its current meaning.

### E. Optional execution modes and accelerators

Behavior-changing choices are deferred until the exact CPU path is measured and
optimized. Future candidates include:

- PyNNDescent or another approximate CPU neighbor backend;
- randomized SVD for explicitly requested fast mode;
- reduced Stage-4 search parameters under an explicit fast preset;
- scikit-learn-intelex for supported CPU estimators;
- RAPIDS for supported sparse, neighbor, and model kernels;
- Rust/PyO3 native kernels where JIT compilation or wheel support is inadequate.

When the first alternative is implemented, introduce explicit public selection:

```python
validrops.validrops(
    adata,
    execution_mode="exact",  # or "fast"
    backend="auto",          # later: "numpy", "numba", "rapids"
)
```

Until then, do not add unused public parameters.

`execution_mode="exact"` may select only implementations satisfying the exact
validation contract. `execution_mode="fast"` may use approximations but must be
opt-in, documented, and evaluated against exact outputs. An explicitly requested
unavailable backend raises an actionable error; `backend="auto"` may fall back only
within the same execution mode.

### F. Backed and multi-sample execution

Out-of-core support follows the removal of dense intermediates:

- stream Stage 1 and Stage 2 reductions from backed CSR/CSC datasets;
- extend deviance and expression summaries to bounded sparse chunks;
- keep only the PCA embedding and small model inputs resident;
- write annotations to a new AnnData file when the backing format cannot persist
  `obs`, `var`, or `uns` changes safely.

Whole-sample throughput may later use a `validrops_many()` helper that assigns one
AnnData per process. Distributed execution inside a single statistical kernel is not
part of this design.

## Provenance

Once alternative implementations exist, record a structured performance section
inside `adata.uns["validrops"]`, for example:

```python
{
    "execution": {
        "mode": "exact",
        "backend": "numba",
        "components": {
            "sn": "numba",
            "svd": "scipy-linear-operator",
            "neighbors": "sklearn-exact",
            "ridge": "sklearn-lbfgs-warm-start",
        },
        "n_jobs": 4,
    }
}
```

Do not record timing by default because it makes otherwise equivalent results depend
on the machine and run. Benchmark commands write timing to separate artifacts.

## Error handling and fallback rules

- Optional acceleration imports are lazy.
- Missing explicitly requested extras raise an error naming the installation extra.
- `auto` fallback is logged once and recorded in provenance.
- Exact mode cannot fall back to approximate neighbors, randomized SVD, reduced
  penalties, or fewer consensus runs.
- Memory-budget rejection occurs before spawning workers when a safe estimate is
  possible.
- Solver non-convergence follows the current warning/error contract and is included
  in benchmark reports.
- Compiled-kernel failures may fall back to the exact reference implementation only
  when inputs have not been mutated and the failure is safely recoverable.

## Dependency policy

The first work package adds no required accelerator dependency.

Potential later dependencies are evaluated independently:

| Dependency | Role | Policy |
|---|---|---|
| `threadpoolctl` | bound nested native threads | declare directly if imported |
| `numba` | exact CPU kernels | optional extra; maintain pure fallback |
| `pynndescent` | approximate neighbors | optional fast-mode extra |
| `scikit-learn-intelex` | vendor-optimized estimators | experimental optional extra |
| Rust/PyO3/maturin | native kernels and wheels | add only after measured need |
| RAPIDS packages | GPU backend | separate optional extra and compatibility matrix |

Support for Python 3.11–3.14 remains mandatory. Optional extras must have a defined
behavior on every supported Python version: install successfully, be excluded with a
clear message, or use the exact fallback.

## Rollout and work-package boundaries

The roadmap is deliberately split so each plan remains reviewable:

1. **Package 1 — measurement and safe exact wins:** benchmark harness, baseline,
   sparse cluster means, graph reuse, and thread-control experiments.
2. **Package 2 — exact fast `Sn`:** reference algorithm, optional compiled kernel if
   justified, scaling tests, and fallback.
3. **Package 3 — sparse Wilcoxon:** exact zero-tie algorithm and chunked execution.
4. **Package 4 — matrix-free SVD:** Stage-3 and Stage-4 operators and downstream
   equivalence validation.
5. **Package 5 — Stage-4 ridge optimization:** profiling, warm starts, caching, and
   bounded parallelism.
6. **Package 6 — explicit fast/out-of-core modes:** separate user-facing design and
   concordance report.
7. **Package 7 — accelerator backends:** RAPIDS or other backends only after stable
   kernel seams exist.

After this design is approved, the immediate implementation plan covers Package 1
only. Each later package receives its own plan using the measurements produced by its
predecessors.

## Alternatives considered

### General backend abstraction first

Rejected for the initial work. NumPy/SciPy, Numba, RAPIDS, graph libraries, and
out-of-core arrays do not share one useful end-to-end contract. A generic abstraction
would add public and internal complexity before the true kernel boundaries are
stable.

### RAPIDS first

Deferred. It can accelerate selected workloads but does not remove the exact CPU
path's quadratic `Sn`, repeated model-path semantics, or avoidable dense copies for
users without compatible GPUs.

### Dask or Ray around the current pipeline

Rejected until algorithms are streamable. Scheduling the existing dense and
quadratic intermediates distributes or duplicates the problem rather than removing
it.

### Global float32 conversion

Rejected because count reductions and thresholds are sensitive to float32 drift.
Sparse storage may remain float32, while reductions and statistical calculations use
float64 as they do today.

### Leiden or approximate neighbors as the new default

Rejected because clustering differences already dominate the measured end-to-end
R-concordance loss. These are valid future fast-mode options only.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Faster code changes numerical results | exact contract, R fixtures, per-stage differential tests |
| Sparse zero handling changes Wilcoxon ties | dense oracle tests including all-zero and duplicate-value genes |
| Matrix-free SVD changes signs or basis orientation | compare subspaces/distances and downstream outputs, not raw signs alone |
| Warm starts choose a different penalty | compare complete deviance paths and selected `C_min`/`C_1se` |
| Parallel workers oversubscribe BLAS | bound inner threads and benchmark process/thread combinations |
| Optional dependency breaks a supported Python version | compatibility matrix, lazy import, exact fallback |
| Benchmark results depend on noisy CI hardware | correctness in shared CI; performance gates on controlled profiles |
| Backend fallback becomes scientifically invisible | explicit mode rules, logging, and AnnData provenance |

## Acceptance

This design is ready for implementation planning when the maintainer approves:

- exact CPU optimization precedes general backend work;
- Package 1 is the first implementation scope;
- no public `backend` or `execution_mode` parameter is added in Package 1;
- the performance and concordance contract is the merge gate;
- later work packages remain separately planned and reviewed.

## References

- [Existing port design](../superpowers/specs/2026-08-09-validrops-port-design.md)
- [Existing port plan](../superpowers/plans/2026-08-09-validrops-port.md)
- SciPy `LinearOperator`: <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.LinearOperator.html>
- SciPy sparse SVD: <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.svds.html>
- scikit-learn parallelism: <https://scikit-learn.org/stable/computing/parallelism.html>
- scikit-learn logistic regression: <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>
- Numba performance guidance: <https://numba.readthedocs.io/en/stable/user/performance-tips.html>
- AnnData partial/lazy reading: <https://anndata.readthedocs.io/en/stable/tutorials/notebooks/getting-started.html>
- robustbase `Sn` implementation: <https://github.com/cran/robustbase/blob/master/src/qn_sn.c>
