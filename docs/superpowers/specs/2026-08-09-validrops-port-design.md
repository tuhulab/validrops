# valiDrops → Python port: design

**Date:** 2026-08-09
**Status:** approved for planning
**Scope:** complete port of R package `valiDrops` (Kavaliauskaite & Madsen, 2023) to `validrops`, all four stages.

R source of truth: `../valiDrops-r-source/R/`. Every algorithm below cites the R file and line
range it is ported from. Where R and the project's `Agentic Engineering Guide` disagree, R wins.

---

## 1. Goals and non-goals

**Goals**

- Faithful port of all five pipeline functions plus the orchestrator: `rank_barcodes`,
  `quality_metrics`, `quality_filter`, `expression_metrics`, `expression_filter`, `label_dead`.
- >95% barcode concordance with R on PBMC 4K; Pearson r > 0.99 on continuous metrics.
- Self-contained package: no network access at runtime, no R dependency at runtime.
- scverse-conventional API: AnnData first, inplace, return `None`.

**Non-goals**

- Bit-for-bit reproduction of R's RNG stream. Stochastic stages are validated distributionally.
- Reimplementing Seurat or presto wholesale. Only the specific behaviours valiDrops depends on.
- Performance work beyond keeping the O(n²) `Sn` viable at <100k cells.

---

## 2. Architecture

```
src/validrops/
  __init__.py              exports pp, tl, pl, validrops()
  _constants.py            frozen R defaults
  _pipeline.py             validrops() orchestrator          <- valiDrops.R
  data/
    annotation.parquet     6-species gene table              <- sysdata.rda
  tl/
    _stats.py              sn()                              <- robustbase::Sn
    _uik.py                uik()                             <- inflection::uik
    _segmented.py          segmented()                       <- segmented::segmented
    _deviance.py           deviance_feature_selection()      <- scry
    _wilcox.py             wilcoxauc()                       <- presto::wilcoxauc
    _snn.py                snn_graph(), louvain()            <- Seurat::FindNeighbors/FindClusters
    _annotation.py         detect_annotation(), gene_sets()
    quality_metrics.py     Stage 2a                          <- quality_metrics.R
    expression_metrics.py  Stage 3a                          <- expression_metrics.R
  pp/
    rank_barcodes.py       Stage 1                           <- rank_barcodes.R
    quality_filter.py      Stage 2b                          <- quality_filter.R
    expression_filter.py   Stage 3b                          <- expression_filter.R
    label_dead.py          Stage 4                           <- label_dead.R
  pl/
    _qc.py                 five diagnostic plots
```

`pp` holds everything that filters or labels barcodes; `tl` holds everything that computes
metrics, plus the shared numerical primitives. Primitives are pure functions over numpy arrays —
no AnnData, no pipeline state — so each is unit-testable against an R fixture in isolation.

**Unit boundaries.** Every primitive answers: what does it do (one R function's behaviour), how do
you call it (numpy in, numpy out), what does it depend on (numpy/scipy only, except `_snn` which
uses igraph). Stage modules depend on primitives and on `_annotation`, never on each other —
the orchestrator wires them.

### 2.1 Dependencies

Runtime: `anndata`, `numpy`, `scipy`, `pandas`, `scikit-learn`, `pyarrow`, `igraph`, `leidenalg`,
`matplotlib`, `joblib`.
Test/doc only: `scanpy` (for `read_10x_h5` in fixtures and notebooks).

`scanpy` is deliberately not a runtime dependency: AnnData is the interop surface, and the two
places the dependency map suggested scanpy (`pp.neighbors`+`leiden`, `rank_genes_groups`) are
replaced by Seurat-compatible implementations for fidelity reasons (§4).

### 2.2 Corrections to CLAUDE.md's dependency map

The map in `CLAUDE.md` is superseded where noted. `CLAUDE.md` must be updated as part of
implementation.

| R | CLAUDE.md said | This design | Why |
|---|---|---|---|
| `segmented` | `pwlf` | own `_segmented.py` | pwlf uses global optimisation over breakpoint locations; R uses Muggeo's iterative linearisation. Different estimates from the same data. Stage 1's threshold *is* a breakpoint. |
| `inflection::uik` | `kneed` | own `_uik.py` | Kneedle and UIK are different estimators. |
| `scry` | custom in `pp/_hvg.py` | `tl/_deviance.py` | location only; `tl` per the pp/tl split. |
| Seurat clustering | `sc.pp.neighbors` + `sc.tl.leiden` | own `_snn.py` + igraph Louvain | Seurat prunes the SNN at 1/15 and optimises modularity via Louvain; scanpy's defaults differ in k, pruning and algorithm. |
| `presto::wilcoxauc` | `sc.tl.rank_genes_groups` | `tl/_wilcox.py` | needs a three-level grouping (target/rest/excluded) that `rank_genes_groups` does not express. |
| `glmnet` | `RidgeCV` | `LogisticRegression` L2 path + 1se rule | `label_dead.R:183` is `family="binomial"` — logistic, not linear. |
| `irlba` | `scipy.sparse.linalg.svds` | unchanged | |
| `robustbase::Sn` | custom | unchanged | |
| `mixtools::normalmixEM` | `GaussianMixture` | unchanged | |
| `BiocParallel` | `joblib` | unchanged | |

---

## 3. Public API

All public functions take `adata` first, mutate it, return `None`.

```python
validrops.validrops(
    adata, *, rank_barcodes=True, stage_three=True, label_dead=False,
    mitochondrial_clusters=3, ribosomal_clusters=3,
    random_state=0, verbose=True, **kwargs,
) -> None

validrops.pp.rank_barcodes(adata, *, type="UMI", psi_min=2, psi_max=5, alpha=0.001,
                           alpha_max=0.05, boot=10, factor=1.5, random_state=0) -> None
validrops.tl.quality_metrics(adata, *, contrast=None, contrast_type="denominator",
                             species="auto", annotation="auto",
                             mito="auto", ribo="auto", coding="auto", verbose=False) -> None
validrops.pp.quality_filter(adata, *, mito=True, distance=True, coding=True, contrast=False,
                            mito_nreps=10, mito_max=0.3, npsi=3, dist_threshold=5,
                            coding_threshold=3, contrast_threshold=3, random_state=0) -> None
validrops.tl.expression_metrics(adata, *, nfeats=5000, npcs=10, k_min=5, res_shallow=0.1,
                                top_n=10, clusters=None, random_state=0) -> None
validrops.pp.expression_filter(adata, *, mito=3, ribo=3, min_significant=1, min_target_pct=0.3,
                               max_background_pct=0.7, min_diff_pct=0.2, min_de_frac=0.01,
                               min_significance_level=None) -> None
validrops.pp.label_dead(adata, *, cor_threshold=None, train=True, rep=10, n_min=8, n_relabel=1,
                        feature_try=3, label_thrs=None, label_frac=0.1, nfeats=2000, alpha=0,
                        npcs=100, weight=True, epochs=20, nfolds=5, nrep=10, fail_weight=0.2,
                        cor_min=0.0001, cor_max=0.005, cor_steps=50, nrep_cor=10, min_dead=100,
                        max_live=500, n_jobs=1, verbose=False, random_state=0) -> None
```

Parameter names are R's with `.` → `_`. Defaults are R's defaults, verbatim — taken from the
function *signatures*, not the roxygen docs, which disagree in one place: `expression_filter`'s
roxygen says `max.background.pct = 0.8` but the signature is `0.7` (`expression_filter.R:11` vs
`:22`). The signature wins, since that is what produced the reference output.

### 3.1 Where results land

Every stage annotates the **full, unsubset** AnnData. Barcodes outside a stage's input set get
`NaN` (float columns) or `False` (bool columns). Matching is always by `obs_names`, never by
position.

`adata.obs`

| column | dtype | written by |
|---|---|---|
| `rank_pass` | bool | Stage 1 |
| `log_umis`, `log_features` | float | Stage 2a |
| `mitochondrial_fraction`, `ribosomal_fraction`, `coding_fraction` | float | Stage 2a |
| `contrast_fraction` | float | Stage 2a, only if `contrast` given |
| `pass_mito`, `pass_distance`, `pass_coding`, `pass_contrast` | bool | Stage 2b |
| `cluster` | category | Stage 3a |
| `qc_pass` | bool | Stage 3b (or Stage 2b if `stage_three=False`) |
| `dead_score` | float | Stage 4 |
| `label` | category `{live, dead, uncertain}` | Stage 4 |

`adata.uns["validrops"]`

| key | contents |
|---|---|
| `rank_threshold` | float, the count cutoff |
| `barcode_ranks` | DataFrame: counts, rank (rank-passing barcodes) |
| `species`, `annotation_column`, `n_mapped` | detected annotation |
| `gene_sets` | dict of `mitochondrial`, `ribosomal`, `protein_coding` gene name arrays |
| `mitochondrial_threshold` | float |
| `mito_threshold_method` | `"gmm_uik"` or `"segmented_fallback"` |
| `cluster_stats` | DataFrame, 11 columns, one row per deep cluster |
| `min_significance_level` | float |
| `label_flag` | `"Success"` / `"Caution"` / `"Failed"` |
| `label_threshold` | float, soft-label score cutoff |
| `params` | dict of every resolved parameter, for provenance |

Also mirrored: `adata.var["mitochondrial"]`, `["ribosomal"]`, `["protein_coding"]` as bool columns.

---

## 4. Stage specifications

Throughout: R's `log()` is **natural log**, R's `rank()` defaults to `ties.method="average"`
(use `scipy.stats.rankdata`), R's `quantile()` defaults to type 7, and R's `var()` uses `n-1`.
AnnData is cells × genes where R's matrix is genes × cells — every axis is transposed.

### 4.1 Primitives

**`sn(x)` — `robustbase::Sn`**
`1.1926 * c_n * lomed_i( himed_j |x_i - x_j| )` where `himed` is the `⌊n/2⌋+1`-th order statistic
and `lomed` the `⌈n/2⌉`-th. Finite-sample correction `c_n`: for `n` in 2..9, the tabulated
constants `{0.743, 1.851, 0.954, 1.351, 0.993, 1.198, 1.005, 1.131}` respectively; for `n > 9`,
`n/(n-0.9)` when `n` is even and `1.0` when odd. Naive O(n²) is acceptable — Stage 2b calls it on
≤ ~10k values. Asserted at `rtol=1e-10`.

**`uik(x, y)` — `inflection::uik`**
Unit-invariant knee. Normalise both axes to [0,1], then apply the `ese`/`bese` double-iteration to
locate the point of maximum curvature and return the corresponding `x`. Ported from the
`inflection` package source rather than approximated. Asserted at `rtol=1e-6`.

**`segmented(x, y, psi_init, ...)` — `segmented::segmented`**
Muggeo's iterative method: at each iteration fit
`y ~ x + Σ_k (x-ψ_k)_+ β_k + Σ_k I(x>ψ_k) γ_k` by OLS, update `ψ_k ← ψ_k + γ_k/β_k`, iterate to
convergence. Supports `n_boot` bootstrap restarts (seeded via `random_state`), `alpha` (the
admissible region for breakpoints, expressed as a quantile bound), and `npsi`. Returns estimated
breakpoints, per-segment slopes, residuals, and a converged flag. Raises `SegmentedFitError` when
no configuration converges — callers replicate R's fallback behaviour. Asserted at `rtol=1e-6` on
breakpoints and slopes.

**`deviance_feature_selection(counts)` — `scry`**
Binomial deviance per gene under a constant-proportion null, with per-cell size factors. Returns a
deviance per gene; callers take the top-`nfeats` by `rank(-dev)`.

**`wilcoxauc(X, y, groups_use)` — `presto::wilcoxauc`**
Vectorised rank-sum over a sparse matrix with a three-level grouping. Normal approximation with tie
correction, no continuity correction — matching presto, which is what the R reference run uses
(presto is installed in the reference environment). Returns AUC, p-value, pct.1, pct.2 per feature.

**`snn_graph(X, k=20, prune=1/15)` / `louvain(g, resolution)` — Seurat**
Exact kNN on the PCA scores, Jaccard-weighted SNN, edges below `prune` dropped. Louvain via
igraph's `RBConfigurationVertexPartition` at the given resolution, which shares Seurat's modularity
objective. Deterministic given `random_state`.

### 4.2 Stage 1 — `pp.rank_barcodes` (`rank_barcodes.R:31-150`)

1. `counts` per barcode: `UMI` → column sums; `Genes` → count of non-zero genes.
2. `rank = rankdata(-counts)` — computed **before** dropping zero-count barcodes (`R:73-74`).
3. Drop zero-count barcodes; sort by `(-counts, -rank)`.
4. Keep the first occurrence of each distinct count value (`R:78`); take natural log of both
   counts and rank.
5. Smooth: centred rolling mean, window `n = ceil(2 * n_unique^(1/3))`. `zoo::rollmean` returns
   `N - n + 1` values, and its centring convention for **even** windows is asymmetric. Rather than
   assume the offset, it is pinned by a dedicated R fixture — an off-by-one here shifts the x-axis
   the breakpoints are estimated on, and therefore the threshold.
6. For `psi` in `psi_min..psi_max`: fit OLS, then `segmented` with initial breakpoints
   `linspace(quantile(x, alpha), quantile(x, 1-alpha), psi)` and
   `seg.control(alpha=alpha - alpha/1000, n_boot=boot)`. On a "psi values too close" failure,
   increment `alpha` by `alpha` and retry until `alpha > alpha_max` (`R:101-116`).
7. Among converged models, pick the **lowest-indexed** model whose RMSE ≤ `factor × min(RMSE)`
   (`R:121`) — not the lowest-RMSE model.
8. Compute per-segment slopes; angles between consecutive slopes
   `atan((s_i - s_{i+1}) / (1 + s_i·s_{i+1})) · 180/π`; choose `argmin(angles[1:]) + 1`, i.e. the
   sharpest turn **excluding the first** (`R:127`).
9. Threshold = `exp(counts)` at the unique-count row minimising `|log(rank) - ψ_best|`.
10. `obs.rank_pass = counts >= threshold`.

If `rank_barcodes=False` in the orchestrator, `rank_pass = counts > 0` (`valiDrops.R:79`).
A warning is emitted when >20,000 barcodes pass (`valiDrops.R:83`).

### 4.3 Stage 2a — `tl.quality_metrics` (`quality_metrics.R:32-217`)

Operates on the rank-passing subset.

1. **Gene ID cleaning.** For names matching `^ENSG00` or `^ENSMUSG00`, strip the version suffix at
   the first `.` (`R:113-121`).
2. **Annotation auto-detection.** For each of the six species tables and — when
   `annotation="auto"` — **each of its seven columns**, count how many cleaned gene names appear in
   that column. Take the argmax. This scans `Chr`, `Type` and `Alias` too (`R:129`); that is
   preserved verbatim because it is what selects the ID space used for gene-set lookup.
3. **Gene sets** from the winning (table, column):
   - protein-coding: rows with `Type == "protein_coding"`
   - mitochondrial: rows with `Chr ∈ {MtDNA, MT, mitochondrion_genome}`
   - ribosomal: rows whose `Symbol` (lowercased) starts with `rpl` or `rps`
4. **Metrics** (`R:186-194`): `log_umis = log(total counts)`,
   `log_features = log(n genes detected)`, and mito/ribo/coding fractions as
   `sum(counts in set) / total counts`.
5. Optional contrast fraction (`R:197-207`).

Gene sets are written to `adata.var` and `adata.uns`; Stage 3 consumes the protein-coding set.

### 4.4 Stage 2b — `pp.quality_filter` (`quality_filter.R:26-216`)

Three sub-filters applied **sequentially** — each sees only the survivors of the previous one
(`R:116`, `R:155`, `R:180`).

**Mitochondrial.** Repeat `mito_nreps=10` times: fit a 2-component Gaussian mixture to
`log_features`; take the barcodes whose max-posterior component is the one with the larger mean;
build `sequence = arange(median(grp mito fraction), 1, 0.001)`; `cnts[i] = #{grp mito ≤ seq[i]}`;
threshold = `uik(sequence, cnts)`. Final threshold = median of the 10.

If that exceeds `mito_max=0.3`, fall back (`R:79-97`): 10 repetitions of — subsample
`min(5000, floor(0.8n))` barcodes, fit `log_features ~ mitochondrial_fraction`, segmented with
`npsi=1`, escalating `npsi` up to 5 until `min(ψ) ≤ mito_max`; take `min(ψ)`. Final = median.
The escalation loop's `stop`-flag logic is transcribed as written. `mito_threshold_method` records
which path ran.

Barcodes with `mitochondrial_fraction ≤ threshold` pass.

**Distance.** Segmented fit of `log_features ~ log_umis` with `npsi=3`, decrementing `npsi` on
convergence failure (`R:126-133`). Residual bounds `median(resid) ± sn(resid) × dist_threshold`
(default 5). Barcodes inside the band pass.

**Coding.** `median(coding_fraction) ± sn(coding_fraction) × coding_threshold` (default 3).
Two-sided (`R:164-168`).

**Contrast.** Same shape, off by default.

`qc_pass` after Stage 2b = survivors of all enabled sub-filters.

### 4.5 Stage 3a — `tl.expression_metrics` (`expression_metrics.R:21-202`)

Input: counts restricted to protein-coding genes × Stage-2b survivors (`valiDrops.R:101`).

1. Drop all-zero genes → `nonzero`.
2. Normalise: `sf = 10000 / colSums(nonzero)`, scale each cell, `log1p` → `norm_transform`.
3. `deviance_feature_selection(nonzero)`; keep genes with `rank(-dev) ≤ nfeats` (5000).
4. Subset to variable features, transpose to cells × genes, then scale **per gene**
   (`means = colMeans(cells×genes)`, `sd` with the `nr/(nr-1)` correction, `sd == 0 → 1`).
5. SVD with `npcs=10`; PCA scores `U · diag(d)`.
6. SNN graph on the scores; Louvain at `res_shallow=0.1` → **shallow** clusters.
7. **Deep** clusters: sweep resolutions 1..20; for each, record the smallest cluster size; find the
   resolution(s) whose smallest cluster is nearest `k_min=5`; refine with a ±0.9 sweep at 0.1
   steps; select the **largest** resolution whose minimum cluster size equals `k_min`, with the
   nearest-achievable fallback of `R:109-113`.
8. Per deep cluster: `target` = its barcodes; `rest` = barcodes **not** in the dominant *shallow*
   cluster of the target's members (`R:130`). Compute `pct.1`, `pct.2` (rounded to 3 dp),
   `pct.diff = (pct.1 - pct.2)/pct.1`, and log2 fold change over `expm1` means. Test genes with
   `max(pct.1, pct.2) ≥ 0.1` **and** `fc ≥ 0.25`; skip the cluster if fewer than 2 qualify.
9. `wilcoxauc` on those genes, target vs rest; Bonferroni over `nrow(nonzero)`.
10. Collect 11 statistics per cluster: `cluster, pct.diff, pct.1, pct.2, n_de, n_total,
    n_negative, min_fdr, de_fraction, mito_fraction, ribo_fraction` — the means in columns 2-4 and
    7 are taken over the top `min(n_de, top_n)` genes by p-value.

**R quirk preserved:** when `n_de == 0`, `1:min(0, 10)` is `c(1, 0)` in R, and index 0 is dropped,
so the "top genes" set silently becomes the single most significant gene (`R:172-177`). Replicated.

A `clusters=` parameter allows injecting a precomputed cluster assignment. This exists so Stage 3b
can be validated against R's own clustering, isolating filter-logic fidelity from clustering
fidelity.

### 4.6 Stage 3b — `pp.expression_filter` (`expression_filter.R:22-113`)

1. **Automatic significance threshold** when `min_significance_level is None` (`R:57-65`): over
   clusters with `min_fdr > 0`, let `y = pct.diff`, `x = -log10(min_fdr)`. Then
   `threshold = median(x[y ≤ 0.4]) + sn(x[y ≤ 0.4]) × 3`, and
   `model_level = segmented(y ~ x, npsi=1).psi`. Take the minimum of the two, ignoring NaN.
2. Sequentially drop clusters failing: `n_negative == 0`, `pct.diff ≥ min_diff_pct`,
   `pct.1 ≥ min_target_pct`, `pct.2 ≤ max_background_pct`, `n_de ≥ min_significant`,
   `-log10(min_fdr) ≥ min_significance_level`, `de_fraction > min_de_frac`.
3. If `mito` is not None, additionally require
   `mito_fraction ≤ median(mito_fraction) + mito × sn(mito_fraction)`; same for `ribo`.
4. `qc_pass` = barcodes in surviving clusters.

### 4.7 Stage 4 — `pp.label_dead` (`label_dead.R:43-483`)

Called with the **full** count matrix and the rank-passing metrics (`valiDrops.R:126`), then
internally subset to the metrics' barcodes.

**Soft labelling (deterministic — asserted exactly).**
Centre `log_umis` and `log_features` (mean-subtract, no scaling); transform the three fractions as
`asin(sqrt(f)) / (π/2)`. Then (`R:50`):

```
score = -11.82·U + 2.08·F + 158.98·R + 18.87·F·C - 125.9·R·C
```

where `U`, `F` are the centred logs and `R`, `C` the transformed ribosomal and coding fractions.
Note the `/(π/2)` normalisation, which the Agentic Engineering Guide omits, and that the
mitochondrial fraction is transformed but does not enter the score.

Threshold search (`R:57-118`): starting at `max_quantile = 0.1`, tabulate
`quantile(score, brk)` for `brk` in `arange(0.0001, max_quantile, 0.0001)`, take
`uik` of that curve, and use it as a quantile to get a score cutoff. Increase `max_quantile` by 0.1
and repeat, comparing the resulting `label × qc` contingency tables, stopping per `R:91-113`.
Barcodes at or below the cutoff are soft-labelled dead.

Early exits (`R:128-143`): <3 dead → skip training, flag `Failed`; 0 QC-passing dead → relabel the
`n_relabel` least-dead-like as QC-pass, flag `Caution`; ≥ `label_frac` (10%) dead → abort, all
live, flag `Failed`.

**Training (stochastic — asserted distributionally).**
Normalise and log1p; then scale **per cell** (`means = colMeans(nonzero)` on a genes×cells matrix
is a per-cell mean — `R:170-174`) and SVD to `npcs=100`.

> **R quirks preserved.** (a) Stage 4 scales per cell where Stage 3 scales per gene — the two
> genuinely differ, and both are reproduced. (b) `R:166-167` computes `var.feats` and never uses
> it; the SVD runs on all non-zero genes. The dead computation is omitted, the behaviour kept.

Initial ridge pass (`R:183-191`): logistic ridge with 5-fold CV on all 100 PCs, predict at the
`lambda.1se` equivalent, build an ROC over QC-passing barcodes, and relabel dead→live below the
smallest threshold with specificity ≥ 0.99.

Then `rep=10` independent runs (joblib), each:
- **Feature selection** by Kendall's τ (`pcaPP::cor.fk` → `scipy.stats.kendalltau`) between each PC
  and the label; keep PCs with `τ² ≥ cor_threshold`.
- **Threshold search** when `cor_threshold is None` (`R:239-352`): 50 log-spaced values in
  [1e-4, 5e-3]; for each, 10 resampled ridge fits; score by specificity, dead count, and
  confusion-matrix ratios; retry up to `feature_try=3` times with a shifted range if nothing hits
  specificity ≥ 0.99. Selection cascade transcribed from `R:331-351`.
- **Epoch loop** (up to 20): resample dead/live × qc-fail/qc-pass with replacement, weighted by
  current probabilities; jitter each feature by `sd/5`; shuffle; fit weighted logistic ridge with
  5-fold CV; median the probabilities over 10 replicates; pick the ROC "best" (Youden) threshold
  with the multi-row fallbacks of `R:412-421`; relabel; stop per the convergence rules at
  `R:429-443`.

Consensus (`R:459`): `live` if ≥ `n_min=8` of 10 runs say live, `dead` if ≥ 8 say dead, else
`uncertain`. Flag escalates to `Caution` at ≥1.25% uncertain among QC-passing barcodes and to
`Failed` at ≥2.5% (`R:471-478`).

`lambda.1se` has no sklearn equivalent and is implemented explicitly: fit the L2 path with
`LogisticRegression` over a `Cs` grid under `StratifiedKFold`, compute mean and standard error of
the per-fold binomial deviance, and select the strongest regularisation whose mean deviance is
within one standard error of the minimum.

### 4.8 Orchestrator — `validrops()` (`valiDrops.R:21-139`)

Stage 1 → subset → Stage 2a → Stage 2b → (if `stage_three`) Stage 3a → Stage 3b →
(if `label_dead`) Stage 4. When `stage_three=False`, `qc_pass` comes straight from Stage 2b.
Status messages mirror R's, via `logging` rather than `print`.

### 4.9 Plotting — `pl`

Five functions, each reproducing an R `plot()` call and reading only from `obs`/`uns`:
`barcode_rank`, `mito_threshold`, `umi_vs_features`, `coding_fraction`, `dead_score`.
Each takes `adata`, returns a matplotlib `Axes`, and accepts `ax=`.

---

## 5. The annotation database

`sysdata.rda` holds `valiDrops:::annotation`, six tables:

| # | species | rows | columns |
|---|---|---|---|
| 1 | human | 106,851 | NCBI, HGNC, Ensembl, Chr, Symbol, Type, Alias |
| 2 | mouse | 82,003 | NCBI, MGI, Ensembl, Chr, Symbol, Type, Alias |
| 3 | rat | 47,954 | NCBI, Ensembl, Chr, Symbol, Type, Alias |
| 4 | zebrafish | 77,216 | as rat |
| 5 | worm | 46,912 | as rat |
| 6 | fly | 67,809 | as rat |

Note `quality_metrics.R:85-90` loads them in the order 1,2,3,**6**,5,**4** and then labels indices
`{1..6}` as `{Human, Mouse, Rat, Zebrafish, Worm, Fly}` (`R:178`). The extraction must reproduce
that permutation, not the raw `sysdata` order.

Extraction: `tests/R/extract_annotation.R` writes a single long-format
`src/validrops/data/annotation.parquet` with a `species` partition column and a nullable
`HGNC`/`MGI` column, dictionary-encoded. Rows are one-per-alias, as in R. Committed to the repo and
regenerable. Loaded lazily and cached at module level; `species != "auto"` reads only one partition.

---

## 6. Error handling

- Invalid parameter values → `ValueError`, mirroring R's `stop()` checks (which are extensive:
  `rank_barcodes.R:32-56`, `quality_filter.R:27-44`, `expression_filter.R:23-54`).
- A metric column absent when a filter needs it → `logging.warning` and skip that sub-filter,
  mirroring `quality_filter.R:118`, `:157`, `:182`.
- `SegmentedFitError` when no segmented model converges. Stage 1 propagates it (there is no
  threshold without it); Stage 2b's distance filter decrements `npsi` first, per R.
- Stage 4's `label_flag` is surfaced in `uns` rather than only logged, so downstream code can
  branch on a failed run.
- `random_state` defaults to `0` and threads through GMM initialisation, segmented bootstrap
  restarts, Louvain, resampling and CV folds. R leaves all of this to the global seed.

---

## 7. Validation

R with `valiDrops` and every dependency is installed locally; CI has neither. So references are
generated locally and committed as CSV, and `tests/R/generate_reference.R` stays in-repo to
regenerate them.

Barcodes in the existing reference are named positionally (`cell_15`, `cell_30`) because R read the
matrix without column names. The fixture loader maps `cell_N` → column `N-1` of the raw matrix;
`scanpy.read_10x_h5` preserves the file's column order, so this is well-defined. A dedicated test
asserts the mapping holds before any concordance test runs.

| Fixture | Generated by | Assertion |
|---|---|---|
| `sn_reference.csv` (exists) | `robustbase::Sn` on 5 vectors | `rtol=1e-10` |
| `uik_reference.csv` | `inflection::uik` on 5 curves | `rtol=1e-6` |
| `segmented_reference.csv` | breakpoints/slopes on 4 synthetic piecewise sets | `rtol=1e-6` |
| `rollmean_reference.csv` | `zoo::rollmean` at odd and even `k` | exact, `rtol=1e-12` |
| `deviance_reference.csv` | `scry::devianceFeatureSelection` on pbmc4k | `r > 0.999`, top-5000 set overlap > 99% |
| `wilcoxauc_reference.csv` | `presto::wilcoxauc` on one cluster | AUC `rtol=1e-6`, p-value `rtol=1e-6` |
| `annotation_genesets.csv` | detected sets for pbmc4k | exact set equality; species/column exact |
| `stage1_threshold.csv` | `rank_barcodes()` | threshold `rtol=1e-6`; `rank_pass` set exact |
| `stage2_metrics.csv` | `quality_metrics()` | all five metrics `r > 0.99`, `rtol=1e-8` |
| `stage2_filters.csv` | survivors per sub-filter + mito threshold | >95% concordance per sub-filter |
| `stage3_clusters.csv` | shallow + deep assignments | ARI > 0.9 |
| `stage3_stats.csv` | the 11-column stats frame | `r > 0.99` per column, given injected R clusters |
| `stage3_barcodes.csv` | `expression_filter()` output | exact set equality given injected R clusters; >95% otherwise |
| `stage4_soft_labels.csv` | score + threshold + soft labels | score `rtol=1e-10`; labels exact |
| `stage4_final.csv` | consensus labels over 10 runs | dead count within ±20%; >90% agreement on non-`uncertain` barcodes |
| `pbmc4k_full_pipeline.csv` (exists) | `valiDrops()` | >95% end-to-end concordance on `qc.pass` |

Stage 3's two rows are the split that matters: injecting R's clusters tests the filter logic, and
the ARI test separately tests the clustering. A clustering shortfall therefore cannot masquerade as
a filter bug, or vice versa.

Beyond reference tests: unit tests for parameter validation, a synthetic small-matrix smoke test
for the whole pipeline (fast, runs on every commit), and property tests for `sn` (scale
equivariance, translation invariance).

---

## 8. Known risks

1. **Stage 3 clustering.** Seurat's SNN+Louvain vs. an igraph reconstruction is the largest source
   of drift, and it propagates into every downstream statistic. Mitigated by the `clusters=` seam
   and by reporting ARI honestly rather than tuning until the number looks good.
2. **`zoo::rollmean` centring for even windows** shifts the axis Stage 1's breakpoints are
   estimated on. Pinned by a dedicated fixture.
3. **Muggeo convergence.** R's `segmented` has years of edge-case handling. A from-scratch
   implementation will fail on inputs R survives. `SegmentedFitError` plus R's own retry ladders
   (alpha escalation, npsi decrement) are the containment.
4. **Stage 4 wall-clock.** 10 runs × up to 20 epochs × 10 replicates × 5-fold CV logistic ridge on
   100 PCs. Parallelised with joblib; if it proves too slow, the `Cs` grid is the first knob.

---

## 9. Build order

Bottom-up, because four of five stages share the primitives.

1. Dependencies, `_constants.py`, annotation extraction → parquet, `_annotation.py`.
2. Primitives with their R fixtures: `sn`, `uik`, `segmented`, `deviance`, `wilcox`, `snn`.
3. Stage 1, validated.
4. Stage 2a, 2b, validated.
5. Stage 3a, 3b, validated (filter logic first via injected clusters, then clustering).
6. Stage 4, validated.
7. Orchestrator, end-to-end concordance.
8. `pl` module.
9. Docs: update `CLAUDE.md`'s dependency map per §2.2, `docs/api.md`, and the example notebook.
