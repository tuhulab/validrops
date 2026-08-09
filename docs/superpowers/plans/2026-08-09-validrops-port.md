# valiDrops Python Port Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the R package `valiDrops` (automated QC for scRNA-seq) to a Python/AnnData package `validrops`, reaching >95% barcode concordance with R on PBMC 4K.

**Architecture:** Six pure-numpy numerical primitives at the bottom (`tl/_*.py`), five pipeline stages on top (`tl/` computes metrics, `pp/` filters barcodes), one orchestrator, one plotting module. Every stage annotates the full unsubset AnnData in place. Build order is bottom-up: primitives are validated against R fixtures before any stage uses them.

**Tech Stack:** Python ≥3.11, anndata, numpy, scipy, pandas, scikit-learn, pyarrow, python-igraph, leidenalg, matplotlib, joblib. Tests: pytest. R (with `valiDrops` installed) is used only to generate fixtures, never at test time.

**Design spec:** `docs/superpowers/specs/2026-08-09-validrops-port-design.md`. Section references below (§4.2 etc.) point into it.

**R source of truth:** `../valiDrops-r-source/R/`. Every algorithm cites the R file and lines it ports.

## Global Constraints

- Python `>=3.11`. Line length 120. NumPy-convention docstrings. Linter/formatter: `ruff`.
- All public functions take `adata` as the first argument, mutate it in place, and return `None`.
- Parameter names are R's with `.` → `_`. Defaults come from the R function **signatures**, not the roxygen docs (they disagree on `max.background.pct`: doc says 0.8, signature says 0.7 — signature wins).
- Every `log()` in R is **natural log**. Never `log10` unless the R source literally says `log10`.
- R's `rank()` defaults to `ties.method="average"` → use `scipy.stats.rankdata(x)`. R's `quantile()` defaults to type 7 → `np.quantile(..., method="linear")`. R's `var()` uses `n-1` → `ddof=1`.
- AnnData is **cells × genes**; R's matrix is **genes × cells**. Every axis is transposed.
- R quirks listed in the spec are preserved verbatim, each with a comment citing the R line. Do not "fix" them.
- Run commands with `uv run` (a bare `pytest` resolves to a Homebrew Python outside the project venv).
- Commit at the end of every task. Never commit a task with failing tests.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/validrops/_constants.py` | Frozen R defaults, one module-level constant per value |
| `src/validrops/_pipeline.py` | `validrops()` orchestrator ← `valiDrops.R` |
| `src/validrops/data/annotation.parquet` | 6-species gene table, extracted from `sysdata.rda` |
| `src/validrops/tl/_stats.py` | `sn()`, `rollmean()` |
| `src/validrops/tl/_uik.py` | `uik()` ← `inflection::uik` |
| `src/validrops/tl/_segmented.py` | `segmented()` ← Muggeo's method |
| `src/validrops/tl/_deviance.py` | `deviance_feature_selection()` ← `scry` |
| `src/validrops/tl/_wilcox.py` | `wilcoxauc()` ← `presto::wilcoxauc` |
| `src/validrops/tl/_snn.py` | `snn_graph()`, `louvain()` ← Seurat |
| `src/validrops/tl/_annotation.py` | Gene-ID cleaning, species/column detection, gene sets |
| `src/validrops/tl/quality_metrics.py` | Stage 2a |
| `src/validrops/tl/expression_metrics.py` | Stage 3a |
| `src/validrops/pp/rank_barcodes.py` | Stage 1 |
| `src/validrops/pp/quality_filter.py` | Stage 2b |
| `src/validrops/pp/expression_filter.py` | Stage 3b |
| `src/validrops/pp/label_dead.py` | Stage 4 |
| `src/validrops/pl/_qc.py` | Five diagnostic plots |
| `tests/R/generate_reference.R` | Regenerates every fixture |
| `tests/R/extract_annotation.R` | `sysdata.rda` → parquet |
| `tests/conftest.py` | Fixture loading, `cell_N` → column mapping |

---

## Task 1: Package skeleton, dependencies, constants

**Files:**
- Modify: `pyproject.toml`
- Create: `src/validrops/_constants.py`
- Modify: `src/validrops/__init__.py`
- Modify: `src/validrops/tl/__init__.py`, `src/validrops/pp/__init__.py`, `src/validrops/pl/__init__.py`
- Delete: `src/validrops/tl/basic.py`, `src/validrops/pp/basic.py`, `src/validrops/pl/basic.py`
- Create: `tests/test_constants.py`
- Modify: `tests/test_basic.py`

**Interfaces:**
- Produces: module `validrops._constants` with the named constants below; `validrops.tl`, `validrops.pp`, `validrops.pl` importable submodules.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_constants.py
import validrops
from validrops import _constants as C


def test_r_defaults_are_frozen():
    assert C.MITO_CAP == 0.3
    assert C.SN_MULTIPLIER_CODING == 3
    assert C.SN_MULTIPLIER_DISTANCE == 5
    assert C.SN_C_SMALL == (0.743, 1.851, 0.954, 1.351, 0.993, 1.198, 1.005, 1.131)
    assert C.RMSE_FACTOR == 1.5
    assert C.BREAKPOINT_RANGE == (2, 5)
    assert C.MITO_SCAN_INCREMENT == 0.001
    assert C.HVG_COUNT == 5000
    assert C.SHALLOW_RESOLUTION == 0.1
    assert C.MIN_CLUSTER_SIZE == 5
    assert C.DEAD_CELL_RUNS == 10
    assert C.DEAD_CELL_CONSENSUS == 8
    assert C.DEAD_SCORE_COEFFICIENTS == {
        "log_umis": -11.82,
        "log_features": 2.08,
        "ribosomal": 158.98,
        "features_x_coding": 18.87,
        "ribosomal_x_coding": -125.9,
    }


def test_submodules_importable():
    assert validrops.tl is not None
    assert validrops.pp is not None
    assert validrops.pl is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_constants.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'validrops._constants'`

- [ ] **Step 3: Add dependencies**

Edit `pyproject.toml`, replacing the `dependencies` array:

```toml
dependencies = [
  "anndata",
  "joblib",
  "matplotlib",
  "numpy",
  "pandas",
  "pyarrow",
  "python-igraph",
  "leidenalg",
  "scikit-learn",
  "scipy",
  # for debug logging (referenced from the issue template)
  "session-info2",
]
```

Add `scanpy` to the `test` and `doc` dependency groups (it is used for `read_10x_h5` in fixtures and notebooks, never at runtime).

Add to `[tool.hatch]` section so the parquet ships in the wheel:

```toml
build.targets.wheel.packages = [ "src/validrops" ]
build.targets.wheel.force-include = { "src/validrops/data" = "validrops/data" }
```

Then run `uv sync --all-extras`.

- [ ] **Step 4: Write the constants module**

```python
# src/validrops/_constants.py
"""Constants transcribed from the R source. Do not change without a matching R change."""

# rank_barcodes.R:31
BREAKPOINT_RANGE = (2, 5)  # psi.min, psi.max
RMSE_FACTOR = 1.5  # factor
RANK_ALPHA = 0.001
RANK_ALPHA_MAX = 0.05
RANK_BOOT = 10

# quality_filter.R:26
MITO_CAP = 0.3  # mito.max
MITO_NREPS = 10
MITO_SCAN_INCREMENT = 0.001
SN_MULTIPLIER_DISTANCE = 5  # dist.threshold
SN_MULTIPLIER_CODING = 3  # coding.threshold
SN_MULTIPLIER_CONTRAST = 3  # contrast.threshold
DISTANCE_NPSI = 3

# robustbase::Sn finite-sample corrections for n = 2..9
SN_C_SMALL = (0.743, 1.851, 0.954, 1.351, 0.993, 1.198, 1.005, 1.131)
SN_CONSTANT = 1.1926

# expression_metrics.R:21
HVG_COUNT = 5000  # nfeats
N_PCS = 10  # npcs
MIN_CLUSTER_SIZE = 5  # k.min
SHALLOW_RESOLUTION = 0.1  # res.shallow
TOP_N_MARKERS = 10  # top.n
SNN_K = 20  # Seurat::FindNeighbors k.param
SNN_PRUNE = 1 / 15  # Seurat::FindNeighbors prune.SNN

# expression_filter.R:22
MIN_SIGNIFICANT = 1
MIN_TARGET_PCT = 0.3
MAX_BACKGROUND_PCT = 0.7  # signature value; roxygen says 0.8 and is wrong
MIN_DIFF_PCT = 0.2
MIN_DE_FRAC = 0.01
CLUSTER_MITO_DEVIATIONS = 3
CLUSTER_RIBO_DEVIATIONS = 3

# label_dead.R:43-50
DEAD_CELL_RUNS = 10  # rep
DEAD_CELL_CONSENSUS = 8  # n.min
DEAD_SCORE_COEFFICIENTS = {
    "log_umis": -11.82,
    "log_features": 2.08,
    "ribosomal": 158.98,
    "features_x_coding": 18.87,
    "ribosomal_x_coding": -125.9,
}
DEAD_LABEL_FRAC = 0.1
DEAD_NFEATS = 2000
DEAD_NPCS = 100
DEAD_EPOCHS = 20
DEAD_NFOLDS = 5
DEAD_NREP = 10
DEAD_FAIL_WEIGHT = 0.2
DEAD_COR_MIN = 0.0001
DEAD_COR_MAX = 0.005
DEAD_COR_STEPS = 50
DEAD_NREP_COR = 10
DEAD_MIN_DEAD = 100
DEAD_MAX_LIVE = 500
DEAD_FEATURE_TRY = 3

# Species table order as loaded by quality_metrics.R:85-90
SPECIES_ORDER = ("human", "mouse", "rat", "zebrafish", "worm", "fly")
MITO_CHROMOSOMES = ("MtDNA", "MT", "mitochondrion_genome")
UNS_KEY = "validrops"
```

- [ ] **Step 5: Wire up the package namespaces**

```python
# src/validrops/__init__.py
from importlib.metadata import version

from . import pl, pp, tl

__all__ = ["pl", "pp", "tl"]
__version__ = version("validrops")
```

Leave `src/validrops/tl/__init__.py`, `pp/__init__.py`, `pl/__init__.py` empty for now — later tasks add exports. Delete the three `basic.py` files and strip the `elaborate_example` test from `tests/test_basic.py`, leaving only `test_package_has_version`.

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/ -v`
Expected: PASS (3 tests)

- [ ] **Step 7: Lint and commit**

```bash
uv run ruff check src/validrops/ --fix && uv run ruff format src/validrops/
git add pyproject.toml uv.lock src/validrops tests/
git rm src/validrops/tl/basic.py src/validrops/pp/basic.py src/validrops/pl/basic.py
git commit -m "feat: package skeleton, dependencies and frozen R constants"
```

---

## Task 2: Extract the annotation database to Parquet

**Files:**
- Create: `tests/R/extract_annotation.R`
- Create: `src/validrops/data/annotation.parquet` (generated, committed)
- Create: `tests/test_annotation_data.py`

**Interfaces:**
- Produces: `src/validrops/data/annotation.parquet` — long-format table with columns `species` (str), `species_index` (int), `column_name` (str), `column_index` (int), `value` (str), `chr` (str), `type` (str). One row per (species, ID column, ID value) triple. `column_name` ∈ {NCBI, HGNC, MGI, Ensembl, Chr, Symbol, Type, Alias}. The two index columns exist so Python can reproduce R's `which.max` tie-breaking, which takes the first table and first column at the maximum.

**Why long format:** `quality_metrics.R:129` scans *every* column of a table looking for the best ID match, including `Chr`, `Type` and `Alias`. A long table makes "count matches per (species, column)" a single groupby, and makes gene-set lookup a filter, rather than six wide frames with differing schemas.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_annotation_data.py
from importlib.resources import files

import pandas as pd
import pytest

EXPECTED_ROWS = {
    "human": 106_851,
    "mouse": 82_003,
    "rat": 47_954,
    "zebrafish": 77_216,
    "worm": 46_912,
    "fly": 67_809,
}


@pytest.fixture(scope="module")
def annotation():
    path = files("validrops.data").joinpath("annotation.parquet")
    return pd.read_parquet(path)


def test_all_six_species_present(annotation):
    assert set(annotation["species"].unique()) == set(EXPECTED_ROWS)


def test_row_counts_match_r(annotation):
    # each source row contributes one row per ID column, so divide back out
    for species, expected in EXPECTED_ROWS.items():
        sub = annotation[annotation["species"] == species]
        n_source = sub.groupby("column_name").size().max()
        assert n_source == expected, species


def test_human_has_expected_columns(annotation):
    human = annotation[annotation["species"] == "human"]
    assert set(human["column_name"].unique()) == {
        "NCBI", "HGNC", "Ensembl", "Chr", "Symbol", "Type", "Alias",
    }


def test_mouse_uses_mgi_not_hgnc(annotation):
    mouse = annotation[annotation["species"] == "mouse"]
    cols = set(mouse["column_name"].unique())
    assert "MGI" in cols
    assert "HGNC" not in cols


def test_human_mitochondrial_gene_count(annotation):
    human = annotation[annotation["species"] == "human"]
    mito = human[(human["column_name"] == "Symbol") & (human["chr"] == "MT")]
    assert len(mito) == 98


def test_human_protein_coding_count(annotation):
    human = annotation[annotation["species"] == "human"]
    pc = human[(human["column_name"] == "Symbol") & (human["type"] == "protein_coding")]
    assert len(pc) == 55_304
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_annotation_data.py -v`
Expected: FAIL — `annotation.parquet` does not exist.

- [ ] **Step 3: Write the extraction script**

```r
# tests/R/extract_annotation.R
# Extracts valiDrops:::annotation (sysdata.rda) into a long-format parquet.
#
# CRITICAL: quality_metrics.R:85-90 loads the six tables in the order
#   1, 2, 3, 6, 5, 4
# and then labels indices 1..6 as Human, Mouse, Rat, Zebrafish, Worm, Fly
# (quality_metrics.R:178). The raw sysdata order is NOT the labelled order.
# Reproduce the permutation, not the raw order.

library(arrow)

raw <- valiDrops:::annotation
load_order <- c(1, 2, 3, 6, 5, 4)
species_names <- c("human", "mouse", "rat", "zebrafish", "worm", "fly")

pieces <- list()
for (i in seq_along(load_order)) {
  d <- as.data.frame(raw[[load_order[i]]])
  stopifnot(all(c("Chr", "Type", "Symbol") %in% colnames(d)))
  for (ci in seq_along(colnames(d))) {
    col <- colnames(d)[ci]
    pieces[[length(pieces) + 1]] <- data.frame(
      species      = species_names[i],
      species_index = i,          # R's which.max ties break toward the first table
      column_name  = col,
      column_index = ci,          # ... and toward the first column within it
      value        = as.character(d[[col]]),
      chr          = as.character(d$Chr),
      type         = as.character(d$Type),
      stringsAsFactors = FALSE
    )
  }
}

out <- do.call(rbind, pieces)
dir.create("src/validrops/data", recursive = TRUE, showWarnings = FALSE)
arrow::write_parquet(out, "src/validrops/data/annotation.parquet", compression = "zstd")

cat("rows:", nrow(out), "\n")
print(table(out$species))
```

- [ ] **Step 4: Run the extraction**

```bash
Rscript -e 'if (!requireNamespace("arrow", quietly=TRUE)) install.packages("arrow", repos="https://cloud.r-project.org")'
Rscript tests/R/extract_annotation.R
ls -lh src/validrops/data/annotation.parquet
```

Expected: a file of roughly 5-15 MB. If it exceeds 40 MB, re-run `write_parquet` with `compression = "zstd", compression_level = 9`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_annotation_data.py -v`
Expected: PASS (6 tests)

- [ ] **Step 6: Commit**

```bash
git add tests/R/extract_annotation.R src/validrops/data/annotation.parquet tests/test_annotation_data.py
git commit -m "feat: extract valiDrops annotation database to parquet"
```

---

## Task 3: Generate all R reference fixtures

**Files:**
- Modify: `tests/R/generate_reference.R` (replace wholesale)
- Create: `tests/reference_outputs/*.csv` (generated, committed)
- Create: `tests/test_fixtures_present.py`

**Interfaces:**
- Produces: the fixture CSVs named in the test below. Every later task consumes one or more.

**Note on regeneration:** the existing `pbmc4k_full_pipeline.csv` was produced without `set.seed`, so the mitochondrial threshold (a median over 10 stochastic GMM fits) is not reproducible. This script seeds everything and **overwrites** it. That is intentional: a reference we cannot regenerate is not a reference.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fixtures_present.py
from pathlib import Path

import pandas as pd
import pytest

REF = Path(__file__).parent / "reference_outputs"

EXPECTED = {
    "sn_reference.csv": ["name", "sn"],
    "rollmean_reference.csv": ["case", "index", "value"],
    "uik_reference.csv": ["case", "knee"],
    "segmented_reference.csv": ["case", "term", "value"],
    "deviance_reference.csv": ["gene", "deviance"],
    "wilcoxauc_reference.csv": ["feature", "auc", "pval", "pct_in", "pct_out"],
    "annotation_genesets.csv": ["gene", "set"],
    "annotation_detection.csv": ["species", "column", "n_mapped"],
    "stage1_threshold.csv": ["barcode", "counts", "rank"],
    "stage1_meta.csv": ["key", "value"],
    "stage2_metrics.csv": [
        "barcode", "logUMIs", "logFeatures",
        "mitochondrial_fraction", "ribosomal_fraction", "coding_fraction",
    ],
    "stage2_filters.csv": ["barcode", "pass_mito", "pass_distance", "pass_coding"],
    "stage2_meta.csv": ["key", "value"],
    "stage3_clusters.csv": ["barcode", "shallow", "deep"],
    "stage3_stats.csv": [
        "cluster", "pct.diff", "pct.1", "pct.2", "n_de", "n_total",
        "n_negative", "min_fdr", "de_fraction", "mito_fraction", "ribo_fraction",
    ],
    "stage3_barcodes.csv": ["barcode"],
    "stage4_soft_labels.csv": ["barcode", "score", "soft_label"],
    "stage4_meta.csv": ["key", "value"],
    "stage4_final.csv": ["barcode", "label"],
    "pbmc4k_full_pipeline.csv": ["barcode", "qc.pass"],
}


@pytest.mark.parametrize(("name", "columns"), EXPECTED.items())
def test_fixture_exists_with_expected_columns(name, columns):
    path = REF / name
    assert path.exists(), f"missing fixture {name}; run tests/R/generate_reference.R"
    df = pd.read_csv(path)
    missing = set(columns) - set(df.columns)
    assert not missing, f"{name} missing columns {missing}"
    assert len(df) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_fixtures_present.py -v`
Expected: FAIL — most fixtures missing.

- [ ] **Step 3: Write the reference generation script**

```r
# tests/R/generate_reference.R
# Regenerates every fixture in tests/reference_outputs/.
# Run from the repository root:  Rscript tests/R/generate_reference.R
#
# Everything is seeded so the fixtures are reproducible. The stochastic
# stages (mitochondrial threshold, clustering, dead-cell training) will
# still differ across R versions; regenerate rather than hand-edit.

library(valiDrops)
library(Matrix)
library(DropletUtils)
library(robustbase)
library(inflection)
library(zoo)
library(segmented)
library(scry)
library(presto)

set.seed(42)
OUT <- "tests/reference_outputs"
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
pdf(NULL)  # swallow plots

## ---------------------------------------------------------------- primitives

# Sn
set.seed(42)
test_vectors <- list(
  normal = rnorm(100), skewed = rexp(100), heavy_tail = rt(100, df = 3),
  small = rnorm(20), large = rnorm(5000)
)
write.csv(
  data.frame(name = names(test_vectors), sn = sapply(test_vectors, Sn)),
  file.path(OUT, "sn_reference.csv"), row.names = FALSE
)

# rollmean at odd and even k
set.seed(1)
rm_x <- cumsum(rnorm(50))
rm_rows <- list()
for (k in c(3, 4, 7, 8)) {
  v <- as.numeric(zoo::rollmean(rm_x, k = k, align = "center"))
  rm_rows[[length(rm_rows) + 1]] <- data.frame(
    case = paste0("k", k), index = seq_along(v), value = v
  )
}
rm_rows[[length(rm_rows) + 1]] <- data.frame(case = "input", index = seq_along(rm_x), value = rm_x)
write.csv(do.call(rbind, rm_rows), file.path(OUT, "rollmean_reference.csv"), row.names = FALSE)

# uik
uik_cases <- list(
  convex_decreasing = list(x = 1:100, y = 100 / (1:100)),
  concave_increasing = list(x = 1:100, y = log(1:100)),
  sigmoid = list(x = seq(-5, 5, length.out = 100), y = 1 / (1 + exp(-seq(-5, 5, length.out = 100)))),
  step = list(x = 1:100, y = c(rep(1, 40), seq(1, 10, length.out = 20), rep(10, 40))),
  noisy_elbow = list(x = 1:200, y = pmax(0, 50 - 0.5 * (1:200)) + 0.02 * (1:200))
)
uik_rows <- data.frame(
  case = names(uik_cases),
  knee = sapply(uik_cases, function(c) uik(c$x, c$y))
)
write.csv(uik_rows, file.path(OUT, "uik_reference.csv"), row.names = FALSE)
# also persist the inputs so Python tests use identical data
uik_in <- do.call(rbind, lapply(names(uik_cases), function(n) {
  data.frame(case = n, x = uik_cases[[n]]$x, y = uik_cases[[n]]$y)
}))
write.csv(uik_in, file.path(OUT, "uik_inputs.csv"), row.names = FALSE)

# segmented
seg_cases <- list()
set.seed(7)
x1 <- seq(0, 10, length.out = 200)
seg_cases$one_break <- list(x = x1, y = ifelse(x1 < 4, 2 * x1, 8 + 0.5 * (x1 - 4)) + rnorm(200, sd = 0.1), npsi = 1)
x2 <- seq(0, 20, length.out = 400)
y2 <- ifelse(x2 < 5, x2, ifelse(x2 < 12, 5 + 3 * (x2 - 5), 26 - 0.5 * (x2 - 12))) + rnorm(400, sd = 0.2)
seg_cases$two_breaks <- list(x = x2, y = y2, npsi = 2)
x3 <- seq(1, 50, length.out = 300)
y3 <- log(x3) * 3 + rnorm(300, sd = 0.15)
seg_cases$smooth_curve <- list(x = x3, y = y3, npsi = 3)
x4 <- seq(0, 1, length.out = 150)
y4 <- ifelse(x4 < 0.3, 1, ifelse(x4 < 0.7, 1 + 5 * (x4 - 0.3), 3)) + rnorm(150, sd = 0.05)
seg_cases$plateau <- list(x = x4, y = y4, npsi = 2)

seg_rows <- list()
seg_in <- list()
for (nm in names(seg_cases)) {
  cs <- seg_cases[[nm]]
  set.seed(99)
  # n.boot = 0 disables bootstrap restart, making the fit deterministic given
  # the default quantile-spaced starting values. Without this the fixture
  # depends on R's RNG stream, which the Python port cannot reproduce.
  fit <- segmented(lm(y ~ x, data = data.frame(x = cs$x, y = cs$y)),
                   npsi = cs$npsi, control = seg.control(n.boot = 0))
  psi <- fit$psi[, 2]        # column 2 is "Est."
  psi0 <- fit$psi[, 1]       # column 1 is "Initial" — Python starts from these
  sl <- slope(fit)$x[, 1]
  seg_rows[[nm]] <- rbind(
    data.frame(case = nm, term = paste0("psi_init", seq_along(psi0)), value = psi0),
    data.frame(case = nm, term = paste0("psi", seq_along(psi)), value = psi),
    data.frame(case = nm, term = paste0("slope", seq_along(sl)), value = sl),
    data.frame(case = nm, term = "rmse", value = sqrt(mean(fit$residuals^2)))
  )
  seg_in[[nm]] <- data.frame(case = nm, x = cs$x, y = cs$y, npsi = cs$npsi)
}
write.csv(do.call(rbind, seg_rows), file.path(OUT, "segmented_reference.csv"), row.names = FALSE)
write.csv(do.call(rbind, seg_in), file.path(OUT, "segmented_inputs.csv"), row.names = FALSE)

## ------------------------------------------------------------------- dataset

sce <- DropletUtils::read10xCounts("tests/data/pbmc4k/raw.h5")
counts <- SingleCellExperiment::counts(sce)
rownames(counts) <- rowData(sce)$Symbol
colnames(counts) <- paste("cell", seq_len(ncol(counts)), sep = "_")

## ------------------------------------------------------------------- stage 1

set.seed(42)
threshold <- valiDrops::rank_barcodes(counts, plot = FALSE)
rank.pass <- rownames(threshold$ranks[threshold$ranks$counts >= threshold$lower.threshold, ])
ranks <- threshold$ranks
ranks$barcode <- rownames(ranks)
write.csv(ranks[ranks$barcode %in% rank.pass, c("barcode", "counts", "rank")],
          file.path(OUT, "stage1_threshold.csv"), row.names = FALSE)
write.csv(data.frame(key = c("lower_threshold", "n_pass", "n_input"),
                     value = c(threshold$lower.threshold, length(rank.pass), ncol(counts))),
          file.path(OUT, "stage1_meta.csv"), row.names = FALSE)

counts.subset <- counts[, colnames(counts) %in% rank.pass]

## ------------------------------------------------------------------ stage 2a

set.seed(42)
metrics <- valiDrops::quality_metrics(counts.subset, verbose = TRUE)
write.csv(metrics$metrics, file.path(OUT, "stage2_metrics.csv"), row.names = FALSE)

genesets <- rbind(
  data.frame(gene = metrics$mitochondrial, set = "mitochondrial"),
  data.frame(gene = metrics$ribosomal, set = "ribosomal"),
  data.frame(gene = metrics$protein_coding, set = "protein_coding")
)
write.csv(genesets, file.path(OUT, "annotation_genesets.csv"), row.names = FALSE)
# detection result: recompute the winning (dataset, column) the same way R does
write.csv(data.frame(species = "human", column = "Symbol",
                     n_mapped = sum(rownames(counts.subset) %in%
                       as.data.frame(valiDrops:::annotation[[1]])$Symbol)),
          file.path(OUT, "annotation_detection.csv"), row.names = FALSE)

## ------------------------------------------------------------------ stage 2b

set.seed(42)
qc.pass <- valiDrops::quality_filter(metrics$metrics, plot = FALSE)
bc <- metrics$metrics$barcode
write.csv(data.frame(
  barcode = bc,
  pass_mito = bc %in% qc.pass$pass.mitochondrial_filter,
  pass_distance = bc %in% qc.pass$pass.distance_filter,
  pass_coding = bc %in% qc.pass$pass.coding_filter,
  final = bc %in% qc.pass$final
), file.path(OUT, "stage2_filters.csv"), row.names = FALSE)
write.csv(data.frame(key = c("mitochondrial_threshold", "n_final"),
                     value = c(qc.pass$mitochondrial.threshold, length(qc.pass$final))),
          file.path(OUT, "stage2_meta.csv"), row.names = FALSE)

## ------------------------------------------------------------------ stage 3a

counts.filtered <- counts.subset[rownames(counts.subset) %in% metrics$protein_coding,
                                 colnames(counts.subset) %in% qc.pass$final]
counts.filtered <- as(counts.filtered, "dgCMatrix")
set.seed(42)
expr <- valiDrops::expression_metrics(counts.filtered,
                                      mito = metrics$mitochondrial,
                                      ribo = metrics$ribosomal)
write.csv(expr$stats, file.path(OUT, "stage3_stats.csv"), row.names = FALSE)

# clusters: re-derive the shallow assignment alongside the deep one
deep <- data.frame(barcode = rownames(expr$clusters), deep = expr$clusters[, 1])
write.csv(deep, file.path(OUT, "stage3_clusters_deep.csv"), row.names = FALSE)

## ------------------------------------------------------------------ stage 3b

set.seed(42)
valid <- valiDrops::expression_filter(stats = expr$stats, clusters = expr$clusters,
                                      mito = 3, ribo = 3, plot = FALSE)
write.csv(data.frame(barcode = valid), file.path(OUT, "stage3_barcodes.csv"), row.names = FALSE)

## ---------------------------------------------------------- deviance, wilcox

nonzero <- counts.filtered[Matrix::rowSums(counts.filtered) > 0, ]
dev <- scry::devianceFeatureSelection(nonzero)
write.csv(data.frame(gene = names(dev), deviance = as.numeric(dev)),
          file.path(OUT, "deviance_reference.csv"), row.names = FALSE)

sf <- 10000 / Matrix::colSums(nonzero)
norm_transform <- Matrix::t(Matrix::t(nonzero) * sf)
norm_transform@x <- log1p(norm_transform@x)
target <- deep$barcode[deep$deep == deep$deep[1]]
y <- rep("rest", ncol(norm_transform))
y[colnames(norm_transform) %in% target] <- "target"
feats <- rownames(norm_transform)[1:500]
wa <- presto::wilcoxauc(X = norm_transform[feats, ], y = y, groups_use = c("target", "rest"))
wa <- wa[wa$group == "target", ]
write.csv(data.frame(feature = wa$feature, auc = wa$auc, pval = wa$pval,
                     pct_in = wa$pct_in, pct_out = wa$pct_out),
          file.path(OUT, "wilcoxauc_reference.csv"), row.names = FALSE)
write.csv(data.frame(barcode = colnames(norm_transform), group = y),
          file.path(OUT, "wilcoxauc_groups.csv"), row.names = FALSE)

## ------------------------------------------------------------------- stage 4

met <- metrics$metrics
met$qc.pass <- "fail"
met[met$barcode %in% valid, "qc.pass"] <- "pass"

# soft labels only: call with train = FALSE to get the deterministic part
set.seed(42)
soft <- valiDrops::label_dead(counts = counts, metrics = met,
                              qc.labels = setNames(as.character(met$qc.pass), met$barcode),
                              train = FALSE, plot = FALSE)
write.csv(data.frame(barcode = soft$metrics$barcode,
                     score = soft$metrics$score,
                     soft_label = as.character(soft$metrics$label)),
          file.path(OUT, "stage4_soft_labels.csv"), row.names = FALSE)
write.csv(data.frame(key = c("flag", "n_dead"),
                     value = c(soft$flag, sum(soft$metrics$label == "dead"))),
          file.path(OUT, "stage4_meta.csv"), row.names = FALSE)

# full trained labels
set.seed(42)
trained <- valiDrops::label_dead(counts = counts, metrics = met,
                                 qc.labels = setNames(as.character(met$qc.pass), met$barcode),
                                 plot = FALSE)
write.csv(data.frame(barcode = trained$metrics$barcode,
                     label = as.character(trained$metrics$label)),
          file.path(OUT, "stage4_final.csv"), row.names = FALSE)

## -------------------------------------------------------------- end-to-end

set.seed(42)
full <- valiDrops(counts, plot = FALSE)
write.csv(full, file.path(OUT, "pbmc4k_full_pipeline.csv"), row.names = FALSE)

dev.off()
cat("done\n")
```

- [ ] **Step 4: Run the script**

```bash
cd /Users/tuhu/Projects/validrops && Rscript tests/R/generate_reference.R 2>&1 | tail -30
ls tests/reference_outputs/
```

If a call errors, fix the script — do not hand-write a fixture. Two known hazards: `label_dead` needs `metrics$qc` named exactly as the R source expects (`valiDrops.R:126` passes `qc.labels`), and `expression_metrics` needs `presto` loaded before it checks `require("presto")`.

- [ ] **Step 5: Reconcile the test's expected columns with what was produced**

The test in Step 1 lists `stage3_clusters.csv` with a `shallow` column, but `expression_metrics` returns only the deep assignment. Update `EXPECTED` in `tests/test_fixtures_present.py` to match reality:

```python
    "stage3_clusters_deep.csv": ["barcode", "deep"],
```

and remove the `stage3_clusters.csv` entry. Also add:

```python
    "uik_inputs.csv": ["case", "x", "y"],
    "segmented_inputs.csv": ["case", "x", "y", "npsi"],
    "wilcoxauc_groups.csv": ["barcode", "group"],
```

`stage2_filters.csv` also gains a `final` column — add it to that entry's list.

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_fixtures_present.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add tests/R/generate_reference.R tests/reference_outputs tests/test_fixtures_present.py
git commit -m "test: generate per-stage R reference fixtures"
```

---

## Task 4: Fixture loading and barcode mapping

**Files:**
- Modify: `tests/conftest.py`
- Create: `tests/test_barcode_mapping.py`

**Interfaces:**
- Produces: pytest fixtures `raw_adata` (AnnData, cells × genes, `obs_names` = `cell_1`…`cell_N`), `ref(name)` (callable returning a DataFrame for a fixture CSV), `ref_dir` (Path).

**Why this task exists:** every concordance test depends on `cell_N` meaning "column N-1 of the raw matrix". If that mapping is wrong, every downstream test is wrong in a way that looks like an algorithm bug. Pin it once, first.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_barcode_mapping.py
import numpy as np


def test_obs_names_are_positional(raw_adata):
    assert raw_adata.obs_names[0] == "cell_1"
    assert raw_adata.obs_names[14] == "cell_15"
    assert raw_adata.n_obs == len(raw_adata.obs_names)


def test_stage2_metrics_match_recomputed_totals(raw_adata, ref):
    """cell_N must be column N-1 of the R matrix: verify via log total counts."""
    m = ref("stage2_metrics.csv").set_index("barcode")
    sub = raw_adata[m.index]
    totals = np.asarray(sub.X.sum(axis=1)).ravel()
    np.testing.assert_allclose(np.log(totals), m["logUMIs"].to_numpy(), rtol=1e-10)


def test_stage2_metrics_match_recomputed_features(raw_adata, ref):
    m = ref("stage2_metrics.csv").set_index("barcode")
    sub = raw_adata[m.index]
    n_genes = np.asarray((sub.X > 0).sum(axis=1)).ravel()
    np.testing.assert_allclose(np.log(n_genes), m["logFeatures"].to_numpy(), rtol=1e-10)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_barcode_mapping.py -v`
Expected: FAIL — fixtures `raw_adata` and `ref` not defined.

- [ ] **Step 3: Write the conftest**

```python
# tests/conftest.py
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

REF_DIR = Path(__file__).parent / "reference_outputs"
DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def ref_dir() -> Path:
    return REF_DIR


@pytest.fixture(scope="session")
def ref():
    """Load a reference CSV by filename."""

    def _load(name: str) -> pd.DataFrame:
        path = REF_DIR / name
        if not path.exists():
            pytest.skip(f"fixture {name} missing; run tests/R/generate_reference.R")
        return pd.read_csv(path)

    return _load


@pytest.fixture(scope="session")
def raw_adata():
    """PBMC 4K raw matrix, cells x genes, with R's positional barcode names.

    R read the matrix without column names, so valiDrops.R:65 assigned
    cell_1 .. cell_N by column index. scanpy.read_10x_h5 preserves the file's
    column order, so cell_N is obs position N-1.
    """
    sc = pytest.importorskip("scanpy")
    path = DATA_DIR / "pbmc4k" / "raw.h5"
    if not path.exists():
        pytest.skip("tests/data/pbmc4k/raw.h5 missing")
    adata = sc.read_10x_h5(path)
    adata.var_names_make_unique()
    adata.obs_names = [f"cell_{i + 1}" for i in range(adata.n_obs)]
    return adata


@pytest.fixture
def adata():
    """Tiny synthetic object for unit tests that need an AnnData but no real data."""
    a = ad.AnnData(X=np.array([[1.2, 2.3], [3.4, 4.5], [5.6, 6.7]]).astype(np.float32))
    a.layers["scaled"] = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]).astype(np.float32)
    return a
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_barcode_mapping.py -v`
Expected: PASS (3 tests)

If `test_stage2_metrics_match_recomputed_totals` fails, the R script used a different gene set or ordering — **stop and fix the mapping before continuing**. Every later task depends on it. Check whether `var_names_make_unique` changed the gene count relative to R's `rownames(counts) <- rowData(sce)$Symbol`, which does not deduplicate.

- [ ] **Step 5: Commit**

```bash
git add tests/conftest.py tests/test_barcode_mapping.py
git commit -m "test: fixture loading and positional barcode mapping"
```

---

## Task 5: `sn()` and `rollmean()`

**Files:**
- Create: `src/validrops/tl/_stats.py`
- Create: `tests/test_stats.py`

**Interfaces:**
- Produces:
  - `sn(x: np.ndarray) -> float` — Rousseeuw–Croux Sn scale estimator
  - `rollmean(x: np.ndarray, k: int) -> np.ndarray` — centred rolling mean, length `len(x) - k + 1`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_stats.py
import numpy as np
import pytest

from validrops.tl._stats import rollmean, sn


def test_sn_matches_r(ref):
    """Reference vectors were generated with the same seed in R; regenerate them here."""
    expected = ref("sn_reference.csv").set_index("name")["sn"]
    rng_vectors = _r_seeded_vectors()
    for name, x in rng_vectors.items():
        np.testing.assert_allclose(sn(x), expected[name], rtol=1e-10, err_msg=name)


def _r_seeded_vectors():
    """R's set.seed(42) streams cannot be reproduced in numpy, so the inputs
    themselves are read back from the fixture written by generate_reference.R."""
    import pandas as pd
    from pathlib import Path

    path = Path(__file__).parent / "reference_outputs" / "sn_inputs.csv"
    df = pd.read_csv(path)
    return {name: g["value"].to_numpy() for name, g in df.groupby("name")}


def test_sn_scale_equivariance():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    np.testing.assert_allclose(sn(3.0 * x), 3.0 * sn(x), rtol=1e-12)


def test_sn_translation_invariance():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    np.testing.assert_allclose(sn(x + 100.0), sn(x), rtol=1e-12)


def test_sn_constant_vector_is_zero():
    assert sn(np.full(50, 7.0)) == 0.0


def test_sn_single_value():
    assert sn(np.array([1.0])) == 0.0


def test_rollmean_matches_r(ref):
    df = ref("rollmean_reference.csv")
    x = df[df["case"] == "input"].sort_values("index")["value"].to_numpy()
    for k in (3, 4, 7, 8):
        expected = df[df["case"] == f"k{k}"].sort_values("index")["value"].to_numpy()
        got = rollmean(x, k)
        assert got.shape == expected.shape, f"k={k}"
        np.testing.assert_allclose(got, expected, rtol=1e-12, err_msg=f"k={k}")


def test_rollmean_window_larger_than_input():
    with pytest.raises(ValueError, match="window"):
        rollmean(np.arange(3.0), 5)
```

- [ ] **Step 2: Add the `sn` inputs to the reference script**

`sn_reference.csv` records only the results, so Python cannot reproduce R's `rnorm` stream. Add to `tests/R/generate_reference.R`, immediately after the `sn_reference.csv` write:

```r
sn_in <- do.call(rbind, lapply(names(test_vectors), function(n) {
  data.frame(name = n, value = test_vectors[[n]])
}))
write.csv(sn_in, file.path(OUT, "sn_inputs.csv"), row.names = FALSE)
```

Re-run: `Rscript tests/R/generate_reference.R`

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_stats.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'validrops.tl._stats'`

- [ ] **Step 4: Write the implementation**

```python
# src/validrops/tl/_stats.py
"""Scale estimation and smoothing primitives ported from R."""

import numpy as np

from .._constants import SN_C_SMALL, SN_CONSTANT


def _finite_sample_correction(n: int) -> float:
    """Rousseeuw & Croux (1993) finite-sample correction factor c_n."""
    if 2 <= n <= 9:
        return SN_C_SMALL[n - 2]
    if n % 2 == 0:
        return n / (n - 0.9)
    return 1.0


def sn(x: np.ndarray) -> float:
    """Rousseeuw-Croux Sn robust scale estimator.

    Ports ``robustbase::Sn``. Unlike the MAD this needs no location estimate
    and has 58% Gaussian efficiency.

    ``Sn = 1.1926 * c_n * lomed_i( himed_j |x_i - x_j| )`` where ``himed`` is
    the ``floor(n/2) + 1``-th order statistic and ``lomed`` the
    ``ceil(n/2)``-th, both 1-based.

    Parameters
    ----------
    x
        One-dimensional sample.

    Returns
    -------
    The scale estimate. Returns 0.0 for samples of fewer than two values.

    Notes
    -----
    The naive O(n^2) form is used. valiDrops calls this on at most a few tens
    of thousands of values, where it costs well under a second.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    if n < 2:
        return 0.0

    diffs = np.abs(x[:, None] - x[None, :])
    hi_idx = n // 2  # 0-based index of the (floor(n/2)+1)-th order statistic
    himed = np.partition(diffs, hi_idx, axis=1)[:, hi_idx]
    lo_idx = (n + 1) // 2 - 1  # 0-based index of the ceil(n/2)-th
    lomed = np.partition(himed, lo_idx)[lo_idx]

    return float(SN_CONSTANT * _finite_sample_correction(n) * lomed)


def rollmean(x: np.ndarray, k: int) -> np.ndarray:
    """Centred rolling mean, matching ``zoo::rollmean(x, k, align="center")``.

    For a plain numeric vector ``zoo`` returns the ``len(x) - k + 1``
    consecutive window means; ``align`` affects only the index of a zoo
    object, not the values.

    Parameters
    ----------
    x
        One-dimensional input.
    k
        Window width.

    Returns
    -------
    Array of length ``len(x) - k + 1``.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if k < 1 or k > x.size:
        raise ValueError(f"window k={k} must be between 1 and len(x)={x.size}")
    cumsum = np.concatenate(([0.0], np.cumsum(x)))
    return (cumsum[k:] - cumsum[:-k]) / k
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_stats.py -v`
Expected: PASS (7 tests)

If `test_sn_matches_r` fails by a constant factor, check `_finite_sample_correction` — the n=2..9 table is 1-indexed in the literature and 0-indexed here.

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check src/validrops/ --fix && uv run ruff format src/validrops/
git add src/validrops/tl/_stats.py tests/test_stats.py tests/R/generate_reference.R tests/reference_outputs/sn_inputs.csv
git commit -m "feat: Sn scale estimator and centred rolling mean"
```

---

## Task 6: `uik()` — unit-invariant knee

**Files:**
- Create: `src/validrops/tl/_uik.py`
- Create: `tests/test_uik.py`

**Interfaces:**
- Produces: `uik(x: np.ndarray, y: np.ndarray) -> float` — returns the x-value at the knee.

**Algorithm (read from the installed `inflection` package source):**

`uik(x, y)` is `x[ede(x, y, check_curve(x, y)$index)[1]]`, which unrolls to:

1. `check_curve` classifies the curve as convex/concave on each side by integrating the signed deviation from the chord at five split points (`1`, `Q1`, `Q2`, `Q3`, `n`), and returns `index = 1` for concave-left curves (`concave_convex` or `concave`), else `0`.
2. `ede` negates `y` when `index == 1`, computes `LF = y - chord(x)` where `chord` linearly interpolates between the first and last points, and returns `argmin(LF)`.
3. The knee is `x` at that index.

It is **not** a curvature method and is not equivalent to `kneed`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_uik.py
import numpy as np
import pytest

from validrops.tl._uik import uik


def test_uik_matches_r(ref):
    inputs = ref("uik_inputs.csv")
    expected = ref("uik_reference.csv").set_index("case")["knee"]
    for case, g in inputs.groupby("case"):
        x = g["x"].to_numpy()
        y = g["y"].to_numpy()
        np.testing.assert_allclose(uik(x, y), expected[case], rtol=1e-6, err_msg=case)


def test_uik_rejects_short_input():
    with pytest.raises(ValueError, match="at least"):
        uik(np.arange(3.0), np.arange(3.0))


def test_uik_returns_an_x_value():
    x = np.arange(1.0, 101.0)
    y = 100.0 / x
    assert uik(x, y) in set(x)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_uik.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# src/validrops/tl/_uik.py
"""Unit-invariant knee detection, ported from the R package ``inflection``."""

import numpy as np


def _chord(x1: float, y1: float, x2: float, y2: float, x: np.ndarray) -> np.ndarray:
    """``inflection::lin2`` — the straight line through two points, evaluated at x."""
    return y1 + (y2 - y1) * (x - x1) / (x2 - x1)


def _signed_areas(x: np.ndarray, y: np.ndarray, j: int) -> tuple[float, float]:
    """``inflection::findipl`` — trapezoidal integral of the chord deviation.

    Splits at 1-based index ``j`` and integrates ``y - chord`` over each side.
    Returns ``(left, right)``.
    """
    n = x.size
    left = slice(0, j)  # R's x[1:j]
    dxl = np.diff(x[left])
    fl = y[left] - _chord(x[0], y[0], x[j - 1], y[j - 1], x[left])
    sl = float(np.sum(dxl * 0.5 * (fl[:-1] + fl[1:])))

    right = slice(j - 1, n)  # R's x[j:n]
    dxr = np.diff(x[right])
    fr = y[right] - _chord(x[j - 1], y[j - 1], x[n - 1], y[n - 1], x[right])
    sr = float(np.sum(dxr * 0.5 * (fr[:-1] + fr[1:])))
    return sl, sr


def _check_curve_index(x: np.ndarray, y: np.ndarray) -> int:
    """``inflection::check_curve`` — returns 1 when the curve is concave on the left."""
    n = x.size
    # R: as.integer(quantile(1:N, p)) truncates toward zero
    quarts = [int(np.quantile(np.arange(1, n + 1), p)) for p in (0.25, 0.5, 0.75)]
    js = [1, *quarts, n]
    areas = [_signed_areas(x, y, j) for j in js]

    left_signs = np.sign([areas[1][0], areas[2][0], areas[3][0], areas[4][0]])
    right_signs = np.sign([areas[0][1], areas[1][1], areas[2][1], areas[3][1]])

    def classify(signs: np.ndarray) -> str:
        unique = np.unique(signs)
        ref_sign = unique[0] if unique.size == 1 else signs[0]
        return "concave" if ref_sign > 0 else "convex"

    left = classify(left_signs)
    right = classify(right_signs)
    # concave_convex and concave both yield index 1; convex_concave and convex yield 0
    return 1 if left == "concave" else 0


def uik(x: np.ndarray, y: np.ndarray) -> float:
    """Unit-invariant knee of a curve, ported from ``inflection::uik``.

    Parameters
    ----------
    x
        Strictly increasing x-coordinates.
    y
        Matching y-coordinates.

    Returns
    -------
    The x-value at the knee. Always one of the input ``x`` values.

    Notes
    -----
    This is the chord-deviation extremum used by ``inflection``, not the
    Kneedle algorithm implemented by the ``kneed`` package. The two disagree.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size <= 3:
        raise ValueError("uik needs at least 4 points; give a vector of 5 or more")
    if x.size != y.size:
        raise ValueError(f"x and y must be the same length, got {x.size} and {y.size}")

    if _check_curve_index(x, y) == 1:
        y = -y
    deviation = y - _chord(x[0], y[0], x[-1], y[-1], x)
    return float(x[int(np.argmin(deviation))])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_uik.py -v`
Expected: PASS (3 tests)

If a case disagrees, print `_check_curve_index` for it and compare against R's `inflection::check_curve(x, y)$index` — a flipped index inverts the answer.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src/validrops/ --fix && uv run ruff format src/validrops/
git add src/validrops/tl/_uik.py tests/test_uik.py
git commit -m "feat: unit-invariant knee detection ported from inflection::uik"
```

---

## Task 7: `segmented()` — Muggeo piecewise regression

**Files:**
- Create: `src/validrops/tl/_segmented.py`
- Create: `tests/test_segmented.py`

**Interfaces:**
- Produces:
  - `class SegmentedFitError(RuntimeError)`
  - `class SegmentedFit` — dataclass with `psi: np.ndarray`, `slopes: np.ndarray`, `residuals: np.ndarray`, `rmse: float`, `converged: bool`, `intercept: float`
  - `segmented(x, y, *, npsi=None, psi_init=None, alpha=0.0, max_iter=30, tol=1e-8, n_boot=0, random_state=0) -> SegmentedFit`

**Algorithm (Muggeo 2003).** At each iteration, with current breakpoints `ψ`, fit by OLS:

```
y ~ 1 + x + Σ_k β_k (x - ψ_k)_+ + Σ_k γ_k I(x > ψ_k)
```

then update `ψ_k ← ψ_k + γ_k / β_k`. At convergence `γ → 0`. The final model drops the `γ` terms. Segment slopes are the cumulative sums `slope_0 = b_x`, `slope_j = b_x + Σ_{k≤j} β_k`.

`alpha` bounds admissible breakpoints to `[quantile(x, alpha), quantile(x, 1-alpha)]`, matching `seg.control(alpha=)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_segmented.py
import numpy as np
import pytest

from validrops.tl._segmented import SegmentedFitError, segmented


def _case(inputs, ref_df, name):
    g = inputs[inputs["case"] == name]
    terms = ref_df[ref_df["case"] == name].set_index("term")["value"]
    return g["x"].to_numpy(), g["y"].to_numpy(), int(g["npsi"].iloc[0]), terms


def test_segmented_breakpoints_match_r(ref):
    inputs = ref("segmented_inputs.csv")
    expected = ref("segmented_reference.csv")
    for name in inputs["case"].unique():
        x, y, npsi, terms = _case(inputs, expected, name)
        psi_init = np.array([terms[f"psi_init{i + 1}"] for i in range(npsi)])
        fit = segmented(x, y, psi_init=psi_init, n_boot=0)
        want = np.array([terms[f"psi{i + 1}"] for i in range(npsi)])
        np.testing.assert_allclose(np.sort(fit.psi), np.sort(want), rtol=1e-6, err_msg=name)


def test_segmented_slopes_match_r(ref):
    inputs = ref("segmented_inputs.csv")
    expected = ref("segmented_reference.csv")
    for name in inputs["case"].unique():
        x, y, npsi, terms = _case(inputs, expected, name)
        psi_init = np.array([terms[f"psi_init{i + 1}"] for i in range(npsi)])
        fit = segmented(x, y, psi_init=psi_init, n_boot=0)
        want = np.array([terms[f"slope{i + 1}"] for i in range(npsi + 1)])
        np.testing.assert_allclose(fit.slopes, want, rtol=1e-6, err_msg=name)


def test_segmented_rmse_matches_r(ref):
    inputs = ref("segmented_inputs.csv")
    expected = ref("segmented_reference.csv")
    for name in inputs["case"].unique():
        x, y, npsi, terms = _case(inputs, expected, name)
        psi_init = np.array([terms[f"psi_init{i + 1}"] for i in range(npsi)])
        fit = segmented(x, y, psi_init=psi_init, n_boot=0)
        np.testing.assert_allclose(fit.rmse, terms["rmse"], rtol=1e-6, err_msg=name)


def test_segmented_recovers_a_known_breakpoint():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 500)
    y = np.where(x < 6.0, 1.0 * x, 6.0 + 4.0 * (x - 6.0)) + rng.normal(scale=0.05, size=500)
    fit = segmented(x, y, npsi=1)
    assert abs(fit.psi[0] - 6.0) < 0.1
    np.testing.assert_allclose(fit.slopes, [1.0, 4.0], atol=0.05)


def test_segmented_raises_when_no_breakpoint_exists():
    x = np.linspace(0, 10, 200)
    y = 2.0 * x  # perfectly linear, no curvature to place a break against
    with pytest.raises(SegmentedFitError):
        segmented(x, y, npsi=3, alpha=0.4)


def test_alpha_bounds_breakpoints():
    rng = np.random.default_rng(1)
    x = np.linspace(0, 10, 400)
    y = np.where(x < 1.0, 0.0, 5.0 * (x - 1.0)) + rng.normal(scale=0.05, size=400)
    fit = segmented(x, y, npsi=1, alpha=0.3)
    lo, hi = np.quantile(x, 0.3), np.quantile(x, 0.7)
    assert lo <= fit.psi[0] <= hi
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_segmented.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# src/validrops/tl/_segmented.py
"""Muggeo's iterative segmented (piecewise linear) regression.

Ports the estimation procedure of the R package ``segmented``. ``pwlf`` is not
a substitute: it locates breakpoints by global optimisation, which converges to
different estimates on the same data.
"""

from dataclasses import dataclass

import numpy as np


class SegmentedFitError(RuntimeError):
    """Raised when no segmented model converges for the given data."""


@dataclass(frozen=True)
class SegmentedFit:
    """Result of a segmented regression."""

    psi: np.ndarray
    """Estimated breakpoints, ascending."""
    slopes: np.ndarray
    """Slope of each segment, length ``len(psi) + 1``."""
    intercept: float
    residuals: np.ndarray
    rmse: float
    converged: bool


def _default_psi_init(x: np.ndarray, npsi: int) -> np.ndarray:
    """R's default starting values: equally spaced interior quantiles."""
    probs = np.linspace(0.0, 1.0, npsi + 2)[1:-1]
    return np.quantile(x, probs)


def _design(x: np.ndarray, psi: np.ndarray) -> np.ndarray:
    """[1, x, (x-psi_k)_+ ..., I(x>psi_k) ...]"""
    n = x.size
    k = psi.size
    out = np.empty((n, 2 + 2 * k), dtype=np.float64)
    out[:, 0] = 1.0
    out[:, 1] = x
    for j, p in enumerate(psi):
        out[:, 2 + j] = np.maximum(x - p, 0.0)
        out[:, 2 + k + j] = -(x > p).astype(np.float64)
    return out


def _fit_once(
    x: np.ndarray, y: np.ndarray, psi: np.ndarray, lo: float, hi: float, max_iter: int, tol: float
) -> tuple[np.ndarray, bool]:
    """Run Muggeo's iteration from ``psi``. Returns (psi, converged)."""
    k = psi.size
    psi = np.sort(np.asarray(psi, dtype=np.float64))
    for _ in range(max_iter):
        design = _design(x, psi)
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        beta = coef[2 : 2 + k]
        gamma = coef[2 + k :]

        if np.any(np.abs(beta) < 1e-12):
            return psi, False
        step = gamma / beta
        new_psi = np.sort(psi + step)

        if np.any(new_psi <= lo) or np.any(new_psi >= hi):
            return psi, False
        if np.any(np.diff(new_psi) < 1e-9):
            return psi, False
        # every segment must retain observations
        edges = np.concatenate(([-np.inf], new_psi, [np.inf]))
        if np.any(np.histogram(x, bins=edges)[0] < 2):
            return psi, False

        converged = float(np.max(np.abs(step))) < tol
        psi = new_psi
        if converged:
            return psi, True
    return psi, False


def _final_model(x: np.ndarray, y: np.ndarray, psi: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Refit without the gamma terms. Returns (intercept, slopes, residuals)."""
    k = psi.size
    design = np.empty((x.size, 2 + k), dtype=np.float64)
    design[:, 0] = 1.0
    design[:, 1] = x
    for j, p in enumerate(psi):
        design[:, 2 + j] = np.maximum(x - p, 0.0)
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    residuals = y - design @ coef
    slopes = coef[1] + np.concatenate(([0.0], np.cumsum(coef[2:])))
    return float(coef[0]), slopes, residuals


def segmented(
    x: np.ndarray,
    y: np.ndarray,
    *,
    npsi: int | None = None,
    psi_init: np.ndarray | None = None,
    alpha: float = 0.0,
    max_iter: int = 30,
    tol: float = 1e-8,
    n_boot: int = 0,
    random_state: int = 0,
) -> SegmentedFit:
    """Fit a piecewise linear model with free breakpoints.

    Parameters
    ----------
    x, y
        Predictor and response, one-dimensional and the same length.
    npsi
        Number of breakpoints. Ignored when ``psi_init`` is given.
    psi_init
        Explicit starting values for the breakpoints.
    alpha
        Breakpoints are constrained to ``[quantile(x, alpha), quantile(x, 1-alpha)]``,
        matching ``segmented::seg.control(alpha=)``.
    max_iter
        Maximum Muggeo iterations per start.
    tol
        Convergence tolerance on the breakpoint update.
    n_boot
        Bootstrap restarts, as in ``seg.control(n.boot=)``. Each restart refits
        on a resampled dataset and uses the result as a new starting value; the
        lowest-RSS solution wins. Set to 0 for a deterministic fit.
    random_state
        Seed for the bootstrap restarts.

    Returns
    -------
    SegmentedFit

    Raises
    ------
    SegmentedFitError
        When no start converges to an admissible set of breakpoints.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size != y.size:
        raise ValueError(f"x and y must be the same length, got {x.size} and {y.size}")

    if psi_init is None:
        if npsi is None or npsi < 1:
            raise ValueError("give either psi_init or npsi >= 1")
        psi_init = _default_psi_init(x, int(npsi))
    psi_init = np.asarray(psi_init, dtype=np.float64).ravel()

    lo = float(np.quantile(x, alpha)) if alpha > 0 else float(x.min())
    hi = float(np.quantile(x, 1.0 - alpha)) if alpha > 0 else float(x.max())

    best: tuple[float, np.ndarray] | None = None
    psi, converged = _fit_once(x, y, psi_init, lo, hi, max_iter, tol)
    if converged:
        _, _, resid = _final_model(x, y, psi)
        best = (float(resid @ resid), psi)

    if n_boot > 0:
        rng = np.random.default_rng(random_state)
        start = psi if converged else psi_init
        for _ in range(n_boot):
            idx = rng.integers(0, x.size, size=x.size)
            boot_psi, boot_ok = _fit_once(x[idx], y[idx], start, lo, hi, max_iter, tol)
            if not boot_ok:
                continue
            cand_psi, cand_ok = _fit_once(x, y, boot_psi, lo, hi, max_iter, tol)
            if not cand_ok:
                continue
            _, _, resid = _final_model(x, y, cand_psi)
            rss = float(resid @ resid)
            if best is None or rss < best[0]:
                best = (rss, cand_psi)

    if best is None:
        raise SegmentedFitError(
            f"segmented regression did not converge for npsi={psi_init.size} "
            f"within [{lo:.6g}, {hi:.6g}]"
        )

    psi = np.sort(best[1])
    intercept, slopes, residuals = _final_model(x, y, psi)
    return SegmentedFit(
        psi=psi,
        slopes=slopes,
        intercept=intercept,
        residuals=residuals,
        rmse=float(np.sqrt(np.mean(residuals**2))),
        converged=True,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_segmented.py -v`
Expected: PASS (6 tests)

If the R-comparison tests are close but outside `rtol=1e-6`, the likely cause is a different convergence criterion, not a different algorithm. Tighten `tol` to `1e-12` and raise `max_iter` to 100 before loosening the assertion — and if you do loosen it, record the achieved tolerance in the commit message.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src/validrops/ --fix && uv run ruff format src/validrops/
git add src/validrops/tl/_segmented.py tests/test_segmented.py
git commit -m "feat: Muggeo segmented regression"
```

---

## Task 8: `deviance_feature_selection()`

**Files:**
- Create: `src/validrops/tl/_deviance.py`
- Create: `tests/test_deviance.py`

**Interfaces:**
- Produces: `deviance_feature_selection(counts: sparse matrix | np.ndarray) -> np.ndarray` — one binomial deviance per gene, given a **cells × genes** matrix.

**Algorithm (read from `scry:::.compute_deviance` and `scry:::sparseBinomialDeviance`).** With `sz` the per-cell totals, `p_ij = X_ij / sz_i`, and `π_j = colsum_j / Σsz`:

```
ll_sat_j  = Σ_i [ X_ij (log p_ij - log1p(-p_ij)) + sz_i log1p(-p_ij) ]
ll_null_j = colsum_j (log π_j - log1p(-π_j)) + Σsz · log1p(-π_j)
deviance_j = 2 (ll_sat_j - ll_null_j)
```

Both `log p` and `log1p(-p)` are evaluated only on structural non-zeros: at `p_ij = 0` the whole contribution is zero, so the sparse structure carries it.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_deviance.py
import numpy as np
import pytest
import scipy.sparse as sp

from validrops.tl._deviance import deviance_feature_selection


def test_deviance_matches_r(ref, raw_adata):
    """Compare against scry on the same protein-coding, QC-passed submatrix."""
    expected = ref("deviance_reference.csv").set_index("gene")["deviance"]
    genes = expected.index.to_numpy()
    barcodes = ref("stage3_barcodes.csv")["barcode"].to_numpy()
    # R built this from qc-passing barcodes; use the stage-2 survivors instead,
    # which is the same submatrix expression_metrics saw.
    filters = ref("stage2_filters.csv")
    keep = filters.loc[filters["final"], "barcode"].to_numpy()
    sub = raw_adata[keep, genes]
    got = deviance_feature_selection(sub.X)
    np.testing.assert_allclose(got, expected.to_numpy(), rtol=1e-8)


def test_top_genes_overlap_r(ref, raw_adata):
    expected = ref("deviance_reference.csv").set_index("gene")["deviance"]
    genes = expected.index.to_numpy()
    filters = ref("stage2_filters.csv")
    keep = filters.loc[filters["final"], "barcode"].to_numpy()
    got = deviance_feature_selection(raw_adata[keep, genes].X)
    top_py = set(genes[np.argsort(-got)[:5000]])
    top_r = set(expected.sort_values(ascending=False).index[:5000])
    assert len(top_py & top_r) / 5000 > 0.99


def test_deviance_is_nonnegative_on_random_counts():
    rng = np.random.default_rng(0)
    X = sp.csr_matrix(rng.poisson(2.0, size=(200, 50)).astype(np.float64))
    dev = deviance_feature_selection(X)
    assert dev.shape == (50,)
    assert np.all(dev >= -1e-8)


def test_all_zero_gene_gives_zero_deviance():
    X = sp.csr_matrix(np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]))
    dev = deviance_feature_selection(X)
    assert dev[1] == 0.0


def test_dense_and_sparse_agree():
    rng = np.random.default_rng(1)
    dense = rng.poisson(1.5, size=(100, 20)).astype(np.float64)
    np.testing.assert_allclose(
        deviance_feature_selection(dense),
        deviance_feature_selection(sp.csr_matrix(dense)),
        rtol=1e-10,
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_deviance.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# src/validrops/tl/_deviance.py
"""Binomial deviance feature selection, ported from the R package ``scry``."""

import numpy as np
import scipy.sparse as sp


def deviance_feature_selection(counts) -> np.ndarray:
    """Per-gene binomial deviance against a constant-proportion null.

    Ports ``scry::devianceFeatureSelection`` with its default
    ``fam = "binomial"``. Genes with the largest deviance are the most
    informative; valiDrops keeps the top 5000.

    Parameters
    ----------
    counts
        Raw counts, **cells x genes** (the R function takes genes x cells;
        this is the AnnData orientation).

    Returns
    -------
    Deviance per gene, length ``counts.shape[1]``. Non-finite results are
    set to 0, matching ``.compute_deviance``'s ``out[is.na(out)] <- 0``.
    """
    is_sparse = sp.issparse(counts)
    X = counts.tocsr().astype(np.float64) if is_sparse else np.asarray(counts, dtype=np.float64)

    sz = np.asarray(X.sum(axis=1)).ravel()  # per-cell totals
    sz_sum = float(sz.sum())
    feature_sums = np.asarray(X.sum(axis=0)).ravel()  # per-gene totals

    if is_sparse:
        rows = X.tocoo().row
        p = X.data / sz[rows]
        log_p = np.log(p)
        log1p_neg = np.log1p(-p)
        contrib = X.data * (log_p - log1p_neg) + sz[rows] * log1p_neg
        ll_sat = np.zeros(X.shape[1], dtype=np.float64)
        np.add.at(ll_sat, X.tocoo().col, contrib)
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            p = X / sz[:, None]
            log_p = np.where(p > 0, np.log(np.where(p > 0, p, 1.0)), 0.0)
            log1p_neg = np.log1p(-p)
            contrib = np.where(p > 0, X * (log_p - log1p_neg) + sz[:, None] * log1p_neg, 0.0)
        ll_sat = contrib.sum(axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        pi = feature_sums / sz_sum
        l1p = np.log1p(-pi)
        ll_null = feature_sums * (np.log(pi) - l1p) + sz_sum * l1p

    deviance = 2.0 * (ll_sat - ll_null)
    deviance[~np.isfinite(deviance)] = 0.0
    return deviance
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_deviance.py -v`
Expected: PASS (5 tests)

`np.add.at` is correct but slow on large matrices. If the pbmc4k test takes more than ~30 s, replace it with `sp.csc_matrix((contrib, (rows, cols)), shape=X.shape).sum(axis=0)` — same result, one pass.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src/validrops/ --fix && uv run ruff format src/validrops/
git add src/validrops/tl/_deviance.py tests/test_deviance.py
git commit -m "feat: binomial deviance feature selection ported from scry"
```

---

## Task 9: `wilcoxauc()`

**Files:**
- Create: `src/validrops/tl/_wilcox.py`
- Create: `tests/test_wilcox.py`

**Interfaces:**
- Produces: `wilcoxauc(X, y: np.ndarray, groups_use: tuple[str, str]) -> pd.DataFrame` with columns `feature`, `auc`, `pval`, `pct_in`, `pct_out`. `X` is **genes × cells** (matching how valiDrops slices `norm_transform`), `y` is a per-cell group label array which may contain a third `"excluded"` level that is dropped.

**Algorithm.** Rank each gene's values across the retained cells (average ties). With `n1` cells in group 1 and `n2` in group 2:

```
R1  = Σ ranks in group 1
U   = R1 - n1(n1+1)/2
AUC = U / (n1 n2)
μ   = n1 n2 / 2
σ²  = n1 n2 (n+1) / 12  ·  tie correction  (1 - Σ(t³-t) / (n³-n))
z   = (U - μ) / σ
p   = 2 (1 - Φ(|z|))
```

Normal approximation with tie correction and **no** continuity correction — this is what `presto::wilcoxauc` does, and presto is what the R reference run uses.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_wilcox.py
import numpy as np
import pytest
import scipy.sparse as sp

from validrops.tl._wilcox import wilcoxauc


def test_wilcoxauc_matches_presto(ref, raw_adata):
    expected = ref("wilcoxauc_reference.csv").set_index("feature")
    groups = ref("wilcoxauc_groups.csv")
    filters = ref("stage2_filters.csv")
    keep = filters.loc[filters["final"], "barcode"].to_numpy()

    sub = raw_adata[groups["barcode"].to_numpy()]
    sf = 10000.0 / np.asarray(sub.X.sum(axis=1)).ravel()
    norm = sp.csr_matrix(sub.X).multiply(sf[:, None]).tocsr()
    norm.data = np.log1p(norm.data)

    genes = expected.index.to_numpy()
    gene_idx = [sub.var_names.get_loc(g) for g in genes]
    X = norm[:, gene_idx].T.tocsr()  # genes x cells

    got = wilcoxauc(X, groups["group"].to_numpy(), ("target", "rest")).set_index("feature")
    got.index = genes

    np.testing.assert_allclose(got["auc"], expected["auc"], rtol=1e-6)
    np.testing.assert_allclose(got["pval"], expected["pval"], rtol=1e-5, atol=1e-300)
    np.testing.assert_allclose(got["pct_in"], expected["pct_in"], rtol=1e-6)
    np.testing.assert_allclose(got["pct_out"], expected["pct_out"], rtol=1e-6)


def test_excluded_cells_are_dropped():
    X = np.array([[1.0, 2.0, 3.0, 100.0]])  # 1 gene, 4 cells
    y = np.array(["target", "target", "rest", "excluded"])
    out = wilcoxauc(X, y, ("target", "rest"))
    assert len(out) == 1
    # cell 4 excluded, so target {1,2} vs rest {3}: target always lower -> AUC 0
    assert out["auc"].iloc[0] == 0.0


def test_perfect_separation_gives_auc_one():
    X = np.array([[10.0, 11.0, 1.0, 2.0]])
    y = np.array(["target", "target", "rest", "rest"])
    out = wilcoxauc(X, y, ("target", "rest"))
    assert out["auc"].iloc[0] == 1.0


def test_all_ties_give_auc_half_and_p_one():
    X = np.array([[5.0, 5.0, 5.0, 5.0]])
    y = np.array(["target", "target", "rest", "rest"])
    out = wilcoxauc(X, y, ("target", "rest"))
    assert out["auc"].iloc[0] == 0.5
    assert out["pval"].iloc[0] == 1.0


def test_pct_columns_count_nonzero_fraction():
    X = np.array([[0.0, 1.0, 0.0, 0.0]])
    y = np.array(["target", "target", "rest", "rest"])
    out = wilcoxauc(X, y, ("target", "rest"))
    assert out["pct_in"].iloc[0] == pytest.approx(50.0)
    assert out["pct_out"].iloc[0] == pytest.approx(0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_wilcox.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# src/validrops/tl/_wilcox.py
"""Vectorised Wilcoxon rank-sum with AUC, ported from ``presto::wilcoxauc``."""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import norm as _normal
from scipy.stats import rankdata


def wilcoxauc(X, y: np.ndarray, groups_use: tuple[str, str]) -> pd.DataFrame:
    """Rank-sum test of ``groups_use[0]`` against ``groups_use[1]``, per feature.

    Parameters
    ----------
    X
        Expression, **genes x cells**. Dense or sparse.
    y
        Group label per cell. Labels outside ``groups_use`` are excluded,
        which is how ``expression_metrics.R:149-151`` uses a third
        ``"excluded"`` level.
    groups_use
        ``(in_group, out_group)``.

    Returns
    -------
    DataFrame with ``feature`` (positional index), ``auc``, ``pval``,
    ``pct_in``, ``pct_out``. Percentages are 0-100, matching presto.

    Notes
    -----
    Normal approximation with tie correction and no continuity correction.
    """
    in_group, out_group = groups_use
    y = np.asarray(y)
    mask = np.isin(y, [in_group, out_group])
    if not mask.any():
        raise ValueError(f"no cells labelled {in_group!r} or {out_group!r}")

    dense = X.toarray() if sp.issparse(X) else np.asarray(X, dtype=np.float64)
    dense = np.atleast_2d(dense)[:, mask]
    labels = y[mask]
    is_in = labels == in_group

    n1 = int(is_in.sum())
    n2 = int((~is_in).sum())
    n = n1 + n2
    if n1 == 0 or n2 == 0:
        raise ValueError(f"both groups must be non-empty, got n1={n1}, n2={n2}")

    ranks = np.apply_along_axis(rankdata, 1, dense)
    r1 = ranks[:, is_in].sum(axis=1)
    u = r1 - n1 * (n1 + 1) / 2.0
    auc = u / (n1 * n2)

    # tie correction: 1 - sum(t^3 - t) / (n^3 - n), per feature
    tie_term = np.empty(dense.shape[0], dtype=np.float64)
    for i in range(dense.shape[0]):
        _, counts = np.unique(dense[i], return_counts=True)
        tie_term[i] = np.sum(counts**3 - counts)
    correction = 1.0 - tie_term / (n**3 - n) if n > 1 else np.zeros_like(tie_term)

    mu = n1 * n2 / 2.0
    var = n1 * n2 * (n + 1) / 12.0 * correction
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (u - mu) / np.sqrt(var)
    z = np.where(np.isfinite(z), z, 0.0)
    pval = 2.0 * _normal.sf(np.abs(z))

    pct_in = (dense[:, is_in] > 0).sum(axis=1) / n1 * 100.0
    pct_out = (dense[:, ~is_in] > 0).sum(axis=1) / n2 * 100.0

    return pd.DataFrame(
        {
            "feature": np.arange(dense.shape[0]),
            "auc": auc,
            "pval": pval,
            "pct_in": pct_in,
            "pct_out": pct_out,
        }
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_wilcox.py -v`
Expected: PASS (5 tests)

The `np.apply_along_axis(rankdata, ...)` and per-feature tie loop densify the matrix. That is acceptable here — valiDrops only ever calls this on a pre-filtered feature set (`expression_metrics.R:144`), typically a few hundred genes. Do **not** optimise it into a sparse ranking; ranks over a sparse matrix must still account for the zeros, and getting that wrong silently changes every p-value.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src/validrops/ --fix && uv run ruff format src/validrops/
git add src/validrops/tl/_wilcox.py tests/test_wilcox.py
git commit -m "feat: vectorised wilcoxauc ported from presto"
```

---

## Task 10: SNN graph and Louvain clustering

**Files:**
- Create: `src/validrops/tl/_snn.py`
- Create: `tests/test_snn.py`
- Modify: `tests/R/generate_reference.R` (add the shallow cluster fixture)

**Interfaces:**
- Produces:
  - `snn_graph(embedding: np.ndarray, k: int = 20, prune: float = 1/15) -> scipy.sparse.csr_matrix` — symmetric Jaccard-weighted SNN adjacency
  - `louvain(adjacency, resolution: float, random_state: int = 0) -> np.ndarray` — integer cluster labels

**Algorithm (Seurat `FindNeighbors` / `FindClusters`).** Exact kNN including self (so each cell has `k` neighbours counting itself); SNN weight between cells `i` and `j` is the Jaccard index of their neighbour sets, `|N_i ∩ N_j| / |N_i ∪ N_j|`; edges with weight below `prune` are dropped. Clustering optimises modularity via Louvain with an RB-configuration resolution parameter.

- [ ] **Step 1: Add the shallow cluster fixture to the R script**

`expression_metrics` returns only the deep assignment, but Stage 3a needs the shallow one internally and the ARI test needs both. Insert into `tests/R/generate_reference.R` after the `expr <- valiDrops::expression_metrics(...)` call:

```r
# Re-derive the intermediate embedding and both clusterings, mirroring
# expression_metrics.R:58-94, so the Python port can be checked stage by stage.
nz <- counts.filtered[Matrix::rowSums(counts.filtered) > 0, ]
sf2 <- 10000 / Matrix::colSums(nz)
nt <- Matrix::t(Matrix::t(nz) * sf2)
nt@x <- log1p(nt@x)
dev2 <- scry::devianceFeatureSelection(nz)
vf <- names(which(rank(-dev2) <= 5000))
dat <- Matrix::t(nt[rownames(nt) %in% vf, ])
mu <- Matrix::colMeans(dat)
nr <- nrow(dat)
sds <- sqrt((Matrix::colMeans(dat * dat) - mu^2) * (nr / (nr - 1)))
sds[sds == 0] <- 1
scaled <- Matrix::t((Matrix::t(dat) - mu) / sds)
set.seed(42)
sv <- irlba::irlba(scaled, nv = 10, nu = 10)
emb <- sv$u %*% diag(sv$d)
rownames(emb) <- rownames(scaled)
colnames(emb) <- paste0("PC_", 1:10)
write.csv(data.frame(barcode = rownames(emb), emb),
          file.path(OUT, "stage3_embedding.csv"), row.names = FALSE)
snn <- Seurat::FindNeighbors(emb, verbose = FALSE)$snn
shallow <- Seurat::FindClusters(snn, verbose = FALSE, res = 0.1)
write.csv(data.frame(barcode = rownames(shallow),
                     shallow = shallow[, 1],
                     deep = expr$clusters[rownames(shallow), 1]),
          file.path(OUT, "stage3_clusters.csv"), row.names = FALSE)
```

Add `library(irlba)` and `library(Seurat)` to the script header. Re-run it, then restore `"stage3_clusters.csv": ["barcode", "shallow", "deep"]` to `EXPECTED` in `tests/test_fixtures_present.py` and add `"stage3_embedding.csv": ["barcode", "PC_1"]`.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_snn.py
import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from validrops.tl._snn import louvain, snn_graph


def test_snn_is_symmetric_and_pruned():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(200, 10))
    g = snn_graph(emb, k=20, prune=1 / 15)
    assert g.shape == (200, 200)
    diff = abs(g - g.T)
    assert diff.max() < 1e-12
    nonzero = g.data[g.data > 0]
    assert nonzero.min() >= 1 / 15


def test_snn_weights_are_jaccard_bounded():
    rng = np.random.default_rng(1)
    g = snn_graph(rng.normal(size=(150, 5)), k=10, prune=0.0)
    assert g.data.max() <= 1.0 + 1e-12
    assert g.data.min() >= 0.0


def test_louvain_separates_well_separated_blobs():
    rng = np.random.default_rng(2)
    emb = np.vstack([
        rng.normal(loc=0.0, scale=0.3, size=(100, 5)),
        rng.normal(loc=10.0, scale=0.3, size=(100, 5)),
        rng.normal(loc=-10.0, scale=0.3, size=(100, 5)),
    ])
    labels = louvain(snn_graph(emb, k=15), resolution=1.0, random_state=0)
    truth = np.repeat([0, 1, 2], 100)
    assert adjusted_rand_score(truth, labels) > 0.95


def test_louvain_is_deterministic_for_a_seed():
    rng = np.random.default_rng(3)
    g = snn_graph(rng.normal(size=(300, 8)), k=20)
    a = louvain(g, resolution=1.0, random_state=7)
    b = louvain(g, resolution=1.0, random_state=7)
    np.testing.assert_array_equal(a, b)


def test_higher_resolution_gives_more_clusters():
    rng = np.random.default_rng(4)
    g = snn_graph(rng.normal(size=(400, 10)), k=20)
    low = len(np.unique(louvain(g, resolution=0.1, random_state=0)))
    high = len(np.unique(louvain(g, resolution=8.0, random_state=0)))
    assert high > low


@pytest.mark.slow
def test_shallow_clusters_agree_with_seurat(ref):
    """The headline fidelity check for stage 3."""
    emb_df = ref("stage3_embedding.csv").set_index("barcode")
    clusters = ref("stage3_clusters.csv").set_index("barcode")
    emb = emb_df.loc[clusters.index].to_numpy()
    got = louvain(snn_graph(emb, k=20, prune=1 / 15), resolution=0.1, random_state=0)
    ari = adjusted_rand_score(clusters["shallow"].to_numpy(), got)
    assert ari > 0.9, f"shallow clustering ARI={ari:.3f} against Seurat"
```

Register the marker in `pyproject.toml` under `[tool.pytest]`:

```toml
markers = [ "slow: tests that take more than a few seconds" ]
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_snn.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Write the implementation**

```python
# src/validrops/tl/_snn.py
"""Shared-nearest-neighbour graph and Louvain clustering, matching Seurat."""

import igraph as ig
import leidenalg
import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

from .._constants import SNN_K, SNN_PRUNE


def snn_graph(embedding: np.ndarray, k: int = SNN_K, prune: float = SNN_PRUNE) -> sp.csr_matrix:
    """Jaccard-weighted shared-nearest-neighbour graph.

    Ports ``Seurat::FindNeighbors``: exact kNN including each cell itself,
    then an edge weight equal to the Jaccard index of the two neighbour sets,
    with weights below ``prune`` dropped.

    Parameters
    ----------
    embedding
        Cells x dimensions, e.g. PCA scores.
    k
        Neighbours per cell, counting the cell itself.
    prune
        Minimum Jaccard weight to retain an edge.

    Returns
    -------
    Symmetric sparse adjacency, cells x cells, zero diagonal.
    """
    n = embedding.shape[0]
    k = min(k, n)
    nn = NearestNeighbors(n_neighbors=k, algorithm="brute").fit(embedding)
    _, indices = nn.kneighbors(embedding)

    rows = np.repeat(np.arange(n), k)
    knn = sp.csr_matrix((np.ones(n * k), (rows, indices.ravel())), shape=(n, n))

    shared = (knn @ knn.T).tocoo()  # |N_i ∩ N_j|
    jaccard = shared.data / (2 * k - shared.data)  # |union| = k + k - |intersection|

    keep = (jaccard >= prune) & (shared.row != shared.col)
    graph = sp.csr_matrix(
        (jaccard[keep], (shared.row[keep], shared.col[keep])), shape=(n, n)
    )
    return graph.maximum(graph.T)


def louvain(adjacency: sp.spmatrix, resolution: float, random_state: int = 0) -> np.ndarray:
    """Modularity clustering at a given resolution.

    Ports ``Seurat::FindClusters``'s default Louvain algorithm. igraph's
    RB-configuration objective is the same modularity Seurat optimises.

    Parameters
    ----------
    adjacency
        Symmetric weighted graph from :func:`snn_graph`.
    resolution
        Higher values give more, smaller clusters.
    random_state
        Seed; the same seed always gives the same partition.

    Returns
    -------
    Integer label per cell, ordered by descending cluster size so that label 0
    is the largest cluster (Seurat's convention).
    """
    coo = sp.triu(adjacency.tocoo(), k=1).tocoo()
    graph = ig.Graph(n=adjacency.shape[0], edges=list(zip(coo.row.tolist(), coo.col.tolist())))
    graph.es["weight"] = coo.data.tolist()

    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        seed=random_state,
        n_iterations=-1,
    )

    raw = np.asarray(partition.membership)
    order = np.argsort(-np.bincount(raw))
    remap = np.empty(order.size, dtype=np.int64)
    remap[order] = np.arange(order.size)
    return remap[raw]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_snn.py -v -m "not slow"` then `uv run pytest tests/test_snn.py -v -m slow`
Expected: PASS (6 tests)

If `test_shallow_clusters_agree_with_seurat` reports ARI between 0.7 and 0.9, try `n_iterations=2` (Seurat runs a fixed number of Louvain passes rather than to convergence) before changing anything else. **Report the achieved ARI in the commit message either way** — this number is the honest measure of stage 3 fidelity and the spec commits to not tuning until it looks good.

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff check src/validrops/ --fix && uv run ruff format src/validrops/
git add src/validrops/tl/_snn.py tests/test_snn.py tests/R/generate_reference.R \
        tests/reference_outputs/stage3_clusters.csv tests/reference_outputs/stage3_embedding.csv \
        tests/test_fixtures_present.py pyproject.toml
git commit -m "feat: Seurat-compatible SNN graph and Louvain clustering (ARI=<measured>)"
```

---

## Task 11: Annotation detection and gene sets

**Files:**
- Create: `src/validrops/tl/_annotation.py`
- Create: `tests/test_annotation.py`

**Interfaces:**
- Produces:
  - `clean_gene_ids(names: np.ndarray) -> np.ndarray` — strips Ensembl version suffixes
  - `class AnnotationMatch` — dataclass with `species: str`, `column: str`, `n_mapped: int`, `n_total: int`
  - `detect_annotation(gene_names, *, species="auto", annotation="auto") -> AnnotationMatch`
  - `gene_sets(gene_names, match: AnnotationMatch) -> dict[str, np.ndarray]` — keys `mitochondrial`, `ribosomal`, `protein_coding`, values are **original** (uncleaned) gene names

- [ ] **Step 1: Write the failing test**

```python
# tests/test_annotation.py
import numpy as np

from validrops.tl._annotation import clean_gene_ids, detect_annotation, gene_sets


def test_clean_strips_ensembl_version():
    got = clean_gene_ids(np.array(["ENSG00000141510.17", "ENSG00000141510"]))
    assert list(got) == ["ENSG00000141510", "ENSG00000141510"]


def test_clean_strips_mouse_ensembl_version():
    got = clean_gene_ids(np.array(["ENSMUSG00000059552.14"]))
    assert list(got) == ["ENSMUSG00000059552"]


def test_clean_leaves_symbols_untouched():
    names = np.array(["TP53", "MT-CO1", "RPL13A", "HLA-DRB1"])
    np.testing.assert_array_equal(clean_gene_ids(names), names)


def test_detect_human_symbols(raw_adata, ref):
    expected = ref("annotation_detection.csv").iloc[0]
    match = detect_annotation(raw_adata.var_names.to_numpy())
    assert match.species == expected["species"]
    assert match.column == expected["column"]


def test_gene_sets_match_r(raw_adata, ref):
    expected = ref("annotation_genesets.csv")
    match = detect_annotation(raw_adata.var_names.to_numpy())
    got = gene_sets(raw_adata.var_names.to_numpy(), match)
    for name in ("mitochondrial", "ribosomal", "protein_coding"):
        want = set(expected.loc[expected["set"] == name, "gene"])
        assert set(got[name]) == want, name


def test_explicit_species_skips_detection(raw_adata):
    match = detect_annotation(raw_adata.var_names.to_numpy(), species="human", annotation="symbol")
    assert match.species == "human"
    assert match.column == "Symbol"


def test_gene_sets_return_original_names():
    names = np.array(["ENSG00000198804.2", "TP53"])
    match = detect_annotation(names, species="human", annotation="ensembl")
    sets = gene_sets(names, match)
    # the versioned name is what came in, so it must be what comes out
    assert "ENSG00000198804.2" in set(sets["mitochondrial"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_annotation.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# src/validrops/tl/_annotation.py
"""Species and gene-annotation detection, ported from ``quality_metrics.R:109-183``."""

import re
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files

import numpy as np
import pandas as pd

from .._constants import MITO_CHROMOSOMES

_ENSEMBL_PREFIX = re.compile(r"^(ENSG00|ENSMUSG00)")

_SPECIES_ALIASES = {
    "human": "human", "sapiens": "human", "h.sapiens": "human",
    "mouse": "mouse", "musculus": "mouse", "m.musculus": "mouse",
    "rat": "rat", "norvegicus": "rat", "r.norvegicus": "rat",
    "worm": "worm", "elegans": "worm", "c.elegans": "worm",
    "fly": "fly", "drosophila": "fly", "d.melanogaster": "fly",
    "zebrafish": "zebrafish", "d.rerio": "zebrafish",
}

_ANNOTATION_ALIASES = {
    "symbol": "Symbol", "entrez": "NCBI", "ncbi": "NCBI",
    "ensembl": "Ensembl", "hgnc": "HGNC", "mgi": "MGI",
}


@dataclass(frozen=True)
class AnnotationMatch:
    """Which species table and ID column best describe a set of gene names."""

    species: str
    column: str
    n_mapped: int
    n_total: int


@lru_cache(maxsize=1)
def _load_annotation() -> pd.DataFrame:
    path = files("validrops.data").joinpath("annotation.parquet")
    return pd.read_parquet(path)


def clean_gene_ids(names: np.ndarray) -> np.ndarray:
    """Strip GENCODE version suffixes from Ensembl identifiers.

    Ports ``quality_metrics.R:113-121``. Only names beginning ``ENSG00`` or
    ``ENSMUSG00`` are touched; everything else passes through unchanged,
    including symbols that legitimately contain dots.
    """
    names = np.asarray(names, dtype=object)
    out = names.copy()
    for i, name in enumerate(names):
        text = str(name)
        if _ENSEMBL_PREFIX.match(text):
            dot = text.find(".")
            if dot > 0:
                out[i] = text[:dot]
    return out.astype(str)


def _resolve_species(species: str) -> str | None:
    if species == "auto":
        return None
    key = species.lower()
    if key not in _SPECIES_ALIASES:
        raise ValueError(
            'species must be "auto", "human", "mouse", "rat", "C.elegans", '
            f'"drosophila" or "zebrafish", got {species!r}'
        )
    return _SPECIES_ALIASES[key]


def _resolve_annotation(annotation: str) -> str | None:
    if annotation == "auto":
        return None
    key = annotation.lower()
    if key not in _ANNOTATION_ALIASES:
        raise ValueError(
            'annotation must be "auto", "symbol", "ensembl", "entrez", "HGNC" or "MGI", '
            f"got {annotation!r}"
        )
    return _ANNOTATION_ALIASES[key]


def detect_annotation(
    gene_names: np.ndarray, *, species: str = "auto", annotation: str = "auto"
) -> AnnotationMatch:
    """Find the species table and ID column that maximises gene-name matches.

    Ports ``quality_metrics.R:123-150``. With ``annotation="auto"`` this scans
    **every** column of each candidate table, including ``Chr``, ``Type`` and
    ``Alias``. That is deliberate: it is what the R source does, and the
    winning column determines which ID space the gene sets are looked up in.

    Ties are broken toward the first table and then the first column, matching
    R's ``which.max``.
    """
    table = _load_annotation()
    cleaned = set(clean_gene_ids(gene_names).tolist())
    n_total = len(gene_names)

    want_species = _resolve_species(species)
    want_column = _resolve_annotation(annotation)

    candidates = table
    if want_species is not None:
        candidates = candidates[candidates["species"] == want_species]
    if want_column is not None:
        candidates = candidates[candidates["column_name"] == want_column]
    if candidates.empty:
        raise ValueError(f"no annotation table for species={species!r}, annotation={annotation!r}")

    hits = (
        candidates[candidates["value"].isin(cleaned)]
        .groupby(["species_index", "species", "column_index", "column_name"], observed=True)["value"]
        .nunique()
        .reset_index(name="n_mapped")
        .sort_values(["n_mapped", "species_index", "column_index"], ascending=[False, True, True])
    )
    if hits.empty:
        raise ValueError(
            "no gene names matched any annotation column; check that gene names are "
            "symbols, Ensembl or Entrez identifiers"
        )

    best = hits.iloc[0]
    return AnnotationMatch(
        species=str(best["species"]),
        column=str(best["column_name"]),
        n_mapped=int(best["n_mapped"]),
        n_total=n_total,
    )


def gene_sets(gene_names: np.ndarray, match: AnnotationMatch) -> dict[str, np.ndarray]:
    """Mitochondrial, ribosomal and protein-coding gene sets for the given names.

    Ports ``quality_metrics.R:152-174``. Returns the **original** input names,
    not the cleaned ones, so the result can index the count matrix directly.
    """
    table = _load_annotation()
    species = table[table["species"] == match.species]
    lookup = species[species["column_name"] == match.column]
    symbols = species[species["column_name"] == "Symbol"]

    cleaned = clean_gene_ids(gene_names)
    by_clean = pd.Series(gene_names, index=cleaned)

    coding_ids = set(lookup.loc[lookup["type"] == "protein_coding", "value"])
    mito_ids = set(lookup.loc[lookup["chr"].isin(MITO_CHROMOSOMES), "value"])

    ribo_rows = symbols["value"].str.lower().str.startswith(("rpl", "rps"))
    ribo_positions = symbols.index[ribo_rows]
    ribo_ids = set(lookup.loc[lookup.index.isin(ribo_positions), "value"])

    def select(ids: set[str]) -> np.ndarray:
        mask = np.isin(cleaned, list(ids))
        return np.asarray(by_clean.to_numpy()[mask], dtype=str)

    return {
        "mitochondrial": select(mito_ids),
        "ribosomal": select(ribo_ids),
        "protein_coding": select(coding_ids),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_annotation.py -v`
Expected: PASS (7 tests)

The ribosomal lookup is the fiddly one: R selects *rows* by `Symbol` prefix and then reads the **winning column's** value from those same rows (`quality_metrics.R:170`). Aligning rows across two column slices of a long table requires the row index to survive the filter — if `test_gene_sets_match_r` fails only on `ribosomal`, that alignment is the cause. Add a `row_id` column to the parquet extraction (`row_id = seq_len(nrow(d))` inside the per-column loop) and join on it instead of relying on the DataFrame index.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src/validrops/ --fix && uv run ruff format src/validrops/
git add src/validrops/tl/_annotation.py tests/test_annotation.py
git commit -m "feat: species and annotation auto-detection with gene sets"
```

---

## Task 12: Stage 1 — `pp.rank_barcodes`

**Files:**
- Create: `src/validrops/pp/rank_barcodes.py`
- Modify: `src/validrops/pp/__init__.py`
- Create: `tests/test_rank_barcodes.py`

**Interfaces:**
- Consumes: `validrops.tl._stats.rollmean`, `validrops.tl._segmented.segmented`, `SegmentedFitError`
- Produces: `rank_barcodes(adata, *, type="UMI", psi_min=2, psi_max=5, alpha=0.001, alpha_max=0.05, boot=10, factor=1.5, random_state=0) -> None`. Writes `adata.obs["rank_pass"]` (bool), `adata.uns["validrops"]["rank_threshold"]` (float), `adata.uns["validrops"]["barcode_ranks"]` (DataFrame indexed by barcode with `counts`, `rank`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rank_barcodes.py
import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

import validrops
from validrops._constants import UNS_KEY


def test_threshold_matches_r(raw_adata, ref):
    meta = ref("stage1_meta.csv").set_index("key")["value"]
    adata = raw_adata.copy()
    validrops.pp.rank_barcodes(adata)
    got = adata.uns[UNS_KEY]["rank_threshold"]
    np.testing.assert_allclose(got, float(meta["lower_threshold"]), rtol=1e-6)


def test_passing_barcodes_match_r(raw_adata, ref):
    expected = set(ref("stage1_threshold.csv")["barcode"])
    adata = raw_adata.copy()
    validrops.pp.rank_barcodes(adata)
    got = set(adata.obs_names[adata.obs["rank_pass"]])
    assert got == expected


def test_returns_none_and_mutates_in_place(raw_adata):
    adata = raw_adata.copy()
    assert validrops.pp.rank_barcodes(adata) is None
    assert "rank_pass" in adata.obs


def test_rank_pass_covers_all_barcodes(raw_adata):
    adata = raw_adata.copy()
    validrops.pp.rank_barcodes(adata)
    assert adata.obs["rank_pass"].shape[0] == adata.n_obs
    assert adata.obs["rank_pass"].dtype == bool


def test_genes_type_uses_detected_gene_counts():
    rng = np.random.default_rng(0)
    counts = rng.poisson(0.4, size=(3000, 200))
    counts[:50] *= 40  # a clear population of real cells
    adata = ad.AnnData(sp.csr_matrix(counts.astype(np.float32)))
    validrops.pp.rank_barcodes(adata, type="Genes")
    assert adata.obs["rank_pass"].sum() > 0
    assert adata.obs["rank_pass"].sum() < adata.n_obs


def test_invalid_type_rejected(adata):
    with pytest.raises(ValueError, match="UMI or Genes"):
        validrops.pp.rank_barcodes(adata, type="protein")


def test_psi_min_must_not_exceed_psi_max(adata):
    with pytest.raises(ValueError, match="psi_min"):
        validrops.pp.rank_barcodes(adata, psi_min=5, psi_max=2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_rank_barcodes.py -v`
Expected: FAIL — `module 'validrops.pp' has no attribute 'rank_barcodes'`

- [ ] **Step 3: Write the implementation**

```python
# src/validrops/pp/rank_barcodes.py
"""Stage 1: barcode-rank filtering. Ports ``rank_barcodes.R:31-150``."""

import logging

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.stats import rankdata

from .._constants import UNS_KEY
from ..tl._segmented import SegmentedFitError, segmented
from ..tl._stats import rollmean

logger = logging.getLogger(__name__)

_UMI_ALIASES = {"UMI", "umi", "UMIS", "umis", "UMIs"}
_GENE_ALIASES = {"Genes", "gene", "genes"}


def rank_barcodes(
    adata: AnnData,
    *,
    type: str = "UMI",
    psi_min: int = 2,
    psi_max: int = 5,
    alpha: float = 0.001,
    alpha_max: float = 0.05,
    boot: int = 10,
    factor: float = 1.5,
    random_state: int = 0,
) -> None:
    """Rank barcodes and detect the cut-off separating cells from empty droplets.

    Fits segmented regressions to the log-rank / log-count curve for a range of
    breakpoint counts, picks the simplest model within ``factor`` of the best
    RMSE, and takes the sharpest turn in the resulting slope sequence as the
    threshold.

    Parameters
    ----------
    adata
        Cells x genes. Not subset; results are written for every barcode.
    type
        ``"UMI"`` to rank by total counts, ``"Genes"`` by detected genes.
        UMI works better at low ambient contamination, genes at high.
    psi_min, psi_max
        Range of breakpoint counts to try.
    alpha
        Breakpoints are sought between the ``alpha`` and ``1-alpha`` quantiles.
        Incremented up to ``alpha_max`` if the fit fails.
    alpha_max
        Ceiling for the ``alpha`` escalation.
    boot
        Bootstrap restarts per segmented fit.
    factor
        How many folds above the best RMSE a simpler model may be.
    random_state
        Seed for the bootstrap restarts.

    Returns
    -------
    None. Writes ``adata.obs["rank_pass"]`` and
    ``adata.uns["validrops"]["rank_threshold"]``.
    """
    if type not in _UMI_ALIASES | _GENE_ALIASES:
        raise ValueError(f"type must be UMI or Genes, got {type!r}")
    if not 0 < psi_min <= psi_max:
        raise ValueError(f"psi_min must be >0 and <= psi_max, got {psi_min} and {psi_max}")

    counts = (
        np.asarray(adata.X.sum(axis=1)).ravel()
        if type in _UMI_ALIASES
        else np.asarray((adata.X > 0).sum(axis=1)).ravel()
    ).astype(np.float64)

    # rank is computed before zero-count barcodes are dropped (rank_barcodes.R:73-74)
    ranks_all = rankdata(-counts)
    nonzero = counts > 0
    frame = pd.DataFrame(
        {"counts": counts[nonzero], "rank": ranks_all[nonzero]},
        index=adata.obs_names[nonzero],
    )
    frame = frame.sort_values(["counts", "rank"], ascending=[False, False])

    unique = frame[~frame["counts"].duplicated()]
    log_counts = np.log(unique["counts"].to_numpy())
    log_ranks = np.log(unique["rank"].to_numpy())

    window = int(np.ceil(2 * len(unique) ** (1 / 3)))
    y = rollmean(log_counts, window)
    x = rollmean(log_ranks, window)

    fit, n_psi = _best_segmented_model(x, y, psi_min, psi_max, alpha, alpha_max, boot, factor, random_state)

    angles = _slope_angles(fit.slopes)
    # skip the first angle (rank_barcodes.R:127)
    best_break = int(np.argmin(angles[1:])) + 1
    nearest = int(np.argmin(np.abs(log_ranks - fit.psi[best_break])))
    threshold = float(np.exp(log_counts[nearest]))

    adata.obs["rank_pass"] = counts >= threshold
    uns = adata.uns.setdefault(UNS_KEY, {})
    uns["rank_threshold"] = threshold
    uns["barcode_ranks"] = frame
    uns["rank_npsi"] = n_psi

    n_pass = int(adata.obs["rank_pass"].sum())
    logger.info("Step 1: %d barcodes passed the rank threshold (%.1f counts)", n_pass, threshold)
    if n_pass > 20000:
        logger.warning(
            "More than 20,000 barcodes passed initial filtering. Breakpoint estimation may "
            "have failed; try increasing alpha, alpha_max or psi_max."
        )


def _best_segmented_model(x, y, psi_min, psi_max, alpha, alpha_max, boot, factor, random_state):
    """Fit each breakpoint count, then take the simplest model within ``factor`` of the best RMSE."""
    fits: list[tuple[int, float, object]] = []
    for npsi in range(psi_min, psi_max + 1):
        current_alpha = alpha
        while current_alpha <= alpha_max:
            psi_init = np.linspace(
                np.quantile(x, current_alpha), np.quantile(x, 1 - current_alpha), npsi
            )
            try:
                fit = segmented(
                    x, y,
                    psi_init=psi_init,
                    alpha=current_alpha - current_alpha / 1000,
                    n_boot=boot,
                    random_state=random_state,
                )
            except SegmentedFitError:
                current_alpha += alpha  # rank_barcodes.R:104
                continue
            fits.append((npsi, fit.rmse, fit))
            break

    if not fits:
        raise SegmentedFitError(
            f"no segmented model converged for psi in {psi_min}..{psi_max}; "
            "try increasing alpha, alpha_max or psi_max"
        )

    best_rmse = min(rmse for _, rmse, _ in fits)
    for npsi, rmse, fit in fits:  # fits are in ascending npsi order, so this is R's min(index)
        if rmse <= best_rmse * factor:
            return fit, npsi
    raise AssertionError("unreachable: the best model always satisfies the factor bound")


def _slope_angles(slopes: np.ndarray) -> np.ndarray:
    """Angle in degrees between each consecutive pair of segment slopes."""
    left = slopes[:-1]
    right = slopes[1:]
    return np.degrees(np.arctan((left - right) / (1 + left * right)))
```

Add to `src/validrops/pp/__init__.py`:

```python
from .rank_barcodes import rank_barcodes

__all__ = ["rank_barcodes"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_rank_barcodes.py -v`
Expected: PASS (7 tests)

If the threshold is close but the barcode set differs, the cause is almost always the rolling-mean window alignment (Task 5) shifting `x` relative to `log_ranks`. Verify `rollmean` against `rollmean_reference.csv` first, then print `fit.psi` alongside R's `out$psi[,2]`.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src/validrops/ --fix && uv run ruff format src/validrops/
git add src/validrops/pp/rank_barcodes.py src/validrops/pp/__init__.py tests/test_rank_barcodes.py
git commit -m "feat: stage 1 barcode-rank filtering"
```

---

## Task 13: Stage 2a — `tl.quality_metrics`

**Files:**
- Create: `src/validrops/tl/quality_metrics.py`
- Modify: `src/validrops/tl/__init__.py`
- Create: `tests/test_quality_metrics.py`

**Interfaces:**
- Consumes: `validrops.tl._annotation.detect_annotation`, `gene_sets`
- Produces: `quality_metrics(adata, *, contrast=None, contrast_type="denominator", species="auto", annotation="auto", mito="auto", ribo="auto", coding="auto", verbose=False) -> None`. Writes `adata.obs["log_umis"]`, `["log_features"]`, `["mitochondrial_fraction"]`, `["ribosomal_fraction"]`, `["coding_fraction"]`, optionally `["contrast_fraction"]`; `adata.var["mitochondrial"]`, `["ribosomal"]`, `["protein_coding"]` (bool); `adata.uns["validrops"]["gene_sets"]`, `["species"]`, `["annotation_column"]`, `["n_mapped"]`.

**Restriction:** metrics are computed only over barcodes where `adata.obs["rank_pass"]` is True, if that column exists; otherwise over all barcodes. Non-computed barcodes get `NaN`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_quality_metrics.py
import numpy as np
import pytest

import validrops
from validrops._constants import UNS_KEY


@pytest.fixture(scope="module")
def staged(raw_adata, ref):
    """Raw object with R's stage-1 result applied, so stage 2a is tested in isolation."""
    adata = raw_adata.copy()
    passing = set(ref("stage1_threshold.csv")["barcode"])
    adata.obs["rank_pass"] = [b in passing for b in adata.obs_names]
    validrops.tl.quality_metrics(adata)
    return adata


@pytest.mark.parametrize(
    ("obs_col", "ref_col"),
    [
        ("log_umis", "logUMIs"),
        ("log_features", "logFeatures"),
        ("mitochondrial_fraction", "mitochondrial_fraction"),
        ("ribosomal_fraction", "ribosomal_fraction"),
        ("coding_fraction", "coding_fraction"),
    ],
)
def test_metric_matches_r(staged, ref, obs_col, ref_col):
    expected = ref("stage2_metrics.csv").set_index("barcode")[ref_col]
    got = staged.obs.loc[expected.index, obs_col]
    np.testing.assert_allclose(got.to_numpy(), expected.to_numpy(), rtol=1e-8)
    assert np.corrcoef(got, expected)[0, 1] > 0.99


def test_non_rank_passing_barcodes_are_nan(staged):
    outside = staged.obs.loc[~staged.obs["rank_pass"], "log_umis"]
    assert outside.isna().all()


def test_gene_sets_written_to_var(staged, ref):
    expected = ref("annotation_genesets.csv")
    for name, col in [
        ("mitochondrial", "mitochondrial"),
        ("ribosomal", "ribosomal"),
        ("protein_coding", "protein_coding"),
    ]:
        want = set(expected.loc[expected["set"] == name, "gene"])
        got = set(staged.var_names[staged.var[col]])
        assert got == want, name


def test_detection_recorded_in_uns(staged):
    uns = staged.uns[UNS_KEY]
    assert uns["species"] == "human"
    assert uns["annotation_column"] == "Symbol"
    assert uns["n_mapped"] > 0


def test_explicit_gene_lists_bypass_detection(raw_adata):
    adata = raw_adata[:200].copy()
    mito = list(adata.var_names[:3])
    validrops.tl.quality_metrics(adata, mito=mito, ribo=list(adata.var_names[3:6]),
                                 coding=list(adata.var_names[6:20]))
    assert set(adata.var_names[adata.var["mitochondrial"]]) == set(mito)


def test_unknown_gene_in_explicit_list_raises(raw_adata):
    adata = raw_adata[:50].copy()
    with pytest.raises(ValueError, match="not present"):
        validrops.tl.quality_metrics(adata, mito=["NOT_A_REAL_GENE"])


def test_contrast_fraction_denominator(raw_adata):
    adata = raw_adata[:100].copy()
    contrast = adata.copy()
    contrast.X = contrast.X * 2
    validrops.tl.quality_metrics(adata, contrast=contrast, contrast_type="denominator")
    np.testing.assert_allclose(adata.obs["contrast_fraction"].to_numpy(), 0.5, rtol=1e-10)


def test_invalid_contrast_type_raises(raw_adata):
    adata = raw_adata[:50].copy()
    with pytest.raises(ValueError, match="denominator"):
        validrops.tl.quality_metrics(adata, contrast=adata.copy(), contrast_type="sideways")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_quality_metrics.py -v`
Expected: FAIL — `module 'validrops.tl' has no attribute 'quality_metrics'`

- [ ] **Step 3: Write the implementation**

```python
# src/validrops/tl/quality_metrics.py
"""Stage 2a: per-barcode quality metrics. Ports ``quality_metrics.R:32-217``."""

import logging

import numpy as np
from anndata import AnnData

from .._constants import UNS_KEY
from ._annotation import detect_annotation, gene_sets

logger = logging.getLogger(__name__)

_METRIC_COLUMNS = (
    "log_umis",
    "log_features",
    "mitochondrial_fraction",
    "ribosomal_fraction",
    "coding_fraction",
)


def quality_metrics(
    adata: AnnData,
    *,
    contrast: AnnData | None = None,
    contrast_type: str = "denominator",
    species: str = "auto",
    annotation: str = "auto",
    mito: str | list[str] = "auto",
    ribo: str | list[str] = "auto",
    coding: str | list[str] = "auto",
    verbose: bool = False,
) -> None:
    """Compute per-barcode quality metrics.

    Parameters
    ----------
    adata
        Cells x genes. Only barcodes with ``obs["rank_pass"]`` True are
        measured, if that column exists.
    contrast
        Optional second matrix for the exon fraction. For scRNA-seq the main
        matrix holds exonic reads and the contrast holds exon+intron, with
        ``contrast_type="numerator"``; for snRNA-seq it is the other way round.
    contrast_type
        ``"denominator"`` or ``"numerator"``.
    species, annotation
        ``"auto"`` to detect, or an explicit name.
    mito, ribo, coding
        ``"auto"`` to look up, or explicit gene-name lists.
    verbose
        Log the detected species, annotation and gene-set sizes.

    Returns
    -------
    None. Writes five metric columns to ``adata.obs`` and three boolean
    columns to ``adata.var``.
    """
    if contrast_type not in ("denominator", "numerator"):
        raise ValueError(f'contrast_type must be "denominator" or "numerator", got {contrast_type!r}')

    mask = (
        adata.obs["rank_pass"].to_numpy(dtype=bool)
        if "rank_pass" in adata.obs
        else np.ones(adata.n_obs, dtype=bool)
    )
    sub = adata[mask]
    gene_names = adata.var_names.to_numpy()

    match = detect_annotation(gene_names, species=species, annotation=annotation)
    detected = gene_sets(gene_names, match)

    sets = {
        "mitochondrial": _resolve_set(mito, detected["mitochondrial"], gene_names, "mitochondrial"),
        "ribosomal": _resolve_set(ribo, detected["ribosomal"], gene_names, "ribosomal"),
        "protein_coding": _resolve_set(coding, detected["protein_coding"], gene_names, "protein-coding"),
    }

    if verbose:
        logger.info(
            "Detected sample origin: %s. Detected gene annotation: %s. Mapped %d/%d (%.3g%%) of input IDs.",
            match.species, match.column, match.n_mapped, match.n_total,
            match.n_mapped / match.n_total * 100,
        )
        logger.info(
            "Found %d mitochondrial genes, %d ribosomal genes, and %d protein-coding genes.",
            len(sets["mitochondrial"]), len(sets["ribosomal"]), len(sets["protein_coding"]),
        )

    totals = np.asarray(sub.X.sum(axis=1)).ravel().astype(np.float64)
    n_features = np.asarray((sub.X > 0).sum(axis=1)).ravel().astype(np.float64)

    values = {
        "log_umis": np.log(totals),
        "log_features": np.log(n_features),
    }
    for key, column in (
        ("mitochondrial", "mitochondrial_fraction"),
        ("ribosomal", "ribosomal_fraction"),
        ("protein_coding", "coding_fraction"),
    ):
        selector = np.isin(gene_names, sets[key])
        values[column] = np.asarray(sub[:, selector].X.sum(axis=1)).ravel() / totals

    for column in _METRIC_COLUMNS:
        out = np.full(adata.n_obs, np.nan)
        out[mask] = values[column]
        adata.obs[column] = out

    if contrast is not None:
        shared = contrast[sub.obs_names]
        contrast_totals = np.asarray(shared.X.sum(axis=1)).ravel().astype(np.float64)
        fraction = totals / contrast_totals if contrast_type == "denominator" else contrast_totals / totals
        out = np.full(adata.n_obs, np.nan)
        out[mask] = fraction
        adata.obs["contrast_fraction"] = out

    for key, column in (
        ("mitochondrial", "mitochondrial"),
        ("ribosomal", "ribosomal"),
        ("protein_coding", "protein_coding"),
    ):
        adata.var[column] = np.isin(gene_names, sets[key])

    uns = adata.uns.setdefault(UNS_KEY, {})
    uns["gene_sets"] = sets
    uns["species"] = match.species
    uns["annotation_column"] = match.column
    uns["n_mapped"] = match.n_mapped


def _resolve_set(given, detected: np.ndarray, gene_names: np.ndarray, label: str) -> np.ndarray:
    """Use the caller's gene list when given, else the detected one."""
    if isinstance(given, str) and given == "auto":
        return detected
    requested = np.asarray(list(given), dtype=str)
    missing = set(requested) - set(gene_names)
    if missing:
        raise ValueError(
            f"{len(missing)} {label} gene(s) not present in the count matrix, "
            f"e.g. {sorted(missing)[:3]}"
        )
    return requested
```

Add to `src/validrops/tl/__init__.py`:

```python
from .quality_metrics import quality_metrics

__all__ = ["quality_metrics"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_quality_metrics.py -v`
Expected: PASS (12 tests)

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src/validrops/ --fix && uv run ruff format src/validrops/
git add src/validrops/tl/quality_metrics.py src/validrops/tl/__init__.py tests/test_quality_metrics.py
git commit -m "feat: stage 2a per-barcode quality metrics"
```

---

## Task 14: Stage 2b — `pp.quality_filter`

**Files:**
- Create: `src/validrops/pp/quality_filter.py`
- Modify: `src/validrops/pp/__init__.py`
- Create: `tests/test_quality_filter.py`

**Interfaces:**
- Consumes: `validrops.tl._stats.sn`, `validrops.tl._uik.uik`, `validrops.tl._segmented.segmented`, `SegmentedFitError`
- Produces: `quality_filter(adata, *, mito=True, distance=True, coding=True, contrast=False, mito_nreps=10, mito_max=0.3, npsi=3, dist_threshold=5, coding_threshold=3, contrast_threshold=3, random_state=0) -> None`. Writes `adata.obs["pass_mito"]`, `["pass_distance"]`, `["pass_coding"]`, `["pass_contrast"]`, `["qc_pass"]` (all bool); `adata.uns["validrops"]["mitochondrial_threshold"]`, `["mito_threshold_method"]`.

**The three sub-filters are sequential** — each sees only the survivors of the previous (`quality_filter.R:116`, `:155`, `:180`). A barcode that fails the mitochondrial filter is not even considered by the distance filter, so its `pass_distance` is `False` by construction, not by test.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_quality_filter.py
import numpy as np
import pytest

import validrops
from validrops._constants import UNS_KEY


@pytest.fixture(scope="module")
def staged(raw_adata, ref):
    adata = raw_adata.copy()
    passing = set(ref("stage1_threshold.csv")["barcode"])
    adata.obs["rank_pass"] = [b in passing for b in adata.obs_names]
    validrops.tl.quality_metrics(adata)
    validrops.pp.quality_filter(adata, random_state=0)
    return adata


def _concordance(got: set, want: set) -> float:
    return len(got & want) / len(got | want)


@pytest.mark.parametrize("column", ["pass_mito", "pass_distance", "pass_coding"])
def test_subfilter_concordance_with_r(staged, ref, column):
    expected = ref("stage2_filters.csv")
    want = set(expected.loc[expected[column], "barcode"])
    got = set(staged.obs_names[staged.obs[column]])
    score = _concordance(got, want)
    assert score > 0.95, f"{column} concordance {score:.4f}"


def test_final_qc_pass_concordance(staged, ref):
    expected = ref("stage2_filters.csv")
    want = set(expected.loc[expected["final"], "barcode"])
    got = set(staged.obs_names[staged.obs["qc_pass"]])
    assert _concordance(got, want) > 0.95


def test_mito_threshold_close_to_r(staged, ref):
    meta = ref("stage2_meta.csv").set_index("key")["value"]
    got = staged.uns[UNS_KEY]["mitochondrial_threshold"]
    np.testing.assert_allclose(got, float(meta["mitochondrial_threshold"]), rtol=0.1)


def test_threshold_respects_the_cap(staged):
    assert staged.uns[UNS_KEY]["mitochondrial_threshold"] <= 0.3 + 1e-12


def test_method_recorded(staged):
    assert staged.uns[UNS_KEY]["mito_threshold_method"] in {"gmm_uik", "segmented_fallback"}


def test_filters_are_sequential(staged):
    """A barcode failing the mito filter cannot pass the distance filter."""
    failed_mito = ~staged.obs["pass_mito"] & staged.obs["rank_pass"]
    assert not staged.obs.loc[failed_mito, "pass_distance"].any()


def test_numeric_mito_uses_the_given_threshold(raw_adata, ref):
    adata = raw_adata.copy()
    passing = set(ref("stage1_threshold.csv")["barcode"])
    adata.obs["rank_pass"] = [b in passing for b in adata.obs_names]
    validrops.tl.quality_metrics(adata)
    validrops.pp.quality_filter(adata, mito=0.2)
    assert adata.uns[UNS_KEY]["mitochondrial_threshold"] == 0.2
    kept = adata.obs.loc[adata.obs["pass_mito"], "mitochondrial_fraction"]
    assert kept.max() <= 0.2


def test_missing_metric_column_skips_filter(raw_adata, caplog):
    adata = raw_adata[:500].copy()
    adata.obs["rank_pass"] = True
    adata.obs["log_umis"] = 1.0
    adata.obs["log_features"] = 1.0
    adata.obs["mitochondrial_fraction"] = 0.05
    # coding_fraction deliberately absent
    validrops.pp.quality_filter(adata, mito=0.3, distance=False, coding=True)
    assert "coding_fraction" in caplog.text


def test_invalid_dist_threshold_rejected(adata):
    with pytest.raises(ValueError, match="dist_threshold"):
        validrops.pp.quality_filter(adata, dist_threshold=-1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_quality_filter.py -v`
Expected: FAIL — `module 'validrops.pp' has no attribute 'quality_filter'`

- [ ] **Step 3: Write the implementation**

```python
# src/validrops/pp/quality_filter.py
"""Stage 2b: filtering on quality metrics. Ports ``quality_filter.R:26-216``."""

import logging

import numpy as np
import pandas as pd
from anndata import AnnData
from sklearn.mixture import GaussianMixture

from .._constants import MITO_SCAN_INCREMENT, UNS_KEY
from ..tl._segmented import SegmentedFitError, segmented
from ..tl._stats import sn
from ..tl._uik import uik

logger = logging.getLogger(__name__)


def quality_filter(
    adata: AnnData,
    *,
    mito: bool | float = True,
    distance: bool = True,
    coding: bool = True,
    contrast: bool = False,
    mito_nreps: int = 10,
    mito_max: float = 0.3,
    npsi: int = 3,
    dist_threshold: float = 5,
    coding_threshold: float = 3,
    contrast_threshold: float = 3,
    random_state: int = 0,
) -> None:
    """Filter barcodes on the metrics from :func:`~validrops.tl.quality_metrics`.

    Three filters run in sequence, each seeing only the survivors of the last:
    a mitochondrial-fraction cap, a residual band around the feature-to-UMI
    relationship, and a band around the protein-coding fraction.

    Parameters
    ----------
    adata
        Must already carry the columns written by ``quality_metrics``.
    mito
        ``True`` to detect the threshold, a float to set it directly,
        ``False`` to skip.
    distance, coding, contrast
        Enable each sub-filter.
    mito_nreps
        Repetitions of the stochastic threshold search; the median wins.
    mito_max
        Above this, fall back to segmented regression (``quality_filter.R:79``).
    npsi
        Breakpoints for the feature-to-UMI fit, decremented on failure.
    dist_threshold, coding_threshold, contrast_threshold
        Multiples of Sn defining each band.
    random_state
        Seed for the mixture fits and subsampling.

    Returns
    -------
    None. Writes ``pass_mito``, ``pass_distance``, ``pass_coding``,
    ``pass_contrast`` and ``qc_pass`` to ``adata.obs``.
    """
    for name, value in (("dist_threshold", dist_threshold), ("coding_threshold", coding_threshold),
                        ("contrast_threshold", contrast_threshold), ("mito_nreps", mito_nreps)):
        if value <= 0:
            raise ValueError(f"{name} must be greater than 0, got {value}")
    if int(npsi) <= 0:
        raise ValueError(f"npsi must be greater than 0, got {npsi}")

    base = (
        adata.obs["rank_pass"].to_numpy(dtype=bool)
        if "rank_pass" in adata.obs
        else np.ones(adata.n_obs, dtype=bool)
    )
    metrics = adata.obs.loc[base]
    surviving = pd.Index(metrics.index)
    uns = adata.uns.setdefault(UNS_KEY, {})
    rng = np.random.default_rng(random_state)

    # ---- mitochondrial ------------------------------------------------------
    if mito is not False:
        if {"mitochondrial_fraction", "log_features"} <= set(metrics.columns):
            if mito is True:
                threshold, method = _detect_mito_threshold(
                    metrics.loc[surviving], mito_nreps, mito_max, rng, random_state
                )
            else:
                threshold, method = float(mito), "user"
            uns["mitochondrial_threshold"] = threshold
            uns["mito_threshold_method"] = method
            keep = metrics.loc[surviving, "mitochondrial_fraction"] <= threshold
            surviving = surviving[keep.to_numpy()]
        else:
            logger.warning(
                "Columns mitochondrial_fraction and log_features do not both exist. "
                "Skipping filtering using the mitochondrial fraction."
            )
    _write_pass(adata, "pass_mito", surviving)

    # ---- distance -----------------------------------------------------------
    if distance:
        if {"log_umis", "log_features"} <= set(metrics.columns):
            sub = metrics.loc[surviving]
            residuals = _feature_umi_residuals(
                sub["log_umis"].to_numpy(), sub["log_features"].to_numpy(), int(npsi)
            )
            spread = sn(residuals)
            centre = float(np.median(residuals))
            inside = (residuals <= centre + spread * dist_threshold) & (
                residuals >= centre - spread * dist_threshold
            )
            surviving = surviving[inside]
        else:
            logger.warning(
                "Columns log_umis and log_features do not both exist. Skipping filtering using the distance."
            )
    _write_pass(adata, "pass_distance", surviving)

    # ---- coding -------------------------------------------------------------
    if coding:
        surviving = _band_filter(metrics, surviving, "coding_fraction", coding_threshold)
    _write_pass(adata, "pass_coding", surviving)

    # ---- contrast -----------------------------------------------------------
    if contrast:
        surviving = _band_filter(metrics, surviving, "contrast_fraction", contrast_threshold)
    _write_pass(adata, "pass_contrast", surviving)

    _write_pass(adata, "qc_pass", surviving)
    logger.info("Step 3: %d barcodes passed quality filtering", len(surviving))


def _write_pass(adata: AnnData, column: str, surviving: pd.Index) -> None:
    adata.obs[column] = adata.obs_names.isin(surviving)


def _band_filter(metrics: pd.DataFrame, surviving: pd.Index, column: str, multiplier: float) -> pd.Index:
    """Keep barcodes within ``median +/- multiplier * Sn`` of ``column``."""
    if column not in metrics.columns:
        logger.warning("Column named %s does not exist. Skipping filtering using it.", column)
        return surviving
    values = metrics.loc[surviving, column].to_numpy()
    spread = sn(values)
    centre = float(np.median(values))
    inside = (values >= centre - spread * multiplier) & (values <= centre + spread * multiplier)
    return surviving[inside]


def _detect_mito_threshold(
    metrics: pd.DataFrame, nreps: int, mito_max: float, rng: np.random.Generator, random_state: int
) -> tuple[float, str]:
    """Threshold on the mitochondrial fraction, with R's segmented fallback."""
    thresholds = []
    log_features = metrics["log_features"].to_numpy().reshape(-1, 1)
    mito_fraction = metrics["mitochondrial_fraction"].to_numpy()

    for rep in range(nreps):
        model = GaussianMixture(n_components=2, random_state=random_state + rep, n_init=1)
        model.fit(log_features)
        high = int(np.argmax(model.means_.ravel()))
        group = model.predict(log_features) == high
        source = mito_fraction[group] if group.any() else mito_fraction
        sequence = np.arange(float(np.median(source)), 1.0, MITO_SCAN_INCREMENT)
        counts = np.array([np.sum(source <= value) for value in sequence], dtype=np.float64)
        thresholds.append(uik(sequence, counts))

    threshold = float(np.median(thresholds))
    if threshold <= mito_max:
        return threshold, "gmm_uik"

    # quality_filter.R:79-97 — subsampled segmented regression fallback
    logger.info("Mitochondrial threshold %.3f exceeded the cap; using segmented fallback", threshold)
    sample_size = min(5000, int(np.floor(len(metrics) * 0.8)))
    fallback = []
    for _ in range(nreps):
        idx = rng.choice(len(metrics), size=sample_size, replace=False)
        x = mito_fraction[idx]
        y = metrics["log_features"].to_numpy()[idx]
        psi_count = 1
        while True:
            try:
                fit = segmented(x, y, npsi=psi_count)
            except SegmentedFitError:
                if psi_count >= 5:
                    break
                psi_count += 1
                continue
            if float(fit.psi.min()) <= mito_max or psi_count >= 5:
                fallback.append(float(fit.psi.min()))
                break
            psi_count += 1
    if not fallback:
        raise SegmentedFitError("mitochondrial threshold fallback failed to converge")
    return float(np.median(fallback)), "segmented_fallback"


def _feature_umi_residuals(log_umis: np.ndarray, log_features: np.ndarray, npsi: int) -> np.ndarray:
    """Residuals of ``log_features ~ log_umis``, decrementing npsi on failure."""
    while npsi >= 1:
        try:
            return segmented(log_umis, log_features, npsi=npsi).residuals
        except SegmentedFitError:
            npsi -= 1
    # quality_filter.R:147 falls back to a plain linear fit
    slope, intercept = np.polyfit(log_umis, log_features, 1)
    return log_features - (slope * log_umis + intercept)
```

Update `src/validrops/pp/__init__.py`:

```python
from .quality_filter import quality_filter
from .rank_barcodes import rank_barcodes

__all__ = ["quality_filter", "rank_barcodes"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_quality_filter.py -v`
Expected: PASS (11 tests)

If `test_mito_threshold_close_to_r` fails, first check which path ran: `uns["mito_threshold_method"]`. R and Python taking different branches is a much larger discrepancy than either branch's numerical noise, and means the GMM component assignment differs. `sklearn`'s `GaussianMixture` defaults to k-means initialisation while `mixtools::normalmixEM` uses a random start — set `init_params="random_from_data"` if the assignment looks unstable across `mito_nreps`.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src/validrops/ --fix && uv run ruff format src/validrops/
git add src/validrops/pp/quality_filter.py src/validrops/pp/__init__.py tests/test_quality_filter.py
git commit -m "feat: stage 2b quality metric filtering"
```

---

## Task 15: Stage 3a — `tl.expression_metrics`

**Files:**
- Create: `src/validrops/tl/expression_metrics.py`
- Modify: `src/validrops/tl/__init__.py`
- Create: `tests/test_expression_metrics.py`

**Interfaces:**
- Consumes: `_deviance.deviance_feature_selection`, `_snn.snn_graph`, `_snn.louvain`, `_wilcox.wilcoxauc`
- Produces: `expression_metrics(adata, *, nfeats=5000, npcs=10, k_min=5, res_shallow=0.1, top_n=10, clusters=None, random_state=0) -> None`. Writes `adata.obs["cluster"]` (category, NaN outside), `adata.obs["cluster_shallow"]` (category), `adata.uns["validrops"]["cluster_stats"]` (DataFrame, 11 columns), `adata.obsm["X_validrops_pca"]` for the QC-passing subset.

**The `clusters=` parameter** accepts a Series mapping barcode → deep cluster label. When given, clustering is skipped and the supplied assignment is used. This exists so Stage 3b can be validated against R's own clustering, separating filter-logic fidelity from clustering fidelity.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_expression_metrics.py
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

import validrops
from validrops._constants import UNS_KEY

STAT_COLUMNS = [
    "cluster", "pct.diff", "pct.1", "pct.2", "n_de", "n_total",
    "n_negative", "min_fdr", "de_fraction", "mito_fraction", "ribo_fraction",
]


@pytest.fixture(scope="module")
def staged(raw_adata, ref):
    """Stage 3a run on R's stage-1 and stage-2 results, with R's clusters injected."""
    adata = raw_adata.copy()
    rank_pass = set(ref("stage1_threshold.csv")["barcode"])
    adata.obs["rank_pass"] = [b in rank_pass for b in adata.obs_names]
    validrops.tl.quality_metrics(adata)
    filters = ref("stage2_filters.csv")
    qc = set(filters.loc[filters["final"], "barcode"])
    adata.obs["qc_pass"] = [b in qc for b in adata.obs_names]

    r_clusters = ref("stage3_clusters.csv").set_index("barcode")
    validrops.tl.expression_metrics(
        adata, clusters=r_clusters["deep"], random_state=0
    )
    return adata


def test_stats_frame_has_r_columns(staged):
    stats = staged.uns[UNS_KEY]["cluster_stats"]
    assert list(stats.columns) == STAT_COLUMNS


@pytest.mark.parametrize("column", ["pct.diff", "pct.1", "pct.2", "de_fraction",
                                    "mito_fraction", "ribo_fraction"])
def test_continuous_stat_correlates_with_r(staged, ref, column):
    want = ref("stage3_stats.csv").set_index("cluster")[column]
    got = staged.uns[UNS_KEY]["cluster_stats"].set_index("cluster")[column]
    shared = want.index.intersection(got.index)
    assert len(shared) >= 0.9 * len(want)
    r = np.corrcoef(got.loc[shared], want.loc[shared])[0, 1]
    assert r > 0.99, f"{column} r={r:.4f}"


def test_de_counts_match_r(staged, ref):
    want = ref("stage3_stats.csv").set_index("cluster")["n_de"]
    got = staged.uns[UNS_KEY]["cluster_stats"].set_index("cluster")["n_de"]
    shared = want.index.intersection(got.index)
    ratio = (got.loc[shared] / want.loc[shared].replace(0, np.nan)).dropna()
    assert ratio.between(0.8, 1.25).mean() > 0.9


@pytest.mark.slow
def test_own_clustering_agrees_with_seurat(raw_adata, ref):
    adata = raw_adata.copy()
    rank_pass = set(ref("stage1_threshold.csv")["barcode"])
    adata.obs["rank_pass"] = [b in rank_pass for b in adata.obs_names]
    validrops.tl.quality_metrics(adata)
    filters = ref("stage2_filters.csv")
    qc = set(filters.loc[filters["final"], "barcode"])
    adata.obs["qc_pass"] = [b in qc for b in adata.obs_names]
    validrops.tl.expression_metrics(adata, random_state=0)

    r_clusters = ref("stage3_clusters.csv").set_index("barcode")
    shared = adata.obs_names.intersection(r_clusters.index)
    ari = adjusted_rand_score(
        r_clusters.loc[shared, "deep"], adata.obs.loc[shared, "cluster"].astype(int)
    )
    assert ari > 0.9, f"deep clustering ARI={ari:.3f}"


def test_barcodes_outside_qc_have_no_cluster(staged):
    outside = staged.obs.loc[~staged.obs["qc_pass"], "cluster"]
    assert outside.isna().all()


def test_injected_clusters_are_used_verbatim(staged, ref):
    r_clusters = ref("stage3_clusters.csv").set_index("barcode")["deep"]
    shared = staged.obs_names.intersection(r_clusters.index)
    np.testing.assert_array_equal(
        staged.obs.loc[shared, "cluster"].astype(int).to_numpy(),
        r_clusters.loc[shared].to_numpy(),
    )


def test_nfeats_must_exceed_npcs(raw_adata):
    adata = raw_adata[:100].copy()
    adata.obs["qc_pass"] = True
    with pytest.raises(ValueError, match="nfeats"):
        validrops.tl.expression_metrics(adata, nfeats=5, npcs=10)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_expression_metrics.py -v -m "not slow"`
Expected: FAIL — `module 'validrops.tl' has no attribute 'expression_metrics'`

- [ ] **Step 3: Write the implementation**

```python
# src/validrops/tl/expression_metrics.py
"""Stage 3a: expression-based cluster metrics. Ports ``expression_metrics.R:21-202``."""

import logging

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData
from scipy.sparse.linalg import svds

from .._constants import SHALLOW_RESOLUTION, UNS_KEY
from ._deviance import deviance_feature_selection
from ._snn import louvain, snn_graph
from ._wilcox import wilcoxauc

logger = logging.getLogger(__name__)

STAT_COLUMNS = [
    "cluster", "pct.diff", "pct.1", "pct.2", "n_de", "n_total",
    "n_negative", "min_fdr", "de_fraction", "mito_fraction", "ribo_fraction",
]


def expression_metrics(
    adata: AnnData,
    *,
    nfeats: int = 5000,
    npcs: int = 10,
    k_min: int = 5,
    res_shallow: float = SHALLOW_RESOLUTION,
    top_n: int = 10,
    clusters: pd.Series | None = None,
    random_state: int = 0,
) -> None:
    """Cluster QC-passing barcodes and compute per-cluster marker statistics.

    Clusters that fail to produce coherent markers are the signal Stage 3b
    filters on: a cluster of debris has no genes that distinguish it.

    Parameters
    ----------
    adata
        Must carry ``obs["qc_pass"]`` and ``var["protein_coding"]``.
    nfeats
        Deviance-selected variable features used for the embedding.
    npcs
        Singular vectors retained.
    k_min
        Target size of the smallest deep cluster.
    res_shallow
        Resolution for the coarse clustering that defines each cluster's
        background set.
    top_n
        Marker genes averaged into the summary percentages.
    clusters
        Optional barcode -> cluster mapping. When given, clustering is skipped.
        Used to validate the downstream filter independently of clustering.
    random_state
        Seed for the SVD and Louvain.

    Returns
    -------
    None. Writes ``obs["cluster"]``, ``obs["cluster_shallow"]`` and
    ``uns["validrops"]["cluster_stats"]``.
    """
    if nfeats <= 0 or nfeats < npcs:
        raise ValueError(f"nfeats must be > 0 and >= npcs, got nfeats={nfeats}, npcs={npcs}")
    if npcs <= 0 or k_min <= 0 or res_shallow <= 0 or top_n <= 0:
        raise ValueError("npcs, k_min, res_shallow and top_n must all be greater than 0")

    qc_mask = adata.obs["qc_pass"].to_numpy(dtype=bool)
    coding_mask = (
        adata.var["protein_coding"].to_numpy(dtype=bool)
        if "protein_coding" in adata.var
        else np.ones(adata.n_vars, dtype=bool)
    )
    sub = adata[qc_mask, coding_mask]
    barcodes = sub.obs_names

    counts = sp.csr_matrix(sub.X, dtype=np.float64)
    keep_genes = np.asarray(counts.sum(axis=0)).ravel() > 0
    counts = counts[:, keep_genes]
    gene_names = sub.var_names.to_numpy()[keep_genes]

    size_factors = 10000.0 / np.asarray(counts.sum(axis=1)).ravel()
    norm = counts.multiply(size_factors[:, None]).tocsr()
    norm.data = np.log1p(norm.data)

    deviance = deviance_feature_selection(counts)
    # R: names(which(rank(-dev) <= nfeats)); average ties, so take the top nfeats
    variable = np.argsort(-deviance, kind="stable")[: min(nfeats, deviance.size)]
    variable.sort()

    embedding = _embed(norm[:, variable], npcs, random_state)
    obsm = np.full((adata.n_obs, npcs), np.nan)
    obsm[qc_mask] = embedding
    adata.obsm["X_validrops_pca"] = obsm

    if clusters is None:
        graph = snn_graph(embedding)
        shallow = louvain(graph, res_shallow, random_state=random_state)
        deep = _deep_clustering(graph, k_min, random_state)
    else:
        aligned = clusters.reindex(barcodes)
        if aligned.isna().any():
            raise ValueError("clusters must cover every QC-passing barcode")
        deep = aligned.to_numpy().astype(int)
        graph = snn_graph(embedding)
        shallow = louvain(graph, res_shallow, random_state=random_state)

    stats = _cluster_stats(norm, gene_names, deep, shallow, adata, counts, top_n)

    _write_categorical(adata, "cluster", barcodes, deep)
    _write_categorical(adata, "cluster_shallow", barcodes, shallow)
    uns = adata.uns.setdefault(UNS_KEY, {})
    uns["cluster_stats"] = stats
    logger.info("Step 4: %d clusters with marker statistics", len(stats))


def _write_categorical(adata: AnnData, column: str, barcodes, values: np.ndarray) -> None:
    series = pd.Series(pd.NA, index=adata.obs_names, dtype="object")
    series.loc[barcodes] = values
    adata.obs[column] = series.astype("category")


def _embed(norm: sp.csr_matrix, npcs: int, random_state: int) -> np.ndarray:
    """Scale per gene, then SVD. Ports ``expression_metrics.R:74-87``."""
    dense = np.asarray(norm.todense(), dtype=np.float64)
    means = dense.mean(axis=0)
    n_rows = dense.shape[0]
    sds = np.sqrt((np.mean(dense * dense, axis=0) - means**2) * (n_rows / (n_rows - 1)))
    sds[sds == 0] = 1.0
    scaled = (dense - means) / sds

    u, s, _ = svds(scaled, k=npcs, random_state=random_state)
    order = np.argsort(-s)
    return u[:, order] * s[order]


def _deep_clustering(graph, k_min: int, random_state: int) -> np.ndarray:
    """Resolution search targeting a smallest cluster of ``k_min``.

    Ports ``expression_metrics.R:97-116``: a coarse 1..20 sweep, a +/-0.9
    refinement at 0.1 steps, then the largest resolution whose smallest
    cluster is exactly ``k_min``, falling back to the nearest achievable size.
    """
    def smallest(resolution: float) -> tuple[int, np.ndarray]:
        labels = louvain(graph, resolution, random_state=random_state)
        return int(np.bincount(labels).min()), labels

    coarse = {float(r): smallest(float(r))[0] for r in range(1, 21)}
    distances = {r: abs(v - k_min) for r, v in coarse.items()}
    closest = min(distances.values())
    near = [r for r, d in distances.items() if d == closest]

    fine = sorted({round(r + delta, 1) for r in near for delta in np.arange(-0.9, 0.95, 0.1)})
    fine = [r for r in fine if r > 0]
    sizes = {r: smallest(r)[0] for r in fine}

    exact = [r for r, v in sizes.items() if v == k_min]
    if not exact:
        target = min(sizes.values(), key=lambda v: abs(v - k_min))
        exact = [r for r, v in sizes.items() if v == target]
    return smallest(max(exact))[1]


def _cluster_stats(norm, gene_names, deep, shallow, adata, counts, top_n) -> pd.DataFrame:
    """Per-cluster marker statistics. Ports ``expression_metrics.R:118-197``."""
    mito = set(adata.uns[UNS_KEY]["gene_sets"]["mitochondrial"])
    ribo = set(adata.uns[UNS_KEY]["gene_sets"]["ribosomal"])
    mito_idx = np.isin(gene_names, list(mito))
    ribo_idx = np.isin(gene_names, list(ribo))
    totals = np.asarray(counts.sum(axis=1)).ravel()
    n_genes_total = norm.shape[1]

    rows = []
    for cluster in np.unique(deep):
        target = deep == cluster
        # background: everything outside the target's dominant shallow cluster
        dominant = np.bincount(shallow[target]).argmax()
        rest = shallow != dominant
        if target.sum() == 0 or rest.sum() == 0:
            continue

        pct1 = np.round(np.asarray((norm[target] > 0).sum(axis=0)).ravel() / target.sum(), 3)
        pct2 = np.round(np.asarray((norm[rest] > 0).sum(axis=0)).ravel() / rest.sum(), 3)
        with np.errstate(divide="ignore", invalid="ignore"):
            pct_diff = (pct1 - pct2) / pct1

        mean_target = np.asarray(np.expm1(norm[target].todense()).mean(axis=0)).ravel()
        mean_rest = np.asarray(np.expm1(norm[rest].todense()).mean(axis=0)).ravel()
        fold_change = np.log2(mean_target + 1) - np.log2(mean_rest + 1)

        eligible = (np.maximum(pct1, pct2) >= 0.1) & (fold_change >= 0.25)
        features = np.flatnonzero(eligible)
        if features.size < 2:
            continue

        labels = np.full(norm.shape[0], "excluded", dtype=object)
        labels[target] = "target"
        labels[rest] = "rest"
        result = wilcoxauc(norm[:, features].T, labels, ("target", "rest"))
        fdr = np.minimum(result["pval"].to_numpy() * n_genes_total, 1.0)  # bonferroni
        order = np.argsort(result["pval"].to_numpy(), kind="stable")
        n_de = int(np.sum(fdr <= 0.05))

        # expression_metrics.R:172 - R's 1:min(n_de, top_n) collapses to a single
        # gene when n_de == 0, because 1:0 is c(1, 0) and index 0 is dropped.
        n_top = min(n_de, top_n)
        top = order[:n_top] if n_top > 0 else order[:1]
        top_genes = features[top]

        rows.append({
            "cluster": int(cluster),
            "pct.diff": float(np.mean(pct_diff[top_genes])),
            "pct.1": float(np.mean(pct1[top_genes])),
            "pct.2": float(np.mean(pct2[top_genes])),
            "n_de": n_de,
            "n_total": int(features.size),
            "n_negative": int(np.sum(pct_diff[top_genes] < -0.01)),
            "min_fdr": float(fdr.min()),
            "de_fraction": float(n_de / features.size),
            "mito_fraction": float(
                np.median(np.asarray(counts[target][:, mito_idx].sum(axis=1)).ravel() / totals[target])
            ) if mito_idx.any() else 0.0,
            "ribo_fraction": float(
                np.median(np.asarray(counts[target][:, ribo_idx].sum(axis=1)).ravel() / totals[target])
            ) if ribo_idx.any() else 0.0,
        })

    return pd.DataFrame(rows, columns=STAT_COLUMNS)
```

Update `src/validrops/tl/__init__.py`:

```python
from .expression_metrics import expression_metrics
from .quality_metrics import quality_metrics

__all__ = ["expression_metrics", "quality_metrics"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_expression_metrics.py -v -m "not slow"` then the slow one separately.
Expected: PASS (11 tests)

`_embed` densifies the variable-feature matrix — for pbmc4k that is roughly 4000 × 5000 float64, about 160 MB. Acceptable. If memory becomes a problem on larger inputs, centre implicitly using a `LinearOperator` rather than materialising `scaled`.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src/validrops/ --fix && uv run ruff format src/validrops/
git add src/validrops/tl/expression_metrics.py src/validrops/tl/__init__.py tests/test_expression_metrics.py
git commit -m "feat: stage 3a expression-based cluster metrics"
```

---

## Task 16: Stage 3b — `pp.expression_filter`

**Files:**
- Create: `src/validrops/pp/expression_filter.py`
- Modify: `src/validrops/pp/__init__.py`
- Create: `tests/test_expression_filter.py`

**Interfaces:**
- Consumes: `_stats.sn`, `_segmented.segmented`, `SegmentedFitError`
- Produces: `expression_filter(adata, *, mito=3, ribo=3, min_significant=1, min_target_pct=0.3, max_background_pct=0.7, min_diff_pct=0.2, min_de_frac=0.01, min_significance_level=None) -> None`. Overwrites `adata.obs["qc_pass"]`; writes `adata.uns["validrops"]["min_significance_level"]` and `["surviving_clusters"]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_expression_filter.py
import numpy as np
import pandas as pd
import pytest

import validrops
from validrops._constants import UNS_KEY


@pytest.fixture
def staged_with_r_inputs(raw_adata, ref):
    """Feed R's own cluster stats and assignments in, so only the filter logic is under test."""
    adata = raw_adata.copy()
    clusters = ref("stage3_clusters.csv").set_index("barcode")
    adata.obs["cluster"] = pd.Series(
        clusters["deep"], index=clusters.index
    ).reindex(adata.obs_names).astype("category")
    adata.uns[UNS_KEY] = {"cluster_stats": ref("stage3_stats.csv")}
    return adata


def test_barcodes_match_r_exactly_given_r_inputs(staged_with_r_inputs, ref):
    validrops.pp.expression_filter(staged_with_r_inputs)
    got = set(staged_with_r_inputs.obs_names[staged_with_r_inputs.obs["qc_pass"]])
    want = set(ref("stage3_barcodes.csv")["barcode"])
    assert got == want


def test_significance_threshold_is_recorded(staged_with_r_inputs):
    validrops.pp.expression_filter(staged_with_r_inputs)
    level = staged_with_r_inputs.uns[UNS_KEY]["min_significance_level"]
    assert np.isfinite(level)
    assert level > 0


def test_explicit_significance_level_is_used(staged_with_r_inputs):
    validrops.pp.expression_filter(staged_with_r_inputs, min_significance_level=1e6)
    # nothing can clear that bar
    assert not staged_with_r_inputs.obs["qc_pass"].any()


def test_disabling_mito_filter_keeps_more_clusters(staged_with_r_inputs, ref):
    a = staged_with_r_inputs
    b = a.copy()
    validrops.pp.expression_filter(a, mito=3, ribo=3)
    validrops.pp.expression_filter(b, mito=None, ribo=None)
    assert set(a.uns[UNS_KEY]["surviving_clusters"]) <= set(b.uns[UNS_KEY]["surviving_clusters"])


def test_negative_enrichment_clusters_are_dropped(staged_with_r_inputs):
    validrops.pp.expression_filter(staged_with_r_inputs)
    stats = staged_with_r_inputs.uns[UNS_KEY]["cluster_stats"]
    survivors = set(staged_with_r_inputs.uns[UNS_KEY]["surviving_clusters"])
    dropped = stats.loc[stats["n_negative"] != 0, "cluster"]
    assert not (set(dropped) & survivors)


def test_invalid_min_target_pct_rejected(staged_with_r_inputs):
    with pytest.raises(ValueError, match="min_target_pct"):
        validrops.pp.expression_filter(staged_with_r_inputs, min_target_pct=1.5)


def test_invalid_mito_argument_rejected(staged_with_r_inputs):
    with pytest.raises(ValueError, match="mito"):
        validrops.pp.expression_filter(staged_with_r_inputs, mito="three")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_expression_filter.py -v`
Expected: FAIL — `module 'validrops.pp' has no attribute 'expression_filter'`

- [ ] **Step 3: Write the implementation**

```python
# src/validrops/pp/expression_filter.py
"""Stage 3b: filtering on expression metrics. Ports ``expression_filter.R:22-113``."""

import logging

import numpy as np
import pandas as pd
from anndata import AnnData

from .._constants import UNS_KEY
from ..tl._segmented import SegmentedFitError, segmented
from ..tl._stats import sn

logger = logging.getLogger(__name__)


def expression_filter(
    adata: AnnData,
    *,
    mito: float | None = 3,
    ribo: float | None = 3,
    min_significant: int = 1,
    min_target_pct: float = 0.3,
    max_background_pct: float = 0.7,
    min_diff_pct: float = 0.2,
    min_de_frac: float = 0.01,
    min_significance_level: float | None = None,
) -> None:
    """Keep barcodes belonging to clusters with coherent marker expression.

    A cluster of real cells has genes that are specifically expressed in it.
    A cluster of debris or ambient RNA does not. Each threshold below encodes
    one aspect of that distinction.

    Parameters
    ----------
    adata
        Must carry ``obs["cluster"]`` and ``uns["validrops"]["cluster_stats"]``.
    mito, ribo
        Deviations above the median cluster mitochondrial/ribosomal content
        beyond which a cluster is dropped. ``None`` disables the check.
    min_significant
        Minimum significant marker genes.
    min_target_pct
        Minimum mean fraction of in-cluster barcodes expressing the top markers.
    max_background_pct
        Maximum mean fraction of out-of-cluster barcodes expressing them.
    min_diff_pct
        Minimum in-versus-out difference.
    min_de_frac
        Minimum fraction of tested genes that must be significant.
    min_significance_level
        ``-log10`` significance the best marker must reach. ``None`` detects it.

    Returns
    -------
    None. Overwrites ``adata.obs["qc_pass"]``.
    """
    for name, value in (("min_target_pct", min_target_pct), ("max_background_pct", max_background_pct),
                        ("min_diff_pct", min_diff_pct)):
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1, got {value}")
    if min_significant < 0:
        raise ValueError(f"min_significant must be >= 0, got {min_significant}")
    for name, value in (("mito", mito), ("ribo", ribo)):
        if value is not None and not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be None or a number, got {value!r}")

    stats = adata.uns[UNS_KEY]["cluster_stats"]
    if min_significance_level is None:
        min_significance_level = _detect_significance_level(stats)
    elif min_significance_level < 0:
        raise ValueError(f"min_significance_level must be >= 0, got {min_significance_level}")

    keep = stats[stats["n_negative"] == 0]
    keep = keep[keep["pct.diff"] >= min_diff_pct]
    keep = keep[keep["pct.1"] >= min_target_pct]
    keep = keep[keep["pct.2"] <= max_background_pct]
    keep = keep[keep["n_de"] >= min_significant]
    keep = keep[-np.log10(keep["min_fdr"]) >= min_significance_level]
    keep = keep[keep["de_fraction"] > min_de_frac]

    if mito is not None:
        cap = np.median(stats["mito_fraction"]) + mito * sn(stats["mito_fraction"].to_numpy())
        allowed = set(stats.loc[stats["mito_fraction"] <= cap, "cluster"])
        keep = keep[keep["cluster"].isin(allowed)]
    if ribo is not None:
        cap = np.median(stats["ribo_fraction"]) + ribo * sn(stats["ribo_fraction"].to_numpy())
        allowed = set(stats.loc[stats["ribo_fraction"] <= cap, "cluster"])
        keep = keep[keep["cluster"].isin(allowed)]

    surviving = keep["cluster"].to_numpy()
    cluster = adata.obs["cluster"]
    valid = cluster.isin(surviving) & cluster.notna()
    adata.obs["qc_pass"] = valid.to_numpy(dtype=bool)

    uns = adata.uns.setdefault(UNS_KEY, {})
    uns["min_significance_level"] = float(min_significance_level)
    uns["surviving_clusters"] = surviving
    logger.info(
        "Step 5: %d of %d clusters passed, keeping %d barcodes",
        len(surviving), len(stats), int(adata.obs["qc_pass"].sum()),
    )


def _detect_significance_level(stats: pd.DataFrame) -> float:
    """Automatic significance threshold. Ports ``expression_filter.R:57-65``."""
    subset = stats[stats["min_fdr"] > 0]
    y = subset["pct.diff"].to_numpy()
    x = -np.log10(subset["min_fdr"].to_numpy())

    low = x[y <= 0.4]
    threshold = np.median(low) + sn(low) * 3 if low.size else np.nan

    try:
        model_level = float(segmented(x, y, npsi=1).psi[0])
    except (SegmentedFitError, ValueError):
        model_level = np.nan

    candidates = [v for v in (threshold, model_level) if np.isfinite(v)]
    if not candidates:
        raise ValueError(
            "could not determine a significance threshold; pass min_significance_level explicitly"
        )
    return float(min(candidates))
```

Update `src/validrops/pp/__init__.py`:

```python
from .expression_filter import expression_filter
from .quality_filter import quality_filter
from .rank_barcodes import rank_barcodes

__all__ = ["expression_filter", "quality_filter", "rank_barcodes"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_expression_filter.py -v`
Expected: PASS (7 tests)

`test_barcodes_match_r_exactly_given_r_inputs` asserts **exact** equality, not concordance, because both the cluster assignments and the cluster statistics come from R. If it fails, the bug is in this file's filter logic — nowhere else.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src/validrops/ --fix && uv run ruff format src/validrops/
git add src/validrops/pp/expression_filter.py src/validrops/pp/__init__.py tests/test_expression_filter.py
git commit -m "feat: stage 3b expression-based cluster filtering"
```

---

## Task 17: Stage 4a — dead-cell score and soft labelling

**Files:**
- Create: `src/validrops/pp/label_dead.py`
- Modify: `src/validrops/pp/__init__.py`
- Create: `tests/test_label_dead_soft.py`

**Interfaces:**
- Consumes: `_uik.uik`
- Produces:
  - `dead_score(log_umis, log_features, ribosomal, coding) -> np.ndarray`
  - `soft_label(score: np.ndarray, qc: np.ndarray, *, label_thrs=None, label_frac=0.1, n_relabel=1) -> tuple[np.ndarray, float, str]` — returns `(labels, threshold, flag)` where labels are `"live"`/`"dead"` and flag is `"Success"`/`"Caution"`/`"Failed"`
  - `label_dead(adata, *, train=True, ...) -> None` — training deferred to Task 18; this task implements it with `train=False` only

**This half is deterministic and is asserted exactly.** The score is a fixed linear form; the threshold search is a deterministic loop over quantiles.

The score (`label_dead.R:45-50`) — note the `/(π/2)` normalisation, which the project's Agentic Engineering Guide omits, and that the mitochondrial fraction is transformed but never enters the score:

```
U = log_umis - mean(log_umis)          # centred, not scaled
F = log_features - mean(log_features)
R = asin(sqrt(ribosomal_fraction)) / (π/2)
C = asin(sqrt(coding_fraction))  / (π/2)
score = -11.82·U + 2.08·F + 158.98·R + 18.87·F·C - 125.9·R·C
```

- [ ] **Step 1: Write the failing test**

```python
# tests/test_label_dead_soft.py
import numpy as np
import pytest

import validrops
from validrops._constants import UNS_KEY
from validrops.pp.label_dead import dead_score, soft_label


def test_score_matches_r(raw_adata, ref):
    expected = ref("stage4_soft_labels.csv").set_index("barcode")
    metrics = ref("stage2_metrics.csv").set_index("barcode").loc[expected.index]
    got = dead_score(
        metrics["logUMIs"].to_numpy(),
        metrics["logFeatures"].to_numpy(),
        metrics["ribosomal_fraction"].to_numpy(),
        metrics["coding_fraction"].to_numpy(),
    )
    np.testing.assert_allclose(got, expected["score"].to_numpy(), rtol=1e-10)


def test_soft_labels_match_r(raw_adata, ref):
    expected = ref("stage4_soft_labels.csv").set_index("barcode")
    filters = ref("stage2_filters.csv").set_index("barcode")
    qc = np.where(filters.loc[expected.index, "final"], "pass", "fail")
    labels, threshold, flag = soft_label(expected["score"].to_numpy(), qc)
    np.testing.assert_array_equal(labels, expected["soft_label"].to_numpy())


def test_score_uses_centred_logs():
    """A constant shift in log_umis must not change the score ordering."""
    kwargs = dict(
        log_features=np.array([1.0, 2.0, 3.0]),
        ribosomal=np.array([0.1, 0.2, 0.3]),
        coding=np.array([0.9, 0.8, 0.7]),
    )
    a = dead_score(np.array([1.0, 2.0, 3.0]), **kwargs)
    b = dead_score(np.array([101.0, 102.0, 103.0]), **kwargs)
    np.testing.assert_allclose(a, b, rtol=1e-12)


def test_score_normalises_fractions_by_half_pi():
    """A ribosomal fraction of 1.0 contributes exactly its coefficient."""
    score_one = dead_score(
        np.array([0.0, 0.0]), np.array([0.0, 0.0]),
        np.array([1.0, 1.0]), np.array([0.0, 0.0]),
    )
    np.testing.assert_allclose(score_one, 158.98, rtol=1e-10)


def test_too_few_dead_sets_failed_flag():
    score = np.linspace(0.0, 1.0, 100)
    qc = np.array(["pass"] * 100)
    _, _, flag = soft_label(score, qc, label_thrs=-1.0)  # nothing below the threshold
    assert flag == "Failed"


def test_too_many_dead_aborts_and_labels_all_live():
    score = np.linspace(0.0, 1.0, 100)
    qc = np.array(["pass"] * 100)
    labels, _, flag = soft_label(score, qc, label_thrs=0.9)  # 90% dead, over label_frac
    assert flag == "Failed"
    assert set(labels) == {"live"}


def test_explicit_threshold_is_used_directly():
    score = np.array([-5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0])
    qc = np.array(["pass"] * 10)
    labels, threshold, _ = soft_label(score, qc, label_thrs=1.0)
    assert threshold == 1.0
    assert list(labels[:2]) == ["dead", "dead"]


def test_label_dead_without_training_writes_obs(raw_adata, ref):
    adata = raw_adata.copy()
    metrics = ref("stage2_metrics.csv").set_index("barcode")
    for col, src in [("log_umis", "logUMIs"), ("log_features", "logFeatures"),
                     ("ribosomal_fraction", "ribosomal_fraction"),
                     ("coding_fraction", "coding_fraction")]:
        adata.obs[col] = metrics[src].reindex(adata.obs_names).to_numpy()
    adata.obs["rank_pass"] = adata.obs["log_umis"].notna()
    filters = ref("stage2_filters.csv").set_index("barcode")
    adata.obs["qc_pass"] = filters["final"].reindex(adata.obs_names).fillna(False).to_numpy()

    validrops.pp.label_dead(adata, train=False)
    assert "dead_score" in adata.obs
    assert set(adata.obs["label"].dropna().unique()) <= {"live", "dead", "uncertain"}
    assert adata.uns[UNS_KEY]["label_flag"] in {"Success", "Caution", "Failed"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_label_dead_soft.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# src/validrops/pp/label_dead.py
"""Stage 4: dead-cell labelling. Ports ``label_dead.R:43-483``."""

import logging

import numpy as np
from anndata import AnnData

from .._constants import DEAD_LABEL_FRAC, DEAD_SCORE_COEFFICIENTS, UNS_KEY
from ..tl._uik import uik

logger = logging.getLogger(__name__)

_QUANTILE_STEP = 0.0001
_QUANTILE_BLOCK = 0.1


def dead_score(
    log_umis: np.ndarray,
    log_features: np.ndarray,
    ribosomal: np.ndarray,
    coding: np.ndarray,
) -> np.ndarray:
    """Heuristic dead-cell score. Ports ``label_dead.R:45-50``.

    Dying cells lose cytoplasmic RNA first, so they show a low UMI count for
    their gene count and an inflated ribosomal fraction. The coefficients are
    fitted constants from the paper and must not be changed.

    Parameters
    ----------
    log_umis, log_features
        Natural logs of total counts and detected genes. Centred internally.
    ribosomal, coding
        Fractions in [0, 1]. Arcsine-square-root transformed and normalised
        by pi/2 internally.

    Returns
    -------
    One score per barcode. Lower means more dead-like.
    """
    u = np.asarray(log_umis, dtype=np.float64)
    f = np.asarray(log_features, dtype=np.float64)
    u = u - np.nanmean(u)
    f = f - np.nanmean(f)
    r = np.arcsin(np.sqrt(np.asarray(ribosomal, dtype=np.float64))) / (np.pi / 2)
    c = np.arcsin(np.sqrt(np.asarray(coding, dtype=np.float64))) / (np.pi / 2)

    k = DEAD_SCORE_COEFFICIENTS
    return (
        k["log_umis"] * u
        + k["log_features"] * f
        + k["ribosomal"] * r
        + k["features_x_coding"] * f * c
        + k["ribosomal_x_coding"] * r * c
    )


def _threshold_at(score: np.ndarray, max_quantile: float) -> float:
    """Knee of the quantile curve up to ``max_quantile``. Ports ``label_dead.R:61-68``."""
    breaks = np.arange(_QUANTILE_STEP, max_quantile + _QUANTILE_STEP / 2, _QUANTILE_STEP)
    values = np.quantile(score, breaks)
    return float(np.quantile(score, uik(breaks, values)))


def _contingency(labels: np.ndarray, qc: np.ndarray) -> np.ndarray:
    """2x2 table of (dead, live) x (fail, pass), in R's alphabetical order."""
    table = np.zeros((2, 2), dtype=np.int64)
    for i, label in enumerate(("dead", "live")):
        for j, status in enumerate(("fail", "pass")):
            table[i, j] = int(np.sum((labels == label) & (qc == status)))
    return table


def soft_label(
    score: np.ndarray,
    qc: np.ndarray,
    *,
    label_thrs: float | None = None,
    label_frac: float = DEAD_LABEL_FRAC,
    n_relabel: int = 1,
) -> tuple[np.ndarray, float, str]:
    """Split barcodes into live and dead by a score threshold.

    Ports ``label_dead.R:56-143``. When no threshold is given, the quantile
    ceiling is raised in steps of 0.1 until the dead/live by pass/fail table
    stops gaining QC-passing dead cells.

    Returns
    -------
    ``(labels, threshold, flag)``. ``flag`` is ``"Success"``, ``"Caution"``
    when barcodes had to be relabelled, or ``"Failed"`` when the split is
    unusable.
    """
    score = np.asarray(score, dtype=np.float64)
    qc = np.asarray(qc)
    flag = "Success"

    if label_thrs is None:
        max_quantile = _QUANTILE_BLOCK
        last_threshold = _threshold_at(score, max_quantile)
        last_table = _contingency(np.where(score <= last_threshold, "dead", "live"), qc)

        while True:
            max_quantile += _QUANTILE_BLOCK
            new_threshold = _threshold_at(score, max_quantile)
            new_table = _contingency(np.where(score <= new_threshold, "dead", "live"), qc)

            if new_table.min() > 0:
                if last_table.min() > 0:
                    if last_table[0, 1] == new_table[0, 1]:
                        last_table, last_threshold = new_table, new_threshold
                    else:  # last_table[0,1] < new_table[0,1]
                        label_thrs = last_threshold
                        break
                else:
                    label_thrs = new_threshold
                    break
            elif max_quantile >= 0.95:
                label_thrs = new_threshold
                flag = "Failed"
                break
            else:
                last_table, last_threshold = new_table, new_threshold

    labels = np.where(score <= label_thrs, "dead", "live")
    n_dead = int(np.sum(labels == "dead"))

    if n_dead < 3:
        logger.info("Soft-labeling identified fewer than 3 dead barcodes; returning soft labels only")
        flag = "Failed"
    elif np.sum((qc == "pass") & (labels == "dead")) == 0:
        logger.info("Soft-labeling labelled 0 QC-passing barcode as dead; relabeling %d", n_relabel)
        dead_idx = np.flatnonzero(labels == "dead")
        least_dead = dead_idx[np.argsort(-score[dead_idx])][:n_relabel]
        qc[least_dead] = "pass"
        flag = "Caution"
    elif n_dead / labels.size >= label_frac:
        logger.info("Soft-labeling identified more than %.0f%% of barcodes as dead; aborting",
                    label_frac * 100)
        labels = np.full(labels.size, "live")
        flag = "Failed"
    else:
        logger.info("Soft-labeling identified %d dead barcodes", n_dead)

    return labels, float(label_thrs), flag


def label_dead(adata: AnnData, *, train: bool = True, label_thrs: float | None = None,
               label_frac: float = DEAD_LABEL_FRAC, n_relabel: int = 1, **kwargs) -> None:
    """Label barcodes as live, dead or uncertain.

    Parameters
    ----------
    adata
        Must carry ``log_umis``, ``log_features``, ``ribosomal_fraction``,
        ``coding_fraction`` and ``qc_pass``.
    train
        Run the consensus training loop. ``False`` returns the soft labels.
    label_thrs
        Explicit score cutoff. ``None`` detects one.
    label_frac
        Abort if more than this fraction is labelled dead.
    n_relabel
        Barcodes to relabel when no QC-passing barcode is soft-labelled dead.

    Returns
    -------
    None. Writes ``obs["dead_score"]`` and ``obs["label"]``.
    """
    mask = (
        adata.obs["rank_pass"].to_numpy(dtype=bool)
        if "rank_pass" in adata.obs
        else np.ones(adata.n_obs, dtype=bool)
    )
    sub = adata.obs.loc[mask]
    score = dead_score(
        sub["log_umis"].to_numpy(),
        sub["log_features"].to_numpy(),
        sub["ribosomal_fraction"].to_numpy(),
        sub["coding_fraction"].to_numpy(),
    )
    qc = np.where(sub["qc_pass"].to_numpy(dtype=bool), "pass", "fail")
    labels, threshold, flag = soft_label(
        score, qc, label_thrs=label_thrs, label_frac=label_frac, n_relabel=n_relabel
    )

    if train and flag != "Failed":
        from ._label_dead_train import train_labels  # Task 18

        labels, flag = train_labels(adata, mask, score, labels, qc, threshold, flag, **kwargs)

    scores_out = np.full(adata.n_obs, np.nan)
    scores_out[mask] = score
    adata.obs["dead_score"] = scores_out

    labels_out = np.full(adata.n_obs, None, dtype=object)
    labels_out[mask] = labels
    adata.obs["label"] = labels_out
    adata.obs["label"] = adata.obs["label"].astype("category")

    uns = adata.uns.setdefault(UNS_KEY, {})
    uns["label_threshold"] = threshold
    uns["label_flag"] = flag
```

Update `src/validrops/pp/__init__.py` to also export `label_dead`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_label_dead_soft.py -v`
Expected: PASS (8 tests). `test_label_dead_without_training_writes_obs` needs `train=False`; the `_label_dead_train` import is inside the function so this task does not require Task 18 to exist.

If `test_soft_labels_match_r` disagrees on a handful of barcodes, the threshold escalation loop terminated one step early or late. Log `max_quantile` and the contingency table each iteration and compare against R by adding the same logging to a scratch copy of `label_dead.R`.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src/validrops/ --fix && uv run ruff format src/validrops/
git add src/validrops/pp/label_dead.py src/validrops/pp/__init__.py tests/test_label_dead_soft.py
git commit -m "feat: stage 4 dead-cell score and soft labelling"
```

---

## Task 18: Stage 4b — consensus training loop

**Files:**
- Create: `src/validrops/tl/_ridge.py`
- Create: `src/validrops/pp/_label_dead_train.py`
- Create: `tests/test_ridge.py`
- Create: `tests/test_label_dead_train.py`

**Interfaces:**
- Consumes: `_deviance` (not used — see quirk below), `scipy.stats.kendalltau`
- Produces:
  - `logistic_ridge_1se(X, y, *, sample_weight=None, nfolds=5, n_alphas=100, random_state=0) -> LogisticRegression` — fits an L2 path and returns the model at the one-standard-error lambda
  - `roc_best_threshold(labels, scores) -> tuple[float, float]` — returns `(threshold, specificity)` at Youden's J, with R's `pROC::coords` tie-handling
  - `train_labels(adata, mask, score, labels, qc, threshold, flag, **kwargs) -> tuple[np.ndarray, str]`

**Two R quirks preserved here, both cited in the spec:**
1. `label_dead.R:170-174` computes `colMeans` on a genes×cells matrix, which is a **per-cell** mean — so Stage 4 scales per cell where Stage 3 scales per gene. Both are reproduced as written.
2. `label_dead.R:166-167` computes `var.feats` and never uses it; the SVD runs on all non-zero genes. The dead computation is omitted, the behaviour kept.

**`lambda.1se`** has no sklearn equivalent: fit the L2 path over an alpha grid under `StratifiedKFold`, compute the mean and standard error of per-fold binomial deviance, and take the strongest regularisation whose mean deviance is within one standard error of the minimum.

- [ ] **Step 1: Write the failing test for the ridge helper**

```python
# tests/test_ridge.py
import numpy as np
import pytest
from sklearn.datasets import make_classification

from validrops.tl._ridge import logistic_ridge_1se, roc_best_threshold


def test_1se_is_more_regularised_than_the_minimum():
    X, y = make_classification(n_samples=300, n_features=20, random_state=0)
    model = logistic_ridge_1se(X, y, nfolds=5, random_state=0)
    assert model.C_1se <= model.C_min + 1e-12


def test_predicts_probabilities_in_range():
    X, y = make_classification(n_samples=200, n_features=10, random_state=1)
    model = logistic_ridge_1se(X, y, nfolds=5, random_state=0)
    probs = model.predict_proba(X)[:, 1]
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0


def test_is_deterministic_for_a_seed():
    X, y = make_classification(n_samples=200, n_features=10, random_state=2)
    a = logistic_ridge_1se(X, y, nfolds=5, random_state=3).predict_proba(X)[:, 1]
    b = logistic_ridge_1se(X, y, nfolds=5, random_state=3).predict_proba(X)[:, 1]
    np.testing.assert_allclose(a, b, rtol=1e-12)


def test_sample_weights_shift_the_fit():
    X, y = make_classification(n_samples=200, n_features=5, random_state=4)
    w = np.where(y == 1, 10.0, 0.1)
    unweighted = logistic_ridge_1se(X, y, nfolds=5, random_state=0).predict_proba(X)[:, 1]
    weighted = logistic_ridge_1se(X, y, sample_weight=w, nfolds=5, random_state=0).predict_proba(X)[:, 1]
    assert weighted.mean() > unweighted.mean()


def test_roc_best_threshold_on_perfect_separation():
    labels = np.array(["live", "live", "dead", "dead"])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    threshold, specificity = roc_best_threshold(labels, scores)
    assert specificity == 1.0
    assert 0.2 <= threshold <= 0.8


def test_roc_best_threshold_on_random_scores_is_finite():
    rng = np.random.default_rng(0)
    labels = rng.choice(["live", "dead"], size=200)
    threshold, specificity = roc_best_threshold(labels, rng.random(200))
    assert np.isfinite(threshold)
    assert 0.0 <= specificity <= 1.0
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run pytest tests/test_ridge.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the ridge helper**

```python
# src/validrops/tl/_ridge.py
"""Logistic ridge with glmnet's lambda.1se rule, and pROC-compatible thresholds."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold


class _Fitted(LogisticRegression):
    """LogisticRegression carrying the two selected penalties."""

    C_min: float
    C_1se: float


def logistic_ridge_1se(
    X: np.ndarray,
    y: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
    nfolds: int = 5,
    n_alphas: int = 100,
    random_state: int = 0,
) -> _Fitted:
    """L2 logistic regression at glmnet's ``lambda.1se``.

    ``cv.glmnet(..., s = "lambda.1se")`` picks the strongest regularisation
    whose cross-validated deviance is within one standard error of the
    minimum. sklearn has no equivalent, so it is implemented here.

    Parameters
    ----------
    X
        Design matrix.
    y
        Binary labels.
    sample_weight
        Per-observation weights.
    nfolds
        Cross-validation folds.
    n_alphas
        Points on the penalty path.
    random_state
        Seed for the fold split.

    Returns
    -------
    A fitted model with ``C_min`` and ``C_1se`` attached.
    """
    grid = np.logspace(-4, 4, n_alphas)
    classes = np.unique(y)
    if classes.size != 2:
        raise ValueError(f"expected two classes, got {classes.size}")
    binary = (y == classes[1]).astype(int)

    splitter = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=random_state)
    deviance = np.zeros((nfolds, grid.size))

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, binary)):
        weights = None if sample_weight is None else sample_weight[train_idx]
        for j, c in enumerate(grid):
            model = LogisticRegression(C=c, penalty="l2", max_iter=1000)
            model.fit(X[train_idx], binary[train_idx], sample_weight=weights)
            probs = np.clip(model.predict_proba(X[test_idx])[:, 1], 1e-10, 1 - 1e-10)
            actual = binary[test_idx]
            deviance[fold, j] = -2 * np.sum(actual * np.log(probs) + (1 - actual) * np.log(1 - probs))

    mean = deviance.mean(axis=0)
    stderr = deviance.std(axis=0, ddof=1) / np.sqrt(nfolds)
    best = int(np.argmin(mean))
    within = np.flatnonzero(mean <= mean[best] + stderr[best])
    # smaller C means stronger regularisation
    chosen = int(within[np.argmin(grid[within])])

    final = _Fitted(C=grid[chosen], penalty="l2", max_iter=1000)
    final.fit(X, binary, sample_weight=sample_weight)
    final.C_min = float(grid[best])
    final.C_1se = float(grid[chosen])
    return final


def roc_best_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Youden-optimal threshold, matching ``pROC::coords(roc, "best")``.

    Ties are broken toward the lowest threshold at the highest specificity,
    which is what ``label_dead.R:295-299`` falls back to.

    Parameters
    ----------
    labels
        ``"live"``/``"dead"`` per observation; ``"dead"`` is the positive class.
    scores
        Predicted probability of being dead.

    Returns
    -------
    ``(threshold, specificity)``.
    """
    positive = (np.asarray(labels) == "dead").astype(int)
    fpr, tpr, thresholds = roc_curve(positive, np.asarray(scores, dtype=np.float64))
    youden = tpr - fpr
    best = np.flatnonzero(youden == youden.max())
    specificity = 1.0 - fpr[best]
    at_max_spec = best[specificity == specificity.max()]
    chosen = at_max_spec[np.argmin(thresholds[at_max_spec])]
    return float(thresholds[chosen]), float(1.0 - fpr[chosen])
```

- [ ] **Step 4: Run the ridge tests**

Run: `uv run pytest tests/test_ridge.py -v`
Expected: PASS (6 tests)

The nested loop is `nfolds × n_alphas` fits. At 5 × 100 that is 500 small fits per call, and Stage 4 calls it 10 runs × 20 epochs × 10 replicates. Reduce `n_alphas` to 30 if the full stage takes more than ~20 minutes — record the change in the commit message, since it changes the penalty grid relative to glmnet's default of 100.

- [ ] **Step 5: Write the failing test for the training loop**

```python
# tests/test_label_dead_train.py
import numpy as np
import pytest

import validrops
from validrops._constants import UNS_KEY


@pytest.fixture(scope="module")
def trained(raw_adata, ref):
    adata = raw_adata.copy()
    metrics = ref("stage2_metrics.csv").set_index("barcode")
    for col, src in [("log_umis", "logUMIs"), ("log_features", "logFeatures"),
                     ("mitochondrial_fraction", "mitochondrial_fraction"),
                     ("ribosomal_fraction", "ribosomal_fraction"),
                     ("coding_fraction", "coding_fraction")]:
        adata.obs[col] = metrics[src].reindex(adata.obs_names).to_numpy()
    adata.obs["rank_pass"] = adata.obs["log_umis"].notna()
    valid = set(ref("stage3_barcodes.csv")["barcode"])
    adata.obs["qc_pass"] = [b in valid for b in adata.obs_names]
    validrops.pp.label_dead(adata, random_state=0, n_jobs=-1)
    return adata


@pytest.mark.slow
def test_dead_count_within_tolerance_of_r(trained, ref):
    want = ref("stage4_final.csv")
    n_want = int((want["label"] == "dead").sum())
    n_got = int((trained.obs["label"] == "dead").sum())
    assert abs(n_got - n_want) <= max(0.2 * n_want, 5), f"got {n_got}, R had {n_want}"


@pytest.mark.slow
def test_confident_labels_agree_with_r(trained, ref):
    want = ref("stage4_final.csv").set_index("barcode")["label"]
    got = trained.obs["label"]
    shared = [b for b in want.index if b in got.index]
    confident = [b for b in shared if want[b] != "uncertain" and got[b] != "uncertain"]
    assert len(confident) > 0.9 * len(shared)
    agreement = np.mean([want[b] == got[b] for b in confident])
    assert agreement > 0.9, f"label agreement {agreement:.3f}"


@pytest.mark.slow
def test_uncertain_fraction_is_small(trained):
    passing = trained.obs.loc[trained.obs["qc_pass"], "label"]
    assert (passing == "uncertain").mean() < 0.05


def test_consensus_requires_eight_of_ten():
    from validrops.pp._label_dead_train import consensus

    runs = np.array([["dead"] * 8 + ["live"] * 2,
                     ["dead"] * 7 + ["live"] * 3,
                     ["live"] * 10]).T
    assert list(consensus(runs, n_min=8)) == ["dead", "uncertain", "live"]


def test_flag_escalates_on_many_uncertain():
    from validrops.pp._label_dead_train import escalate_flag

    assert escalate_flag("Success", 0.001) == "Success"
    assert escalate_flag("Success", 0.02) == "Caution"
    assert escalate_flag("Success", 0.05) == "Failed"
    assert escalate_flag("Failed", 0.001) == "Failed"
```

- [ ] **Step 6: Run it to verify it fails**

Run: `uv run pytest tests/test_label_dead_train.py -v -m "not slow"`
Expected: FAIL with `ModuleNotFoundError: validrops.pp._label_dead_train`

- [ ] **Step 7: Write the training loop**

```python
# src/validrops/pp/_label_dead_train.py
"""Stage 4 consensus training. Ports ``label_dead.R:155-479``."""

import logging

import numpy as np
import scipy.sparse as sp
from anndata import AnnData
from joblib import Parallel, delayed
from scipy.sparse.linalg import svds
from scipy.stats import kendalltau

from .._constants import (
    DEAD_CELL_CONSENSUS, DEAD_CELL_RUNS, DEAD_COR_MAX, DEAD_COR_MIN,
    DEAD_EPOCHS, DEAD_FAIL_WEIGHT, DEAD_MAX_LIVE, DEAD_MIN_DEAD,
    DEAD_NFOLDS, DEAD_NPCS, DEAD_NREP,
)
from ..tl._ridge import logistic_ridge_1se, roc_best_threshold

logger = logging.getLogger(__name__)


def consensus(runs: np.ndarray, n_min: int = DEAD_CELL_CONSENSUS) -> np.ndarray:
    """Combine per-run labels. Ports ``label_dead.R:459``.

    Parameters
    ----------
    runs
        ``(n_barcodes, n_runs)`` array of ``"live"``/``"dead"``.
    n_min
        Runs that must agree for a confident call.

    Returns
    -------
    ``"live"``, ``"dead"`` or ``"uncertain"`` per barcode.
    """
    n_live = (runs == "live").sum(axis=1)
    n_dead = (runs == "dead").sum(axis=1)
    return np.where(n_live >= n_min, "live", np.where(n_dead >= n_min, "dead", "uncertain"))


def escalate_flag(flag: str, uncertain_fraction: float) -> str:
    """Downgrade the run flag when too many barcodes are uncertain (``label_dead.R:471-478``)."""
    if flag == "Failed":
        return "Failed"
    if uncertain_fraction >= 0.025:
        return "Failed"
    if uncertain_fraction >= 0.0125:
        return "Caution"
    return flag


def _embed(adata: AnnData, barcodes, npcs: int, random_state: int) -> np.ndarray:
    """Normalise, scale **per cell**, and take the SVD.

    ``label_dead.R:170-174`` takes ``colMeans`` of a genes-by-cells matrix,
    which is a per-cell mean, so this scales cells rather than genes. Stage 3
    scales genes. The difference is in the R source and is preserved.

    ``label_dead.R:166-167`` also computes variable features and never uses
    them; the SVD runs on all non-zero genes. That computation is omitted.
    """
    counts = sp.csr_matrix(adata[barcodes].X, dtype=np.float64)
    counts = counts[:, np.asarray(counts.sum(axis=0)).ravel() > 0]
    dense = np.asarray(counts.todense())

    means = dense.mean(axis=1, keepdims=True)  # per cell
    n_genes = dense.shape[1]
    sds = np.sqrt((np.mean(dense * dense, axis=1, keepdims=True) - means**2) * (n_genes / (n_genes - 1)))
    sds[sds == 0] = 1.0
    scaled = (dense - means) / sds

    npcs = min(npcs, min(scaled.shape) - 1)
    u, s, _ = svds(scaled, k=npcs, random_state=random_state)
    order = np.argsort(-s)
    return u[:, order]


def _resample(metrics_qc, labels, probs, rng, min_dead, max_live):
    """Weighted resampling with replacement. Ports ``label_dead.R:386-389``."""

    def draw(qc_value, label_value, size_fn, weights):
        idx = np.flatnonzero((metrics_qc == qc_value) & (labels == label_value))
        if idx.size == 0:
            return idx
        size = size_fn(idx.size)
        w = weights[idx]
        total = w.sum()
        p = w / total if total > 0 else None
        return rng.choice(idx, size=size, replace=True, p=p)

    dead_fail = draw("fail", "dead", lambda n: max(min_dead, n), probs)
    dead_pass = draw("pass", "dead", lambda n: max(min_dead, n), probs)
    live_fail = draw("fail", "live", lambda n: min(max_live, n), np.abs(probs - 1))
    live_pass = draw("pass", "live", lambda n: min(max_live, n), np.abs(probs - 1))
    return np.concatenate([dead_fail, live_fail]), np.concatenate([dead_pass, live_pass])


def _one_run(embedding, labels, qc, fail_weight, cor_threshold,
             epochs, nrep, nfolds, min_dead, max_live, seed):
    """One independent optimisation run. Ports the ``bplapply`` body, ``label_dead.R:200-448``.

    Notes
    -----
    ``label_dead.R:225-236`` builds a score- and QC-derived weight vector, but
    every replicate then overwrites ``weights`` wholesale at ``R:276-277`` with
    ``rep(1, ...)`` and ``fail_weight`` for the QC-failing block. The initial
    vector is therefore dead code in R, so it is not computed here — only the
    per-replicate weights that actually reach the model.
    """
    rng = np.random.default_rng(seed)
    labels = labels.copy()
    probs = np.where(labels == "live", 0.0, 1.0)

    spec_old = 0.0
    balance_old = 0.0
    relabel_old = labels.size
    trigger = False

    for _ in range(epochs):
        taus = np.array([
            kendalltau(embedding[:, d], (labels == "dead").astype(float)).statistic
            for d in range(embedding.shape[1])
        ])
        taus = np.nan_to_num(taus)
        if trigger:
            cor_threshold += DEAD_COR_MIN
        selected = np.flatnonzero(taus**2 >= cor_threshold)
        if selected.size == 0:
            break
        X = embedding[:, selected]

        prob_matrix = np.empty((labels.size, nrep))
        for rep in range(nrep):
            fail_idx, pass_idx = _resample(qc, labels, probs, rng, min_dead, max_live)
            idx = np.concatenate([fail_idx, pass_idx])
            if idx.size == 0 or np.unique(labels[idx]).size < 2:
                return labels, "Failed"
            sample_X = X[idx].copy()
            sample_y = labels[idx]
            weights = np.ones(idx.size)
            weights[: fail_idx.size] = fail_weight
            for d in range(sample_X.shape[1]):  # jitter, label_dead.R:401
                amount = np.std(sample_X[:, d], ddof=1) / 5
                sample_X[:, d] += rng.uniform(-amount, amount, size=idx.size)
            order = rng.permutation(idx.size)
            # encode explicitly so column 1 of predict_proba is always P(dead);
            # passing the strings would make "live" the positive class, since
            # np.unique sorts "dead" before "live"
            binary = (sample_y == "dead").astype(int)
            model = logistic_ridge_1se(
                sample_X[order], binary[order], sample_weight=weights[order],
                nfolds=nfolds, random_state=seed + rep,
            )
            prob_matrix[:, rep] = model.predict_proba(X)[:, 1]

        new_prob = np.median(prob_matrix, axis=1)
        passing = qc == "pass"
        cut, specificity = roc_best_threshold(labels[passing], new_prob[passing])
        prediction = np.where(new_prob > cut, "dead", "live")

        disagree = prediction != labels
        relabel = int(disagree.sum())
        balance = (
            np.sum((prediction == "dead") & (labels == "live")) / relabel if relabel else 0.0
        )

        if relabel <= labels.size * 0.002 or relabel >= relabel_old * 2:
            break
        if (spec_old > specificity or balance_old >= balance * 1.5) and not trigger:
            trigger = True
        else:
            balance_old = balance
            spec_old = specificity
            probs = new_prob
            labels = prediction
            if min(np.sum(labels == "dead"), np.sum(labels == "live")) == 0:
                return labels, "Failed"
        relabel_old = relabel

    return labels, "Success"


def train_labels(adata, mask, score, labels, qc, threshold, flag, *, rep=DEAD_CELL_RUNS,
                 n_min=DEAD_CELL_CONSENSUS, npcs=DEAD_NPCS, epochs=DEAD_EPOCHS,
                 nrep=DEAD_NREP, nfolds=DEAD_NFOLDS, fail_weight=DEAD_FAIL_WEIGHT,
                 cor_threshold=None, cor_min=DEAD_COR_MIN, cor_max=DEAD_COR_MAX,
                 min_dead=DEAD_MIN_DEAD, max_live=DEAD_MAX_LIVE, n_jobs=1, random_state=0):
    """Run ``rep`` independent optimisations and take the consensus.

    ``score`` and ``threshold`` are accepted so the call site mirrors R's, but
    are unused: see the note in :func:`_one_run` about R's dead weight vector.

    Returns
    -------
    ``(labels, flag)``.
    """
    barcodes = adata.obs_names[mask]
    embedding = _embed(adata, barcodes, npcs, random_state)

    if cor_threshold is None:
        # A full port of label_dead.R:239-352 searches cor_steps thresholds with
        # nrep_cor fits each, per run. That is 50 x 10 x 10 = 5000 model fits before
        # training even starts. Use the geometric midpoint as the starting value and
        # let the per-epoch escalation at label_dead.R:373 adapt it.
        cor_threshold = float(np.sqrt(cor_min * cor_max))
        logger.info("Using cor_threshold=%.5g (geometric midpoint of the R search range)", cor_threshold)

    results = Parallel(n_jobs=n_jobs)(
        delayed(_one_run)(
            embedding, labels, qc, fail_weight, cor_threshold,
            epochs, nrep, nfolds, min_dead, max_live, random_state + run,
        )
        for run in range(rep)
    )

    runs = np.column_stack([r[0] for r in results])
    flags = [r[1] for r in results]
    final = consensus(runs, n_min=n_min)

    if "Failed" in flags:
        flag = "Failed"
    passing = qc == "pass"
    uncertain_fraction = float(np.mean(final[passing] == "uncertain")) if passing.any() else 0.0
    flag = escalate_flag(flag, uncertain_fraction)

    logger.info("Step 6: %d dead, %d uncertain (flag=%s)",
                int(np.sum(final == "dead")), int(np.sum(final == "uncertain")), flag)
    return final, flag
```

- [ ] **Step 8: Run the tests**

Run: `uv run pytest tests/test_label_dead_train.py -v -m "not slow"` then `uv run pytest tests/test_label_dead_train.py -v -m slow`
Expected: PASS (5 tests)

**This step contains a deliberate simplification, and it must be reported.** The `cor_threshold` search of `label_dead.R:239-352` is replaced by the geometric midpoint of R's search range plus the existing per-epoch escalation. The full search is roughly 5000 extra model fits *per run*, 50,000 across the ten runs. If the slow tests pass, note the simplification in the commit message and open a follow-up issue. **If they fail on the dead-count tolerance, implement the full search before loosening any assertion** — the simplification is the first suspect, not the tolerance.

- [ ] **Step 9: Lint and commit**

```bash
uv run ruff check src/validrops/ --fix && uv run ruff format src/validrops/
git add src/validrops/tl/_ridge.py src/validrops/pp/_label_dead_train.py tests/test_ridge.py tests/test_label_dead_train.py
git commit -m "feat: stage 4 consensus training loop

cor_threshold search simplified to the geometric midpoint of R's range;
see label_dead.R:239-352 for the full search."
```

---

## Task 19: The `validrops()` orchestrator

**Files:**
- Create: `src/validrops/_pipeline.py`
- Modify: `src/validrops/__init__.py`
- Create: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: every stage function
- Produces: `validrops(adata, *, rank_barcodes=True, stage_three=True, label_dead=False, mitochondrial_clusters=3, ribosomal_clusters=3, random_state=0, verbose=True, **kwargs) -> None`, exported as `validrops.validrops`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pipeline.py
import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

import validrops
from validrops._constants import UNS_KEY


@pytest.fixture(scope="module")
def result(raw_adata):
    adata = raw_adata.copy()
    validrops.validrops(adata, random_state=0)
    return adata


@pytest.mark.slow
def test_end_to_end_concordance_with_r(result, ref):
    expected = ref("pbmc4k_full_pipeline.csv")
    want = set(expected.loc[expected["qc.pass"] == "pass", "barcode"])
    got = set(result.obs_names[result.obs["qc_pass"]])
    concordance = len(got & want) / len(got | want)
    assert concordance > 0.95, f"end-to-end concordance {concordance:.4f}"


def test_all_expected_columns_written(result):
    for column in ["rank_pass", "log_umis", "log_features", "mitochondrial_fraction",
                   "ribosomal_fraction", "coding_fraction", "pass_mito", "pass_distance",
                   "pass_coding", "cluster", "qc_pass"]:
        assert column in result.obs, column


def test_params_recorded_for_provenance(result):
    params = result.uns[UNS_KEY]["params"]
    assert params["random_state"] == 0
    assert params["stage_three"] is True


def test_returns_none(raw_adata):
    adata = raw_adata[:2000].copy()
    assert validrops.validrops(adata, stage_three=False) is None


def test_stage_three_false_uses_stage_two_qc_pass(raw_adata):
    adata = raw_adata.copy()
    validrops.validrops(adata, stage_three=False)
    assert "cluster" not in adata.obs
    np.testing.assert_array_equal(
        adata.obs["qc_pass"].to_numpy(), adata.obs["pass_coding"].to_numpy()
    )


def test_rank_barcodes_false_keeps_all_nonzero(raw_adata):
    adata = raw_adata.copy()
    validrops.validrops(adata, rank_barcodes=False, stage_three=False)
    totals = np.asarray(adata.X.sum(axis=1)).ravel()
    np.testing.assert_array_equal(adata.obs["rank_pass"].to_numpy(), totals > 0)


def test_label_dead_adds_labels(raw_adata):
    """Small synthetic run so the slow training loop stays out of the fast suite."""
    rng = np.random.default_rng(0)
    counts = rng.poisson(0.5, size=(1500, 400))
    counts[:300] *= 30
    adata = ad.AnnData(sp.csr_matrix(counts.astype(np.float32)))
    adata.var_names = [f"gene_{i}" for i in range(400)]
    validrops.validrops(adata, stage_three=False, label_dead=True,
                        mito=["gene_0"], ribo=["gene_1"], coding=list(adata.var_names[2:200]))
    assert "label" in adata.obs
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pipeline.py -v -m "not slow"`
Expected: FAIL — `module 'validrops' has no attribute 'validrops'`

- [ ] **Step 3: Write the implementation**

```python
# src/validrops/_pipeline.py
"""End-to-end pipeline. Ports ``valiDrops.R:21-139``."""

import inspect
import logging

import numpy as np
from anndata import AnnData

from . import pp, tl
from ._constants import UNS_KEY

logger = logging.getLogger(__name__)


def validrops(
    adata: AnnData,
    *,
    rank_barcodes: bool = True,
    stage_three: bool = True,
    label_dead: bool = False,
    mitochondrial_clusters: float | None = 3,
    ribosomal_clusters: float | None = 3,
    random_state: int = 0,
    verbose: bool = True,
    **kwargs,
) -> None:
    """Run the complete valiDrops quality-control pipeline.

    Parameters
    ----------
    adata
        Raw, unfiltered counts, cells x genes. Annotated in place; nothing is
        removed, so call ``adata[adata.obs["qc_pass"]].copy()`` afterwards to
        get the clean object.
    rank_barcodes
        Run stage 1. When ``False``, every barcode with a non-zero count
        proceeds (``valiDrops.R:79``).
    stage_three
        Run the expression-based stages. When ``False``, ``qc_pass`` comes
        straight from stage 2.
    label_dead
        Run the dead-cell prediction. Off by default: it is stochastic and
        much slower than the rest of the pipeline.
    mitochondrial_clusters, ribosomal_clusters
        Deviations above the median cluster content beyond which a cluster is
        dropped in stage 3b. ``None`` disables.
    random_state
        Seed threaded through every stochastic step.
    verbose
        Log progress.
    **kwargs
        Forwarded to the individual stage functions by parameter name.

    Returns
    -------
    None.

    Examples
    --------
    >>> import scanpy as sc, validrops  # doctest: +SKIP
    >>> adata = sc.read_10x_h5("raw.h5")  # doctest: +SKIP
    >>> validrops.validrops(adata)  # doctest: +SKIP
    >>> clean = adata[adata.obs["qc_pass"]].copy()  # doctest: +SKIP
    """
    if verbose:
        logging.getLogger("validrops").setLevel(logging.INFO)

    if rank_barcodes:
        logger.info("Step 1: Filtering on the barcode-rank plot.")
        pp.rank_barcodes(adata, random_state=random_state, **_for(pp.rank_barcodes, kwargs))
    else:
        logger.info("Step 1: Removing barcodes with zero counts.")
        adata.obs["rank_pass"] = np.asarray(adata.X.sum(axis=1)).ravel() > 0

    logger.info("Step 2: Collecting quality metrics.")
    tl.quality_metrics(adata, **_for(tl.quality_metrics, kwargs))

    logger.info("Step 3: Filtering on quality metrics.")
    pp.quality_filter(adata, random_state=random_state, **_for(pp.quality_filter, kwargs))

    if stage_three:
        logger.info("Step 4: Collecting expression-based metrics.")
        tl.expression_metrics(adata, random_state=random_state, **_for(tl.expression_metrics, kwargs))

        logger.info("Step 5: Filtering on expression-based metrics.")
        pp.expression_filter(
            adata,
            mito=mitochondrial_clusters,
            ribo=ribosomal_clusters,
            **_for(pp.expression_filter, kwargs, exclude={"mito", "ribo"}),
        )

    if label_dead:
        logger.info("Step %d: Predicting dead cells.", 6 if stage_three else 4)
        if not stage_three:
            logger.warning("Predicting dead cells without stage 3. CAUTION: this has not been tested.")
        pp.label_dead(adata, random_state=random_state, **_for(pp.label_dead, kwargs))

    uns = adata.uns.setdefault(UNS_KEY, {})
    uns["params"] = {
        "rank_barcodes": rank_barcodes,
        "stage_three": stage_three,
        "label_dead": label_dead,
        "mitochondrial_clusters": mitochondrial_clusters,
        "ribosomal_clusters": ribosomal_clusters,
        "random_state": random_state,
        **kwargs,
    }

    logger.info("\t%d barcodes passed quality control.", int(adata.obs["qc_pass"].sum()))
    if label_dead:
        dead = (adata.obs["qc_pass"] & (adata.obs["label"] == "dead")).sum()
        logger.info("\t%d barcodes that passed quality control are predicted to be dead.", int(dead))


def _for(func, kwargs: dict, exclude: set[str] | None = None) -> dict:
    """Select the kwargs a stage function actually accepts (R does this with ``doCall``).

    Functions declaring ``**kwargs`` receive everything not explicitly excluded,
    which is how ``label_dead``'s training parameters reach it.
    """
    parameters = inspect.signature(func).parameters
    blocked = {"adata", "random_state"} | (exclude or set())
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return {k: v for k, v in kwargs.items() if k not in blocked}
    accepted = set(parameters) - blocked
    return {k: v for k, v in kwargs.items() if k in accepted}
```

Update `src/validrops/__init__.py`:

```python
from importlib.metadata import version

from . import pl, pp, tl
from ._pipeline import validrops

__all__ = ["pl", "pp", "tl", "validrops"]
__version__ = version("validrops")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_pipeline.py -v -m "not slow"` then the slow test.
Expected: PASS (7 tests)

`test_end_to_end_concordance_with_r` is the headline number for the whole project. If it lands between 0.90 and 0.95, run each stage's concordance test to find where the loss enters — do not adjust the threshold.

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src/validrops/ --fix && uv run ruff format src/validrops/
git add src/validrops/_pipeline.py src/validrops/__init__.py tests/test_pipeline.py
git commit -m "feat: validrops() pipeline orchestrator (end-to-end concordance=<measured>)"
```

---

## Task 20: Plotting

**Files:**
- Create: `src/validrops/pl/_qc.py`
- Modify: `src/validrops/pl/__init__.py`
- Create: `tests/test_plotting.py`

**Interfaces:**
- Produces five functions, each `(adata, *, ax=None, **kwargs) -> matplotlib.axes.Axes`:
  `barcode_rank`, `mito_threshold`, `umi_vs_features`, `coding_fraction`, `dead_score`

Each reads only from `obs`/`uns` — no recomputation — and mirrors one R `plot()` call.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_plotting.py
import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import validrops  # noqa: E402


@pytest.fixture(scope="module")
def plotted(raw_adata):
    adata = raw_adata.copy()
    validrops.validrops(adata, stage_three=False, random_state=0)
    return adata


@pytest.mark.parametrize(
    "name", ["barcode_rank", "mito_threshold", "umi_vs_features", "coding_fraction"]
)
def test_plot_returns_axes(plotted, name):
    ax = getattr(validrops.pl, name)(plotted)
    assert isinstance(ax, plt.Axes)
    assert ax.get_xlabel()
    assert ax.get_ylabel()
    plt.close(ax.figure)


def test_plot_accepts_an_existing_axes(plotted):
    fig, ax = plt.subplots()
    returned = validrops.pl.barcode_rank(plotted, ax=ax)
    assert returned is ax
    plt.close(fig)


def test_dead_score_requires_label_dead(plotted):
    with pytest.raises(KeyError, match="dead_score"):
        validrops.pl.dead_score(plotted)


def test_mito_threshold_draws_the_cutoff_line(plotted):
    ax = validrops.pl.mito_threshold(plotted)
    assert len(ax.lines) >= 1
    plt.close(ax.figure)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_plotting.py -v`
Expected: FAIL — `module 'validrops.pl' has no attribute 'barcode_rank'`

- [ ] **Step 3: Write the implementation**

```python
# src/validrops/pl/_qc.py
"""Diagnostic plots mirroring valiDrops' base-R plots."""

import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData
from matplotlib.axes import Axes

from .._constants import UNS_KEY


def _axes(ax: Axes | None) -> Axes:
    return ax if ax is not None else plt.subplots(figsize=(5, 4))[1]


def barcode_rank(adata: AnnData, *, ax: Axes | None = None, **kwargs) -> Axes:
    """Log-rank against log-count, with the detected threshold marked.

    The knee separates cell-containing droplets from ambient background.
    Mirrors ``rank_barcodes.R:132-139``.
    """
    ax = _axes(ax)
    ranks = adata.uns[UNS_KEY]["barcode_ranks"]
    threshold = adata.uns[UNS_KEY]["rank_threshold"]
    ax.scatter(np.log(ranks["rank"]), np.log(ranks["counts"]), s=4, c="#CDCDCD", **kwargs)
    ax.axhline(np.log(threshold), color="red", lw=1)
    ax.set_xlabel("log Rank")
    ax.set_ylabel("log Counts")
    ax.set_title(f"n = {int(adata.obs['rank_pass'].sum())} barcodes above threshold")
    return ax


def mito_threshold(adata: AnnData, *, ax: Axes | None = None, **kwargs) -> Axes:
    """Mitochondrial fraction against detected features, with the cutoff.

    Mirrors ``quality_filter.R:109-112``.
    """
    ax = _axes(ax)
    threshold = adata.uns[UNS_KEY]["mitochondrial_threshold"]
    obs = adata.obs.dropna(subset=["mitochondrial_fraction", "log_features"])
    colours = np.where(obs["mitochondrial_fraction"] > threshold, "red", "black")
    ax.scatter(obs["log_features"], obs["mitochondrial_fraction"], s=4, c=colours, **kwargs)
    ax.axhline(threshold, color="black", lw=1)
    ax.set_xlabel("log Total features")
    ax.set_ylabel("Mitochondrial fraction")
    ax.set_title(f"Threshold = {threshold:.3f}")
    return ax


def umi_vs_features(adata: AnnData, *, ax: Axes | None = None, **kwargs) -> Axes:
    """Detected features against total UMIs, coloured by the distance filter.

    Barcodes far from the trend are doublets or damaged cells. Mirrors
    ``quality_filter.R:144-150``.
    """
    ax = _axes(ax)
    obs = adata.obs.dropna(subset=["log_umis", "log_features"])
    colours = np.where(obs["pass_distance"], "grey", "red")
    ax.scatter(obs["log_umis"], obs["log_features"], s=4, c=colours, **kwargs)
    ax.set_xlabel("log Total UMIs")
    ax.set_ylabel("log Total features")
    ax.set_title(f"Kept {int(obs['pass_distance'].sum())} barcodes")
    return ax


def coding_fraction(adata: AnnData, *, ax: Axes | None = None, **kwargs) -> Axes:
    """Histogram of the protein-coding fraction. Mirrors ``quality_filter.R:172-175``."""
    ax = _axes(ax)
    values = adata.obs["coding_fraction"].dropna()
    ax.hist(values, bins="auto", color="#4C72B0", **kwargs)
    ax.set_xlabel("Fraction of UMIs from protein-coding genes")
    ax.set_ylabel("Barcodes")
    ax.set_title(f"Kept {int(adata.obs['pass_coding'].sum())} barcodes")
    return ax


def dead_score(adata: AnnData, *, ax: Axes | None = None, **kwargs) -> Axes:
    """Sorted dead-cell score with the soft-labelling cutoff.

    Mirrors ``label_dead.R:146-149``. Requires ``label_dead`` to have run.
    """
    if "dead_score" not in adata.obs:
        raise KeyError("dead_score not found; run validrops.pp.label_dead first")
    ax = _axes(ax)
    values = np.sort(adata.obs["dead_score"].dropna().to_numpy())
    ax.scatter(np.arange(values.size), values, s=4, c="black", **kwargs)
    ax.axhline(adata.uns[UNS_KEY]["label_threshold"], color="red", ls="--")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Score")
    return ax
```

Write `src/validrops/pl/__init__.py`:

```python
from ._qc import barcode_rank, coding_fraction, dead_score, mito_threshold, umi_vs_features

__all__ = ["barcode_rank", "coding_fraction", "dead_score", "mito_threshold", "umi_vs_features"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_plotting.py -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff check src/validrops/ --fix && uv run ruff format src/validrops/
git add src/validrops/pl tests/test_plotting.py
git commit -m "feat: QC diagnostic plots"
```

---

## Task 21: Documentation and final verification

**Files:**
- Modify: `CLAUDE.md`
- Modify: `docs/api.md`
- Modify: `DEVELOPMENT.md`
- Modify: `README.md`
- Create: `tests/test_smoke.py`

- [ ] **Step 1: Write a fast synthetic smoke test**

```python
# tests/test_smoke.py
"""Fast end-to-end run on synthetic data, so every commit exercises the pipeline."""

import anndata as ad
import numpy as np
import scipy.sparse as sp

import validrops


def _synthetic(n_cells=1200, n_genes=300, seed=0):
    rng = np.random.default_rng(seed)
    counts = rng.poisson(0.3, size=(n_cells, n_genes)).astype(np.float32)
    counts[:250] *= 25  # a clear population of real cells
    adata = ad.AnnData(sp.csr_matrix(counts))
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    adata.obs_names = [f"cell_{i + 1}" for i in range(n_cells)]
    return adata


def test_pipeline_runs_without_stage_three():
    adata = _synthetic()
    validrops.validrops(
        adata, stage_three=False,
        mito=["gene_0", "gene_1"], ribo=["gene_2", "gene_3"],
        coding=list(adata.var_names[4:200]),
    )
    assert adata.obs["qc_pass"].sum() > 0
    assert adata.obs["qc_pass"].sum() < adata.n_obs


def test_pipeline_is_reproducible():
    a, b = _synthetic(), _synthetic()
    for adata in (a, b):
        validrops.validrops(
            adata, stage_three=False, random_state=7,
            mito=["gene_0"], ribo=["gene_1"], coding=list(adata.var_names[2:200]),
        )
    np.testing.assert_array_equal(a.obs["qc_pass"].to_numpy(), b.obs["qc_pass"].to_numpy())


def test_nothing_is_removed_from_the_object():
    adata = _synthetic()
    before = adata.shape
    validrops.validrops(
        adata, stage_three=False,
        mito=["gene_0"], ribo=["gene_1"], coding=list(adata.var_names[2:200]),
    )
    assert adata.shape == before
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/test_smoke.py -v`
Expected: PASS (3 tests)

- [ ] **Step 3: Update `CLAUDE.md`'s dependency map**

Replace the "R → Python dependency map" table with the corrected version from spec §2.2. Specifically: `segmented` → own `tl/_segmented.py` (not `pwlf`), `inflection` → own `tl/_uik.py` (not `kneed`), Seurat clustering → own `tl/_snn.py` (not `sc.pp.neighbors`+`sc.tl.leiden`), `presto::wilcoxauc` → own `tl/_wilcox.py`, `glmnet` → `LogisticRegression` L2 path with a 1se rule (**not** `RidgeCV` — the R call is `family="binomial"`), `scry` → `tl/_deviance.py`.

Add a short "R quirks preserved" section listing the four from spec §2, each with its R line reference, so a future session does not "fix" them.

Update the "Key constants" line to point at `src/validrops/_constants.py` as the single source of truth.

- [ ] **Step 4: Update `docs/api.md`**

```markdown
# API

## Pipeline

```{eval-rst}
.. autofunction:: validrops.validrops
```

## Preprocessing: `pp`

```{eval-rst}
.. autofunction:: validrops.pp.rank_barcodes
.. autofunction:: validrops.pp.quality_filter
.. autofunction:: validrops.pp.expression_filter
.. autofunction:: validrops.pp.label_dead
```

## Tools: `tl`

```{eval-rst}
.. autofunction:: validrops.tl.quality_metrics
.. autofunction:: validrops.tl.expression_metrics
```

## Plotting: `pl`

```{eval-rst}
.. autofunction:: validrops.pl.barcode_rank
.. autofunction:: validrops.pl.mito_threshold
.. autofunction:: validrops.pl.umi_vs_features
.. autofunction:: validrops.pl.coding_fraction
.. autofunction:: validrops.pl.dead_score
```
```

- [ ] **Step 5: Update `DEVELOPMENT.md` and `README.md`**

In `DEVELOPMENT.md`, replace the in-progress Step 2 notes with: how to regenerate fixtures (`Rscript tests/R/extract_annotation.R` then `Rscript tests/R/generate_reference.R`), which R packages that needs, and the measured concordance per stage.

In `README.md`, add a usage example:

```python
import scanpy as sc
import validrops

adata = sc.read_10x_h5("raw_feature_bc_matrix.h5")
adata.var_names_make_unique()

validrops.validrops(adata)

clean = adata[adata.obs["qc_pass"]].copy()
```

and a note that the object is annotated rather than subset.

- [ ] **Step 6: Full verification**

```bash
uv run ruff check src/validrops/ && uv run ruff format --check src/validrops/
uv run pytest tests/ -v
uv run pytest tests/ -v -m slow
uv run python -m build && uv run twine check --strict dist/*
uv run hatch run docs:build
```

Every command must pass. Record the measured numbers — per-stage concordance, stage 3 ARI, end-to-end concordance — and check them against the spec's targets. **Report any target that was missed rather than adjusting the assertion.**

- [ ] **Step 7: Commit**

```bash
git add CLAUDE.md docs/api.md DEVELOPMENT.md README.md tests/test_smoke.py
git commit -m "docs: update dependency map, API reference and development notes"
```

---

## Verification Summary

At completion, these are the numbers that matter. Fill them in from the final run:

| Check | Target | Measured |
|---|---|---|
| `sn` vs `robustbase::Sn` | rtol 1e-10 | |
| `uik` vs `inflection::uik` | rtol 1e-6 | |
| `segmented` vs R | rtol 1e-6 | |
| `deviance` vs `scry` | rtol 1e-8, top-5000 overlap >99% | |
| `wilcoxauc` vs `presto` | rtol 1e-6 | |
| Gene sets | exact set equality | |
| Stage 1 threshold | rtol 1e-6, barcode set exact | |
| Stage 2a metrics | r > 0.99 | |
| Stage 2b sub-filters | >95% concordance each | |
| Stage 3 clustering | ARI > 0.9 | |
| Stage 3b filter (R inputs) | exact | |
| Stage 4 soft labels | exact | |
| Stage 4 trained labels | dead count ±20%, agreement >90% | |
| End-to-end | >95% concordance | |
