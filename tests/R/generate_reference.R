# Regenerates every fixture in tests/reference_outputs/.
# Run from the repository root:  Rscript tests/R/generate_reference.R
#
# Everything is seeded so the fixtures are reproducible. The stochastic
# stages (mitochondrial threshold, clustering, dead-cell training) will
# still differ across R versions; regenerate rather than hand-edit.
#
# Stage 4 (label_dead) is deliberately last, and the earlier stages
# deliberately do NOT convert the full (unfiltered) `counts` object to
# dgCMatrix, so that they exercise valiDrops on the object shape
# read10xCounts() actually hands back for an .h5 file (a DelayedMatrix).
# Immediately before stage 4, `counts` is coerced to dgCMatrix — see the
# comment at that assignment for why: label_dead.R:163 does
# `norm_transform@x <- ...`, which needs a dgCMatrix slot, and
# valiDrops.R:54 lists dgCMatrix as a supported input class, so this is
# ordinary supported usage, not a workaround. Even so, valiDrops also has a
# documented soft-label threshold bug independent of matrix class
# (label_dead.R:56-125: `max.quantile` can climb past 1.0 and
# `quantile(metrics$score, brk)` then errors with 'probs' outside [0,1]) —
# see project docs. We do NOT patch the package or wrap the label_dead()
# calls in tryCatch to hide a failure from either of these — both calls are
# attempted for real, and if the trained call errors, it is left to halt
# the script. Because stage 4 runs last, every other fixture (including the
# score column, which is safe pure arithmetic computed directly from
# label_dead.R:45-50) is already written to disk by then.

library(valiDrops)
library(Matrix)
library(DropletUtils)
library(robustbase)
library(inflection)
library(zoo)
library(segmented)
library(scry)
library(presto)
library(irlba)
library(Seurat)

set.seed(42)
OUT <- "tests/reference_outputs"
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
pdf(NULL) # swallow plots

## ---------------------------------------------------------------- primitives

# Sn
set.seed(42)
test_vectors <- list(
  normal = rnorm(100), skewed = rexp(100), heavy_tail = rt(100, df = 3),
  small = rnorm(20), large = rnorm(5000),
  odd_medium = rnorm(21)
)
write.csv(
  data.frame(name = names(test_vectors), sn = sapply(test_vectors, Sn)),
  file.path(OUT, "sn_reference.csv"), row.names = FALSE
)
sn_in <- do.call(rbind, lapply(names(test_vectors), function(n) {
  data.frame(name = n, value = test_vectors[[n]])
}))
write.csv(sn_in, file.path(OUT, "sn_inputs.csv"), row.names = FALSE)

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
  noisy_elbow = list(x = 1:200, y = pmax(0, 50 - 0.5 * (1:200)) + 0.02 * (1:200)),
  # Oscillating curve whose four sampled chord-deviation signs (at j=25,50,75,100)
  # disagree ([+,-,+,-]), exercising check_curve's mixed-sign tie-break branch
  # (uses signs[1] rather than a unanimous sign). Deterministic, no RNG draw.
  mixed_sign_wave = list(x = 1:100, y = sin((1:100) / 8) * 5 + (1:100) * 0.1)
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
  K <- cs$npsi
  # segmented.lm always overwrites fit$psi[, "Initial"] with NA before
  # returning (see segmented.lm source, the line `objF$psi[, "Initial"] <-
  # NA`), so the starting values can't be read off the fitted object. They
  # have to be recomputed. With seg.control()'s default quant = FALSE, the
  # default start for K breakpoints is K equally spaced points across the
  # *range* of x (not quantiles of its distribution):
  #   psiE = min(x) + diff(range(x)) * (1:K) / (K + 1)
  # This is exactly the formula segmented.lm evaluates internally before
  # fitting, so it reproduces R's actual starting values deterministically.
  psi0 <- min(cs$x) + diff(range(cs$x)) * (1:K) / (K + 1)
  set.seed(99)
  # n.boot = 0 disables bootstrap restart, making the fit deterministic given
  # the default starting values above. Without this the fixture depends on
  # R's RNG stream, which the Python port cannot reproduce.
  fit <- segmented(lm(y ~ x, data = data.frame(x = cs$x, y = cs$y)),
    npsi = cs$npsi, control = seg.control(n.boot = 0)
  )
  psi <- fit$psi[, 2] # column 2 is "Est."
  # slope()'s default digits = max(4, getOption("digits") - 2) = 5 rounds the
  # Est. column to 5 significant figures via signif() before returning it;
  # request full precision explicitly so the fixture isn't display-rounded.
  sl <- slope(fit, digits = 15)$x[, 1]
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

# segmented, bootstrap variant: same four cases (seg_cases, above), but with
# seg.control(n.boot = 10) — R's own default restart count — instead of
# n.boot = 0. This validates the Python port's simplified n_boot path (no
# evolving start / stagnation kick / random-restart fallback) against R's
# actual seg.lm.fit.boot output; see Task 7's report for the achieved
# agreement (exact for well-posed cases, seed-dependent for smooth_curve,
# which has no true breakpoints and a genuinely multi-modal RSS surface).
# Not meant for a Python-side exact-match comparison in general — R's RNG
# stream can't be reproduced from Python — only to check where the Python
# bootstrap's breakpoints land relative to R's.
boot_rows <- list()
for (nm in names(seg_cases)) {
  cs <- seg_cases[[nm]]
  set.seed(99)
  fit <- segmented(lm(y ~ x, data = data.frame(x = cs$x, y = cs$y)),
    npsi = cs$npsi, control = seg.control(n.boot = 10)
  )
  psi <- fit$psi[, 2]
  sl <- slope(fit, digits = 15)$x[, 1]
  boot_rows[[nm]] <- rbind(
    data.frame(case = nm, term = paste0("psi", seq_along(psi)), value = psi),
    data.frame(case = nm, term = paste0("slope", seq_along(sl)), value = sl),
    data.frame(case = nm, term = "rmse", value = sqrt(mean(fit$residuals^2)))
  )
}
write.csv(do.call(rbind, boot_rows), file.path(OUT, "segmented_boot_reference.csv"), row.names = FALSE)

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
  file.path(OUT, "stage1_threshold.csv"),
  row.names = FALSE
)
write.csv(data.frame(
  key = c("lower_threshold", "n_pass", "n_input"),
  value = c(threshold$lower.threshold, length(rank.pass), ncol(counts))
),
file.path(OUT, "stage1_meta.csv"),
row.names = FALSE
)

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
write.csv(data.frame(
  species = "human", column = "Symbol",
  n_mapped = sum(rownames(counts.subset) %in%
    as.data.frame(valiDrops:::annotation[[1]])$Symbol)
),
file.path(OUT, "annotation_detection.csv"),
row.names = FALSE
)

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
write.csv(data.frame(
  key = c("mitochondrial_threshold", "n_final"),
  value = c(qc.pass$mitochondrial.threshold, length(qc.pass$final))
),
file.path(OUT, "stage2_meta.csv"),
row.names = FALSE
)

## ------------------------------------------------------------------ stage 3a

counts.filtered <- counts.subset[
  rownames(counts.subset) %in% metrics$protein_coding,
  colnames(counts.subset) %in% qc.pass$final
]
counts.filtered <- as(counts.filtered, "dgCMatrix")
set.seed(42)
expr <- valiDrops::expression_metrics(counts.filtered,
  mito = metrics$mitochondrial,
  ribo = metrics$ribosomal
)
write.csv(expr$stats, file.path(OUT, "stage3_stats.csv"), row.names = FALSE)

# clusters: expression_metrics only returns the deep assignment
deep <- data.frame(barcode = rownames(expr$clusters), deep = expr$clusters[, 1])
write.csv(deep, file.path(OUT, "stage3_clusters_deep.csv"), row.names = FALSE)

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

## ------------------------------------------------------------------ stage 3b

set.seed(42)
valid <- valiDrops::expression_filter(
  stats = expr$stats, clusters = expr$clusters,
  mito = 3, ribo = 3, plot = FALSE
)
write.csv(data.frame(barcode = valid), file.path(OUT, "stage3_barcodes.csv"), row.names = FALSE)

## ---------------------------------------------------------- deviance, wilcox

nonzero <- counts.filtered[Matrix::rowSums(counts.filtered) > 0, ]
dev <- scry::devianceFeatureSelection(nonzero)
write.csv(data.frame(gene = names(dev), deviance = as.numeric(dev)),
  file.path(OUT, "deviance_reference.csv"),
  row.names = FALSE
)

sf <- 10000 / Matrix::colSums(nonzero)
norm_transform <- Matrix::t(Matrix::t(nonzero) * sf)
norm_transform@x <- log1p(norm_transform@x)
target <- deep$barcode[deep$deep == deep$deep[1]]
y <- rep("rest", ncol(norm_transform))
y[colnames(norm_transform) %in% target] <- "target"
feats <- rownames(norm_transform)[1:500]
wa <- presto::wilcoxauc(X = norm_transform[feats, ], y = y, groups_use = c("target", "rest"))
wa <- wa[wa$group == "target", ]
write.csv(data.frame(
  feature = wa$feature, auc = wa$auc, pval = wa$pval,
  pct_in = wa$pct_in, pct_out = wa$pct_out
),
file.path(OUT, "wilcoxauc_reference.csv"),
row.names = FALSE
)
write.csv(data.frame(barcode = colnames(norm_transform), group = y),
  file.path(OUT, "wilcoxauc_groups.csv"),
  row.names = FALSE
)

## -------------------------------------------------------------- end-to-end
##
## This runs before stage 4 (deliberately) so that pbmc4k_full_pipeline.csv
## is on disk regardless of whether the stage-4 label_dead() bug below
## halts the script. label_dead = FALSE by default, so this call does not
## touch label_dead() at all.

set.seed(42)
full <- valiDrops(counts, plot = FALSE)
write.csv(full, file.path(OUT, "pbmc4k_full_pipeline.csv"), row.names = FALSE)

## ------------------------------------------------------------------- stage 4

## Stage 4 needs an in-memory sparse matrix: read10xCounts on .h5 returns an
## HDF5-backed DelayedMatrix, and label_dead.R:163 does norm_transform@x <- ...,
## which needs a dgCMatrix slot. valiDrops.R:54 accepts dgCMatrix as an input
## class, so this is ordinary supported usage, not a patch.
counts <- as(counts, "dgCMatrix")

met <- metrics$metrics
met$qc.pass <- "fail"
met[met$barcode %in% valid, "qc.pass"] <- "pass"

# The dead-cell score (label_dead.R:45-50) is pure arithmetic on `met` and
# does not touch the buggy threshold-search loop below. Compute it directly
# so we have a trustworthy `score` column even if label_dead() itself
# crashes on this dataset. `soft_label` is intentionally omitted: it is only
# known once the (buggy) threshold search below picks a cutoff.
score_metrics <- met
score_metrics$logUMIs <- scale(score_metrics$logUMIs, scale = FALSE)
score_metrics$logFeatures <- scale(score_metrics$logFeatures, scale = FALSE)
score_metrics$ribosomal_fraction <- asin(sqrt(score_metrics$ribosomal_fraction)) / (pi / 2)
score_metrics$coding_fraction <- asin(sqrt(score_metrics$coding_fraction)) / (pi / 2)
score_metrics$mitochondrial_fraction <- asin(sqrt(score_metrics$mitochondrial_fraction)) / (pi / 2)
score_metrics$score <- score_metrics$logUMIs * -11.82 +
  score_metrics$logFeatures * 2.08 +
  score_metrics$ribosomal_fraction * 158.98 +
  score_metrics$logFeatures * score_metrics$coding_fraction * 18.87 +
  score_metrics$ribosomal_fraction * score_metrics$coding_fraction * -125.9
write.csv(data.frame(barcode = score_metrics$barcode, score = score_metrics$score),
  file.path(OUT, "stage4_soft_labels.csv"),
  row.names = FALSE
)

# Checkpoint for debugging the label_dead() crash without recomputing the
# whole pipeline. Not a fixture: written outside tests/reference_outputs.
saveRDS(list(counts = counts, met = met),
  file.path(tempdir(), "stage4_checkpoint.rds")
)
message(paste("Stage 4 checkpoint saved to", file.path(tempdir(), "stage4_checkpoint.rds")))

# soft labels only: call with train = FALSE to get the deterministic part.
# This has succeeded in prior runs on this dataset/seed (flag = "Succes"),
# overwriting stage4_soft_labels.csv with the full soft_label column and
# producing stage4_meta.csv. Attempted for real either way, not wrapped in
# tryCatch — if this ever fails, its error is left to halt the script rather
# than being papered over.
set.seed(42)
soft <- valiDrops::label_dead(
  counts = counts, metrics = met,
  qc.labels = setNames(as.character(met$qc.pass), met$barcode),
  train = FALSE, plot = FALSE
)
write.csv(data.frame(
  barcode = soft$metrics$barcode,
  score = soft$metrics$score,
  soft_label = as.character(soft$metrics$label)
),
file.path(OUT, "stage4_soft_labels.csv"),
row.names = FALSE
)
write.csv(data.frame(
  key = c("flag", "n_dead"),
  value = c(soft$flag, sum(soft$metrics$label == "dead"))
),
file.path(OUT, "stage4_meta.csv"),
row.names = FALSE
)

# full trained labels
set.seed(42)
trained <- valiDrops::label_dead(
  counts = counts, metrics = met,
  qc.labels = setNames(as.character(met$qc.pass), met$barcode),
  plot = FALSE
)
write.csv(data.frame(
  barcode = trained$metrics$barcode,
  label = as.character(trained$metrics$label)
),
file.path(OUT, "stage4_final.csv"),
row.names = FALSE
)

dev.off()
cat("done\n")
