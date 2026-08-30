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
  row_ids <- seq_len(nrow(d))  # stable row identity, shared across every column of this table
  for (ci in seq_along(colnames(d))) {
    col <- colnames(d)[ci]
    pieces[[length(pieces) + 1]] <- data.frame(
      species      = species_names[i],
      species_index = i,          # R's which.max ties break toward the first table
      column_name  = col,
      column_index = ci,          # ... and toward the first column within it
      row_id       = row_ids,     # position within this species' table; aligns column slices
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
