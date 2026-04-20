# In R — run once to generate reference data
pak::pkg_install(c("madsen-lab/valiDrops", "immunogenomics/presto", "DropletTestFiles", "DropletUtils"))
library(valiDrops)
library(DropletTestFiles)
library(DropletUtils)

# Load PBMC 4K test dataset
path <- getTestFile("tenx-2.1.0-pbmc4k/1.0.0/raw.h5", prefix=TRUE)
data <- DropletUtils::read10xCounts(path, type = "HDF5")

# Run full pipeline, capture intermediate results
valid <- valiDrops(data)
dir.create("tests/reference_outputs", recursive = TRUE, showWarnings = FALSE)
write.csv(valid, "tests/reference_outputs/pbmc4k_full_pipeline.csv")

# Also run with dead cell labeling
# Encounters an error when predicting dead cells
# Error in quantile.default(metrics$score, brk) : 'probs' outside [0,1]


# valid_dead <- valiDrops(data, label_dead = TRUE)
# write.csv(valid_dead, "tests/reference_outputs/pbmc4k_with_dead_labels.csv")

# Generate Sn estimator reference values
library(robustbase)
set.seed(42)
test_vectors <- list(
  normal = rnorm(100),
  skewed = rexp(100),
  heavy_tail = rt(100, df=3),
  small = rnorm(20),
  large = rnorm(5000)
)
sn_values <- sapply(test_vectors, Sn)
write.csv(data.frame(name=names(sn_values), sn=sn_values),
          "tests/reference_outputs/sn_reference.csv")
