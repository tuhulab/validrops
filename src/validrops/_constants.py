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
