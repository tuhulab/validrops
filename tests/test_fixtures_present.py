from pathlib import Path

import pandas as pd
import pytest

REF = Path(__file__).parent / "reference_outputs"

EXPECTED = {
    "sn_reference.csv": ["name", "sn"],
    "rollmean_reference.csv": ["case", "index", "value"],
    "uik_reference.csv": ["case", "knee"],
    "segmented_reference.csv": ["case", "term", "value"],
    "segmented_boot_reference.csv": ["case", "term", "value"],
    "deviance_reference.csv": ["gene", "deviance"],
    "wilcoxauc_reference.csv": ["feature", "auc", "pval", "pct_in", "pct_out"],
    "annotation_genesets.csv": ["gene", "set"],
    "annotation_detection.csv": ["species", "column", "n_mapped"],
    "stage1_threshold.csv": ["barcode", "counts", "rank"],
    "stage1_meta.csv": ["key", "value"],
    "stage2_metrics.csv": [
        "barcode",
        "logUMIs",
        "logFeatures",
        "mitochondrial_fraction",
        "ribosomal_fraction",
        "coding_fraction",
    ],
    "stage2_filters.csv": ["barcode", "pass_mito", "pass_distance", "pass_coding", "final"],
    "stage2_meta.csv": ["key", "value"],
    "stage3_clusters_deep.csv": ["barcode", "deep"],
    "stage3_stats.csv": [
        "cluster",
        "pct.diff",
        "pct.1",
        "pct.2",
        "n_de",
        "n_total",
        "n_negative",
        "min_fdr",
        "de_fraction",
        "mito_fraction",
        "ribo_fraction",
    ],
    "stage3_barcodes.csv": ["barcode"],
    "stage4_soft_labels.csv": ["barcode", "score", "soft_label"],
    "stage4_meta.csv": ["key", "value"],
    "stage4_final.csv": ["barcode", "label"],
    "pbmc4k_full_pipeline.csv": ["barcode", "qc.pass"],
    "uik_inputs.csv": ["case", "x", "y"],
    "segmented_inputs.csv": ["case", "x", "y", "npsi"],
    "wilcoxauc_groups.csv": ["barcode", "group"],
}


@pytest.mark.parametrize(("name", "columns"), EXPECTED.items())
def test_fixture_exists_with_expected_columns(name, columns):
    path = REF / name
    assert path.exists(), f"missing fixture {name}; run tests/R/generate_reference.R"
    df = pd.read_csv(path)
    missing = set(columns) - set(df.columns)
    assert not missing, f"{name} missing columns {missing}"
    assert len(df) > 0
