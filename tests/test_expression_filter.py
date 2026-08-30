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
    adata.obs["cluster"] = pd.Series(clusters["deep"], index=clusters.index).reindex(adata.obs_names).astype("category")
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
    stats = staged_with_r_inputs.uns[UNS_KEY]["cluster_stats"]
    survivors = set(staged_with_r_inputs.uns[UNS_KEY]["surviving_clusters"])
    # R's -log10(0) = Inf (expression_filter.R:77), so the zero-min_fdr
    # clusters survive any finite significance bar. This asserts our boundary
    # handling matches R's Inf semantics, which a naively "nothing survives"
    # assertion would contradict for this dataset (11 such clusters).
    zero_fdr = set(stats.loc[stats["min_fdr"] == 0, "cluster"])
    assert survivors == zero_fdr


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
