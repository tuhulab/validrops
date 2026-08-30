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
