import numpy as np


def test_obs_names_are_positional(raw_adata):
    assert raw_adata.obs_names[0] == "cell_1"
    assert raw_adata.obs_names[14] == "cell_15"
    assert raw_adata.n_obs == len(raw_adata.obs_names)


def test_stage2_metrics_match_recomputed_totals(raw_adata, ref):
    """cell_N must be column N-1 of the R matrix: verify via log total counts.

    Sum with an explicit float64 accumulator: raw_adata.X is float32, and the individual
    10x counts are exact in float32 (integers well under 2^24), but summing ~33,694
    float32 values per cell in float32 accumulates ~1e-7 relative error against R's
    float64 arithmetic. A float64 accumulator makes the sum exact.
    """
    m = ref("stage2_metrics.csv").set_index("barcode")
    sub = raw_adata[m.index]
    totals = np.asarray(sub.X.sum(axis=1, dtype=np.float64)).ravel()
    np.testing.assert_allclose(np.log(totals), m["logUMIs"].to_numpy(), rtol=1e-10)


def test_stage2_metrics_match_recomputed_features(raw_adata, ref):
    m = ref("stage2_metrics.csv").set_index("barcode")
    sub = raw_adata[m.index]
    n_genes = np.asarray((sub.X > 0).sum(axis=1, dtype=np.float64)).ravel()
    np.testing.assert_allclose(np.log(n_genes), m["logFeatures"].to_numpy(), rtol=1e-10)
