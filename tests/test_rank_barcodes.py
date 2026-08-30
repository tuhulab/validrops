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
