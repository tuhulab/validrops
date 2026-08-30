"""Fast end-to-end run on synthetic data, so every commit exercises the pipeline."""

from importlib.resources import files

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

import validrops


def _synthetic(n_cells=1200, n_genes=300, seed=0):
    rng = np.random.default_rng(seed)
    counts = rng.poisson(0.3, size=(n_cells, n_genes)).astype(np.float32)
    counts[:250] *= 25  # a clear population of real cells
    adata = ad.AnnData(sp.csr_matrix(counts))
    # real human symbols so quality_metrics' species detection succeeds
    table = pd.read_parquet(files("validrops.data").joinpath("annotation.parquet"))
    names = table.loc[table["species"] == "human", "value"].dropna().unique()
    adata.var_names = list(names[:n_genes])
    adata.obs_names = [f"cell_{i + 1}" for i in range(n_cells)]
    return adata


def test_pipeline_runs_without_stage_three():
    adata = _synthetic()
    validrops.validrops(adata, stage_three=False)
    assert adata.obs["qc_pass"].sum() > 0
    assert adata.obs["qc_pass"].sum() < adata.n_obs


def test_pipeline_is_reproducible():
    a, b = _synthetic(), _synthetic()
    for adata in (a, b):
        validrops.validrops(adata, stage_three=False, random_state=7)
    np.testing.assert_array_equal(a.obs["qc_pass"].to_numpy(), b.obs["qc_pass"].to_numpy())


def test_nothing_is_removed_from_the_object():
    adata = _synthetic()
    before = adata.shape
    validrops.validrops(adata, stage_three=False)
    assert adata.shape == before
