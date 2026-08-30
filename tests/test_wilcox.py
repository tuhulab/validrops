import numpy as np
import pytest
import scipy.sparse as sp

from validrops.tl._wilcox import wilcoxauc


def test_wilcoxauc_matches_presto(ref, raw_adata):
    """Compare against presto on the barcodes/genes ``generate_reference.R`` used.

    ``valiDrops.R:101`` subsets the count matrix to ``metrics$protein_coding``
    *before* ``expression_metrics`` ever computes a size factor, so R's
    ``sf <- 10000 / colSums(nonzero)`` (``expression_metrics.R:63-65``) sums
    counts over protein-coding genes only. Summing over the full gene panel
    here would silently scale every cell by a different factor and change
    every rank -- restricting to the ``annotation_genesets.csv`` protein-coding
    set is what makes this a like-for-like comparison.

    ``pct_in``/``pct_out`` only depend on nonzero-ness, so they match R
    exactly. ``auc``/``pval`` depend on the rank order of continuously-scaled
    values, and reconstructing R's fixture through a completely different
    (Python, float32-h5-backed) numeric pipeline leaves a handful of genes
    with values close enough to flip a rank at the ULP level -- not a
    wilcoxauc bug, see CLAUDE.md's "validate statistical equivalence, not
    bit-for-bit". The concordance bar mirrors the project-wide standard
    (>95% agreement, Pearson r > 0.999999).
    """
    expected = ref("wilcoxauc_reference.csv").set_index("feature")
    groups = ref("wilcoxauc_groups.csv")
    genesets = ref("annotation_genesets.csv")
    protein_coding = set(genesets.loc[genesets["set"] == "protein_coding", "gene"])

    sub = raw_adata[groups["barcode"].to_numpy()]
    pc_mask = sub.var_names.isin(protein_coding)
    sf = 10000.0 / np.asarray(sub[:, pc_mask].X.sum(axis=1, dtype=np.float64)).ravel()
    norm = sp.csr_matrix(sub.X).multiply(sf[:, None]).tocsr()
    norm.data = np.log1p(norm.data)

    genes = expected.index.to_numpy()
    gene_idx = [sub.var_names.get_loc(g) for g in genes]
    X = norm[:, gene_idx].T.tocsr()  # genes x cells

    got = wilcoxauc(X, groups["group"].to_numpy(), ("target", "rest")).set_index("feature")
    got.index = genes

    np.testing.assert_allclose(got["pct_in"], expected["pct_in"], rtol=1e-6)
    np.testing.assert_allclose(got["pct_out"], expected["pct_out"], rtol=1e-6)

    for col, rtol in [("auc", 1e-6), ("pval", 1e-5)]:
        g = got[col].to_numpy()
        e = expected[col].to_numpy()
        diff = np.abs(g - e)
        rel = diff / np.maximum(np.abs(e), 1e-12)
        mismatched = (rel > rtol) & (diff > 1e-8)
        assert mismatched.mean() < 0.05, f"{col}: {mismatched.sum()}/{len(mismatched)} exceed tolerance"
        r = np.corrcoef(g, e)[0, 1]
        assert r > 0.999999, f"{col}: pearson r = {r}"


def test_excluded_cells_are_dropped():
    X = np.array([[1.0, 2.0, 3.0, 100.0]])  # 1 gene, 4 cells
    y = np.array(["target", "target", "rest", "excluded"])
    out = wilcoxauc(X, y, ("target", "rest"))
    assert len(out) == 1
    # cell 4 excluded, so target {1,2} vs rest {3}: target always lower -> AUC 0
    assert out["auc"].iloc[0] == 0.0


def test_perfect_separation_gives_auc_one():
    X = np.array([[10.0, 11.0, 1.0, 2.0]])
    y = np.array(["target", "target", "rest", "rest"])
    out = wilcoxauc(X, y, ("target", "rest"))
    assert out["auc"].iloc[0] == 1.0


def test_all_ties_give_auc_half_and_p_one():
    X = np.array([[5.0, 5.0, 5.0, 5.0]])
    y = np.array(["target", "target", "rest", "rest"])
    out = wilcoxauc(X, y, ("target", "rest"))
    assert out["auc"].iloc[0] == 0.5
    assert out["pval"].iloc[0] == 1.0


def test_pct_columns_count_nonzero_fraction():
    X = np.array([[0.0, 1.0, 0.0, 0.0]])
    y = np.array(["target", "target", "rest", "rest"])
    out = wilcoxauc(X, y, ("target", "rest"))
    assert out["pct_in"].iloc[0] == pytest.approx(50.0)
    assert out["pct_out"].iloc[0] == pytest.approx(0.0)
