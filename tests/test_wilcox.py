import warnings
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from validrops.tl._wilcox import wilcoxauc

DATA_DIR = Path(__file__).parent / "data"


def test_wilcoxauc_matches_presto(ref, raw_adata):
    """Compare against presto on the barcodes/genes ``generate_reference.R`` used.

    ``valiDrops.R:101`` subsets the count matrix to ``metrics$protein_coding``
    *before* ``expression_metrics`` ever computes a size factor, so R's
    ``sf <- 10000 / colSums(nonzero)`` (``expression_metrics.R:63-65``) sums
    counts over protein-coding genes only. Summing over the full gene panel
    here would silently scale every cell by a different factor and change
    every rank -- restricting to the ``annotation_genesets.csv`` protein-coding
    set is what makes this a like-for-like comparison.

    The protein-coding mask itself must be built against the *original*
    (pre-``var_names_make_unique``) gene symbols, not ``sub.var_names``: R's
    ``rownames(counts.subset) %in% metrics$protein_coding``
    (``generate_reference.R:248-251``) matches every raw row whose symbol is
    protein-coding, including both copies of each of the 34 gene symbols
    duplicated in the raw h5 -- ``conftest.py``'s ``raw_adata`` fixture has
    already renamed those duplicates (``X``, ``X-1``, ...), so matching on
    ``sub.var_names`` silently drops the second copy's counts from the sum.
    Re-reading the h5 file without deduplication (mirroring
    ``test_deviance.py``'s ``_reconstruct_counts_filtered``) gets the mask
    right; column order is unaffected by ``var_names_make_unique()``, so the
    resulting boolean mask lines up positionally with ``sub`` as-is.

    ``pct_in``/``pct_out`` only depend on nonzero-ness, so they match R
    exactly. ``auc``/``pval`` depend on the rank order of continuously-scaled
    values, and reconstructing R's fixture through a completely different
    (Python, float32-h5-backed) numeric pipeline leaves a handful of genes
    with values close enough to flip a rank at the ULP level -- not a
    wilcoxauc bug, see CLAUDE.md's "validate statistical equivalence, not
    bit-for-bit". With the corrected mask the residual is tiny (a few genes
    out of 500, worst-case relative error on the order of 1e-5), so the
    assertions below combine a near-brief-literal per-gene magnitude cap
    with a count-based gate, rather than relying on a loose correlation
    check that wouldn't catch a badly-wrong individual gene.
    """
    sc = pytest.importorskip("scanpy")
    h5_path = DATA_DIR / "pbmc4k" / "raw.h5"
    if not h5_path.exists():
        pytest.skip("tests/data/pbmc4k/raw.h5 missing")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Variable names are not unique")
        raw_names = sc.read_10x_h5(h5_path).var_names.to_numpy()  # undeduplicated, same column order

    expected = ref("wilcoxauc_reference.csv").set_index("feature")
    groups = ref("wilcoxauc_groups.csv")
    genesets = ref("annotation_genesets.csv")
    protein_coding = set(genesets.loc[genesets["set"] == "protein_coding", "gene"])

    sub = raw_adata[groups["barcode"].to_numpy()]
    pc_mask = np.isin(raw_names, list(protein_coding))
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

    # auc/pval: bound both *how many* genes may miss the brief's literal tolerance and
    # *how far* any single one may miss it, so a real bug (large error, or many genes)
    # cannot hide behind averaging the way a bare correlation check would.
    for col, rtol, worst_case_rtol in [("auc", 1e-6, 1e-5), ("pval", 1e-5, 5e-4)]:
        g = got[col].to_numpy()
        e = expected[col].to_numpy()
        diff = np.abs(g - e)
        rel = diff / np.maximum(np.abs(e), 1e-12)
        mismatched = (rel > rtol) & (diff > 1e-8)
        assert mismatched.mean() < 0.01, f"{col}: {mismatched.sum()}/{len(mismatched)} exceed rtol={rtol}"
        worst = rel[mismatched].max(initial=0.0)
        assert worst < worst_case_rtol, f"{col}: worst mismatch {worst:.3e} exceeds the {worst_case_rtol:.0e} cap"
    # pval specifically: a bare Pearson correlation on raw p-values is dominated by the
    # many values near 1 and would not catch a magnitude error on a small p-value, so
    # also check agreement on -log10(pval), which weights small (significant) p-values.
    got_p = got["pval"].to_numpy()
    exp_p = expected["pval"].to_numpy()
    neglog_r = np.corrcoef(-np.log10(np.maximum(got_p, 1e-300)), -np.log10(np.maximum(exp_p, 1e-300)))[0, 1]
    assert neglog_r > 0.999999999, f"pval: -log10 correlation = {neglog_r}"


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
