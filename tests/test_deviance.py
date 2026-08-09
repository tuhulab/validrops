import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from validrops.tl._deviance import deviance_feature_selection

DATA_DIR = Path(__file__).parent / "data"


def _reconstruct_counts_filtered(raw_adata, keep_barcodes: np.ndarray, gene_symbols: np.ndarray):
    """Rebuild R's ``nonzero`` submatrix by raw column position, not by name.

    ``generate_reference.R`` builds ``counts.filtered`` from a genes x cells matrix
    whose row names are *raw* gene symbols -- R never deduplicates them the way
    ``AnnData.var_names_make_unique()`` does. Its row selection is a boolean
    ``rownames(...) %in% metrics$protein_coding``, which keeps *every* raw row
    matching a requested symbol, in original row order; the final
    ``rowSums(counts.filtered) > 0`` filter then drops whichever duplicate-symbol
    rows are all-zero over the kept barcodes.

    34 gene symbols in the pbmc4k panel are duplicated in the raw (undeduplicated)
    var list; of the ones that survive into the fixture's protein-coding set, 5
    have *both* raw occurrences genuinely expressed (they appear twice in
    ``deviance_reference.csv``) and ~21 have exactly one all-zero occurrence that
    R's nonzero filter silently drops. Naive indexing by symbol name on
    ``raw_adata.var_names`` (post-dedup: ``SYMBOL``, ``SYMBOL-1``, ...) always binds
    every requested occurrence of a symbol to the same first raw column -- which is
    wrong whenever R's surviving row is the second occurrence, or when R kept both.

    This walks the raw column order once, and for each requested symbol occurrence
    (in fixture row order) consumes the next candidate raw column with nonzero sum
    over ``keep_barcodes`` -- exactly mirroring R's row-order-preserving ``%in%``
    subset followed by the nonzero filter. Read directly from
    ``tests/R/generate_reference.R`` (the ``counts.filtered`` / ``nonzero`` block).
    """
    sc = pytest.importorskip("scanpy")
    path = DATA_DIR / "pbmc4k" / "raw.h5"
    if not path.exists():
        pytest.skip("tests/data/pbmc4k/raw.h5 missing")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Variable names are not unique")
        raw = sc.read_10x_h5(path)  # undeduplicated var_names, same column order as raw_adata
    raw_names = raw.var_names.to_numpy()

    # pandas' Index.isin is hash-based; plain np.isin between raw_adata's fixed-width
    # unicode obs_names and keep_barcodes' object dtype silently falls back to an
    # O(n*m) broadcast comparison (737280 x 4790 ~= 3.5 billion pairs), taking well
    # over a minute -- this is the reason to avoid it here.
    keep_mask = raw_adata.obs_names.isin(keep_barcodes)
    colsum_all = np.asarray(raw.X[keep_mask, :].sum(axis=0, dtype=np.float64)).ravel()

    candidates_by_symbol = defaultdict(list)
    for pos, name in enumerate(raw_names):
        if colsum_all[pos] > 0:
            candidates_by_symbol[name].append(pos)
    # consume in ascending raw-position order, matching R's row-order-preserving subset
    next_candidate_index = defaultdict(int)

    resolved_idx = np.empty(len(gene_symbols), dtype=np.int64)
    for i, symbol in enumerate(gene_symbols):
        candidates = candidates_by_symbol.get(symbol, [])
        j = next_candidate_index[symbol]
        if j >= len(candidates):
            pytest.fail(
                f"gene symbol {symbol!r} (fixture occurrence {j + 1}) has no remaining nonzero raw "
                "column among the kept barcodes -- reconstruction assumption violated."
            )
        resolved_idx[i] = candidates[j]
        next_candidate_index[symbol] += 1

    return raw.X[np.ix_(keep_mask, resolved_idx)]


def test_deviance_matches_r(ref, raw_adata):
    """Compare against scry on the same protein-coding, QC-passed submatrix."""
    expected = ref("deviance_reference.csv").set_index("gene")["deviance"]
    genes = expected.index.to_numpy()
    # R built this from qc-passing barcodes; stage-2 survivors are the same
    # submatrix expression_metrics saw (see counts.filtered in generate_reference.R).
    filters = ref("stage2_filters.csv")
    keep = filters.loc[filters["final"], "barcode"].to_numpy()
    sub_X = _reconstruct_counts_filtered(raw_adata, keep, genes)
    got = deviance_feature_selection(sub_X)
    np.testing.assert_allclose(got, expected.to_numpy(), rtol=1e-8)


def test_top_genes_overlap_r(ref, raw_adata):
    expected = ref("deviance_reference.csv").set_index("gene")["deviance"]
    genes = expected.index.to_numpy()
    filters = ref("stage2_filters.csv")
    keep = filters.loc[filters["final"], "barcode"].to_numpy()
    sub_X = _reconstruct_counts_filtered(raw_adata, keep, genes)
    got = deviance_feature_selection(sub_X)
    top_py = set(genes[np.argsort(-got)[:5000]])
    top_r = set(expected.sort_values(ascending=False).index[:5000])
    assert len(top_py & top_r) / 5000 > 0.99


def test_deviance_is_nonnegative_on_random_counts():
    rng = np.random.default_rng(0)
    X = sp.csr_matrix(rng.poisson(2.0, size=(200, 50)).astype(np.float64))
    dev = deviance_feature_selection(X)
    assert dev.shape == (50,)
    assert np.all(dev >= -1e-8)


def test_all_zero_gene_gives_zero_deviance():
    X = sp.csr_matrix(np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]))
    dev = deviance_feature_selection(X)
    assert dev[1] == 0.0


def test_all_zero_gene_gives_zero_deviance_dense():
    """Same edge case as the sparse version, but through the dense branch.

    Gene 1 is all-zero, so gene 0 alone carries every cell's total (p == 1 for
    every entry), which makes log1p(-p) = -inf and the resulting deviance
    non-finite for gene 0 too -- exercising the dense path's own
    non-finite-to-zero fallback, not just the sparse one.
    """
    X = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    dev = deviance_feature_selection(X)
    assert dev[1] == 0.0


def test_dense_and_sparse_agree():
    rng = np.random.default_rng(1)
    dense = rng.poisson(1.5, size=(100, 20)).astype(np.float64)
    np.testing.assert_allclose(
        deviance_feature_selection(dense),
        deviance_feature_selection(sp.csr_matrix(dense)),
        rtol=1e-10,
    )
