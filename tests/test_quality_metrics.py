import numpy as np
import pytest

import validrops
from validrops._constants import UNS_KEY


@pytest.fixture(scope="module")
def staged(raw_adata, ref):
    """Raw object with R's stage-1 result applied, so stage 2a is tested in isolation."""
    adata = raw_adata.copy()
    passing = set(ref("stage1_threshold.csv")["barcode"])
    adata.obs["rank_pass"] = [b in passing for b in adata.obs_names]
    validrops.tl.quality_metrics(adata)
    return adata


@pytest.mark.parametrize(
    ("obs_col", "ref_col"),
    [
        ("log_umis", "logUMIs"),
        ("log_features", "logFeatures"),
        ("mitochondrial_fraction", "mitochondrial_fraction"),
        ("ribosomal_fraction", "ribosomal_fraction"),
    ],
)
def test_metric_matches_r(staged, ref, obs_col, ref_col):
    expected = ref("stage2_metrics.csv").set_index("barcode")[ref_col]
    got = staged.obs.loc[expected.index, obs_col]
    np.testing.assert_allclose(got.to_numpy(), expected.to_numpy(), rtol=1e-8)
    assert np.corrcoef(got, expected)[0, 1] > 0.99


def test_coding_fraction_matches_r_within_bound(staged, ref):
    """``coding_fraction`` has a known, bounded drift from R -- not exact like the other four.

    Root cause (task-4 ledger finding, confirmed by direct measurement): of the 34 gene
    symbols duplicated in the raw h5, 23 fall in the protein-coding set (0 in mito/ribo,
    which is why those four metrics above are exact). scanpy's ``var_names_make_unique()``
    renames each duplicate's second occurrence to ``SYMBOL-1``, and ``clean_gene_ids()``
    only strips Ensembl version suffixes (not the ``-1``/``-2`` dedup suffix), so
    ``gene_sets()`` matches only the base-name row and misses the renamed one. R's
    un-deduplicated ``rownames`` sums both rows via ``%in%``, so R's numerator is larger
    wherever a droplet has nonzero counts on the renamed duplicate. This is PARKED per
    the task-13 ledger ruling: a real fix needs either recovering pre-dedup names through
    the whole adata lifecycle (bigger than this task -- var_names_make_unique() is a
    standard scanpy step applied before valiDrops sees the data) or a suffix-stripping
    heuristic with real false-positive risk for symbols that legitimately end ``-<digit>``.

    Measured against the full pbmc4k fixture: Pearson r = 0.9999963 (>> the project's
    0.99 bar), 320/5189 barcodes (6.2%) differ by more than 1e-6, worst single-barcode
    absolute difference 0.005587 (~0.56 percentage points). The bounds below give real
    headroom above those measured values so a future regression that widens the drift
    (e.g. a bug that breaks gene-set matching more broadly, not just on the known 23
    duplicates) still fails loudly, rather than this staying an unbounded xfail forever.
    """
    expected = ref("stage2_metrics.csv").set_index("barcode")["coding_fraction"]
    got = staged.obs.loc[expected.index, "coding_fraction"]

    r = np.corrcoef(got, expected)[0, 1]
    assert r > 0.99, f"coding_fraction: Pearson r = {r} against R"

    diff = np.abs(got.to_numpy() - expected.to_numpy())
    assert diff.max() < 0.01, f"coding_fraction: worst single-barcode diff {diff.max()} exceeds the 0.01 cap"


def test_non_rank_passing_barcodes_are_nan(staged):
    outside = staged.obs.loc[~staged.obs["rank_pass"], "log_umis"]
    assert outside.isna().all()


def test_gene_sets_written_to_var(staged, ref):
    expected = ref("annotation_genesets.csv")
    for name, col in [
        ("mitochondrial", "mitochondrial"),
        ("ribosomal", "ribosomal"),
        ("protein_coding", "protein_coding"),
    ]:
        want = set(expected.loc[expected["set"] == name, "gene"])
        got = set(staged.var_names[staged.var[col]])
        assert got == want, name


def test_detection_recorded_in_uns(staged):
    uns = staged.uns[UNS_KEY]
    assert uns["species"] == "human"
    assert uns["annotation_column"] == "Symbol"
    assert uns["n_mapped"] > 0


def test_explicit_gene_lists_bypass_detection(raw_adata):
    adata = raw_adata[:200].copy()
    mito = list(adata.var_names[:3])
    validrops.tl.quality_metrics(adata, mito=mito, ribo=list(adata.var_names[3:6]), coding=list(adata.var_names[6:20]))
    assert set(adata.var_names[adata.var["mitochondrial"]]) == set(mito)


def test_unknown_gene_in_explicit_list_raises(raw_adata):
    adata = raw_adata[:50].copy()
    with pytest.raises(ValueError, match="not present"):
        validrops.tl.quality_metrics(adata, mito=["NOT_A_REAL_GENE"])


def test_contrast_fraction_denominator(raw_adata):
    # raw_adata is the unfiltered droplet matrix (barcode order is not count order), so a
    # plain [:100] slice is ~50% zero-total barcodes and produces 0/0 = NaN, matching R's
    # math but not this test's intent. Select 100 barcodes with nonzero counts instead.
    totals = np.asarray(raw_adata.X.sum(axis=1, dtype=np.float64)).ravel()
    nonzero = np.flatnonzero(totals)[:100]
    adata = raw_adata[nonzero].copy()
    contrast = adata.copy()
    contrast.X = contrast.X * 2
    validrops.tl.quality_metrics(adata, contrast=contrast, contrast_type="denominator")
    np.testing.assert_allclose(adata.obs["contrast_fraction"].to_numpy(), 0.5, rtol=1e-10)


def test_invalid_contrast_type_raises(raw_adata):
    adata = raw_adata[:50].copy()
    with pytest.raises(ValueError, match="denominator"):
        validrops.tl.quality_metrics(adata, contrast=adata.copy(), contrast_type="sideways")
