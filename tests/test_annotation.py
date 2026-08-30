import numpy as np

from validrops.tl._annotation import clean_gene_ids, detect_annotation, gene_sets


def test_clean_strips_ensembl_version():
    got = clean_gene_ids(np.array(["ENSG00000141510.17", "ENSG00000141510"]))
    assert list(got) == ["ENSG00000141510", "ENSG00000141510"]


def test_clean_strips_mouse_ensembl_version():
    got = clean_gene_ids(np.array(["ENSMUSG00000059552.14"]))
    assert list(got) == ["ENSMUSG00000059552"]


def test_clean_leaves_symbols_untouched():
    names = np.array(["TP53", "MT-CO1", "RPL13A", "HLA-DRB1"])
    np.testing.assert_array_equal(clean_gene_ids(names), names)


def test_detect_human_symbols(raw_adata, ref):
    expected = ref("annotation_detection.csv").iloc[0]
    match = detect_annotation(raw_adata.var_names.to_numpy())
    assert match.species == expected["species"]
    assert match.column == expected["column"]


def test_gene_sets_match_r(raw_adata, ref):
    expected = ref("annotation_genesets.csv")
    match = detect_annotation(raw_adata.var_names.to_numpy())
    got = gene_sets(raw_adata.var_names.to_numpy(), match)
    for name in ("mitochondrial", "ribosomal", "protein_coding"):
        want = set(expected.loc[expected["set"] == name, "gene"])
        assert set(got[name]) == want, name


def test_explicit_species_skips_detection(raw_adata):
    match = detect_annotation(raw_adata.var_names.to_numpy(), species="human", annotation="symbol")
    assert match.species == "human"
    assert match.column == "Symbol"


def test_gene_sets_return_original_names():
    names = np.array(["ENSG00000198804.2", "TP53"])
    match = detect_annotation(names, species="human", annotation="ensembl")
    sets = gene_sets(names, match)
    # the versioned name is what came in, so it must be what comes out
    assert "ENSG00000198804.2" in set(sets["mitochondrial"])
