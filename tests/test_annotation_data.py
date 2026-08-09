from importlib.resources import files

import pandas as pd
import pytest

EXPECTED_ROWS = {
    "human": 106_851,
    "mouse": 82_003,
    "rat": 47_954,
    # NOTE: zebrafish/fly counts corrected relative to the task brief's draft values
    # (which had them swapped). Verified against valiDrops:::annotation content:
    # the 67,809-row table contains zebrafish marker genes (sox2, myod1, pax6a) and
    # lacks fly markers (white); the 77,216-row table contains fly marker genes
    # (per, Adh) and lacks zebrafish markers (sox2). See task-2-report.md for detail.
    "zebrafish": 67_809,
    "worm": 46_912,
    "fly": 77_216,
}


@pytest.fixture(scope="module")
def annotation():
    path = files("validrops.data").joinpath("annotation.parquet")
    return pd.read_parquet(path)


def test_all_six_species_present(annotation):
    assert set(annotation["species"].unique()) == set(EXPECTED_ROWS)


def test_row_counts_match_r(annotation):
    # each source row contributes one row per ID column, so divide back out
    for species, expected in EXPECTED_ROWS.items():
        sub = annotation[annotation["species"] == species]
        n_source = sub.groupby("column_name").size().max()
        assert n_source == expected, species


def test_human_has_expected_columns(annotation):
    human = annotation[annotation["species"] == "human"]
    assert set(human["column_name"].unique()) == {
        "NCBI",
        "HGNC",
        "Ensembl",
        "Chr",
        "Symbol",
        "Type",
        "Alias",
    }


def test_mouse_uses_mgi_not_hgnc(annotation):
    mouse = annotation[annotation["species"] == "mouse"]
    cols = set(mouse["column_name"].unique())
    assert "MGI" in cols
    assert "HGNC" not in cols


def test_human_mitochondrial_gene_count(annotation):
    human = annotation[annotation["species"] == "human"]
    mito = human[(human["column_name"] == "Symbol") & (human["chr"] == "MT")]
    assert len(mito) == 98


def test_human_protein_coding_count(annotation):
    human = annotation[annotation["species"] == "human"]
    pc = human[(human["column_name"] == "Symbol") & (human["type"] == "protein_coding")]
    assert len(pc) == 55_304
