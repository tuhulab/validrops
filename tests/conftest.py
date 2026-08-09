import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

REF_DIR = Path(__file__).parent / "reference_outputs"
DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def ref_dir() -> Path:
    return REF_DIR


@pytest.fixture(scope="session")
def ref():
    """Load a reference CSV by filename."""

    def _load(name: str) -> pd.DataFrame:
        path = REF_DIR / name
        if not path.exists():
            pytest.skip(f"fixture {name} missing; run tests/R/generate_reference.R")
        return pd.read_csv(path)

    return _load


@pytest.fixture(scope="session")
def raw_adata():
    """PBMC 4K raw matrix, cells x genes, with R's positional barcode names.

    R read the matrix without column names, so valiDrops.R:65 assigned
    cell_1 .. cell_N by column index. scanpy.read_10x_h5 preserves the file's
    column order, so cell_N is obs position N-1.
    """
    sc = pytest.importorskip("scanpy")
    path = DATA_DIR / "pbmc4k" / "raw.h5"
    if not path.exists():
        pytest.skip("tests/data/pbmc4k/raw.h5 missing")
    with warnings.catch_warnings():
        # read_10x_h5 warns about the 34 duplicated gene symbols; var_names_make_unique()
        # on the next line is precisely the remedy, so the warning is noise here.
        warnings.filterwarnings("ignore", message="Variable names are not unique")
        adata = sc.read_10x_h5(path)
    adata.var_names_make_unique()
    adata.obs_names = [f"cell_{i + 1}" for i in range(adata.n_obs)]
    return adata


@pytest.fixture
def adata():
    """Tiny synthetic object for unit tests that need an AnnData but no real data."""
    a = ad.AnnData(X=np.array([[1.2, 2.3], [3.4, 4.5], [5.6, 6.7]]).astype(np.float32))
    a.layers["scaled"] = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]).astype(np.float32)
    return a
