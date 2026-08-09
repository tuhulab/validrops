from anndata import AnnData


def rank_barcode(adata: AnnData) -> AnnData:
    """Rank barcodes according to number of UMIs (or optionlly number of genes) and detect a
    cut-off point for putative cell- (or nuclei-) containing barcodes.

    Parameters
    ----------
    adata
        The AnnData object to preprocess.

    Returns
    -------
    Some integer value.
    """
    adata.X.sum(axis=1)
    print("Implement a tool to run on the AnnData object.")
    return adata


import scanpy as sc

adata = sc.read_10x_h5("tests/data/pbmc4k/raw.h5")

print(adata)
