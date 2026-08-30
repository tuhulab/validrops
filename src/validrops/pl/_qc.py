"""Diagnostic plots mirroring valiDrops' base-R plots."""

import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData
from matplotlib.axes import Axes

from validrops._constants import UNS_KEY


def _axes(ax: Axes | None) -> Axes:
    return ax if ax is not None else plt.subplots(figsize=(5, 4))[1]


def barcode_rank(adata: AnnData, *, ax: Axes | None = None, **kwargs) -> Axes:
    """Log-rank against log-count, with the detected threshold marked.

    The knee separates cell-containing droplets from ambient background.
    Mirrors ``rank_barcodes.R:132-139``.

    Parameters
    ----------
    adata
        Must have run :func:`~validrops.pp.rank_barcodes`.
    ax
        Axes to draw on; a new one is created when ``None``.
    **kwargs
        Forwarded to the scatter.

    Returns
    -------
    The axes drawn on.
    """
    ax = _axes(ax)
    ranks = adata.uns[UNS_KEY]["barcode_ranks"]
    threshold = adata.uns[UNS_KEY]["rank_threshold"]
    ax.scatter(np.log(ranks["rank"]), np.log(ranks["counts"]), s=4, c="#CDCDCD", **kwargs)
    ax.axhline(np.log(threshold), color="red", lw=1)
    ax.set_xlabel("log Rank")
    ax.set_ylabel("log Counts")
    ax.set_title(f"n = {int(adata.obs['rank_pass'].sum())} barcodes above threshold")
    return ax


def mito_threshold(adata: AnnData, *, ax: Axes | None = None, **kwargs) -> Axes:
    """Mitochondrial fraction against detected features, with the cutoff.

    Mirrors ``quality_filter.R:109-112``.

    Parameters
    ----------
    adata
        Must have run :func:`~validrops.pp.quality_filter` with the mito
        filter enabled.
    ax
        Axes to draw on; a new one is created when ``None``.
    **kwargs
        Forwarded to the scatter.

    Returns
    -------
    The axes drawn on.
    """
    ax = _axes(ax)
    threshold = adata.uns[UNS_KEY]["mitochondrial_threshold"]
    obs = adata.obs.dropna(subset=["mitochondrial_fraction", "log_features"])
    colours = np.where(obs["mitochondrial_fraction"] > threshold, "red", "black")
    ax.scatter(obs["log_features"], obs["mitochondrial_fraction"], s=4, c=colours, **kwargs)
    ax.axhline(threshold, color="black", lw=1)
    ax.set_xlabel("log Total features")
    ax.set_ylabel("Mitochondrial fraction")
    ax.set_title(f"Threshold = {threshold:.3f}")
    return ax


def umi_vs_features(adata: AnnData, *, ax: Axes | None = None, **kwargs) -> Axes:
    """Detected features against total UMIs, coloured by the distance filter.

    Barcodes far from the trend are doublets or damaged cells. Mirrors
    ``quality_filter.R:144-150``.

    Parameters
    ----------
    adata
        Must have run :func:`~validrops.pp.quality_filter`.
    ax
        Axes to draw on; a new one is created when ``None``.
    **kwargs
        Forwarded to the scatter.

    Returns
    -------
    The axes drawn on.
    """
    ax = _axes(ax)
    obs = adata.obs.dropna(subset=["log_umis", "log_features"])
    colours = np.where(obs["pass_distance"], "grey", "red")
    ax.scatter(obs["log_umis"], obs["log_features"], s=4, c=colours, **kwargs)
    ax.set_xlabel("log Total UMIs")
    ax.set_ylabel("log Total features")
    ax.set_title(f"Kept {int(obs['pass_distance'].sum())} barcodes")
    return ax


def coding_fraction(adata: AnnData, *, ax: Axes | None = None, **kwargs) -> Axes:
    """Histogram of the protein-coding fraction. Mirrors ``quality_filter.R:172-175``.

    Parameters
    ----------
    adata
        Must have run :func:`~validrops.pp.quality_filter`.
    ax
        Axes to draw on; a new one is created when ``None``.
    **kwargs
        Forwarded to the histogram.

    Returns
    -------
    The axes drawn on.
    """
    ax = _axes(ax)
    values = adata.obs["coding_fraction"].dropna()
    ax.hist(values, bins="auto", color="#4C72B0", **kwargs)
    ax.set_xlabel("Fraction of UMIs from protein-coding genes")
    ax.set_ylabel("Barcodes")
    ax.set_title(f"Kept {int(adata.obs['pass_coding'].sum())} barcodes")
    return ax


def dead_score(adata: AnnData, *, ax: Axes | None = None, **kwargs) -> Axes:
    """Sorted dead-cell score with the soft-labelling cutoff.

    Mirrors ``label_dead.R:146-149``. Requires ``label_dead`` to have run.

    Parameters
    ----------
    adata
        Must have run :func:`~validrops.pp.label_dead`.
    ax
        Axes to draw on; a new one is created when ``None``.
    **kwargs
        Forwarded to the scatter.

    Returns
    -------
    The axes drawn on.
    """
    if "dead_score" not in adata.obs:
        raise KeyError("dead_score not found; run validrops.pp.label_dead first")
    ax = _axes(ax)
    values = np.sort(adata.obs["dead_score"].dropna().to_numpy())
    ax.scatter(np.arange(values.size), values, s=4, c="black", **kwargs)
    ax.axhline(adata.uns[UNS_KEY]["label_threshold"], color="red", ls="--")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Score")
    return ax
