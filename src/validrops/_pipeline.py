"""End-to-end pipeline. Ports ``valiDrops.R:21-139``."""

import inspect
import logging

import numpy as np
from anndata import AnnData

from . import pp, tl
from ._constants import UNS_KEY

logger = logging.getLogger(__name__)


def validrops(
    adata: AnnData,
    *,
    rank_barcodes: bool = True,
    stage_three: bool = True,
    label_dead: bool = False,
    mitochondrial_clusters: float | None = 3,
    ribosomal_clusters: float | None = 3,
    random_state: int = 0,
    verbose: bool = True,
    **kwargs,
) -> None:
    """Run the complete valiDrops quality-control pipeline.

    Parameters
    ----------
    adata
        Raw, unfiltered counts, cells x genes. Annotated in place; nothing is
        removed, so call ``adata[adata.obs["qc_pass"]].copy()`` afterwards to
        get the clean object.
    rank_barcodes
        Run stage 1. When ``False``, every barcode with a non-zero count
        proceeds (``valiDrops.R:79``).
    stage_three
        Run the expression-based stages. When ``False``, ``qc_pass`` comes
        straight from stage 2.
    label_dead
        Run the dead-cell prediction. Off by default: it is stochastic and
        much slower than the rest of the pipeline.
    mitochondrial_clusters, ribosomal_clusters
        Deviations above the median cluster content beyond which a cluster is
        dropped in stage 3b. ``None`` disables.
    random_state
        Seed threaded through every stochastic step.
    verbose
        Log progress.
    **kwargs
        Forwarded to the individual stage functions by parameter name.

    Returns
    -------
    None.

    Examples
    --------
    >>> import scanpy as sc, validrops  # doctest: +SKIP
    >>> adata = sc.read_10x_h5("raw.h5")  # doctest: +SKIP
    >>> validrops.validrops(adata)  # doctest: +SKIP
    >>> clean = adata[adata.obs["qc_pass"]].copy()  # doctest: +SKIP
    """
    if verbose:
        logging.getLogger("validrops").setLevel(logging.INFO)

    if rank_barcodes:
        logger.info("Step 1: Filtering on the barcode-rank plot.")
        pp.rank_barcodes(adata, random_state=random_state, **_for(pp.rank_barcodes, kwargs))
    else:
        logger.info("Step 1: Removing barcodes with zero counts.")
        adata.obs["rank_pass"] = np.asarray(adata.X.sum(axis=1, dtype=np.float64)).ravel() > 0

    logger.info("Step 2: Collecting quality metrics.")
    tl.quality_metrics(adata, **_for(tl.quality_metrics, kwargs))

    logger.info("Step 3: Filtering on quality metrics.")
    pp.quality_filter(adata, random_state=random_state, **_for(pp.quality_filter, kwargs))

    if stage_three:
        logger.info("Step 4: Collecting expression-based metrics.")
        tl.expression_metrics(adata, random_state=random_state, **_for(tl.expression_metrics, kwargs))

        logger.info("Step 5: Filtering on expression-based metrics.")
        pp.expression_filter(
            adata,
            mito=mitochondrial_clusters,
            ribo=ribosomal_clusters,
            **_for(pp.expression_filter, kwargs, exclude={"mito", "ribo"}),
        )

    if label_dead:
        logger.info("Step %d: Predicting dead cells.", 6 if stage_three else 4)
        if not stage_three:
            logger.warning("Predicting dead cells without stage 3. CAUTION: this has not been tested.")
        pp.label_dead(adata, random_state=random_state, **_for(pp.label_dead, kwargs))

    uns = adata.uns.setdefault(UNS_KEY, {})
    uns["params"] = {
        "rank_barcodes": rank_barcodes,
        "stage_three": stage_three,
        "label_dead": label_dead,
        "mitochondrial_clusters": mitochondrial_clusters,
        "ribosomal_clusters": ribosomal_clusters,
        "random_state": random_state,
        **kwargs,
    }

    logger.info("\t%d barcodes passed quality control.", int(adata.obs["qc_pass"].sum()))
    if label_dead:
        dead = (adata.obs["qc_pass"] & (adata.obs["label"] == "dead")).sum()
        logger.info("\t%d barcodes that passed quality control are predicted to be dead.", int(dead))


def _for(func, kwargs: dict, exclude: set[str] | None = None) -> dict:
    """Select the kwargs a stage function actually accepts (R does this with ``doCall``).

    Functions declaring ``**kwargs`` receive everything not explicitly excluded,
    which is how ``label_dead``'s training parameters reach it.
    """
    parameters = inspect.signature(func).parameters
    blocked = {"adata", "random_state"} | (exclude or set())
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return {k: v for k, v in kwargs.items() if k not in blocked}
    accepted = set(parameters) - blocked
    return {k: v for k, v in kwargs.items() if k in accepted}
