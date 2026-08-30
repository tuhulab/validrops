"""Stage 2a: per-barcode quality metrics. Ports ``quality_metrics.R:32-217``."""

import logging

import numpy as np
from anndata import AnnData

from validrops._constants import UNS_KEY
from validrops.tl._annotation import detect_annotation, gene_sets

logger = logging.getLogger(__name__)

_METRIC_COLUMNS = (
    "log_umis",
    "log_features",
    "mitochondrial_fraction",
    "ribosomal_fraction",
    "coding_fraction",
)


def quality_metrics(
    adata: AnnData,
    *,
    contrast: AnnData | None = None,
    contrast_type: str = "denominator",
    species: str = "auto",
    annotation: str = "auto",
    mito: str | list[str] = "auto",
    ribo: str | list[str] = "auto",
    coding: str | list[str] = "auto",
    verbose: bool = False,
) -> None:
    """Compute per-barcode quality metrics.

    Parameters
    ----------
    adata
        Cells x genes. Only barcodes with ``obs["rank_pass"]`` True are
        measured, if that column exists.
    contrast
        Optional second matrix for the exon fraction. For scRNA-seq the main
        matrix holds exonic reads and the contrast holds exon+intron, with
        ``contrast_type="numerator"``; for snRNA-seq it is the other way round.
    contrast_type
        ``"denominator"`` or ``"numerator"``.
    species, annotation
        ``"auto"`` to detect, or an explicit name.
    mito, ribo, coding
        ``"auto"`` to look up, or explicit gene-name lists.
    verbose
        Log the detected species, annotation and gene-set sizes.

    Returns
    -------
    None. Writes five metric columns to ``adata.obs`` and three boolean
    columns to ``adata.var``.
    """
    if contrast_type not in ("denominator", "numerator"):
        raise ValueError(f'contrast_type must be "denominator" or "numerator", got {contrast_type!r}')

    mask = adata.obs["rank_pass"].to_numpy(dtype=bool) if "rank_pass" in adata.obs else np.ones(adata.n_obs, dtype=bool)
    sub = adata[mask]
    gene_names = adata.var_names.to_numpy()

    match = detect_annotation(gene_names, species=species, annotation=annotation)
    detected = gene_sets(gene_names, match)

    sets = {
        "mitochondrial": _resolve_set(mito, detected["mitochondrial"], gene_names, "mitochondrial"),
        "ribosomal": _resolve_set(ribo, detected["ribosomal"], gene_names, "ribosomal"),
        "protein_coding": _resolve_set(coding, detected["protein_coding"], gene_names, "protein-coding"),
    }

    if verbose:
        logger.info(
            "Detected sample origin: %s. Detected gene annotation: %s. Mapped %d/%d (%.3g%%) of input IDs.",
            match.species,
            match.column,
            match.n_mapped,
            match.n_total,
            match.n_mapped / match.n_total * 100,
        )
        logger.info(
            "Found %d mitochondrial genes, %d ribosomal genes, and %d protein-coding genes.",
            len(sets["mitochondrial"]),
            len(sets["ribosomal"]),
            len(sets["protein_coding"]),
        )

    # float64 accumulator: adata.X from scanpy.read_10x_h5 is float32, and summing
    # ~33,694 values per barcode in float32 drifts ~1e-7 relative against R's doubles.
    # dtype=np.float64 must be passed INTO .sum() so the accumulator itself runs in
    # double precision -- casting the float32-accumulated result afterwards does not
    # undo the drift already baked in (see task-4 ledger ruling; binds this task).
    totals = np.asarray(sub.X.sum(axis=1, dtype=np.float64)).ravel()
    n_features = np.asarray((sub.X > 0).sum(axis=1, dtype=np.float64)).ravel()

    # Zero-count barcodes (e.g. unfiltered droplets with no reads) yield log(0) = -inf and
    # 0/0 = nan, exactly as R's log()/division would -- R does not warn on either, so match
    # that silence here instead of raising spurious RuntimeWarnings for correct arithmetic.
    with np.errstate(divide="ignore", invalid="ignore"):
        values = {
            "log_umis": np.log(totals),
            "log_features": np.log(n_features),
        }
        for key, column in (
            ("mitochondrial", "mitochondrial_fraction"),
            ("ribosomal", "ribosomal_fraction"),
            ("protein_coding", "coding_fraction"),
        ):
            selector = np.isin(gene_names, sets[key])
            values[column] = np.asarray(sub[:, selector].X.sum(axis=1, dtype=np.float64)).ravel() / totals

    for column in _METRIC_COLUMNS:
        out = np.full(adata.n_obs, np.nan)
        out[mask] = values[column]
        adata.obs[column] = out

    if contrast is not None:
        shared = contrast[sub.obs_names]
        contrast_totals = np.asarray(shared.X.sum(axis=1, dtype=np.float64)).ravel()
        with np.errstate(divide="ignore", invalid="ignore"):
            fraction = totals / contrast_totals if contrast_type == "denominator" else contrast_totals / totals
        out = np.full(adata.n_obs, np.nan)
        out[mask] = fraction
        adata.obs["contrast_fraction"] = out

    for key, column in (
        ("mitochondrial", "mitochondrial"),
        ("ribosomal", "ribosomal"),
        ("protein_coding", "protein_coding"),
    ):
        adata.var[column] = np.isin(gene_names, sets[key])

    uns = adata.uns.setdefault(UNS_KEY, {})
    uns["gene_sets"] = sets
    uns["species"] = match.species
    uns["annotation_column"] = match.column
    uns["n_mapped"] = match.n_mapped


def _resolve_set(given, detected: np.ndarray, gene_names: np.ndarray, label: str) -> np.ndarray:
    """Use the caller's gene list when given, else the detected one."""
    if isinstance(given, str) and given == "auto":
        return detected
    requested = np.asarray(list(given), dtype=str)
    missing = set(requested) - set(gene_names)
    if missing:
        raise ValueError(f"{len(missing)} {label} gene(s) not present in the count matrix, e.g. {sorted(missing)[:3]}")
    return requested
