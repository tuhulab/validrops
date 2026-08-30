"""Stage 3b: filtering on expression metrics. Ports ``expression_filter.R:22-113``."""

import logging

import numpy as np
import pandas as pd
from anndata import AnnData

from validrops._constants import UNS_KEY
from validrops.tl._segmented import SegmentedFitError, segmented
from validrops.tl._stats import sn

logger = logging.getLogger(__name__)


def expression_filter(
    adata: AnnData,
    *,
    mito: float | None = 3,
    ribo: float | None = 3,
    min_significant: int = 1,
    min_target_pct: float = 0.3,
    max_background_pct: float = 0.7,
    min_diff_pct: float = 0.2,
    min_de_frac: float = 0.01,
    min_significance_level: float | None = None,
) -> None:
    """Keep barcodes belonging to clusters with coherent marker expression.

    A cluster of real cells has genes that are specifically expressed in it.
    A cluster of debris or ambient RNA does not. Each threshold below encodes
    one aspect of that distinction.

    Parameters
    ----------
    adata
        Must carry ``obs["cluster"]`` and ``uns["validrops"]["cluster_stats"]``.
    mito, ribo
        Deviations above the median cluster mitochondrial/ribosomal content
        beyond which a cluster is dropped. ``None`` disables the check.
    min_significant
        Minimum significant marker genes.
    min_target_pct
        Minimum mean fraction of in-cluster barcodes expressing the top markers.
    max_background_pct
        Maximum mean fraction of out-of-cluster barcodes expressing them.
    min_diff_pct
        Minimum in-versus-out difference.
    min_de_frac
        Minimum fraction of tested genes that must be significant.
    min_significance_level
        ``-log10`` significance the best marker must reach. ``None`` detects it.

    Returns
    -------
    None. Overwrites ``adata.obs["qc_pass"]``.
    """
    for name, value in (
        ("min_target_pct", min_target_pct),
        ("max_background_pct", max_background_pct),
        ("min_diff_pct", min_diff_pct),
    ):
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1, got {value}")
    if min_significant < 0:
        raise ValueError(f"min_significant must be >= 0, got {min_significant}")
    for name, value in (("mito", mito), ("ribo", ribo)):
        if value is not None and not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be None or a number, got {value!r}")

    stats = adata.uns[UNS_KEY]["cluster_stats"]
    if min_significance_level is None:
        min_significance_level = _detect_significance_level(stats)
    elif not isinstance(min_significance_level, (int, float)) or min_significance_level < 0:
        raise ValueError(f"min_significance_level must be >= 0, got {min_significance_level!r}")

    # expression_filter.R:72-78 — the six sequential filters, in R's order
    keep = stats[stats["n_negative"] == 0]
    keep = keep[keep["pct.diff"] >= min_diff_pct]
    keep = keep[keep["pct.1"] >= min_target_pct]
    keep = keep[keep["pct.2"] <= max_background_pct]
    keep = keep[keep["n_de"] >= min_significant]
    # R's -log10(0) = Inf (no warning), so zero-min_fdr clusters clear any
    # finite bar; np.errstate silences numpy's matching divide warning.
    with np.errstate(divide="ignore"):
        sig = -np.log10(keep["min_fdr"].to_numpy())
    keep = keep[sig >= min_significance_level]
    keep = keep[keep["de_fraction"] > min_de_frac]

    # expression_filter.R:79-88 — mito/ribo caps use the FULL stats (median
    # and Sn over every cluster, not just the survivors so far)
    if mito is not None:
        cap = np.median(stats["mito_fraction"]) + mito * sn(stats["mito_fraction"].to_numpy())
        allowed = set(stats.loc[stats["mito_fraction"] <= cap, "cluster"])
        keep = keep[keep["cluster"].isin(allowed)]
    if ribo is not None:
        cap = np.median(stats["ribo_fraction"]) + ribo * sn(stats["ribo_fraction"].to_numpy())
        allowed = set(stats.loc[stats["ribo_fraction"] <= cap, "cluster"])
        keep = keep[keep["cluster"].isin(allowed)]

    surviving = keep["cluster"].to_numpy()
    cluster = adata.obs["cluster"]
    valid = cluster.isin(surviving) & cluster.notna()
    adata.obs["qc_pass"] = valid.to_numpy(dtype=bool)

    uns = adata.uns.setdefault(UNS_KEY, {})
    uns["min_significance_level"] = float(min_significance_level)
    uns["surviving_clusters"] = surviving
    logger.info(
        "Step 5: %d of %d clusters passed, keeping %d barcodes",
        len(surviving),
        len(stats),
        int(adata.obs["qc_pass"].sum()),
    )


def _detect_significance_level(stats: pd.DataFrame) -> float:
    """Automatic significance threshold. Ports ``expression_filter.R:57-65``."""
    subset = stats[stats["min_fdr"] > 0]
    y = subset["pct.diff"].to_numpy()
    x = -np.log10(subset["min_fdr"].to_numpy())

    low = x[y <= 0.4]
    threshold = np.median(low) + sn(low) * 3 if low.size else np.nan

    try:
        model_level = float(segmented(x, y, npsi=1).psi[0])
    except (SegmentedFitError, ValueError):
        model_level = np.nan

    candidates = [v for v in (threshold, model_level) if np.isfinite(v)]
    if not candidates:
        raise ValueError("could not determine a significance threshold; pass min_significance_level explicitly")
    return float(min(candidates))
