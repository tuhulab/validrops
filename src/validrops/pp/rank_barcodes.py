"""Stage 1: barcode-rank filtering. Ports ``rank_barcodes.R:31-150``."""

import logging

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.stats import rankdata

from validrops._constants import UNS_KEY
from validrops.tl._segmented import SegmentedFitError, segmented
from validrops.tl._stats import rollmean

logger = logging.getLogger(__name__)

_UMI_ALIASES = {"UMI", "umi", "UMIS", "umis", "UMIs"}
_GENE_ALIASES = {"Genes", "gene", "genes"}


def rank_barcodes(
    adata: AnnData,
    *,
    type: str = "UMI",
    psi_min: int = 2,
    psi_max: int = 5,
    alpha: float = 0.001,
    alpha_max: float = 0.05,
    boot: int = 10,
    factor: float = 1.5,
    random_state: int = 0,
) -> None:
    """Rank barcodes and detect the cut-off separating cells from empty droplets.

    Fits segmented regressions to the log-rank / log-count curve for a range of
    breakpoint counts, picks the simplest model within ``factor`` of the best
    RMSE, and takes the sharpest turn in the resulting slope sequence as the
    threshold.

    Parameters
    ----------
    adata
        Cells x genes. Not subset; results are written for every barcode.
    type
        ``"UMI"`` to rank by total counts, ``"Genes"`` by detected genes.
        UMI works better at low ambient contamination, genes at high.
    psi_min, psi_max
        Range of breakpoint counts to try.
    alpha
        Breakpoints are sought between the ``alpha`` and ``1-alpha`` quantiles.
        Incremented up to ``alpha_max`` if the fit fails.
    alpha_max
        Ceiling for the ``alpha`` escalation.
    boot
        Bootstrap restarts per segmented fit.
    factor
        How many folds above the best RMSE a simpler model may be.
    random_state
        Seed for the bootstrap restarts.

    Returns
    -------
    None. Writes ``adata.obs["rank_pass"]`` and
    ``adata.uns["validrops"]["rank_threshold"]``.
    """
    if type not in _UMI_ALIASES | _GENE_ALIASES:
        raise ValueError(f"type must be UMI or Genes, got {type!r}")
    if not 0 < psi_min <= psi_max:
        raise ValueError(f"psi_min must be >0 and <= psi_max, got {psi_min} and {psi_max}")

    # float64 accumulator: adata.X from scanpy.read_10x_h5 is float32, and summing
    # ~33,694 values per barcode in float32 drifts ~1e-7 relative against R's doubles
    # — enough to flip rank order near the threshold. Summing in float64 is exact and
    # costs nothing (see task-4 ledger ruling). The Genes branch sums a boolean array,
    # which is exact regardless of dtype, but dtype=np.float64 is harmless there too.
    counts = (
        np.asarray(adata.X.sum(axis=1, dtype=np.float64)).ravel()
        if type in _UMI_ALIASES
        else np.asarray((adata.X > 0).sum(axis=1, dtype=np.float64)).ravel()
    )

    # rank is computed before zero-count barcodes are dropped (rank_barcodes.R:73-74)
    ranks_all = rankdata(-counts)
    nonzero = counts > 0
    frame = pd.DataFrame(
        {"counts": counts[nonzero], "rank": ranks_all[nonzero]},
        index=adata.obs_names[nonzero],
    )
    frame = frame.sort_values(["counts", "rank"], ascending=[False, False])

    unique = frame[~frame["counts"].duplicated()]
    log_counts = np.log(unique["counts"].to_numpy())
    log_ranks = np.log(unique["rank"].to_numpy())

    window = int(np.ceil(2 * len(unique) ** (1 / 3)))
    y = rollmean(log_counts, window)
    x = rollmean(log_ranks, window)

    fit, n_psi = _best_segmented_model(x, y, psi_min, psi_max, alpha, alpha_max, boot, factor, random_state)

    angles = _slope_angles(fit.slopes)
    # skip the first angle (rank_barcodes.R:127)
    best_break = int(np.argmin(angles[1:])) + 1
    nearest = int(np.argmin(np.abs(log_ranks - fit.psi[best_break])))
    threshold = float(np.exp(log_counts[nearest]))

    adata.obs["rank_pass"] = counts >= threshold
    uns = adata.uns.setdefault(UNS_KEY, {})
    uns["rank_threshold"] = threshold
    uns["barcode_ranks"] = frame
    uns["rank_npsi"] = n_psi

    n_pass = int(adata.obs["rank_pass"].sum())
    logger.info("Step 1: %d barcodes passed the rank threshold (%.1f counts)", n_pass, threshold)
    if n_pass > 20000:
        logger.warning(
            "More than 20,000 barcodes passed initial filtering. Breakpoint estimation may "
            "have failed; try increasing alpha, alpha_max or psi_max."
        )


def _best_segmented_model(x, y, psi_min, psi_max, alpha, alpha_max, boot, factor, random_state):
    """Fit each breakpoint count, then take the simplest model within ``factor`` of the best RMSE."""
    fits: list[tuple[int, float, object]] = []
    for npsi in range(psi_min, psi_max + 1):
        current_alpha = alpha
        while current_alpha <= alpha_max:
            psi_init = np.linspace(np.quantile(x, current_alpha), np.quantile(x, 1 - current_alpha), npsi)
            try:
                fit = segmented(
                    x,
                    y,
                    psi_init=psi_init,
                    alpha=current_alpha - current_alpha / 1000,
                    n_boot=boot,
                    random_state=random_state,
                )
            except SegmentedFitError:
                current_alpha += alpha  # rank_barcodes.R:104
                continue
            fits.append((npsi, fit.rmse, fit))
            break

    if not fits:
        raise SegmentedFitError(
            f"no segmented model converged for psi in {psi_min}..{psi_max}; try increasing alpha, alpha_max or psi_max"
        )

    best_rmse = min(rmse for _, rmse, _ in fits)
    for npsi, rmse, fit in fits:  # fits are in ascending npsi order, so this is R's min(index)
        if rmse <= best_rmse * factor:
            return fit, npsi
    raise AssertionError("unreachable: the best model always satisfies the factor bound")


def _slope_angles(slopes: np.ndarray) -> np.ndarray:
    """Angle in degrees between each consecutive pair of segment slopes."""
    left = slopes[:-1]
    right = slopes[1:]
    return np.degrees(np.arctan((left - right) / (1 + left * right)))
