"""Stage 2b: filtering on quality metrics. Ports ``quality_filter.R:26-216``."""

import logging

import numpy as np
import pandas as pd
from anndata import AnnData
from sklearn.mixture import GaussianMixture

from validrops._constants import MITO_SCAN_INCREMENT, UNS_KEY
from validrops.tl._segmented import SegmentedFitError, segmented
from validrops.tl._stats import sn
from validrops.tl._uik import uik

logger = logging.getLogger(__name__)


def quality_filter(
    adata: AnnData,
    *,
    mito: bool | float = True,
    distance: bool = True,
    coding: bool = True,
    contrast: bool = False,
    mito_nreps: int = 10,
    mito_max: float = 0.3,
    npsi: int = 3,
    dist_threshold: float = 5,
    coding_threshold: float = 3,
    contrast_threshold: float = 3,
    random_state: int = 0,
) -> None:
    """Filter barcodes on the metrics from :func:`~validrops.tl.quality_metrics`.

    Three filters run in sequence, each seeing only the survivors of the last:
    a mitochondrial-fraction cap, a residual band around the feature-to-UMI
    relationship, and a band around the protein-coding fraction.

    Parameters
    ----------
    adata
        Must already carry the columns written by ``quality_metrics``.
    mito
        ``True`` to detect the threshold, a float to set it directly,
        ``False`` to skip.
    distance, coding, contrast
        Enable each sub-filter.
    mito_nreps
        Repetitions of the stochastic threshold search; the median wins.
    mito_max
        Above this, fall back to segmented regression (``quality_filter.R:79``).
    npsi
        Breakpoints for the feature-to-UMI fit, decremented on failure.
    dist_threshold, coding_threshold, contrast_threshold
        Multiples of Sn defining each band.
    random_state
        Seed for the mixture fits and subsampling.

    Returns
    -------
    None. Writes ``pass_mito``, ``pass_distance``, ``pass_coding``,
    ``pass_contrast`` and ``qc_pass`` to ``adata.obs``.
    """
    for name, value in (
        ("dist_threshold", dist_threshold),
        ("coding_threshold", coding_threshold),
        ("contrast_threshold", contrast_threshold),
        ("mito_nreps", mito_nreps),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be greater than 0, got {value}")
    if int(npsi) <= 0:
        raise ValueError(f"npsi must be greater than 0, got {npsi}")

    base = adata.obs["rank_pass"].to_numpy(dtype=bool) if "rank_pass" in adata.obs else np.ones(adata.n_obs, dtype=bool)
    metrics = adata.obs.loc[base]
    surviving = pd.Index(metrics.index)
    uns = adata.uns.setdefault(UNS_KEY, {})
    rng = np.random.default_rng(random_state)

    # ---- mitochondrial ------------------------------------------------------
    if mito is not False:
        if {"mitochondrial_fraction", "log_features"} <= set(metrics.columns):
            if mito is True:
                threshold, method = _detect_mito_threshold(
                    metrics.loc[surviving], mito_nreps, mito_max, rng, random_state
                )
            else:
                threshold, method = float(mito), "user"
            uns["mitochondrial_threshold"] = threshold
            uns["mito_threshold_method"] = method
            keep = metrics.loc[surviving, "mitochondrial_fraction"] <= threshold
            surviving = surviving[keep.to_numpy()]
        else:
            logger.warning(
                "Columns mitochondrial_fraction and log_features do not both exist. "
                "Skipping filtering using the mitochondrial fraction."
            )
    _write_pass(adata, "pass_mito", surviving)

    # ---- distance -----------------------------------------------------------
    if distance:
        if {"log_umis", "log_features"} <= set(metrics.columns):
            sub = metrics.loc[surviving]
            residuals = _feature_umi_residuals(sub["log_umis"].to_numpy(), sub["log_features"].to_numpy(), int(npsi))
            spread = sn(residuals)
            centre = float(np.median(residuals))
            inside = (residuals <= centre + spread * dist_threshold) & (residuals >= centre - spread * dist_threshold)
            surviving = surviving[inside]
        else:
            logger.warning(
                "Columns log_umis and log_features do not both exist. Skipping filtering using the distance."
            )
    _write_pass(adata, "pass_distance", surviving)

    # ---- coding -------------------------------------------------------------
    if coding:
        surviving = _band_filter(metrics, surviving, "coding_fraction", coding_threshold)
    _write_pass(adata, "pass_coding", surviving)

    # ---- contrast -----------------------------------------------------------
    if contrast:
        surviving = _band_filter(metrics, surviving, "contrast_fraction", contrast_threshold)
    _write_pass(adata, "pass_contrast", surviving)

    _write_pass(adata, "qc_pass", surviving)
    logger.info("Step 3: %d barcodes passed quality filtering", len(surviving))


def _write_pass(adata: AnnData, column: str, surviving: pd.Index) -> None:
    adata.obs[column] = adata.obs_names.isin(surviving)


def _band_filter(metrics: pd.DataFrame, surviving: pd.Index, column: str, multiplier: float) -> pd.Index:
    """Keep barcodes within ``median +/- multiplier * Sn`` of ``column``."""
    if column not in metrics.columns:
        logger.warning("Column named %s does not exist. Skipping filtering using it.", column)
        return surviving
    values = metrics.loc[surviving, column].to_numpy()
    spread = sn(values)
    centre = float(np.median(values))
    inside = (values >= centre - spread * multiplier) & (values <= centre + spread * multiplier)
    return surviving[inside]


def _detect_mito_threshold(
    metrics: pd.DataFrame, nreps: int, mito_max: float, rng: np.random.Generator, random_state: int
) -> tuple[float, str]:
    """Threshold on the mitochondrial fraction, with R's segmented fallback."""
    thresholds = []
    log_features = metrics["log_features"].to_numpy().reshape(-1, 1)
    mito_fraction = metrics["mitochondrial_fraction"].to_numpy()

    for rep in range(nreps):
        # init_params="random" (fully random responsibilities), not sklearn's kmeans
        # default: mixtools::normalmixEM's default start (mu/sigma unspecified) is a
        # random draw, not a k-means seed, and matching that initialisation strategy
        # is what reproduces R's group assignment and threshold (see task-14 ledger).
        model = GaussianMixture(n_components=2, random_state=random_state + rep, n_init=1, init_params="random")
        model.fit(log_features)
        high = int(np.argmax(model.means_.ravel()))
        group = model.predict(log_features) == high
        source = mito_fraction[group] if group.any() else mito_fraction
        sequence = np.arange(float(np.median(source)), 1.0, MITO_SCAN_INCREMENT)
        counts = np.array([np.sum(source <= value) for value in sequence], dtype=np.float64)
        thresholds.append(uik(sequence, counts))

    threshold = float(np.median(thresholds))
    if threshold <= mito_max:
        return threshold, "gmm_uik"

    # quality_filter.R:79-97 — subsampled segmented regression fallback
    logger.info("Mitochondrial threshold %.3f exceeded the cap; using segmented fallback", threshold)
    sample_size = min(5000, int(np.floor(len(metrics) * 0.8)))
    fallback = []
    for _ in range(nreps):
        idx = rng.choice(len(metrics), size=sample_size, replace=False)
        x = mito_fraction[idx]
        y = metrics["log_features"].to_numpy()[idx]
        psi_count = 1
        while True:
            try:
                fit = segmented(x, y, npsi=psi_count)
            except SegmentedFitError:
                if psi_count >= 5:
                    break
                psi_count += 1
                continue
            if float(fit.psi.min()) <= mito_max or psi_count >= 5:
                fallback.append(float(fit.psi.min()))
                break
            psi_count += 1
    if not fallback:
        raise SegmentedFitError("mitochondrial threshold fallback failed to converge")
    return float(np.median(fallback)), "segmented_fallback"


def _feature_umi_residuals(log_umis: np.ndarray, log_features: np.ndarray, npsi: int) -> np.ndarray:
    """Residuals of ``log_features ~ log_umis``, decrementing npsi on failure."""
    while npsi >= 1:
        try:
            return segmented(log_umis, log_features, npsi=npsi).residuals
        except SegmentedFitError:
            npsi -= 1
    # quality_filter.R:147 falls back to a plain linear fit
    slope, intercept = np.polyfit(log_umis, log_features, 1)
    return log_features - (slope * log_umis + intercept)
