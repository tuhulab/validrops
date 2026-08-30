"""Stage 4: dead-cell labelling. Ports ``label_dead.R:43-483``.

This module holds the deterministic half of the stage: the heuristic dead
score and the soft-labelling quantile search. The stochastic consensus
training loop lives in ``._label_dead_train`` and is wired up in
:func:`label_dead` (Task 18).
"""

import logging

import numpy as np
from anndata import AnnData

from validrops._constants import DEAD_LABEL_FRAC, DEAD_SCORE_COEFFICIENTS, UNS_KEY
from validrops.tl._uik import uik

logger = logging.getLogger(__name__)

_QUANTILE_STEP = 0.0001
_QUANTILE_BLOCK = 0.1


def dead_score(
    log_umis: np.ndarray,
    log_features: np.ndarray,
    ribosomal: np.ndarray,
    coding: np.ndarray,
) -> np.ndarray:
    """Heuristic dead-cell score. Ports ``label_dead.R:45-50``.

    Dying cells lose cytoplasmic RNA first, so they show a low UMI count for
    their gene count and an inflated ribosomal fraction. The coefficients are
    fitted constants from the paper and must not be changed.

    Parameters
    ----------
    log_umis, log_features
        Natural logs of total counts and detected genes. Centred internally
        (mean-subtracted, not scaled).
    ribosomal, coding
        Fractions in [0, 1]. Arcsine-square-root transformed and normalised
        by ``pi/2`` internally.

    Returns
    -------
    One score per barcode. Lower means more dead-like.
    """
    u = np.asarray(log_umis, dtype=np.float64)
    f = np.asarray(log_features, dtype=np.float64)
    u = u - np.nanmean(u)
    f = f - np.nanmean(f)
    r = np.arcsin(np.sqrt(np.asarray(ribosomal, dtype=np.float64))) / (np.pi / 2)
    c = np.arcsin(np.sqrt(np.asarray(coding, dtype=np.float64))) / (np.pi / 2)

    k = DEAD_SCORE_COEFFICIENTS
    return (
        k["log_umis"] * u
        + k["log_features"] * f
        + k["ribosomal"] * r
        + k["features_x_coding"] * f * c
        + k["ribosomal_x_coding"] * r * c
    )


def _threshold_at(score: np.ndarray, max_quantile: float) -> float:
    """Knee of the quantile curve up to ``max_quantile``.

    Ports ``label_dead.R:61-68`` and the equivalent block inside the search
    loop: build the (break, quantile) curve from 1e-4 up to ``max_quantile``
    in 1e-4 steps, take the unit-invariant knee, and use that break as the
    probability at which to read the score threshold.
    """
    breaks = np.arange(_QUANTILE_STEP, max_quantile + _QUANTILE_STEP / 2, _QUANTILE_STEP)
    values = np.quantile(score, breaks)
    return float(np.quantile(score, uik(breaks, values)))


def _contingency(labels: np.ndarray, qc: np.ndarray) -> np.ndarray:
    """2x2 table of (dead, live) x (fail, pass).

    Matches R's ``table(labels, qc)`` alphabetical level ordering: rows
    ``dead`` then ``live``, columns ``fail`` then ``pass``. ``[0, 1]`` is the
    count that drives the stop decision at ``label_dead.R:87`` (QC-passing
    dead cells).
    """
    table = np.zeros((2, 2), dtype=np.int64)
    for i, label in enumerate(("dead", "live")):
        for j, status in enumerate(("fail", "pass")):
            table[i, j] = int(np.sum((labels == label) & (qc == status)))
    return table


def soft_label(
    score: np.ndarray,
    qc: np.ndarray,
    *,
    label_thrs: float | None = None,
    label_frac: float = DEAD_LABEL_FRAC,
    n_relabel: int = 1,
) -> tuple[np.ndarray, float, str]:
    """Split barcodes into live and dead by a score threshold.

    Ports ``label_dead.R:56-143``. When no threshold is given, the quantile
    ceiling is raised in steps of 0.1 until the dead/live by pass/fail table
    stops gaining QC-passing dead cells; the previous threshold is kept.

    Returns
    -------
    ``(labels, threshold, flag)``. ``flag`` is ``"Success"``, ``"Caution"``
    when barcodes had to be relabelled, or ``"Failed"`` when the split is
    unusable.
    """
    score = np.asarray(score, dtype=np.float64)
    qc = np.asarray(qc)
    flag = "Success"

    if label_thrs is None:
        max_quantile = _QUANTILE_BLOCK
        last_threshold = _threshold_at(score, max_quantile)
        last_table = _contingency(np.where(score <= last_threshold, "dead", "live"), qc)

        while True:
            max_quantile += _QUANTILE_BLOCK
            new_threshold = _threshold_at(score, max_quantile)
            new_table = _contingency(np.where(score <= new_threshold, "dead", "live"), qc)

            if new_table.min() > 0:
                if last_table.min() > 0:
                    if last_table[0, 1] == new_table[0, 1]:
                        last_table, last_threshold = new_table, new_threshold
                    else:  # last_table[0, 1] < new_table[0, 1]
                        label_thrs = last_threshold
                        break
                else:
                    label_thrs = new_threshold
                    break
            elif max_quantile >= 0.95:
                label_thrs = new_threshold
                flag = "Failed"
                break
            else:
                last_table, last_threshold = new_table, new_threshold

    labels = np.where(score <= label_thrs, "dead", "live")
    n_dead = int(np.sum(labels == "dead"))

    if n_dead < 3:
        # label_dead.R:122-125 — too few dead to train on
        logger.info("Soft-labeling identified fewer than 3 dead barcodes")
        flag = "Failed"
    elif np.sum((qc == "pass") & (labels == "dead")) == 0:
        # label_dead.R:126-132 — no QC-passing dead cell; relabel the
        # n_relabel least-dead-like dead barcodes as pass
        logger.info("Soft-labeling labelled 0 QC-passing barcode as dead; relabeling %d", n_relabel)
        dead_idx = np.flatnonzero(labels == "dead")
        least_dead = dead_idx[np.argsort(-score[dead_idx])][:n_relabel]
        qc[least_dead] = "pass"
        flag = "Caution"
    elif n_dead / labels.size >= label_frac:
        # label_dead.R:133-139 — too many dead; abort and relabel all live
        logger.info("Soft-labeling identified more than %.0f%% of barcodes as dead; aborting", label_frac * 100)
        labels = np.full(labels.size, "live")
        flag = "Failed"
    else:
        logger.info("Soft-labeling identified %d dead barcodes", n_dead)

    return labels, float(label_thrs), flag


def label_dead(
    adata: AnnData,
    *,
    train: bool = True,
    label_thrs: float | None = None,
    label_frac: float = DEAD_LABEL_FRAC,
    n_relabel: int = 1,
    **kwargs,
) -> None:
    """Label barcodes as live, dead or uncertain.

    Parameters
    ----------
    adata
        Must carry ``log_umis``, ``log_features``, ``ribosomal_fraction``,
        ``coding_fraction`` and ``qc_pass`` in ``obs``.
    train
        Run the consensus training loop. ``False`` returns the soft labels.
    label_thrs
        Explicit score cutoff. ``None`` detects one.
    label_frac
        Abort if more than this fraction is labelled dead.
    n_relabel
        Barcodes to relabel when no QC-passing barcode is soft-labelled dead.

    Returns
    -------
    None. Writes ``obs["dead_score"]`` and ``obs["label"]``.
    """
    mask = adata.obs["rank_pass"].to_numpy(dtype=bool) if "rank_pass" in adata.obs else np.ones(adata.n_obs, dtype=bool)
    sub = adata.obs.loc[mask]
    score = dead_score(
        sub["log_umis"].to_numpy(),
        sub["log_features"].to_numpy(),
        sub["ribosomal_fraction"].to_numpy(),
        sub["coding_fraction"].to_numpy(),
    )
    qc = np.where(sub["qc_pass"].to_numpy(dtype=bool), "pass", "fail")
    labels, threshold, flag = soft_label(score, qc, label_thrs=label_thrs, label_frac=label_frac, n_relabel=n_relabel)

    if train and flag != "Failed":
        from ._label_dead_train import train_labels  # Task 18

        labels, flag = train_labels(adata, mask, score, labels, qc, threshold, flag, **kwargs)

    scores_out = np.full(adata.n_obs, np.nan)
    scores_out[mask] = score
    adata.obs["dead_score"] = scores_out

    labels_out = np.full(adata.n_obs, None, dtype=object)
    labels_out[mask] = labels
    adata.obs["label"] = labels_out
    adata.obs["label"] = adata.obs["label"].astype("category")

    uns = adata.uns.setdefault(UNS_KEY, {})
    uns["label_threshold"] = threshold
    uns["label_flag"] = flag
