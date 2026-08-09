"""Binomial deviance feature selection, ported from the R package ``scry``."""

import numpy as np
import scipy.sparse as sp


def deviance_feature_selection(counts) -> np.ndarray:
    """Per-gene binomial deviance against a constant-proportion null.

    Ports ``scry::devianceFeatureSelection`` with its default
    ``fam = "binomial"``. Genes with the largest deviance are the most
    informative; valiDrops keeps the top 5000.

    R's ``scry:::.compute_deviance`` works on a genes x cells matrix: it
    computes ``sz <- colSums(m)`` (per-cell totals) before transposing to
    cells x genes for ``sparseBinomialDeviance``. This function already takes
    the AnnData orientation (cells x genes) directly, so ``sz`` is a row sum
    and the per-gene totals are a column sum -- no transpose needed.

    Parameters
    ----------
    counts
        Raw counts, **cells x genes** (the R function takes genes x cells;
        this is the AnnData orientation).

    Returns
    -------
    Deviance per gene, length ``counts.shape[1]``. Non-finite results (e.g.
    an all-zero gene, where both the saturated and null log-likelihoods are
    degenerate) are set to 0, matching ``.compute_deviance``'s
    ``out[is.na(out)] <- 0``.
    """
    is_sparse = sp.issparse(counts)
    X = counts.tocsr().astype(np.float64) if is_sparse else np.asarray(counts, dtype=np.float64)

    sz = np.asarray(X.sum(axis=1, dtype=np.float64)).ravel()  # per-cell totals
    sz_sum = float(sz.sum(dtype=np.float64))
    feature_sums = np.asarray(X.sum(axis=0, dtype=np.float64)).ravel()  # per-gene totals

    if is_sparse:
        coo = X.tocoo()
        rows, cols = coo.row, coo.col
        p = coo.data / sz[rows]
        # p == 1 whenever a cell's entire total sits in this one gene, giving
        # log1p(-p) = log(0) = -inf; that -inf later makes ll_sat/deviance
        # non-finite for that gene, and the isfinite mask below zeroes it out
        # (matching .compute_deviance's out[is.na(out)] <- 0). p == 0 cannot
        # occur here since coo.data only holds structural non-zeros.
        with np.errstate(divide="ignore", invalid="ignore"):
            log_p = np.log(p)
            log1p_neg = np.log1p(-p)
            contrib = coo.data * (log_p - log1p_neg) + sz[rows] * log1p_neg
        # Sum contributions per gene (column). A sparse matrix build-and-sum
        # is a single vectorised pass, unlike np.add.at which loops in Python
        # and is prohibitively slow on matrices the size of pbmc4k.
        ll_sat = np.asarray(sp.csc_matrix((contrib, (rows, cols)), shape=X.shape).sum(axis=0, dtype=np.float64)).ravel()
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            p = X / sz[:, None]
            log_p = np.where(p > 0, np.log(np.where(p > 0, p, 1.0)), 0.0)
            log1p_neg = np.log1p(-p)
            contrib = np.where(p > 0, X * (log_p - log1p_neg) + sz[:, None] * log1p_neg, 0.0)
        ll_sat = contrib.sum(axis=0, dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        pi = feature_sums / sz_sum
        l1p = np.log1p(-pi)
        ll_null = feature_sums * (np.log(pi) - l1p) + sz_sum * l1p

    deviance = 2.0 * (ll_sat - ll_null)
    deviance[~np.isfinite(deviance)] = 0.0
    return deviance
