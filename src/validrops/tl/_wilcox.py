"""Vectorised Wilcoxon rank-sum with AUC, ported from ``presto::wilcoxauc``."""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import norm as _normal
from scipy.stats import rankdata


def wilcoxauc(X, y: np.ndarray, groups_use: tuple[str, str]) -> pd.DataFrame:
    """Rank-sum test of ``groups_use[0]`` against ``groups_use[1]``, per feature.

    Parameters
    ----------
    X
        Expression, **genes x cells**. Dense or sparse.
    y
        Group label per cell. Labels outside ``groups_use`` are excluded,
        which is how ``expression_metrics.R:149-151`` uses a third
        ``"excluded"`` level.
    groups_use
        ``(in_group, out_group)``.

    Returns
    -------
    DataFrame with ``feature`` (positional index), ``auc``, ``pval``,
    ``pct_in``, ``pct_out``. Percentages are 0-100, matching presto.

    Notes
    -----
    Normal approximation with tie correction *and* presto's half-integer
    continuity correction: ``compute_pval`` in presto's ``R/utils.R`` computes
    ``z <- ustat - .5 * n1n2; z <- z - sign(z) * .5`` before dividing by the
    tie-corrected sigma, i.e. it shrinks ``|U - mu|`` by 0.5 towards zero.
    Fully-tied features (zero rank variance, e.g. an all-zero row) give
    ``z = 0/0``; presto reports ``p = 1`` for these rather than ``NaN``,
    which is what the ``np.isfinite`` fallback below reproduces.
    """
    in_group, out_group = groups_use
    y = np.asarray(y)
    mask = np.isin(y, [in_group, out_group])
    if not mask.any():
        raise ValueError(f"no cells labelled {in_group!r} or {out_group!r}")

    dense = X.toarray() if sp.issparse(X) else np.asarray(X, dtype=np.float64)
    dense = np.atleast_2d(dense)[:, mask]
    labels = y[mask]
    is_in = labels == in_group

    n1 = int(is_in.sum())
    n2 = int((~is_in).sum())
    n = n1 + n2
    if n1 == 0 or n2 == 0:
        raise ValueError(f"both groups must be non-empty, got n1={n1}, n2={n2}")

    ranks = np.apply_along_axis(rankdata, 1, dense)
    r1 = ranks[:, is_in].sum(axis=1)
    u = r1 - n1 * (n1 + 1) / 2.0
    auc = u / (n1 * n2)

    # tie correction: 1 - sum(t^3 - t) / (n^3 - n), per feature
    tie_term = np.empty(dense.shape[0], dtype=np.float64)
    for i in range(dense.shape[0]):
        _, counts = np.unique(dense[i], return_counts=True)
        tie_term[i] = np.sum(counts**3 - counts)
    correction = 1.0 - tie_term / (n**3 - n) if n > 1 else np.zeros_like(tie_term)

    mu = n1 * n2 / 2.0
    var = n1 * n2 * (n + 1) / 12.0 * correction
    z_raw = u - mu
    z_continuity = z_raw - np.sign(z_raw) * 0.5
    with np.errstate(divide="ignore", invalid="ignore"):
        z = z_continuity / np.sqrt(var)
    z = np.where(np.isfinite(z), z, 0.0)
    pval = 2.0 * _normal.sf(np.abs(z))

    pct_in = (dense[:, is_in] > 0).sum(axis=1) / n1 * 100.0
    pct_out = (dense[:, ~is_in] > 0).sum(axis=1) / n2 * 100.0

    return pd.DataFrame(
        {
            "feature": np.arange(dense.shape[0]),
            "auc": auc,
            "pval": pval,
            "pct_in": pct_in,
            "pct_out": pct_out,
        }
    )
