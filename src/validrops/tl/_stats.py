"""Scale estimation and smoothing primitives ported from R."""

import numpy as np

from validrops._constants import SN_C_SMALL, SN_CONSTANT


def _finite_sample_correction(n: int) -> float:
    """Rousseeuw & Croux (1993) finite-sample correction factor c_n.

    Verified against R: ``deparse(robustbase::Sn)`` shows
    ``if (n <= 9) <table> else if (n %% 2) n / (n - 0.9) else 1`` — R's
    ``n %% 2`` is truthy for *odd* n, so the correction applies to odd n
    and even n gets a bare 1. Swapping the parity here reproduces every
    reference Sn value to only ~2 significant digits (e.g. 1.4197 instead
    of 1.4069 for the heavy_tail fixture, n=100) rather than the ~1e-15
    floating-point noise expected from a correct port.
    """
    if 2 <= n <= 9:
        return SN_C_SMALL[n - 2]
    if n % 2 == 1:
        return n / (n - 0.9)
    return 1.0


def sn(x: np.ndarray) -> float:
    """Rousseeuw-Croux Sn robust scale estimator.

    Ports ``robustbase::Sn``. Unlike the MAD this needs no location estimate
    and has 58% Gaussian efficiency.

    ``Sn = 1.1926 * c_n * lomed_i( himed_j |x_i - x_j| )`` where ``himed`` is
    the ``floor(n/2) + 1``-th order statistic and ``lomed`` the
    ``ceil(n/2)``-th, both 1-based.

    Parameters
    ----------
    x
        One-dimensional sample.

    Returns
    -------
    The scale estimate. Returns 0.0 for samples of fewer than two values.

    Notes
    -----
    The naive O(n^2) form is used. valiDrops calls this on at most a few tens
    of thousands of values, where it costs well under a second.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    if n < 2:
        return 0.0

    diffs = np.abs(x[:, None] - x[None, :])
    hi_idx = n // 2  # 0-based index of the (floor(n/2)+1)-th order statistic
    himed = np.partition(diffs, hi_idx, axis=1)[:, hi_idx]
    lo_idx = (n + 1) // 2 - 1  # 0-based index of the ceil(n/2)-th
    lomed = np.partition(himed, lo_idx)[lo_idx]

    return float(SN_CONSTANT * _finite_sample_correction(n) * lomed)


def rollmean(x: np.ndarray, k: int) -> np.ndarray:
    """Centred rolling mean, matching ``zoo::rollmean(x, k, align="center")``.

    For a plain numeric vector ``zoo`` returns the ``len(x) - k + 1``
    consecutive window means; ``align`` affects only the index of a zoo
    object, not the values.

    Parameters
    ----------
    x
        One-dimensional input.
    k
        Window width.

    Returns
    -------
    Array of length ``len(x) - k + 1``.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if k < 1 or k > x.size:
        raise ValueError(f"window k={k} must be between 1 and len(x)={x.size}")
    cumsum = np.concatenate(([0.0], np.cumsum(x)))
    return (cumsum[k:] - cumsum[:-k]) / k
