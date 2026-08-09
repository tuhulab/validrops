"""Unit-invariant knee detection, ported from the R package ``inflection``."""

import numpy as np


def _chord(x1: float, y1: float, x2: float, y2: float, x: np.ndarray) -> np.ndarray:
    """``inflection::lin2`` — the straight line through two points, evaluated at x.

    ``_signed_areas`` calls this on single-point segments (``j == 1`` or
    ``j == n``), where ``x1 == x2``. The result is always multiplied by an
    empty diff array there, so any finite value is fine; guard it explicitly
    rather than let it raise a spurious 0/0 ``RuntimeWarning``.
    """
    if x2 == x1:
        return np.full_like(x, y1, dtype=np.float64)
    return y1 + (y2 - y1) * (x - x1) / (x2 - x1)


def _signed_areas(x: np.ndarray, y: np.ndarray, j: int) -> tuple[float, float]:
    """``inflection::findipl`` — trapezoidal integral of the chord deviation.

    Splits at 1-based index ``j`` and integrates ``y - chord`` over each side.
    Returns ``(left, right)``.
    """
    n = x.size
    left = slice(0, j)  # R's x[1:j]
    dxl = np.diff(x[left])
    fl = y[left] - _chord(x[0], y[0], x[j - 1], y[j - 1], x[left])
    sl = float(np.sum(dxl * 0.5 * (fl[:-1] + fl[1:])))

    right = slice(j - 1, n)  # R's x[j:n]
    dxr = np.diff(x[right])
    fr = y[right] - _chord(x[j - 1], y[j - 1], x[n - 1], y[n - 1], x[right])
    sr = float(np.sum(dxr * 0.5 * (fr[:-1] + fr[1:])))
    return sl, sr


def _check_curve_index(x: np.ndarray, y: np.ndarray) -> int:
    """``inflection::check_curve`` — returns 1 when the curve is concave on the left.

    R's ``check_curve`` also classifies the right side (``cright``) to build a
    ``ctype`` string (``"convex_concave"``, ``"concave"``, ...), but its
    ``index`` — the only part ``ede``/``uik`` consume — depends solely on the
    left classification: concave-left always yields 1, convex-left always
    yields 0, regardless of ``cright``. The right side is intentionally not
    computed here.
    """
    n = x.size
    # R: as.integer(quantile(1:N, p)) truncates toward zero
    quarts = [int(np.quantile(np.arange(1, n + 1), p)) for p in (0.25, 0.5, 0.75)]
    left_js = [*quarts, n]  # j1, j2, j3, jn
    left_areas = [_signed_areas(x, y, j)[0] for j in left_js]  # sl1, sl2, sl3, sln

    left_signs = np.sign(left_areas)
    unique = np.unique(left_signs)
    ref_sign = unique[0] if unique.size == 1 else left_signs[0]
    return 1 if ref_sign > 0 else 0


def uik(x: np.ndarray, y: np.ndarray) -> float:
    """Unit-invariant knee of a curve, ported from ``inflection::uik``.

    Parameters
    ----------
    x
        Strictly increasing x-coordinates.
    y
        Matching y-coordinates.

    Returns
    -------
    The x-value at the knee. Always one of the input ``x`` values.

    Notes
    -----
    This is the chord-deviation extremum used by ``inflection`` (argmin of the
    deviation from the chord between the first and last points, after
    orienting the curve via ``check_curve``), not the Kneedle algorithm
    implemented by the ``kneed`` package. The two are different algorithms
    and can disagree; do not substitute ``kneed`` here.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size <= 3:
        raise ValueError("uik needs at least 4 points; give a vector of 5 or more")
    if x.size != y.size:
        raise ValueError(f"x and y must be the same length, got {x.size} and {y.size}")

    if _check_curve_index(x, y) == 1:
        y = -y
    deviation = y - _chord(x[0], y[0], x[-1], y[-1], x)
    return float(x[int(np.argmin(deviation))])
