"""Muggeo's iterative segmented (piecewise linear) regression.

Ports the estimation procedure of the R package ``segmented``. ``pwlf`` is not
a substitute: it locates breakpoints by global optimisation, which converges
to different estimates on the same data.

This is a faithful port of ``segmented:::seg.lm.fit``, not the textbook
Muggeo (2003) description of a plain, undamped Newton step. Reading the R
source (there is no vignette for this part) shows three things a naive
reimplementation misses, all needed to reproduce R's numbers to float64
precision:

1. The raw update ``psi + gamma/beta`` is only a *direction*. R scales it by
   ``h = 1.25`` (``seg.control(h=)``) and then does a bounded 1-D line search
   (R's ``optimize``, Brent's method) along that direction for the step
   fraction that minimises the RSS of the model *without* the gamma terms —
   the actual objective, not the linearised one.
2. Convergence is judged on relative RSS improvement,
   ``epsilon = (L0 - L1) / (|L0| + 0.1)``, against ``tol`` — not on the size
   of the psi update.
3. Candidate breakpoints are clipped into ``[quantile(x, alpha), quantile(x,
   1-alpha)]`` at every iteration (R's ``adj.psi``), not just checked at the
   end.

Matching this algorithm (rather than a plain undamped Newton step) brings
breakpoints and RMSE within ~1e-9 relative of R across the reference fixtures
(verified against a live R session reading ``segmented:::seg.lm.fit``'s
source directly) — tighter than R's own default convergence tolerance
(``seg.control(tol=1e-5)``). A residual ~1e-9 relative difference in psi
remains even so, consistent with ordinary cross-implementation floating-point
noise (different BLAS/LAPACK routines, R's QR-based ``.lm.fit`` vs NumPy's
SVD-based ``lstsq``) rather than an algorithmic gap; it is invisible on
well-conditioned segment slopes but can exceed a 1e-6 *relative* tolerance on
a near-zero slope (see the ``plateau`` fixture case), where a tiny psi
perturbation is hugely leveraged. See task-7-report.md for the full
diagnosis.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

# seg.control() defaults that R does not expose as segmented() arguments either.
_STEP_SCALE = 1.25  # seg.control(h=)
_MIN_SEGMENT_N = 2  # seg.control(min.n=)


class SegmentedFitError(RuntimeError):
    """Raised when no segmented model converges for the given data."""


@dataclass(frozen=True)
class SegmentedFit:
    """Result of a segmented regression."""

    psi: np.ndarray
    """Estimated breakpoints, ascending."""
    slopes: np.ndarray
    """Slope of each segment, length ``len(psi) + 1``."""
    intercept: float
    residuals: np.ndarray
    rmse: float
    converged: bool


def _default_psi_init(x: np.ndarray, npsi: int) -> np.ndarray:
    """R's default starting values.

    ``segmented.lm``'s default ``seg.control(quant = FALSE)`` starts from
    ``npsi`` points equally spaced across the *range* of ``x`` — not
    quantiles of its distribution:

        psi_k = min(x) + range(x) * k / (npsi + 1),  k = 1..npsi

    (segmented.lm source, the ``psiE`` branch of the ``if (control$quant)``
    switch). This only coincides with a quantile-based start when ``x`` is
    itself evenly spaced; for skewed data — e.g. Stage 1's barcode-rank
    curve — the two disagree substantially, so using ``np.quantile`` here
    would silently pick different starting breakpoints than R.
    """
    lo, hi = float(np.min(x)), float(np.max(x))
    return lo + (hi - lo) * np.arange(1, npsi + 1, dtype=np.float64) / (npsi + 1)


def _segment_counts(x: np.ndarray, psi: np.ndarray) -> np.ndarray:
    """Number of points strictly within each of the ``len(psi) + 1`` segments."""
    edges = np.concatenate(([-np.inf], np.sort(psi), [np.inf]))
    return np.histogram(x, bins=edges)[0]


def _rss_no_gamma(x: np.ndarray, y: np.ndarray, psi: np.ndarray) -> float:
    """RSS of the broken-line fit (no gamma/indicator terms) at ``psi``.

    This is what R's inner line search (``search.min``) actually minimises:
    the real objective, not the model linearised around the current psi.
    """
    k = psi.size
    design = np.empty((x.size, 2 + k), dtype=np.float64)
    design[:, 0] = 1.0
    design[:, 1] = x
    for j, p in enumerate(psi):
        design[:, 2 + j] = np.maximum(x - p, 0.0)
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid = y - design @ coef
    return float(resid @ resid)


def _design_with_gamma(x: np.ndarray, psi: np.ndarray) -> np.ndarray:
    """[1, x, (x-psi_k)_+ ..., I(x>psi_k) ...] — the direction-finding design."""
    n = x.size
    k = psi.size
    out = np.empty((n, 2 + 2 * k), dtype=np.float64)
    out[:, 0] = 1.0
    out[:, 1] = x
    for j, p in enumerate(psi):
        out[:, 2 + j] = np.maximum(x - p, 0.0)
        out[:, 2 + k + j] = -(x > p).astype(np.float64)
    return out


def _fit_once(
    x: np.ndarray, y: np.ndarray, psi_init: np.ndarray, lo: float, hi: float, max_iter: int, tol: float
) -> tuple[np.ndarray, bool]:
    """Run R's damped-Newton-with-line-search iteration from ``psi_init``.

    Returns ``(psi, converged)``. ``converged`` is only ``True`` when the
    RSS-relative-change criterion was satisfied on an admissible fit; a
    degenerate direction (a beta or gamma coefficient collapsing to ~0, per
    R's ``isZero`` check under ``fix.npsi = TRUE``) or a psi whose segments
    lose the minimum point count is reported as non-convergence rather than
    silently returned, so callers can retry with a different start/alpha.
    """
    k = psi_init.size
    psi = np.clip(np.sort(np.asarray(psi_init, dtype=np.float64)), lo, hi)

    if np.any(_segment_counts(x, psi) < _MIN_SEGMENT_N):
        return psi, False

    l0 = _rss_no_gamma(x, y, psi)
    epsilon = 10.0
    it = 0
    while abs(epsilon) > tol:
        it += 1
        design = _design_with_gamma(x, psi)
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        beta = coef[2 : 2 + k]
        gamma = coef[2 + k :]

        if np.any(np.abs(beta) < 1e-12) or np.any(np.abs(gamma) < 1e-12):
            return psi, False

        psi_old = psi
        # R: psi <- psi.old + h * gamma.c / beta.c; psi <- adj.psi(psi, limZ)
        psi_dir = np.clip(np.sort(psi_old + _STEP_SCALE * gamma / beta), lo, hi)

        def rss_along(step: float, _psi_dir: np.ndarray = psi_dir, _psi_old: np.ndarray = psi_old) -> float:
            candidate = _psi_dir * step + _psi_old * (1.0 - step)
            return _rss_no_gamma(x, y, candidate)

        search = minimize_scalar(rss_along, bounds=(0.0, 1.0), method="bounded")
        use_k = float(search.x)
        l1 = float(search.fun)

        psi = np.clip(psi_dir * use_k + psi_old * (1.0 - use_k), lo, hi)
        epsilon = (l0 - l1) / (abs(l0) + 0.1)
        l0 = l1

        if np.any(_segment_counts(x, psi) < _MIN_SEGMENT_N):
            return psi, False
        if it >= max_iter:
            return np.sort(psi), False

    return np.sort(psi), True


def _final_model(x: np.ndarray, y: np.ndarray, psi: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Refit without the gamma terms. Returns (intercept, slopes, residuals).

    Muggeo's gamma terms are only there to find the update direction for psi;
    the reported fit — and the RMSE the caller compares against R's
    ``sqrt(mean(fit$residuals^2))`` — comes from a plain broken-line OLS fit
    at the converged breakpoints, matching R's ``slope()`` and
    ``residuals()`` on the final ``segmented`` object.
    """
    k = psi.size
    design = np.empty((x.size, 2 + k), dtype=np.float64)
    design[:, 0] = 1.0
    design[:, 1] = x
    for j, p in enumerate(psi):
        design[:, 2 + j] = np.maximum(x - p, 0.0)
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    residuals = y - design @ coef
    # Segment slopes accumulate: slope_0 = coef_x, slope_j = coef_x + sum(beta_1..beta_j),
    # matching segmented::slope(fit)$x[, 1].
    slopes = coef[1] + np.concatenate(([0.0], np.cumsum(coef[2:])))
    return float(coef[0]), slopes, residuals


def segmented(
    x: np.ndarray,
    y: np.ndarray,
    *,
    npsi: int | None = None,
    psi_init: np.ndarray | None = None,
    alpha: float = 0.0,
    max_iter: int = 30,
    tol: float = 1e-8,
    n_boot: int = 0,
    random_state: int = 0,
) -> SegmentedFit:
    """Fit a piecewise linear model with free breakpoints.

    Parameters
    ----------
    x, y
        Predictor and response, one-dimensional and the same length.
    npsi
        Number of breakpoints. Ignored when ``psi_init`` is given.
    psi_init
        Explicit starting values for the breakpoints.
    alpha
        Breakpoints are constrained to ``[quantile(x, alpha), quantile(x, 1-alpha)]``,
        matching ``segmented::seg.control(alpha=)``.
    max_iter
        Maximum Muggeo iterations per start.
    tol
        Convergence tolerance on the relative RSS improvement between
        iterations (R's ``seg.control(tol=)``, default ``1e-5``). The
        default here is tighter; empirically this makes no difference to the
        converged psi (verified up to ``tol=1e-14``), since the fit already
        reaches a stable fixed point in a handful of iterations.
    n_boot
        Bootstrap restarts, as in ``seg.control(n.boot=)``. Each restart refits
        on a resampled dataset and uses the result as a new starting value; the
        lowest-RSS solution wins. Set to 0 for a deterministic fit.
    random_state
        Seed for the bootstrap restarts.

    Returns
    -------
    SegmentedFit

    Raises
    ------
    SegmentedFitError
        When no start converges to an admissible set of breakpoints.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size != y.size:
        raise ValueError(f"x and y must be the same length, got {x.size} and {y.size}")

    if psi_init is None:
        if npsi is None or npsi < 1:
            raise ValueError("give either psi_init or npsi >= 1")
        psi_init = _default_psi_init(x, int(npsi))
    psi_init = np.asarray(psi_init, dtype=np.float64).ravel()

    lo = float(np.quantile(x, alpha)) if alpha > 0 else float(x.min())
    hi = float(np.quantile(x, 1.0 - alpha)) if alpha > 0 else float(x.max())

    best: tuple[float, np.ndarray] | None = None
    psi, converged = _fit_once(x, y, psi_init, lo, hi, max_iter, tol)
    if converged:
        best = (_rss_no_gamma(x, y, psi), psi)

    if n_boot > 0:
        rng = np.random.default_rng(random_state)
        start = psi if converged else psi_init
        for _ in range(n_boot):
            idx = rng.integers(0, x.size, size=x.size)
            boot_psi, boot_ok = _fit_once(x[idx], y[idx], start, lo, hi, max_iter, tol)
            if not boot_ok:
                continue
            cand_psi, cand_ok = _fit_once(x, y, boot_psi, lo, hi, max_iter, tol)
            if not cand_ok:
                continue
            rss = _rss_no_gamma(x, y, cand_psi)
            if best is None or rss < best[0]:
                best = (rss, cand_psi)

    if best is None:
        raise SegmentedFitError(
            f"segmented regression did not converge for npsi={psi_init.size} within [{lo:.6g}, {hi:.6g}]"
        )

    psi = np.sort(best[1])
    intercept, slopes, residuals = _final_model(x, y, psi)
    return SegmentedFit(
        psi=psi,
        slopes=slopes,
        intercept=intercept,
        residuals=residuals,
        rmse=float(np.sqrt(np.mean(residuals**2))),
        converged=True,
    )
