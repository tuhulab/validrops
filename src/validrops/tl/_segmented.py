"""Muggeo's iterative segmented (piecewise linear) regression.

Ports the estimation procedure of the R package ``segmented``. ``pwlf`` is not
a substitute: it locates breakpoints by global optimisation, which converges
to different estimates on the same data.

This is a faithful port of ``segmented:::seg.lm.fit``, not the textbook
Muggeo (2003) description of a plain, undamped Newton step. Reading the R
source (there is no vignette for this part) shows several things a naive
reimplementation misses, all needed to reproduce R's numbers and R's
admissibility behaviour:

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
4. A mid-iteration breakpoint whose neighbouring segment loses too many
   points is *not* a failure in R: it's nudged back by a multiplicative
   rescale (``far.psi``'s ``ifelse(diff(nj) > 0, 1/fc, fc)``) and the loop
   continues. Only two things are genuinely unrecoverable in R: the
   *starting* breakpoints being inadmissible (an unconditional ``stop()``,
   no rescue), and a beta or gamma coefficient collapsing to an *exact*
   zero (R's ``isZero`` is ``identical(.x, 0)``, bitwise equality — not an
   epsilon test) under R's default ``fix.npsi = TRUE``. Exhausting
   ``max_iter`` is not a failure either; R sets ``id.warn`` and returns the
   fit as-is.

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

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

# seg.control() defaults that R does not expose as segmented() arguments either.
_STEP_SCALE = 1.25  # seg.control(h=)
_MIN_SEGMENT_N = 2  # seg.control(min.n=)
_RESCALE_FACTOR = 0.95  # seg.control(fc=)


class SegmentedFitError(RuntimeError):
    """Raised when no admissible segmented fit exists for the given data.

    Reserved for what R itself treats as unrecoverable: the starting
    breakpoints are inadmissible (a segment has fewer than
    ``seg.control(min.n=2)`` points even before iterating), or a beta/gamma
    coefficient collapses to an exact zero mid-iteration. A breakpoint
    drifting into a small segment *during* iteration, or exhausting
    ``max_iter``, are not failures — R rescues the former by rescaling and
    returns the latter as a fit with ``converged=False``.
    """


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
    """``False`` when the fit is only being returned because ``max_iter`` was
    exhausted before the RSS-relative-change criterion was met (R's
    ``id.warn``) — the breakpoints are usable but not fully settled."""


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


def _rescale_inadmissible(psi: np.ndarray, seg_counts: np.ndarray, lo: float, hi: float, fc: float) -> np.ndarray:
    """R's ``far.psi`` rescue for a mid-iteration breakpoint in a thin segment.

    A breakpoint ``psi[j]`` is flagged when the segment to its *left* (the
    ``j``-th of the ``k+1`` segments) has fewer than ``min.n`` points — R's
    ``id.far.ok <- id.ok[-length(id.ok)]``, which keeps all but the last of
    the ``k+1`` per-segment flags. Flagged breakpoints are rescaled — not
    rejected — by ``1/fc`` when the segment to the right is more populated
    than the one to the left (push right, toward where the points are) or
    ``fc`` otherwise (push left): ``ff <- ifelse(diff(nj) > 0, 1/fc, fc)``.
    Also folds in R's (separately computed, but factor-blind) bounds check —
    ``id.psi.ok <- id.psi.in & id.psi.far`` — since after a previous
    unclipped rescale, psi can legitimately sit outside ``[lo, hi]`` here.

    Deliberately **not** re-clipped to ``[lo, hi]`` afterwards: R doesn't
    either (``seg.lm.fit`` only calls ``adj.psi`` right after the raw Newton
    step and right after the line-search blend); the next iteration's clip
    handles it.
    """
    far_ok = seg_counts[:-1] >= _MIN_SEGMENT_N  # length k: left-segment population per breakpoint
    in_ok = (psi >= lo) & (psi <= hi)
    if np.all(far_ok & in_ok):
        return psi
    diff_nj = np.diff(seg_counts)  # length k
    factor = np.where(diff_nj > 0, 1.0 / fc, fc)
    return psi * np.where(far_ok, 1.0, factor)


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

    Returns ``(psi, converged)``, where ``converged`` is only ``False`` when
    the loop exhausted ``max_iter`` before the RSS-relative-change criterion
    was met — R still returns that fit (with a warning), and so do we; the
    caller decides whether to keep it or try another start.

    Raises
    ------
    SegmentedFitError
        Only for the two configurations R itself cannot rescue: the starting
        breakpoints (after clipping to ``[lo, hi]``) leave a segment with
        fewer than ``min.n`` points, or a beta/gamma coefficient is exactly
        zero mid-iteration. Everything else — a breakpoint drifting into a
        thin segment — is nudged back via ``_rescale_inadmissible`` and the
        loop continues, matching R under its default ``fix.npsi = TRUE``.
    """
    k = psi_init.size
    psi = np.clip(np.sort(np.asarray(psi_init, dtype=np.float64)), lo, hi)

    if np.any(_segment_counts(x, psi) < _MIN_SEGMENT_N):
        raise SegmentedFitError(
            "starting psi too close together or at the boundary — no admissible configuration "
            "(R: 'psi starting values too close each other or at the boundaries')"
        )

    l0 = _rss_no_gamma(x, y, psi)
    epsilon = 10.0
    it = 0
    while abs(epsilon) > tol:
        it += 1
        design = _design_with_gamma(x, psi)
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        beta = coef[2 : 2 + k]
        gamma = coef[2 + k :]

        # R's isZero is `identical(.x, 0)` — exact bitwise equality, not an
        # epsilon test — and under fix.npsi=TRUE (R's default) this is a
        # hard stop, not something an epsilon threshold or rescue handles.
        if np.any(beta == 0.0) or np.any(gamma == 0.0):
            raise SegmentedFitError(
                "breakpoint estimate collapsed to an exact zero coefficient "
                "(R: 'breakpoint estimate too close or at the boundary causing NA estimates')"
            )

        psi_old = psi
        # R: psi <- psi.old + h * gamma.c / beta.c; psi <- adj.psi(psi, limZ); sort
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

        # Rescue, not reject: R applies this unconditionally under fix.npsi=TRUE.
        psi = _rescale_inadmissible(psi, _segment_counts(x, psi), lo, hi, _RESCALE_FACTOR)

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
    alpha: float | None = None,
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
        matching ``segmented::seg.control(alpha=)``. ``None`` (the default) resolves to
        R's own internal default, ``max(0.05, 1/len(y))`` — ``segmented.lm``'s
        ``if (is.null(alpha)) alpha <- max(0.05, 1/length(y))``. Callers that relied on
        R's implicit default (i.e. that never passed ``alpha=`` themselves) need this
        primitive to resolve it, not the call site. Pass ``0.0`` explicitly for no trim.
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
        lowest-RSS solution wins. Set to 0 for a deterministic fit. This is a
        simplification of R's ``seg.lm.fit.boot`` (no evolving start, stagnation
        kick, or random-restart fallback); see task-7-report.md for the achieved
        agreement with R's bootstrap output.
    random_state
        Seed for the bootstrap restarts.

    Returns
    -------
    SegmentedFit

    Raises
    ------
    SegmentedFitError
        When no start reaches an admissible set of breakpoints at all — see
        ``_fit_once`` for exactly which configurations that covers. A fit
        that merely exhausts ``max_iter`` is still returned, with
        ``converged=False`` and a warning, matching R.
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

    if alpha is None:
        alpha = max(0.05, 1.0 / y.size)
    lo = float(np.quantile(x, alpha)) if alpha > 0 else float(x.min())
    hi = float(np.quantile(x, 1.0 - alpha)) if alpha > 0 else float(x.max())

    best: tuple[float, np.ndarray, bool] | None = None
    try:
        psi, converged = _fit_once(x, y, psi_init, lo, hi, max_iter, tol)
        best = (_rss_no_gamma(x, y, psi), psi, converged)
    except SegmentedFitError:
        pass

    if n_boot > 0:
        rng = np.random.default_rng(random_state)
        start = best[1] if best is not None else psi_init
        for _ in range(n_boot):
            idx = rng.integers(0, x.size, size=x.size)
            try:
                boot_psi, _ = _fit_once(x[idx], y[idx], start, lo, hi, max_iter, tol)
            except SegmentedFitError:
                continue
            try:
                cand_psi, cand_ok = _fit_once(x, y, boot_psi, lo, hi, max_iter, tol)
            except SegmentedFitError:
                continue
            rss = _rss_no_gamma(x, y, cand_psi)
            if best is None or rss < best[0]:
                best = (rss, cand_psi, cand_ok)

    if best is None:
        raise SegmentedFitError(
            f"segmented regression did not converge for npsi={psi_init.size} within [{lo:.6g}, {hi:.6g}]"
        )

    _, psi, converged = best
    psi = np.sort(psi)
    if not converged:
        warnings.warn(f"segmented: max number of iterations ({max_iter}) attained", stacklevel=2)
    intercept, slopes, residuals = _final_model(x, y, psi)
    return SegmentedFit(
        psi=psi,
        slopes=slopes,
        intercept=intercept,
        residuals=residuals,
        rmse=float(np.sqrt(np.mean(residuals**2))),
        converged=converged,
    )
