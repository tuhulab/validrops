import numpy as np
import pytest

from validrops.tl._segmented import SegmentedFitError, segmented


def _case(inputs, ref_df, name):
    g = inputs[inputs["case"] == name]
    terms = ref_df[ref_df["case"] == name].set_index("term")["value"]
    return g["x"].to_numpy(), g["y"].to_numpy(), int(g["npsi"].iloc[0]), terms


def test_segmented_breakpoints_match_r(ref):
    inputs = ref("segmented_inputs.csv")
    expected = ref("segmented_reference.csv")
    for name in inputs["case"].unique():
        x, y, npsi, terms = _case(inputs, expected, name)
        psi_init = np.array([terms[f"psi_init{i + 1}"] for i in range(npsi)])
        fit = segmented(x, y, psi_init=psi_init, n_boot=0)
        want = np.array([terms[f"psi{i + 1}"] for i in range(npsi)])
        np.testing.assert_allclose(np.sort(fit.psi), np.sort(want), rtol=1e-6, err_msg=name)


def test_segmented_slopes_match_r(ref):
    inputs = ref("segmented_inputs.csv")
    expected = ref("segmented_reference.csv")
    for name in inputs["case"].unique():
        x, y, npsi, terms = _case(inputs, expected, name)
        psi_init = np.array([terms[f"psi_init{i + 1}"] for i in range(npsi)])
        fit = segmented(x, y, psi_init=psi_init, n_boot=0)
        want = np.array([terms[f"slope{i + 1}"] for i in range(npsi + 1)])
        # atol=1e-8: relative tolerance alone is the wrong metric for a
        # near-zero slope (plateau's slope3 is ~0.0045, a nearly-flat
        # segment). psi itself agrees with R to 1.3e-9 relative and the
        # slope arithmetic is exact given matching psi (verified by feeding
        # R's own converged psi into _final_model, which reproduces R's
        # slopes to ten significant figures) — so the residual disagreement
        # here is float64 solver noise in psi (~1.3e-9 relative, R's QR vs
        # NumPy's SVD), not a formula error. atol=1e-8 is far below any
        # slope magnitude that carries real meaning, so it can't mask a
        # genuine disagreement in the other eleven slope values.
        np.testing.assert_allclose(fit.slopes, want, rtol=1e-6, atol=1e-8, err_msg=name)


def test_segmented_rmse_matches_r(ref):
    inputs = ref("segmented_inputs.csv")
    expected = ref("segmented_reference.csv")
    for name in inputs["case"].unique():
        x, y, npsi, terms = _case(inputs, expected, name)
        psi_init = np.array([terms[f"psi_init{i + 1}"] for i in range(npsi)])
        fit = segmented(x, y, psi_init=psi_init, n_boot=0)
        np.testing.assert_allclose(fit.rmse, terms["rmse"], rtol=1e-6, err_msg=name)


def test_segmented_recovers_a_known_breakpoint():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 500)
    y = np.where(x < 6.0, 1.0 * x, 6.0 + 4.0 * (x - 6.0)) + rng.normal(scale=0.05, size=500)
    fit = segmented(x, y, npsi=1)
    assert abs(fit.psi[0] - 6.0) < 0.1
    np.testing.assert_allclose(fit.slopes, [1.0, 4.0], atol=0.05)


def test_segmented_returns_degenerate_fit_when_no_curvature_exists():
    """Perfectly linear data with no real breakpoint does *not* raise in R:
    verified live (Rscript, segmented 2.2.1) that
    ``segmented(lm(y~x), npsi=3, control=seg.control(n.boot=0, alpha=0.4))``
    returns a fit (psi=[4.0, 4.80449, 6.0], it=1, id.warn=FALSE) rather than
    erroring — R's isZero admissibility check is exact bitwise equality
    (``identical(.x, 0)``), and beta/gamma here are ~1e-16 (floating-point
    noise from an exactly-collinear fit), never exactly 0.0.

    The *locations* of interior breakpoints are not reproducible bit-for-bit
    against R even in principle: with beta/gamma dominated by rounding
    noise, the gamma/beta search direction is essentially arbitrary, so R's
    and NumPy's different solvers push it different ways (confirmed: our
    own psi is [4.56, 4.97, 6.0], not R's [4.0, 4.80, 6.0] — both valid,
    neither reproducible from the other). What R's source *does* guarantee,
    and what's asserted here: no exception, psi confined to
    [quantile(x,alpha), quantile(x,1-alpha)], and the true (constant) slope
    recovered in every segment regardless of where the breakpoints land.
    """
    x = np.linspace(0, 10, 200)
    y = 2.0 * x  # perfectly linear, no curvature to place a break against
    lo, hi = np.quantile(x, 0.4), np.quantile(x, 0.6)
    fit = segmented(x, y, npsi=3, alpha=0.4)
    assert np.all(fit.psi >= lo) and np.all(fit.psi <= hi)
    np.testing.assert_allclose(fit.slopes, 2.0, atol=1e-4)


def test_segmented_raises_for_a_genuinely_unrecoverable_fit():
    """An all-zero response is genuinely unrecoverable, in R too: verified
    live that ``segmented(lm(y~x), npsi=2, control=seg.control(n.boot=0))``
    with ``y <- rep(0, 200)`` raises R's own error, verbatim: 'breakpoint
    estimate too close or at the boundary causing NA estimates.. too many
    breakpoints being estimated?'. Unlike the merely-degenerate case above,
    a zero response drives beta/gamma to *exactly* 0.0 (matrix-vector
    products against an exact-zero vector stay exactly zero in IEEE
    arithmetic, no rounding involved), which is precisely what R's isZero
    (identical(.x, 0)) is designed to catch, and R's default
    fix.npsi=TRUE gives no rescue for it.
    """
    x = np.linspace(0, 10, 200)
    y = np.zeros_like(x)
    with pytest.raises(SegmentedFitError):
        segmented(x, y, npsi=2)


def test_converged_is_false_when_max_iter_is_exhausted():
    """SegmentedFit.converged must carry real information: False means the
    fit was only returned because max_iter ran out before the
    RSS-relative-change criterion was met (R's id.warn), matching R
    returning the fit anyway rather than erroring. A max_iter=1 with a very
    tight tol forces this on a case that legitimately needs more than one
    iteration."""
    rng = np.random.default_rng(4)
    x = np.linspace(0, 10, 500)
    y = np.where(x < 6.0, 1.0 * x, 6.0 + 4.0 * (x - 6.0)) + rng.normal(scale=0.05, size=500)
    with pytest.warns(UserWarning, match="max number of iterations"):
        fit = segmented(x, y, npsi=1, psi_init=np.array([3.0]), max_iter=1, tol=1e-14)
    assert fit.converged is False


def test_alpha_bounds_breakpoints():
    rng = np.random.default_rng(1)
    x = np.linspace(0, 10, 400)
    y = np.where(x < 1.0, 0.0, 5.0 * (x - 1.0)) + rng.normal(scale=0.05, size=400)
    fit = segmented(x, y, npsi=1, alpha=0.3)
    lo, hi = np.quantile(x, 0.3), np.quantile(x, 0.7)
    assert lo <= fit.psi[0] <= hi


def test_alpha_none_matches_r_default_trim():
    """segmented.lm resolves a missing alpha internally:
    ``if (is.null(alpha)) alpha <- max(0.05, 1/length(y))``. Stage 2b and
    Stage 3b both call segmented() with no explicit alpha and rely on that
    default, so it has to live here rather than at each call site (Task 7
    Finding 2). Construct data whose true breakpoint sits inside R's default
    trim band but outside what the trim would allow to move further left,
    so the default-alpha fit is forced to the trim boundary while an
    explicit alpha=0.0 (no trim) is free to find the true, closer-to-the-edge
    breakpoint — the two must disagree for this test to mean anything."""
    rng = np.random.default_rng(3)
    n = 400
    x = np.linspace(0, 10, n)
    y = np.where(x < 0.2, 0.0, 5.0 * (x - 0.2)) + rng.normal(scale=0.02, size=n)

    alpha_r = max(0.05, 1.0 / n)
    lo, hi = np.quantile(x, alpha_r), np.quantile(x, 1.0 - alpha_r)

    fit_default = segmented(x, y, npsi=1)  # alpha=None
    assert lo <= fit_default.psi[0] <= hi
    assert fit_default.psi[0] > 0.3  # pinned near the trim boundary, well past the true 0.2

    fit_no_trim = segmented(x, y, npsi=1, alpha=0.0)  # explicit override still works
    assert fit_no_trim.psi[0] < 0.3
    assert abs(fit_no_trim.psi[0] - 0.2) < 0.1


def test_n_boot_matches_r_bootstrap_result(ref):
    """Task 7 Finding 3: the n_boot path here is a deliberate simplification
    of R's seg.lm.fit.boot (no evolving start, stagnation kick, or
    random-restart fallback — just resample, refit from the current best
    start, refit on the full data, keep the lowest RSS). Validated against
    tests/reference_outputs/segmented_boot_reference.csv, generated with
    R's own ``seg.control(n.boot = 10)`` on the same four cases
    (tests/R/generate_reference.R's segmented-bootstrap block).

    Restricted to the three well-posed cases (one_break, two_breaks,
    plateau — each has one unambiguous global optimum): this simplification
    agrees with R's bootstrap result to ~1e-9 relative, and — checked over
    20 random_state values per case — that agreement does not depend on the
    seed at all, so an exact rtol=1e-6 match at the default random_state is
    a meaningful assertion here.

    smooth_curve is deliberately excluded from this loop and covered
    separately by test_n_boot_smooth_curve_is_bimodal_across_seeds below —
    it is genuinely bimodal under this simplified restart (measured: only
    18/30 random seeds reach R's optimum), so pinning it to one seed here
    would assert a result that's true by luck, not by property, and
    couldn't catch the very flakiness it's meant to guard against.
    """
    inputs = ref("segmented_inputs.csv")
    boot_ref = ref("segmented_boot_reference.csv")
    for name in ("one_break", "two_breaks", "plateau"):
        g = inputs[inputs["case"] == name]
        x = g["x"].to_numpy()
        y = g["y"].to_numpy()
        npsi = int(g["npsi"].iloc[0])
        terms = boot_ref[boot_ref["case"] == name].set_index("term")["value"]
        want_psi = np.sort(np.array([terms[f"psi{i + 1}"] for i in range(npsi)]))
        fit = segmented(x, y, npsi=npsi, n_boot=10, random_state=0)
        np.testing.assert_allclose(np.sort(fit.psi), want_psi, rtol=1e-6, err_msg=name)


def test_n_boot_smooth_curve_is_bimodal_across_seeds(ref):
    """smooth_curve is an artificial 3-breakpoint fit to log(x)*3, a smooth
    curve with no true breakpoints, so the RSS surface is genuinely
    multi-modal — unlike the other three fixture cases (see
    test_n_boot_matches_r_bootstrap_result), a single random_state cannot
    stand in for "the bootstrap path works" here.

    Measured (not chosen) by sweeping random_state 0..29 against
    tests/reference_outputs/segmented_boot_reference.csv's R-bootstrap psi:
    18/30 seeds reach R's exact optimum (<1e-3 relative), and the other
    12/30 land on one specific nearby local optimum, 4.7-6.2% relative away
    — not scattered garbage, the same alternate basin every time. This test
    encodes exactly that: a floor on the hit rate well below the measured
    18/30 (so it doesn't flake on ordinary cross-run RNG variation, e.g. a
    numpy version change), and a check that every miss still lands in the
    documented alternate-optimum band rather than somewhere unexpected —
    which would indicate the search itself broke, not just which basin it
    found.

    R's fuller seg.lm.fit.boot (evolving start, stagnation kick,
    random-restart fallback) apparently escapes this second basin more
    reliably than plain resample-and-refit; this Python port doesn't port
    that machinery (see the n_boot docstring on segmented()), so this
    bimodality is an accepted, documented limitation, not a bug to chase.
    """
    inputs = ref("segmented_inputs.csv")
    boot_ref = ref("segmented_boot_reference.csv")
    g = inputs[inputs["case"] == "smooth_curve"]
    x = g["x"].to_numpy()
    y = g["y"].to_numpy()
    npsi = int(g["npsi"].iloc[0])
    terms = boot_ref[boot_ref["case"] == "smooth_curve"].set_index("term")["value"]
    want_psi = np.sort(np.array([terms[f"psi{i + 1}"] for i in range(npsi)]))

    n_seeds = 30
    n_hits = 0
    for seed in range(n_seeds):
        fit = segmented(x, y, npsi=npsi, n_boot=10, random_state=seed)
        rel = np.max(np.abs((np.sort(fit.psi) - want_psi) / want_psi))
        if rel < 1e-3:
            n_hits += 1
        else:
            assert 0.01 < rel < 0.10, f"seed {seed}: relative error {rel} outside the documented alternate basin"

    assert n_hits >= 12, f"only {n_hits}/{n_seeds} seeds reached R's optimum (measured baseline: 18/30)"


def test_n_boot_restarts_still_recover_the_breakpoint():
    """seg.control(n.boot=) restarts on resampled data; exercise that path directly
    since none of the fixture/known-breakpoint tests above pass n_boot > 0."""
    rng = np.random.default_rng(2)
    x = np.linspace(0, 10, 500)
    y = np.where(x < 6.0, 1.0 * x, 6.0 + 4.0 * (x - 6.0)) + rng.normal(scale=0.05, size=500)
    fit = segmented(x, y, npsi=1, n_boot=5, random_state=0)
    assert abs(fit.psi[0] - 6.0) < 0.1
    np.testing.assert_allclose(fit.slopes, [1.0, 4.0], atol=0.05)


def test_segmented_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        segmented(np.arange(10.0), np.arange(5.0), npsi=1)


def test_segmented_requires_npsi_or_psi_init():
    with pytest.raises(ValueError):
        segmented(np.arange(10.0), np.arange(10.0))
