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


def test_segmented_raises_when_no_breakpoint_exists():
    x = np.linspace(0, 10, 200)
    y = 2.0 * x  # perfectly linear, no curvature to place a break against
    with pytest.raises(SegmentedFitError):
        segmented(x, y, npsi=3, alpha=0.4)


def test_alpha_bounds_breakpoints():
    rng = np.random.default_rng(1)
    x = np.linspace(0, 10, 400)
    y = np.where(x < 1.0, 0.0, 5.0 * (x - 1.0)) + rng.normal(scale=0.05, size=400)
    fit = segmented(x, y, npsi=1, alpha=0.3)
    lo, hi = np.quantile(x, 0.3), np.quantile(x, 0.7)
    assert lo <= fit.psi[0] <= hi


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
