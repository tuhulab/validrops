import numpy as np
import pytest

from validrops.tl._stats import _finite_sample_correction, rollmean, sn


def test_sn_matches_r(ref):
    """Reference vectors were generated with the same seed in R; regenerate them here."""
    expected = ref("sn_reference.csv").set_index("name")["sn"]
    rng_vectors = _r_seeded_vectors()
    for name, x in rng_vectors.items():
        np.testing.assert_allclose(sn(x), expected[name], rtol=1e-10, err_msg=name)


def _r_seeded_vectors():
    """R's set.seed(42) streams cannot be reproduced in numpy, so the inputs
    themselves are read back from the fixture written by generate_reference.R."""
    from pathlib import Path

    import pandas as pd

    path = Path(__file__).parent / "reference_outputs" / "sn_inputs.csv"
    df = pd.read_csv(path)
    return {name: g["value"].to_numpy() for name, g in df.groupby("name")}


def test_sn_scale_equivariance():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    np.testing.assert_allclose(sn(3.0 * x), 3.0 * sn(x), rtol=1e-12)


def test_sn_translation_invariance():
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    np.testing.assert_allclose(sn(x + 100.0), sn(x), rtol=1e-12)


def test_sn_constant_vector_is_zero():
    assert sn(np.full(50, 7.0)) == 0.0


def test_sn_single_value():
    assert sn(np.array([1.0])) == 0.0


def test_rollmean_matches_r(ref):
    df = ref("rollmean_reference.csv")
    x = df[df["case"] == "input"].sort_values("index")["value"].to_numpy()
    for k in (3, 4, 7, 8):
        expected = df[df["case"] == f"k{k}"].sort_values("index")["value"].to_numpy()
        got = rollmean(x, k)
        assert got.shape == expected.shape, f"k={k}"
        np.testing.assert_allclose(got, expected, rtol=1e-12, err_msg=f"k={k}")


def test_rollmean_window_larger_than_input():
    with pytest.raises(ValueError, match="window"):
        rollmean(np.arange(3.0), 5)


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (2, 0.743),
        (9, 1.131),  # tabulated bounds
        (11, 11 / 10.1),
        (21, 21 / 20.1),  # odd > 9 -> n / (n - 0.9)
        (10, 1.0),
        (20, 1.0),  # even > 9 -> 1.0
    ],
)
def test_finite_sample_correction(n, expected):
    """Direct unit test on the private correction factor.

    No fixture in ``sn_reference.csv`` before this test exercised the odd-n>9
    branch (all five original vectors were even-length), so a silently
    re-inverted parity there would have passed every other test in this file.
    """
    assert _finite_sample_correction(n) == pytest.approx(expected)
