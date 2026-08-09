import numpy as np
import pytest

from validrops.tl._uik import uik


def test_uik_matches_r(ref):
    inputs = ref("uik_inputs.csv")
    expected = ref("uik_reference.csv").set_index("case")["knee"]
    for case, g in inputs.groupby("case"):
        x = g["x"].to_numpy()
        y = g["y"].to_numpy()
        np.testing.assert_allclose(uik(x, y), expected[case], rtol=1e-6, err_msg=case)


def test_uik_rejects_short_input():
    with pytest.raises(ValueError, match="at least"):
        uik(np.arange(3.0), np.arange(3.0))


def test_uik_returns_an_x_value():
    x = np.arange(1.0, 101.0)
    y = 100.0 / x
    assert uik(x, y) in set(x)
