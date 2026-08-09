import validrops
from validrops import _constants as C


def test_r_defaults_are_frozen():
    assert C.MITO_CAP == 0.3
    assert C.SN_MULTIPLIER_CODING == 3
    assert C.SN_MULTIPLIER_DISTANCE == 5
    assert C.SN_C_SMALL == (0.743, 1.851, 0.954, 1.351, 0.993, 1.198, 1.005, 1.131)
    assert C.RMSE_FACTOR == 1.5
    assert C.BREAKPOINT_RANGE == (2, 5)
    assert C.MITO_SCAN_INCREMENT == 0.001
    assert C.HVG_COUNT == 5000
    assert C.SHALLOW_RESOLUTION == 0.1
    assert C.MIN_CLUSTER_SIZE == 5
    assert C.DEAD_CELL_RUNS == 10
    assert C.DEAD_CELL_CONSENSUS == 8
    assert C.DEAD_SCORE_COEFFICIENTS == {
        "log_umis": -11.82,
        "log_features": 2.08,
        "ribosomal": 158.98,
        "features_x_coding": 18.87,
        "ribosomal_x_coding": -125.9,
    }


def test_submodules_importable():
    assert validrops.tl is not None
    assert validrops.pp is not None
    assert validrops.pl is not None
