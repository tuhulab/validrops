import numpy as np

import validrops
from validrops._constants import UNS_KEY
from validrops.pp.label_dead import dead_score, soft_label


def test_score_matches_r(raw_adata, ref):
    expected = ref("stage4_soft_labels.csv").set_index("barcode")
    metrics = ref("stage2_metrics.csv").set_index("barcode").loc[expected.index]
    got = dead_score(
        metrics["logUMIs"].to_numpy(),
        metrics["logFeatures"].to_numpy(),
        metrics["ribosomal_fraction"].to_numpy(),
        metrics["coding_fraction"].to_numpy(),
    )
    np.testing.assert_allclose(got, expected["score"].to_numpy(), rtol=1e-10)


def test_soft_labels_match_r(raw_adata, ref):
    expected = ref("stage4_soft_labels.csv").set_index("barcode")
    filters = ref("stage2_filters.csv").set_index("barcode")
    qc = np.where(filters.loc[expected.index, "final"], "pass", "fail")
    labels, threshold, flag = soft_label(expected["score"].to_numpy(), qc)
    np.testing.assert_array_equal(labels, expected["soft_label"].to_numpy())


def test_score_uses_centred_logs():
    """A constant shift in log_umis must not change the score ordering."""
    kwargs = {
        "log_features": np.array([1.0, 2.0, 3.0]),
        "ribosomal": np.array([0.1, 0.2, 0.3]),
        "coding": np.array([0.9, 0.8, 0.7]),
    }
    a = dead_score(np.array([1.0, 2.0, 3.0]), **kwargs)
    b = dead_score(np.array([101.0, 102.0, 103.0]), **kwargs)
    np.testing.assert_allclose(a, b, rtol=1e-12)


def test_score_normalises_fractions_by_half_pi():
    """A ribosomal fraction of 1.0 contributes exactly its coefficient."""
    score_one = dead_score(
        np.array([0.0, 0.0]),
        np.array([0.0, 0.0]),
        np.array([1.0, 1.0]),
        np.array([0.0, 0.0]),
    )
    np.testing.assert_allclose(score_one, 158.98, rtol=1e-10)


def test_too_few_dead_sets_failed_flag():
    score = np.linspace(0.0, 1.0, 100)
    qc = np.array(["pass"] * 100)
    _, _, flag = soft_label(score, qc, label_thrs=-1.0)  # nothing below the threshold
    assert flag == "Failed"


def test_too_many_dead_aborts_and_labels_all_live():
    score = np.linspace(0.0, 1.0, 100)
    qc = np.array(["pass"] * 100)
    labels, _, flag = soft_label(score, qc, label_thrs=0.9)  # 90% dead, over label_frac
    assert flag == "Failed"
    assert set(labels) == {"live"}


def test_explicit_threshold_is_used_directly():
    score = np.array([-5.0, 0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0])
    qc = np.array(["pass"] * 10)
    labels, threshold, _ = soft_label(score, qc, label_thrs=1.0)
    assert threshold == 1.0
    assert list(labels[:2]) == ["dead", "dead"]


def test_label_dead_without_training_writes_obs(raw_adata, ref):
    adata = raw_adata.copy()
    metrics = ref("stage2_metrics.csv").set_index("barcode")
    for col, src in [
        ("log_umis", "logUMIs"),
        ("log_features", "logFeatures"),
        ("ribosomal_fraction", "ribosomal_fraction"),
        ("coding_fraction", "coding_fraction"),
    ]:
        adata.obs[col] = metrics[src].reindex(adata.obs_names).to_numpy()
    adata.obs["rank_pass"] = adata.obs["log_umis"].notna()
    filters = ref("stage2_filters.csv").set_index("barcode")
    adata.obs["qc_pass"] = filters["final"].reindex(adata.obs_names, fill_value=False).astype(bool).to_numpy()

    validrops.pp.label_dead(adata, train=False)
    assert "dead_score" in adata.obs
    assert set(adata.obs["label"].dropna().unique()) <= {"live", "dead", "uncertain"}
    assert adata.uns[UNS_KEY]["label_flag"] in {"Success", "Caution", "Failed"}
