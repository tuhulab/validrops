import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from validrops.tl._snn import louvain, snn_graph


def test_snn_is_symmetric_and_pruned():
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(200, 10))
    g = snn_graph(emb, k=20, prune=1 / 15)
    assert g.shape == (200, 200)
    diff = abs(g - g.T)
    assert diff.max() < 1e-12
    nonzero = g.data[g.data > 0]
    assert nonzero.min() >= 1 / 15


def test_snn_weights_are_jaccard_bounded():
    rng = np.random.default_rng(1)
    g = snn_graph(rng.normal(size=(150, 5)), k=10, prune=0.0)
    assert g.data.max() <= 1.0 + 1e-12
    assert g.data.min() >= 0.0


def test_louvain_separates_well_separated_blobs():
    rng = np.random.default_rng(2)
    emb = np.vstack(
        [
            rng.normal(loc=0.0, scale=0.3, size=(100, 5)),
            rng.normal(loc=10.0, scale=0.3, size=(100, 5)),
            rng.normal(loc=-10.0, scale=0.3, size=(100, 5)),
        ]
    )
    labels = louvain(snn_graph(emb, k=15), resolution=1.0, random_state=0)
    truth = np.repeat([0, 1, 2], 100)
    assert adjusted_rand_score(truth, labels) > 0.95


def test_louvain_is_deterministic_for_a_seed():
    rng = np.random.default_rng(3)
    g = snn_graph(rng.normal(size=(300, 8)), k=20)
    a = louvain(g, resolution=1.0, random_state=7)
    b = louvain(g, resolution=1.0, random_state=7)
    np.testing.assert_array_equal(a, b)


def test_higher_resolution_gives_more_clusters():
    rng = np.random.default_rng(4)
    g = snn_graph(rng.normal(size=(400, 10)), k=20)
    low = len(np.unique(louvain(g, resolution=0.1, random_state=0)))
    high = len(np.unique(louvain(g, resolution=8.0, random_state=0)))
    assert high > low


@pytest.mark.slow
def test_shallow_clusters_agree_with_seurat(ref):
    """The headline fidelity check for stage 3."""
    emb_df = ref("stage3_embedding.csv").set_index("barcode")
    clusters = ref("stage3_clusters.csv").set_index("barcode")
    emb = emb_df.loc[clusters.index].to_numpy()
    got = louvain(snn_graph(emb, k=20, prune=1 / 15), resolution=0.1, random_state=0)
    ari = adjusted_rand_score(clusters["shallow"].to_numpy(), got)
    assert ari > 0.9, f"shallow clustering ARI={ari:.3f} against Seurat"
