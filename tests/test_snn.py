import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.metrics import adjusted_rand_score

from validrops.tl._snn import louvain, prepare_louvain_graph, snn_graph


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


# ---------------------------------------------------------------------------
# Prepared-graph Louvain (Package 1, Performance)
# ---------------------------------------------------------------------------


def _clique_adjacency() -> sp.csr_matrix:
    """Two disconnected unit-weight cliques: {0, 1} (size 2) and {2, 3, 4, 5}
    (size 4), the minimal adversarial case for deterministic label output."""
    n = 6
    rows, cols, data = [], [], []
    for a, b in [(0, 1), (2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (4, 5)]:
        rows += [a, b]
        cols += [b, a]
        data += [1.0, 1.0]
    return sp.csr_matrix((data, (rows, cols)), shape=(n, n))


def test_louvain_golden_cliques_pre_refactor():
    """Pre-refactor golden: two disconnected cliques of sizes 2 and 4 give
    labels [1, 1, 0, 0, 0, 0] -- the descending-cluster-size remap assigns
    label 0 to the larger clique. Locked before the graph-reuse refactor so
    both the wrapper and prepared paths must reproduce it."""
    labels = louvain(_clique_adjacency(), resolution=1.0, random_state=0)
    np.testing.assert_array_equal(labels, np.array([1, 1, 0, 0, 0, 0]))


def test_prepare_louvain_graph_matches_upper_triangle():
    rng = np.random.default_rng(7)
    emb = rng.normal(size=(120, 8))
    adj = snn_graph(emb, k=15, prune=1 / 15)
    graph = prepare_louvain_graph(adj)
    coo = sp.triu(adj.tocoo(), k=1).tocoo()
    assert graph.vcount() == adj.shape[0]
    assert graph.ecount() == coo.nnz
    edges = {(e.source, e.target) for e in graph.es}
    assert edges == set(zip(coo.row.tolist(), coo.col.tolist(), strict=True))
    assert sorted(graph.es["weight"]) == sorted(coo.data.tolist())


def test_wrapper_and_prepared_paths_agree_across_resolutions_and_seeds():
    from validrops.tl._snn import louvain_prepared

    rng = np.random.default_rng(6)
    emb = rng.normal(size=(250, 10))
    adj = snn_graph(emb, k=20, prune=1 / 15)
    graph = prepare_louvain_graph(adj)
    for res in (0.1, 1.0, 8.0):
        for seed in (0, 3, 7):
            expected = louvain(adj, res, random_state=seed)
            got = louvain_prepared(graph, res, random_state=seed)
            np.testing.assert_array_equal(expected, got)


def test_prepared_repeated_calls_deterministic_and_graph_not_mutated():
    from validrops.tl._snn import louvain_prepared

    rng = np.random.default_rng(5)
    emb = rng.normal(size=(200, 10))
    adj = snn_graph(emb, k=20, prune=1 / 15)
    graph = prepare_louvain_graph(adj)
    before = {"vcount": graph.vcount(), "ecount": graph.ecount(), "weights": list(graph.es["weight"])}
    # Representative span of the production deep-clustering resolution sweep
    # (coarse 1..20 plus the +/-0.9, 0.1-step refinement around 1, 5, and 20).
    resolutions = [0.1, 0.2, 0.9, 1.0, 1.1, 4.1, 4.9, 5.0, 5.1, 7.9, 8.0, 8.1, 19.1, 19.9, 20.0, 20.1, 20.9]
    first = {res: louvain_prepared(graph, res, random_state=0) for res in resolutions}
    second = {res: louvain_prepared(graph, res, random_state=0) for res in resolutions}
    after = {"vcount": graph.vcount(), "ecount": graph.ecount(), "weights": list(graph.es["weight"])}
    assert before == after, "the prepared sweep must not mutate the igraph object"
    for res in resolutions:
        np.testing.assert_array_equal(first[res], second[res])
