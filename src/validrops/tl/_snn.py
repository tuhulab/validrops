"""Shared-nearest-neighbour graph and Louvain clustering, matching Seurat."""

import random

import igraph as ig
import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

from validrops._constants import SNN_K, SNN_PRUNE


def snn_graph(embedding: np.ndarray, k: int = SNN_K, prune: float = SNN_PRUNE) -> sp.csr_matrix:
    """Jaccard-weighted shared-nearest-neighbour graph.

    Ports ``Seurat::FindNeighbors``: exact kNN including each cell itself,
    then an edge weight equal to the Jaccard index of the two neighbour sets,
    with weights below ``prune`` dropped.

    Parameters
    ----------
    embedding
        Cells x dimensions, e.g. PCA scores.
    k
        Neighbours per cell, counting the cell itself.
    prune
        Minimum Jaccard weight to retain an edge.

    Returns
    -------
    Symmetric sparse adjacency, cells x cells, zero diagonal.
    """
    n = embedding.shape[0]
    k = min(k, n)
    nn = NearestNeighbors(n_neighbors=k, algorithm="brute").fit(embedding)
    _, indices = nn.kneighbors(embedding)

    rows = np.repeat(np.arange(n), k)
    knn = sp.csr_matrix((np.ones(n * k), (rows, indices.ravel())), shape=(n, n))

    shared = (knn @ knn.T).tocoo()  # |N_i ∩ N_j|
    jaccard = shared.data / (2 * k - shared.data)  # |union| = k + k - |intersection|

    keep = (jaccard >= prune) & (shared.row != shared.col)
    graph = sp.csr_matrix((jaccard[keep], (shared.row[keep], shared.col[keep])), shape=(n, n))
    return graph.maximum(graph.T)


def louvain(adjacency: sp.spmatrix, resolution: float, random_state: int = 0) -> np.ndarray:
    """Modularity clustering at a given resolution.

    Ports ``Seurat::FindClusters``'s default algorithm: classic Louvain
    (Blondel et al. 2008), not Leiden. ``igraph.Graph.community_multilevel``
    implements the same Blondel et al. algorithm Seurat's C++
    ``RunModularityClustering`` runs, whereas ``leidenalg``'s partition
    types run the Leiden algorithm — a different (later) local-search
    procedure that, on this codebase's PBMC 4K fixture, was found to
    converge to a visibly different partition of the largest, most nearly
    degenerate communities at low resolution (shallow clustering ARI 0.71
    against Seurat vs. >0.9 with classic Louvain). Use this function, not
    ``leidenalg``, wherever the R port needs Seurat-equivalent clustering.

    Parameters
    ----------
    adjacency
        Symmetric weighted graph from :func:`snn_graph`.
    resolution
        Higher values give more, smaller clusters.
    random_state
        Seed; the same seed always gives the same partition. igraph's
        community detection draws randomness through Python's ``random``
        module by default (see ``igraph.set_random_number_generator``), so
        seeding ``random`` directly is what makes this deterministic.

    Returns
    -------
    Integer label per cell, ordered by descending cluster size so that label 0
    is the largest cluster (Seurat's convention).
    """
    coo = sp.triu(adjacency.tocoo(), k=1).tocoo()
    graph = ig.Graph(n=adjacency.shape[0], edges=list(zip(coo.row.tolist(), coo.col.tolist(), strict=True)))
    graph.es["weight"] = coo.data.tolist()

    random.seed(random_state)
    clustering = graph.community_multilevel(weights="weight", resolution=resolution)

    raw = np.asarray(clustering.membership)
    order = np.argsort(-np.bincount(raw))
    remap = np.empty(order.size, dtype=np.int64)
    remap[order] = np.arange(order.size)
    return remap[raw]
