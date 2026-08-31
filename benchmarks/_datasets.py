"""Deterministic synthetic inputs for the Package-1 benchmarks.

Every generator builds sparse structures directly -- no dense ``cells x genes``
intermediate -- and is fully determined by ``(profile, seed)`` so that the
parent process, spawned benchmark children, and correctness oracles all see
bit-identical inputs. Matrices are float32 CSR exactly like
``scanpy.read_10x_h5`` output; all reductions inside the benchmarks accumulate
in float64 (see the R->Python translation rules in ``AGENTS.md``).
"""

from __future__ import annotations

import numpy as np
from scipy import sparse as sp

#: Benchmark workload profiles (cells, genes, requested density).
#: Only ``smoke`` and ``S`` are exercised automatically by the test suite;
#: the larger tiers are explicit manual choices.
PROFILES = {
    "smoke": {"cells": 60, "genes": 120, "density": 0.05},
    "S": {"cells": 500, "genes": 2_000, "density": 0.02},
    "M-synthetic": {"cells": 4_000, "genes": 20_000, "density": 0.01},
    "L": {"cells": 10_000, "genes": 25_000, "density": 0.005},
    "XL-50": {"cells": 50_000, "genes": 30_000, "density": 0.002},
    "XL-100": {"cells": 100_000, "genes": 30_000, "density": 0.001},
}

_N_DUP_SYMBOLS = 6
_N_MITO = 8
_N_RIBO = 8
_N_CLUSTERS = 4
_CLUSTER_FACTORS = np.array([0.6, 0.9, 1.2, 1.5])


def _base_gene_names(genes: int) -> list[str]:
    return [f"gene_{i:06d}" for i in range(genes)]


def _gene_names_with_duplicates(genes: int) -> list[str]:
    """Build deterministic duplicate-symbol names.

    ``_N_DUP_SYMBOLS`` names appear exactly twice at the tail of the name
    vector. Names are AnnData metadata/labels only -- every matrix operation in
    the pipeline stays positional.
    """
    names = _base_gene_names(genes)
    tail = genes - 2 * _N_DUP_SYMBOLS
    if tail < 0:
        return names
    symbols = [f"dup_sym_{i:02d}" for i in range(_N_DUP_SYMBOLS)]
    names[tail : tail + _N_DUP_SYMBOLS] = symbols
    names[tail + _N_DUP_SYMBOLS : tail + 2 * _N_DUP_SYMBOLS] = symbols
    return names


def make_sparse_counts(profile: str, seed: int = 0) -> tuple[sp.csr_matrix, dict]:
    """Generate a deterministic synthetic count matrix (cells x genes).

    Row-by-row counts are Poisson draws whose rate scales with a per-cluster
    factor, and each simulated cluster additionally elevates its own block of
    marker genes, so the matrix carries real low-dimensional structure the way
    a real filtered matrix does. Coordinates are drawn directly with
    ``numpy.random.Generator`` and canonicalized with ``sum_duplicates``;
    no dense intermediate is ever allocated.

    Returns
    -------
    (matrix, meta)
        ``matrix`` is float32 CSR. ``meta`` records the exact generator
        parameters, realized density, duplicate-name pattern, simulated
        cluster layout and the mito/ribo gene name lists used by the
        Stage-3 benchmark kernels to build ``uns["gene_sets"]``.
    """
    if profile not in PROFILES:
        raise KeyError(f"unknown profile {profile!r}; choose from {sorted(PROFILES)}")
    cells, genes, density = PROFILES[profile]["cells"], PROFILES[profile]["genes"], PROFILES[profile]["density"]

    rng = np.random.default_rng(seed)

    # Simulated cluster layout: contiguous blocks, deterministic sizes.
    sizes = np.full(_N_CLUSTERS, cells // _N_CLUSTERS)
    sizes[: cells % _N_CLUSTERS] += 1
    cluster_ids = np.repeat(np.arange(_N_CLUSTERS), sizes)[:cells]

    lam = genes * density * _CLUSTER_FACTORS[cluster_ids]
    nnz_per_cell = rng.poisson(lam).astype(np.int64)
    nnz_per_cell = np.minimum(nnz_per_cell, genes)
    total = int(nnz_per_cell.sum())
    if total == 0:
        nnz_per_cell[0] = 1
        total = 1

    rows = np.repeat(np.arange(cells), nnz_per_cell)
    cols = rng.integers(0, genes, size=total)
    values = rng.integers(1, 20, size=total).astype(np.float32)
    base = sp.csr_matrix((values, (rows, cols)), shape=(cells, genes), dtype=np.float32)

    # Cluster-specific marker genes: each cluster raises its own disjoint block
    # of ``_MARKER_BLOCK`` genes well above the background, so the Stage-3
    # marker-statistics path has real signal to find. Blocks sit at a fixed
    # offset, away from the mito/ribo name heads.
    marker = _elevated_marker_matrix(rng, cluster_ids, cells, genes)
    matrix = (base + marker).tocsr()
    matrix.sum_duplicates()

    names = _gene_names_with_duplicates(genes)
    meta = {
        "profile": profile,
        "cells": int(cells),
        "genes": int(genes),
        "density": float(density),
        "realized_density": float(matrix.nnz / (cells * genes)),
        "nnz": int(matrix.nnz),
        "dtype": str(matrix.dtype),
        "seed": int(seed),
        "duplicate_names": {
            "n_symbols_duplicated": _N_DUP_SYMBOLS,
            "pattern": f"{_N_DUP_SYMBOLS} symbols duplicated once at the tail of the name "
            "vector; all matrix operations are positional",
        },
        "clusters": {
            "n_clusters": int(_N_CLUSTERS),
            "sizes": sizes.tolist(),
            "mean_count_factors": _CLUSTER_FACTORS.tolist(),
            "marker_block_size": _MARKER_BLOCK,
            "marker_offset": _MARKER_OFFSET,
        },
        "mitochondrial": names[:_N_MITO],
        "ribosomal": names[_N_MITO : _N_MITO + _N_RIBO],
    }
    return matrix, meta


_MARKER_OFFSET = 64
_MARKER_BLOCK = 15
_MARKER_LAM = 14.0


def _elevated_marker_matrix(rng, cluster_ids: np.ndarray, cells: int, genes: int) -> sp.csr_matrix:
    """Sparse marker block per simulated cluster, drawn directly (no dense)."""
    n_clusters = int(cluster_ids.max()) + 1
    if cells == 0 or _MARKER_OFFSET + n_clusters * _MARKER_BLOCK > genes:
        return sp.csr_matrix((cells, genes), dtype=np.float32)
    per_cell = rng.poisson(lam=_MARKER_LAM, size=cells).astype(np.int64)
    total = int(per_cell.sum())
    rows = np.repeat(np.arange(cells), per_cell)
    block = np.arange(n_clusters * _MARKER_BLOCK).reshape(n_clusters, _MARKER_BLOCK)
    # one marker pick per row: entry index -> gene within the cell's cluster block
    entry_in_block = rng.integers(0, _MARKER_BLOCK, size=total)
    cols = (_MARKER_OFFSET + block[cluster_ids[rows], entry_in_block]).astype(np.int64)
    values = rng.integers(3, 12, size=total).astype(np.float32)
    return sp.csr_matrix((values, (rows, cols)), shape=(cells, genes), dtype=np.float32)


def make_clustered_embedding(n: int, dim: int, n_clusters: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray, dict]:
    """Deterministic well-separated Gaussian blobs for graph benchmarks.

    Returns
    -------
    (embedding, labels, meta)
        ``embedding`` is float64 ``(n, dim)``, ``labels`` maps each row to an
        integer cluster in ``range(n_clusters)`` (contiguous blocks), and
        ``meta`` records every generation parameter.
    """
    rng = np.random.default_rng(seed)
    sizes = np.full(n_clusters, n // n_clusters)
    sizes[: n % n_clusters] += 1
    labels = np.repeat(np.arange(n_clusters), sizes)
    centers = rng.normal(size=(n_clusters, dim)) * 5.0
    noise = rng.normal(size=(n, dim)) * 0.3
    embedding = centers[labels] + noise
    meta = {
        "n": int(n),
        "dim": int(dim),
        "n_clusters": int(n_clusters),
        "seed": int(seed),
        "cluster_sizes": sizes.tolist(),
        "noise_scale": 0.3,
        "center_scale": 5.0,
    }
    return embedding, labels, meta
