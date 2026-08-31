"""Stage 3a: expression-based cluster metrics. Ports ``expression_metrics.R:21-202``."""

import logging

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData
from scipy.sparse.linalg import svds
from scipy.stats import rankdata

from validrops._constants import (
    HVG_COUNT,
    MIN_CLUSTER_SIZE,
    N_PCS,
    SHALLOW_RESOLUTION,
    TOP_N_MARKERS,
    UNS_KEY,
)
from validrops.tl._deviance import deviance_feature_selection
from validrops.tl._snn import louvain_prepared, prepare_louvain_graph, snn_graph
from validrops.tl._wilcox import wilcoxauc

logger = logging.getLogger(__name__)

STAT_COLUMNS = [
    "cluster",
    "pct.diff",
    "pct.1",
    "pct.2",
    "n_de",
    "n_total",
    "n_negative",
    "min_fdr",
    "de_fraction",
    "mito_fraction",
    "ribo_fraction",
]


def expression_metrics(
    adata: AnnData,
    *,
    nfeats: int = HVG_COUNT,
    npcs: int = N_PCS,
    k_min: int = MIN_CLUSTER_SIZE,
    res_shallow: float = SHALLOW_RESOLUTION,
    top_n: int = TOP_N_MARKERS,
    clusters: pd.Series | None = None,
    random_state: int = 0,
) -> None:
    """Cluster QC-passing barcodes and compute per-cluster marker statistics.

    Clusters that fail to produce coherent markers are the signal Stage 3b
    filters on: a cluster of debris has no genes that distinguish it.

    Parameters
    ----------
    adata
        Must carry ``obs["qc_pass"]`` and the gene sets written by
        :func:`~validrops.tl.quality_metrics` into ``uns["validrops"]``.
        ``var["protein_coding"]``, when present, restricts the input to
        protein-coding genes exactly as ``valiDrops.R:101`` does.
    nfeats
        Deviance-selected variable features used for the embedding.
    npcs
        Singular vectors retained.
    k_min
        Target size of the smallest deep cluster.
    res_shallow
        Resolution for the coarse clustering that defines each cluster's
        background set.
    top_n
        Marker genes averaged into the summary percentages.
    clusters
        Optional barcode -> cluster mapping. When given, clustering is skipped
        and the supplied assignment is used verbatim, so Stage 3b can be
        validated against R's own clustering independently of clustering
        fidelity.
    random_state
        Seed for the SVD and Louvain.

    Returns
    -------
    None. Writes ``obs["cluster"]``, ``obs["cluster_shallow"]``,
    ``obsm["X_validrops_pca"]`` and ``uns["validrops"]["cluster_stats"]``.
    """
    if nfeats <= 0 or nfeats < npcs:
        raise ValueError(f"nfeats must be > 0 and >= npcs, got nfeats={nfeats}, npcs={npcs}")
    if npcs <= 0 or k_min <= 0 or res_shallow <= 0 or top_n <= 0:
        raise ValueError("npcs, k_min, res_shallow and top_n must all be greater than 0")

    uns = adata.uns.get(UNS_KEY, {})
    gene_sets = uns.get("gene_sets")
    if gene_sets is None:
        raise ValueError("uns['validrops']['gene_sets'] missing; run quality_metrics first")

    qc_mask = adata.obs["qc_pass"].to_numpy(dtype=bool)
    coding_mask = (
        adata.var["protein_coding"].to_numpy(dtype=bool)
        if "protein_coding" in adata.var
        else np.ones(adata.n_vars, dtype=bool)
    )
    sub = adata[qc_mask, coding_mask]
    barcodes = sub.obs_names

    # Counts stay float32 (read_10x_h5's dtype): every reduction below passes
    # dtype=np.float64 into .sum() so accumulation runs in double precision,
    # mirroring R's double matrix without doubling memory (task-4 ledger rule).
    counts = sp.csr_matrix(sub.X)
    keep_genes = np.asarray(counts.sum(axis=0, dtype=np.float64)).ravel() > 0
    counts = counts[:, keep_genes]
    gene_names = sub.var_names.to_numpy()[keep_genes]

    # expression_metrics.R:58-61 — per-cell size factor, then log1p of the
    # non-zero entries only, exactly like R's log1p(norm_transform@x).
    size_factors = 10000.0 / np.asarray(counts.sum(axis=1, dtype=np.float64)).ravel()
    norm = counts.multiply(size_factors[:, None]).tocsr()
    norm.data = np.log1p(norm.data)

    # R: dev <- scry::devianceFeatureSelection(nonzero); var.feats <- names(which(rank(-dev) <= nfeats))
    # scry's deviance is computed on raw counts; R's rank() uses ties.method="average".
    deviance = deviance_feature_selection(counts)
    variable = np.flatnonzero(rankdata(-deviance, method="average") <= nfeats)

    embedding = _embed(norm[:, variable], npcs, random_state)
    obsm = np.full((adata.n_obs, npcs), np.nan)
    obsm[qc_mask] = embedding
    adata.obsm["X_validrops_pca"] = obsm

    if clusters is None:
        adjacency = snn_graph(embedding)
        # Package 1: convert the SNN adjacency to igraph exactly once and reuse
        # the same object for the shallow call AND every deep-sweep call.
        prepared = prepare_louvain_graph(adjacency)
        shallow = louvain_prepared(prepared, res_shallow, random_state=random_state)
        deep = _deep_clustering(prepared, k_min, random_state)
    else:
        aligned = clusters.reindex(barcodes)
        if aligned.isna().any():
            raise ValueError("clusters must cover every QC-passing barcode")
        deep = aligned.to_numpy().astype(int)
        adjacency = snn_graph(embedding)
        prepared = prepare_louvain_graph(adjacency)
        shallow = louvain_prepared(prepared, res_shallow, random_state=random_state)

    stats = _cluster_stats(norm, gene_names, deep, shallow, gene_sets, counts, top_n)

    _write_categorical(adata, "cluster", barcodes, deep)
    _write_categorical(adata, "cluster_shallow", barcodes, shallow)
    uns = adata.uns.setdefault(UNS_KEY, {})
    uns["cluster_stats"] = stats
    logger.info("Step 4: %d clusters with marker statistics", len(stats))


def _write_categorical(adata: AnnData, column: str, barcodes, values: np.ndarray) -> None:
    series = pd.Series(pd.NA, index=adata.obs_names, dtype="object")
    series.loc[barcodes] = values
    adata.obs[column] = series.astype("category")


def _embed(norm: sp.csr_matrix, npcs: int, random_state: int) -> np.ndarray:
    """Scale per gene, then SVD. Ports ``expression_metrics.R:74-87``."""
    dense = np.asarray(norm.todense(), dtype=np.float64)
    means = dense.mean(axis=0)
    n_rows = dense.shape[0]
    # sample sd (n/(n-1) correction) like R's colMeans-based formula
    sds = np.sqrt((np.mean(dense * dense, axis=0) - means**2) * (n_rows / (n_rows - 1)))
    sds[sds == 0] = 1.0
    scaled = (dense - means) / sds

    # irlba::irlba(data.scaled, nv=npcs, nu=npcs) -> sv <- u %*% diag(d). svds
    # returns ascending singular values, so flip to descending to match irlba's.
    u, s, _ = svds(scaled, k=npcs)
    order = np.argsort(-s)
    return u[:, order] * s[order]


def _deep_clustering(prepared_graph, k_min: int, random_state: int) -> np.ndarray:
    """Resolution search targeting a smallest cluster of ``k_min``.

    Ports ``expression_metrics.R:97-116``: a coarse 1..20 sweep, a +/-0.9
    refinement at 0.1 steps, then the largest resolution whose smallest
    cluster is exactly ``k_min``, falling back to the nearest achievable size.
    R's seq() values are kept unrounded because FindClusters consumes them as
    doubles and the resolution selection is order-sensitive. All Louvain calls
    reuse the prepared igraph passed in (Package 1).
    """

    def smallest(resolution: float) -> tuple[int, np.ndarray]:
        labels = louvain_prepared(prepared_graph, resolution, random_state=random_state)
        return int(np.bincount(labels).min()), labels

    # coarse sweep, res = 1..20 (R:104-106)
    coarse_mins = {float(r): smallest(float(r))[0] for r in range(1, 21)}
    closest = min(abs(v - k_min) for v in coarse_mins.values())
    close_res = [r for r, v in coarse_mins.items() if abs(v - k_min) == closest]

    # fine sweep (R:107-110): seq(res - 0.9, res + 0.9, by = 0.1) per coarse hit,
    # deduplicated preserving first-occurrence order
    fine = []
    for r in close_res:
        for delta in np.arange(-0.9, 0.9 + 1e-9, 0.1):
            value = r + delta
            if value not in fine:
                fine.append(value)

    # R:111-113 — resol. = largest fine resolution whose smallest cluster == k.min;
    # if none, re-target the nearest achievable size (which.min picks the first)
    fine_mins_keys = [smallest(res) for res in fine]
    fine_mins = {r: m for r, (m, _) in zip(fine, fine_mins_keys, strict=True)}
    exact = [r for r, v in fine_mins.items() if v == k_min]
    if not exact:
        target = min(fine_mins.values(), key=lambda v: abs(v - k_min))
        exact = [r for r, v in fine_mins.items() if v == target]
    _, labels = fine_mins_keys[fine.index(max(exact))]
    return labels


def _sparse_expm1_col_mean(matrix: sp.spmatrix) -> np.ndarray:
    """Column means of ``expm1`` over a sparse matrix, accumulated in float64.

    The log-normalized matrix stores ``log1p`` values sparsely; the marker
    fold-change needs ``mean(expm1(x))`` per gene. Implicit zeros contribute
    ``expm1(0) - 1 = 0`` for free, so only the stored entries are transformed.
    Duplicate coordinates are coalesced with ``sum_duplicates`` *before*
    ``expm1``, matching what a dense assembly (``toarray``) would do:
    ``expm1(a + b) != expm1(a) + expm1(b)``, so the order matters. The input is
    never mutated: ``tocsr(copy=True)`` always copies the arrays.

    Parameters
    ----------
    matrix
        Sparse cells x genes matrix. May be non-canonical CSR/CSC with
        duplicate coordinates and any storage dtype.

    Returns
    -------
    float64 array of per-gene ``mean(expm1)``, length ``n_genes``.

    Raises
    ------
    ValueError
        If the matrix has zero rows (no valid mean). ``_cluster_stats`` never
        passes this because it skips empty target/rest groups.
    """
    if matrix.shape[0] == 0:
        raise ValueError("cannot take a column mean of a zero-row sparse matrix")
    transformed = matrix.tocsr(copy=True).astype(np.float64, copy=False)
    transformed.sum_duplicates()
    transformed.data = np.expm1(transformed.data)
    return np.asarray(transformed.sum(axis=0, dtype=np.float64)).ravel() / transformed.shape[0]


def _cluster_stats(norm, gene_names, deep, shallow, gene_sets, counts, top_n) -> pd.DataFrame:
    """Per-cluster marker statistics. Ports ``expression_metrics.R:118-197``."""
    mito_idx = np.isin(gene_names, list(gene_sets["mitochondrial"]))
    ribo_idx = np.isin(gene_names, list(gene_sets["ribosomal"]))
    totals = np.asarray(counts.sum(axis=1, dtype=np.float64)).ravel()
    n_genes_total = norm.shape[1]

    rows = []
    for cluster in np.unique(deep):
        target = deep == cluster
        # background: everything outside the target's dominant shallow cluster
        dominant = np.bincount(shallow[target]).argmax()
        rest = shallow != dominant
        if target.sum() == 0 or rest.sum() == 0:
            continue

        # expression_metrics.R:138-139 — percentages rounded to 3 dp
        pct1 = np.round(np.asarray((norm[target] > 0).sum(axis=0)).ravel() / target.sum(), 3)
        pct2 = np.round(np.asarray((norm[rest] > 0).sum(axis=0)).ravel() / rest.sum(), 3)
        with np.errstate(divide="ignore", invalid="ignore"):
            pct_diff = (pct1 - pct2) / pct1

        # expression_metrics.R:142-143 — log2 fold change over expm1 means.
        # Package 1: sparse float64 reduction (no todense); mathematically
        # identical to the dense oracle, with float64 accumulation like R.
        mean_target = _sparse_expm1_col_mean(norm[target])
        mean_rest = _sparse_expm1_col_mean(norm[rest])
        fold_change = np.log2(mean_target + 1) - np.log2(mean_rest + 1)

        # expression_metrics.R:146-150 — max(pct.1, pct.2) >= 0.1 & fc >= 0.25
        eligible = (np.maximum(pct1, pct2) >= 0.1) & (fold_change >= 0.25)
        features = np.flatnonzero(eligible)
        if features.size < 2:
            continue

        # expression_metrics.R:152-159 — target/rest/excluded, presto::wilcoxauc
        labels = np.full(norm.shape[0], "excluded", dtype=object)
        labels[target] = "target"
        labels[rest] = "rest"
        result = wilcoxauc(norm[:, features].T, labels, ("target", "rest"))
        fdr = np.minimum(result["pval"].to_numpy() * n_genes_total, 1.0)  # bonferroni
        order = np.argsort(result["pval"].to_numpy(), kind="stable")
        n_de = int(np.sum(fdr <= 0.05))

        # expression_metrics.R:172 — R's 1:min(n_de, top_n) collapses to a single
        # gene when n_de == 0, because 1:0 is c(1, 0) and index 0 is dropped.
        n_top = min(n_de, top_n)
        top = order[:n_top] if n_top > 0 else order[:1]
        top_genes = features[top]

        rows.append(
            {
                "cluster": int(cluster),
                "pct.diff": float(np.mean(pct_diff[top_genes])),
                "pct.1": float(np.mean(pct1[top_genes])),
                "pct.2": float(np.mean(pct2[top_genes])),
                "n_de": n_de,
                "n_total": int(features.size),
                "n_negative": int(np.sum(pct_diff[top_genes] < -0.01)),
                "min_fdr": float(fdr.min()),
                "de_fraction": float(n_de / features.size),
                # expression_metrics.R:182-191 — mito/ribo fractions come from the
                # RAW counts, not the normalized matrix, medians over target cells.
                "mito_fraction": float(
                    np.median(
                        np.asarray(counts[target][:, mito_idx].sum(axis=1, dtype=np.float64)).ravel() / totals[target]
                    )
                )
                if mito_idx.any()
                else 0.0,
                "ribo_fraction": float(
                    np.median(
                        np.asarray(counts[target][:, ribo_idx].sum(axis=1, dtype=np.float64)).ravel() / totals[target]
                    )
                )
                if ribo_idx.any()
                else 0.0,
            }
        )

    return pd.DataFrame(rows, columns=STAT_COLUMNS)
