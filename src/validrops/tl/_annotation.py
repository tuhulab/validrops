"""Species and gene-annotation detection, ported from ``quality_metrics.R:109-183``."""

import re
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files

import numpy as np
import pandas as pd

from validrops._constants import MITO_CHROMOSOMES

_ENSEMBL_PREFIX = re.compile(r"^(ENSG00|ENSMUSG00)")

_SPECIES_ALIASES = {
    "human": "human",
    "sapiens": "human",
    "h.sapiens": "human",
    "mouse": "mouse",
    "musculus": "mouse",
    "m.musculus": "mouse",
    "rat": "rat",
    "norvegicus": "rat",
    "r.norvegicus": "rat",
    "worm": "worm",
    "elegans": "worm",
    "c.elegans": "worm",
    "fly": "fly",
    "drosophila": "fly",
    "d.melanogaster": "fly",
    "zebrafish": "zebrafish",
    "d.rerio": "zebrafish",
}

_ANNOTATION_ALIASES = {
    "symbol": "Symbol",
    "entrez": "NCBI",
    "ncbi": "NCBI",
    "ensembl": "Ensembl",
    "hgnc": "HGNC",
    "mgi": "MGI",
}


@dataclass(frozen=True)
class AnnotationMatch:
    """Which species table and ID column best describe a set of gene names."""

    species: str
    column: str
    n_mapped: int
    n_total: int


@lru_cache(maxsize=1)
def _load_annotation() -> pd.DataFrame:
    path = files("validrops.data").joinpath("annotation.parquet")
    return pd.read_parquet(path)


def clean_gene_ids(names: np.ndarray) -> np.ndarray:
    """Strip GENCODE version suffixes from Ensembl identifiers.

    Ports ``quality_metrics.R:113-121``. Only names beginning ``ENSG00`` or
    ``ENSMUSG00`` are touched; everything else passes through unchanged,
    including symbols that legitimately contain dots.
    """
    names = np.asarray(names, dtype=object)
    out = names.copy()
    for i, name in enumerate(names):
        text = str(name)
        if _ENSEMBL_PREFIX.match(text):
            dot = text.find(".")
            if dot > 0:
                out[i] = text[:dot]
    return out.astype(str)


def _resolve_species(species: str) -> str | None:
    if species == "auto":
        return None
    key = species.lower()
    if key not in _SPECIES_ALIASES:
        raise ValueError(
            'species must be "auto", "human", "mouse", "rat", "C.elegans", '
            f'"drosophila" or "zebrafish", got {species!r}'
        )
    return _SPECIES_ALIASES[key]


def _resolve_annotation(annotation: str) -> str | None:
    if annotation == "auto":
        return None
    key = annotation.lower()
    if key not in _ANNOTATION_ALIASES:
        raise ValueError(
            f'annotation must be "auto", "symbol", "ensembl", "entrez", "HGNC" or "MGI", got {annotation!r}'
        )
    return _ANNOTATION_ALIASES[key]


def detect_annotation(gene_names: np.ndarray, *, species: str = "auto", annotation: str = "auto") -> AnnotationMatch:
    """Find the species table and ID column that maximises gene-name matches.

    Ports ``quality_metrics.R:123-150``. With ``annotation="auto"`` this scans
    **every** column of each candidate table, including ``Chr``, ``Type`` and
    ``Alias``. That is deliberate: it is what the R source does, and the
    winning column determines which ID space the gene sets are looked up in.

    Ties are broken toward the first table and then the first column, matching
    R's ``which.max``.
    """
    table = _load_annotation()
    cleaned = set(clean_gene_ids(gene_names).tolist())
    n_total = len(gene_names)

    want_species = _resolve_species(species)
    want_column = _resolve_annotation(annotation)

    candidates = table
    if want_species is not None:
        candidates = candidates[candidates["species"] == want_species]
    if want_column is not None:
        candidates = candidates[candidates["column_name"] == want_column]
    if candidates.empty:
        raise ValueError(f"no annotation table for species={species!r}, annotation={annotation!r}")

    hits = (
        candidates[candidates["value"].isin(cleaned)]
        .groupby(["species_index", "species", "column_index", "column_name"], observed=True)["value"]
        .nunique()
        .reset_index(name="n_mapped")
        .sort_values(["n_mapped", "species_index", "column_index"], ascending=[False, True, True])
    )
    if hits.empty:
        raise ValueError(
            "no gene names matched any annotation column; check that gene names are "
            "symbols, Ensembl or Entrez identifiers"
        )

    best = hits.iloc[0]
    return AnnotationMatch(
        species=str(best["species"]),
        column=str(best["column_name"]),
        n_mapped=int(best["n_mapped"]),
        n_total=n_total,
    )


def gene_sets(gene_names: np.ndarray, match: AnnotationMatch) -> dict[str, np.ndarray]:
    """Mitochondrial, ribosomal and protein-coding gene sets for the given names.

    Ports ``quality_metrics.R:152-174``. Returns the **original** input names,
    not the cleaned ones, so the result can index the count matrix directly.
    """
    table = _load_annotation()
    species = table[table["species"] == match.species]
    lookup = species[species["column_name"] == match.column]
    symbols = species[species["column_name"] == "Symbol"]

    cleaned = clean_gene_ids(gene_names)
    by_clean = pd.Series(gene_names, index=cleaned)

    coding_ids = set(lookup.loc[lookup["type"] == "protein_coding", "value"])
    mito_ids = set(lookup.loc[lookup["chr"].isin(MITO_CHROMOSOMES), "value"])

    ribo_rows = symbols["value"].str.lower().str.startswith(("rpl", "rps"))
    ribo_row_ids = set(symbols.loc[ribo_rows, "row_id"])
    ribo_ids = set(lookup.loc[lookup["row_id"].isin(ribo_row_ids), "value"])

    def select(ids: set[str]) -> np.ndarray:
        mask = np.isin(cleaned, list(ids))
        return np.asarray(by_clean.to_numpy()[mask], dtype=str)

    return {
        "mitochondrial": select(mito_ids),
        "ribosomal": select(ribo_ids),
        "protein_coding": select(coding_ids),
    }
