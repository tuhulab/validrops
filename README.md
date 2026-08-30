# validrops

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/INFIMM-Bioinformatics/validrops/test.yaml?branch=main
[badge-docs]: https://img.shields.io/readthedocs/validrops

automated QC for scRNA-seq data

`validrops` is a Python/scanpy port of the R package
[valiDrops](https://doi.org/10.1093/nargab/lqad101) (Kavaliauskaite & Madsen,
2023): it ranks barcodes, filters on quality metrics, filters on expression-based
cluster metrics, and optionally labels dead cells — emitting an in-place
annotated `AnnData` rather than a new object.

## Getting started

Please refer to the [documentation][], in particular, the [API documentation][].

### Usage

```python
import scanpy as sc
import validrops

adata = sc.read_10x_h5("raw_feature_bc_matrix.h5")
adata.var_names_make_unique()

validrops.validrops(adata)          # annotates in place

clean = adata[adata.obs["qc_pass"]].copy()
```

The object is **annotated in place** — nothing is removed from `adata`. The QC
verdict lands in `adata.obs["qc_pass"]`, per-stage diagnostics in
`adata.obs` (e.g. `rank_pass`, `pass_mito`, `cluster`) and
`adata.uns["validrops"]` (thresholds, gene sets, params). Subset afterwards with
`adata[adata.obs["qc_pass"]].copy()`.

For very large datasets run `validrops.validrops(adata)` as-is first; the
`label_dead` stage (slow, stochastic) is off by default.

## Installation

You need to have Python 3.11 or newer installed on your system.
If you don't have Python installed, we recommend installing [uv][].

There are several alternative options to install validrops:

<!--
1) Install the latest release of `validrops` from [PyPI][]:

```bash
pip install validrops
```
-->

1. Install the latest development version:

```bash
pip install git+https://github.com/INFIMM-Bioinformatics/validrops.git@main
```

## Release notes

See the [changelog][].

## Contact

For questions and help requests, you can reach out in the [scverse discourse][].
If you found a bug, please use the [issue tracker][].

## Citation

> t.b.a

[uv]: https://github.com/astral-sh/uv
[scverse discourse]: https://discourse.scverse.org/
[issue tracker]: https://github.com/INFIMM-Bioinformatics/validrops/issues
[tests]: https://github.com/INFIMM-Bioinformatics/validrops/actions/workflows/test.yaml
[documentation]: https://validrops.readthedocs.io
[changelog]: https://validrops.readthedocs.io/en/latest/changelog.html
[api documentation]: https://validrops.readthedocs.io/en/latest/api.html
[pypi]: https://pypi.org/project/validrops
