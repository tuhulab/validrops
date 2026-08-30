# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [Unreleased]

### Added

- Stage 1 `pp.rank_barcodes`: barcode ranking and knee-point detection separating cells from ambient droplets
- Stage 2a `tl.quality_metrics`: per-barcode mito/ribo/coding fractions and log UMI/feature counts
- Stage 2b `pp.quality_filter`: mito-fraction cap, feature↔UMI residual band, coding-fraction band
- Stage 3a `tl.expression_metrics`: deviance HVG selection, SVD embedding, SNN + Louvain clustering, per-cluster marker statistics
- Stage 3b `pp.expression_filter`: expression-based cluster outlier filtering
- Stage 4 `pp.label_dead`: heuristic dead-cell score and consensus ridge-training loop (off by default)
- `validrops()` end-to-end pipeline orchestrator annotating an `AnnData` in place
- `validrops.pl`: QC diagnostic plots (barcode rank, mito threshold, UMI vs features, coding fraction, dead score)
- Validation suite against R-generated fixtures on PBMC 4K, with a bounded end-to-end barcode concordance of 0.9193
- Package scaffolding: CI (test/build/release), Sphinx docs on readthedocs, pre-commit, codecov
