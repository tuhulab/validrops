# graph-conversion / {'label': 'baseline', 'source_hashes': {'src/validrops/_constants.py': 'd5b85f7e71247b1d', 'src/validrops/tl/_snn.py': 'ec34e0600e1820f3'}, 'harness_hashes': {'benchmarks/_datasets.py': '500d54e5eb650f36', 'benchmarks/_measure.py': 'c4069013023cdbd6', 'benchmarks/run.py': '1efcffac3a1dba0c'}}

- revision: c9a1c8ceb578795d8288704fe9696a349b86c8c7 (dirty=True)
- profile: S
- input fingerprint: a61af7e34eb99d0d
- parameters: `{"profile": "S", "n_cells": 500, "n_pcs": 10, "n_clusters": 4, "knn": 20, "prune": 0.06666666666666667, "k_min": 5, "seed": 0}`
- timing seconds: median=6.199671, min=6.152155, mad=0.007233, reps=7, CV=0.0042
- peak RSS bytes: worker_max=261537792, unit=bytes
- correctness: pass
- environment: {"os": "Darwin", "os_release": "25.5.0", "architecture": "arm64", "python_version": "3.12.12", "python_implementation": "CPython", "logical_cpus": 14, "physical_cpus": 14, "available_memory_bytes": 25769803776, "numpy_version": "2.4.4", "scipy_version": "1.17.1", "sklearn_version": "unavailable", "anndata_version": "0.12.10", "igraph_version": "1.0.0", "blas_config": {}, "blas_threadpool": "[{'num
