# tier-m-pbmc / {'label': 'current', 'source_hashes': {'src/validrops/_constants.py': 'd5b85f7e71247b1d', 'src/validrops/tl/_snn.py': 'c0536038a02c3c1b', 'src/validrops/tl/expression_metrics.py': '91cc4303871f5a7c'}, 'harness_hashes': {'benchmarks/_datasets.py': '500d54e5eb650f36', 'benchmarks/_measure.py': 'ebec6fcabaa59b9a', 'benchmarks/run.py': '1efcffac3a1dba0c'}}

- revision: c9a1c8ceb578795d8288704fe9696a349b86c8c7 (dirty=True)
- profile: S
- input fingerprint: 1fc1df2c37f13a7a
- parameters: `{"profile": "S", "nfeats": 300, "npcs": 10, "k_min": 150, "res_shallow": 0.1, "top_n": 10, "n_cells_cap": 1200, "workload": "Stage-3 PBMC4K", "seed": 0}`
- timing seconds: median=81.819761, min=81.457796, mad=0.252676, reps=3, CV=0.0038
- peak RSS bytes: worker_max=1735884800, unit=bytes
- correctness: pass
- environment: {"os": "Darwin", "os_release": "25.5.0", "architecture": "arm64", "python_version": "3.12.12", "python_implementation": "CPython", "logical_cpus": 14, "physical_cpus": 14, "available_memory_bytes": 25769803776, "numpy_version": "2.4.4", "scipy_version": "1.17.1", "sklearn_version": "unavailable", "anndata_version": "0.12.10", "igraph_version": "1.0.0", "blas_config": {}, "blas_threadpool": "[{'num

## Warnings
- warm-up vs timed output fingerprints differ: pre-existing scipy svds OS-entropy starting vector (expression_metrics._embed); dense SVD is unchanged and out of Package-1 scope
- warm-up vs timed output fingerprints differ: pre-existing scipy svds OS-entropy starting vector (expression_metrics._embed); dense SVD is unchanged and out of Package-1 scope
