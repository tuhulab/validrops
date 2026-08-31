# tier-s / {'label': 'current', 'source_hashes': {'src/validrops/_constants.py': 'd5b85f7e71247b1d', 'src/validrops/tl/_snn.py': 'c0536038a02c3c1b', 'src/validrops/tl/expression_metrics.py': '91cc4303871f5a7c'}, 'harness_hashes': {'benchmarks/_datasets.py': '500d54e5eb650f36', 'benchmarks/_measure.py': 'c4069013023cdbd6', 'benchmarks/run.py': '1efcffac3a1dba0c'}}

- revision: c9a1c8ceb578795d8288704fe9696a349b86c8c7 (dirty=True)
- profile: S
- input fingerprint: 5f4161d14d14f250
- parameters: `{"profile": "S", "nfeats": 300, "npcs": 10, "k_min": 5, "res_shallow": 0.1, "top_n": 10, "workload": "Stage-3 synthetic", "seed": 0}`
- timing seconds: median=3.618745, min=3.606460, mad=0.006179, reps=3, CV=0.0026
- peak RSS bytes: worker_max=270712832, unit=bytes
- correctness: pass
- environment: {"os": "Darwin", "os_release": "25.5.0", "architecture": "arm64", "python_version": "3.12.12", "python_implementation": "CPython", "logical_cpus": 14, "physical_cpus": 14, "available_memory_bytes": 25769803776, "numpy_version": "2.4.4", "scipy_version": "1.17.1", "sklearn_version": "unavailable", "anndata_version": "0.12.10", "igraph_version": "1.0.0", "blas_config": {}, "blas_threadpool": "[{'num
