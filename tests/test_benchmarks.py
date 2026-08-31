"""Benchmark harness and deterministic dataset tests (Performance Package 1).

The benchmark tooling lives outside the installed package in ``benchmarks/``
and writes artifacts only under the gitignored ``benchmark-results/``. These
tests exercise the generator contracts, the PBMC4K fixture detection, the
spawn-isolated measurement schema, and artifact-comparison rejection rules.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

# benchmarks/ is a top-level package next to tests/; make it importable from
# the pytest process (pytest only inserts tests/ into sys.path with importlib
# mode).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks import _datasets, _measure

REQUIRED_TOP_LEVEL = [
    "schema_version",
    "benchmark",
    "revision",
    "dirty",
    "environment",
    "input",
    "implementation",
    "timing_seconds",
    "peak_rss_bytes",
    "correctness",
    "warnings",
]


def test_tier_s_generator_is_deterministic_and_sparse():
    first, first_meta = _datasets.make_sparse_counts("S", seed=0)
    second, second_meta = _datasets.make_sparse_counts("S", seed=0)
    assert sp.isspmatrix_csr(first)
    assert first.shape == (500, 2_000)
    assert first.dtype == np.float32
    assert (first != second).nnz == 0
    assert first_meta == second_meta


def test_smoke_profile_is_deterministic_sparse_and_metadata_complete():
    matrix, meta = _datasets.make_sparse_counts("smoke", seed=1)
    assert sp.isspmatrix_csr(matrix)
    assert matrix.shape == (60, 120)
    assert matrix.dtype == np.float32
    assert isinstance(matrix, sp.csr_matrix) and not isinstance(matrix, np.ndarray)
    assert meta["profile"] == "smoke"
    assert meta["cells"] == 60
    assert meta["genes"] == 120
    assert meta["seed"] == 1
    assert meta["dtype"] == "float32"
    assert meta["realized_density"] > 0
    # generators never return a dense array
    assert not isinstance(matrix, np.ndarray)
    assert not hasattr(matrix, "toarray") or sp.issparse(matrix)


def test_profiles_table_matches_contract():
    assert _datasets.PROFILES["smoke"] == {"cells": 60, "genes": 120, "density": 0.05}
    assert _datasets.PROFILES["S"] == {"cells": 500, "genes": 2_000, "density": 0.02}
    assert _datasets.PROFILES["M-synthetic"] == {"cells": 4_000, "genes": 20_000, "density": 0.01}
    assert _datasets.PROFILES["L"] == {"cells": 10_000, "genes": 25_000, "density": 0.005}
    assert _datasets.PROFILES["XL-50"] == {"cells": 50_000, "genes": 30_000, "density": 0.002}
    assert _datasets.PROFILES["XL-100"] == {"cells": 100_000, "genes": 30_000, "density": 0.001}


def test_clustered_embedding_is_deterministic():
    first, labels_a, meta_a = _datasets.make_clustered_embedding(300, 10, 4, seed=0)
    second, labels_b, meta_b = _datasets.make_clustered_embedding(300, 10, 4, seed=0)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(labels_a, labels_b)
    assert meta_a == meta_b
    assert first.shape == (300, 10)
    assert set(labels_a.tolist()) == set(range(4))


def test_meta_records_duplicate_name_pattern():
    _, meta = _datasets.make_sparse_counts("S", seed=0)
    assert "duplicate_names" in meta
    dup = meta["duplicate_names"]
    assert dup["n_symbols_duplicated"] >= 1
    assert dup["pattern"]  # non-empty description


def test_pbmc_detection_rooted_at_repo():
    state = _measure.resolve_pbmc_state({"pbmc_override": "auto", "pbmc_path": None})
    assert state["available"] is True  # tests/data/pbmc4k/raw.h5 is present in this checkout
    assert state["path"].endswith("tests/data/pbmc4k/raw.h5")
    assert state["override"] == "auto"


def test_pbmc_override_yes_and_no(tmp_path):
    missing = tmp_path / "nope.h5"
    state = _measure.resolve_pbmc_state({"pbmc_override": "no", "pbmc_path": str(missing)})
    assert state["available"] is False
    assert state["path"] == str(missing)
    assert state["override"] == "no"
    with pytest.raises(FileNotFoundError, match="PBMC4K fixture"):
        _measure.resolve_pbmc_state({"pbmc_override": "yes", "pbmc_path": str(missing)})


def test_one_rep_smoke_run_produces_contract_schema(tmp_path):
    artifact = _measure.run_benchmark(
        benchmark="cluster-means",
        implementation="baseline",
        profile="smoke",
        seed=0,
        repetitions=1,
        timeout=120,
        output_dir=None,
        pbmc_override="auto",
        pbmc_path=None,
    )
    for key in REQUIRED_TOP_LEVEL:
        assert key in artifact, f"missing top-level key {key}"
    assert artifact["schema_version"] == 1
    assert artifact["benchmark"] == "cluster-means"
    assert artifact["revision"]  # sha or "unavailable"
    assert isinstance(artifact["dirty"], bool)
    assert artifact["environment"]
    assert artifact["input"]
    assert artifact["implementation"]
    assert artifact["timing_seconds"]["repetitions"] == 1
    assert len(artifact["timing_seconds"]["raw"]) == 1
    assert artifact["timing_seconds"]["median"] > 0.0
    assert artifact["timing_seconds"]["minimum"] == artifact["timing_seconds"]["median"]
    assert artifact["timing_seconds"]["mad"] == 0.0
    assert artifact["peak_rss_bytes"]["worker_max"] > 0
    assert artifact["correctness"]["status"] == "pass"
    assert isinstance(artifact["warnings"], list)

    # JSON round trip
    dumped = json.dumps(artifact, indent=2)
    assert json.loads(dumped) == artifact

    # Markdown summary writes next to the JSON when an output dir is given
    out_dir = tmp_path / "artifacts"
    _measure.write_artifacts(artifact, output_dir=str(out_dir))
    json_path = out_dir / "cluster-means-baseline.json"
    md_path = out_dir / "cluster-means-baseline.md"
    assert json_path.exists()
    assert md_path.exists()
    assert "cluster-means" in md_path.read_text()


def test_compare_rejects_incompatible_artifacts():
    base = _measure.run_benchmark(
        benchmark="cluster-means",
        implementation="baseline",
        profile="smoke",
        seed=0,
        repetitions=1,
        timeout=120,
        output_dir=None,
        pbmc_override="no",
        pbmc_path=None,
    )
    other_profile = dict(base)
    other_profile["input"] = {**base["input"], "params": {**base["input"]["params"], "profile": "S"}}
    with pytest.raises(ValueError, match="incompatible"):
        _measure.compare_artifacts(base, other_profile)
    other_bench = dict(base)
    other_bench["benchmark"] = "graph-conversion"
    with pytest.raises(ValueError, match="incompatible"):
        _measure.compare_artifacts(base, other_bench)


def test_compare_same_artifacts_reports_identity():
    a = _measure.run_benchmark(
        benchmark="cluster-means",
        implementation="baseline",
        profile="smoke",
        seed=0,
        repetitions=1,
        timeout=120,
        output_dir=None,
        pbmc_override="no",
        pbmc_path=None,
    )
    summary = _measure.compare_artifacts(a, a)
    assert summary["compatible"] is True
    assert summary["median_speedup"] == 1.0


def test_pbmc_absence_yields_explicit_skip_artifact(tmp_path):
    artifact = _measure.run_benchmark(
        benchmark="tier-m-pbmc",
        implementation="baseline",
        profile="S",
        seed=0,
        repetitions=3,
        timeout=60,
        output_dir=None,
        pbmc_override="no",
        pbmc_path=None,
    )
    assert artifact["pbmc"]["available"] is False
    assert artifact["pbmc"]["path"]
    assert artifact["correctness"]["status"] == "skipped"
    assert any("PBMC4K" in w and "tests/data/pbmc4k/raw.h5" in w for w in artifact["warnings"])
