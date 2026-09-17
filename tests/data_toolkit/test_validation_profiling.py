"""Profiling contracts for data validation internals."""

import json

import numpy as np

from data_toolkit.pipeline.validation import validate_sparse_latent


def test_sparse_validation_profiles_io_and_array_scans(
    monkeypatch, capsys, tmp_path
):
    path = tmp_path / "valid.npz"
    np.savez(
        path,
        feats=np.ones((2, 4), np.float32),
        coords=np.asarray([[0, 1, 2], [15, 15, 15]], np.uint8),
    )
    monkeypatch.setenv("PIXAL3D_PROFILE_STAGES", "1")

    validate_sparse_latent(path, 16, 8192)

    names = [
        json.loads(line.removeprefix("[PIXAL3D_PROFILE] "))["name"]
        for line in capsys.readouterr().out.splitlines()
    ]
    assert names == [
        "validation.npz.open",
        "validation.npz.materialize.feats",
        "validation.npz.materialize.coords",
        "validation.sparse.shape",
        "validation.sparse.token_limit",
        "validation.sparse.features.finite",
        "validation.sparse.coordinates.finite",
        "validation.sparse.coordinates.integral",
        "validation.sparse.coordinates.bounds",
        "validation.sparse.coordinates.unique",
    ]
