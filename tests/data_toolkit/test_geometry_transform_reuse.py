import json
import pickle

import numpy as np
import torch

from data_toolkit import dual_grid_view, voxelize_pbr_view


def _write_transforms(root, sha):
    path = root / sha
    path.mkdir(parents=True)
    (path / "transforms.json").write_text(
        json.dumps({"frames": [{"transform_matrix": np.eye(4).tolist()}]})
    )


def test_dual_grid_reuses_one_view_transform_across_resolutions(
    monkeypatch, tmp_path
):
    sha = "a" * 64
    mesh_root = tmp_path / "mesh"
    transform_root = tmp_path / "transforms"
    output_root = tmp_path / "output"
    (mesh_root / "mesh_dumps").mkdir(parents=True)
    _write_transforms(transform_root, sha)
    with (mesh_root / "mesh_dumps" / f"{sha}.pickle").open("wb") as stream:
        pickle.dump(
            {
                "objects": [
                    {
                        "vertices": np.array(
                            [[-1, -1, -1], [1, 1, 1], [-1, 1, -1]],
                            dtype=np.float32,
                        ),
                        "faces": np.array([[0, 1, 2]], dtype=np.int64),
                    }
                ]
            },
            stream,
        )

    transform_calls = []
    grid_sizes = []

    def track_transform(vertices, frame):
        transform_calls.append(frame)
        return vertices.clone()

    def fake_dual_grid(vertices, faces, *, grid_size, **_kwargs):
        grid_sizes.append(grid_size)
        coordinates = torch.zeros((1, 3), dtype=torch.int32)
        dual_vertices = torch.full((1, 3), 0.5 / grid_size)
        intersected = torch.zeros((1, 3), dtype=torch.uint8)
        return coordinates, dual_vertices, intersected

    monkeypatch.setattr(dual_grid_view, "transform_mesh", track_transform)
    monkeypatch.setattr(
        dual_grid_view.o_voxel.convert,
        "mesh_to_flexible_dual_grid",
        fake_dual_grid,
    )
    monkeypatch.setattr(dual_grid_view, "_publish_vxz_pair", lambda *_a, **_k: None)

    result = dual_grid_view._dual_grid_mesh_view(
        None,
        sha,
        mesh_root,
        transform_root,
        output_root,
        (256, 512, 1024),
        1,
        [0],
    )

    assert "error" not in result
    assert len(transform_calls) == 1
    assert grid_sizes == [256, 512, 1024]


def test_pbr_reuses_one_deep_copy_and_transform_across_resolutions(
    monkeypatch, tmp_path
):
    sha = "b" * 64
    pbr_root = tmp_path / "pbr"
    transform_root = tmp_path / "transforms"
    output_root = tmp_path / "output"
    (pbr_root / "pbr_dumps").mkdir(parents=True)
    _write_transforms(transform_root, sha)
    with (pbr_root / "pbr_dumps" / f"{sha}.pickle").open("wb") as stream:
        pickle.dump({"fixture": True}, stream)

    transform_calls = []
    grid_sizes = []
    prepared = {"objects": [{"vertices": np.zeros((1, 3), dtype=np.float32)}]}
    transformed = {"objects": [{"vertices": np.ones((1, 3), dtype=np.float32)}]}

    def track_transform(dump, frame):
        transform_calls.append((dump, frame))
        return transformed, 0.25

    def fake_voxelize(dump, *, grid_size, **_kwargs):
        assert dump is transformed
        grid_sizes.append(grid_size)
        coordinates = torch.zeros((1, 3), dtype=torch.int32)
        attributes = {
            "base_color": torch.zeros((1, 3), dtype=torch.uint8),
            "normal": torch.zeros((1, 3), dtype=torch.uint8),
            "emissive": torch.zeros((1, 3), dtype=torch.uint8),
        }
        return coordinates, attributes

    monkeypatch.setattr(voxelize_pbr_view, "prepare_pbr_dump", lambda _dump: prepared)
    monkeypatch.setattr(voxelize_pbr_view, "transform_pbr_dump", track_transform)
    monkeypatch.setattr(
        voxelize_pbr_view.o_voxel.convert,
        "blender_dump_to_volumetric_attr",
        fake_voxelize,
    )
    monkeypatch.setattr(
        voxelize_pbr_view,
        "_publish_vxz_pair",
        lambda *_args, **_kwargs: None,
    )

    result = voxelize_pbr_view._pbr_voxelize_view(
        None,
        sha,
        pbr_root,
        transform_root,
        output_root,
        (256, 512, 1024),
        1,
        [0],
    )

    assert "error" not in result
    assert len(transform_calls) == 1
    assert grid_sizes == [256, 512, 1024]
