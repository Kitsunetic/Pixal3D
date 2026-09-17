import json

import numpy as np
import torch

from data_toolkit.pipeline.loader_preprocessing import (
    load_pbr_vxz,
    load_shape_vxz,
    load_ss_coordinates,
)


def _profile_names(output: str) -> list[str]:
    return [
        json.loads(line.removeprefix("[PIXAL3D_PROFILE] "))["name"]
        for line in output.splitlines()
    ]


def test_vxz_loaders_preserve_payloads_and_emit_ordered_stages_when_enabled(
    monkeypatch, capsys, tmp_path
):
    # Given: representative raw VXZ tensors and enabled stage profiling.
    monkeypatch.setenv("PIXAL3D_PROFILE_STAGES", "1")
    coords = torch.tensor([[1, 2, 3]], dtype=torch.int32)
    vertices = torch.tensor([[1]], dtype=torch.uint8)
    intersected = torch.tensor([[5]], dtype=torch.uint8)
    attributes = {
        "base_color": torch.tensor([[2]], dtype=torch.uint8),
        "metallic": torch.tensor([[3]], dtype=torch.uint8),
        "roughness": torch.tensor([[4]], dtype=torch.uint8),
        "alpha": torch.tensor([[5]], dtype=torch.uint8),
    }
    calls: list[tuple[str, int]] = []

    def read_vxz(path: str, *, num_threads: int):
        calls.append((path, num_threads))
        return coords, {"vertices": vertices, "intersected": intersected, **attributes}

    # When: production payload selection loads Shape and PBR VXZ files.
    shape_payload = load_shape_vxz(tmp_path / "shape.vxz", read_vxz)
    pbr_payload = load_pbr_vxz(tmp_path / "pbr.vxz", read_vxz)

    # Then: payload types/order remain exact and every loader phase is attributable.
    assert shape_payload == (coords, vertices, intersected)
    assert pbr_payload == (
        coords,
        (
            attributes["base_color"],
            attributes["metallic"],
            attributes["roughness"],
            attributes["alpha"],
        ),
    )
    assert calls == [(str(tmp_path / "shape.vxz"), 1), (str(tmp_path / "pbr.vxz"), 1)]
    assert _profile_names(capsys.readouterr().out) == [
        "shape.loader.vxz.read_decode",
        "shape.loader.payload.select",
        "pbr.loader.vxz.read_decode",
        "pbr.loader.payload.select",
    ]


def test_ss_loader_preserves_compact_coordinates_and_emits_ordered_stages_when_enabled(
    monkeypatch, capsys, tmp_path
):
    # Given: a sparse-latent NPZ with a non-long coordinate dtype.
    monkeypatch.setenv("PIXAL3D_PROFILE_STAGES", "true")
    expected = np.array([[1, 2, 3]], dtype=np.uint8)
    path = tmp_path / "view00.npz"
    np.savez(path, coords=expected)

    # When: SS coordinates are loaded for dense-volume preprocessing.
    actual = load_ss_coordinates(path)

    # Then: coordinates stay compact until the encoder-device transfer casts them.
    assert actual.dtype is torch.uint8
    assert torch.equal(actual, torch.from_numpy(expected))
    assert _profile_names(capsys.readouterr().out) == [
        "ss.loader.npz.open",
        "ss.loader.npz.coords_materialize",
        "ss.loader.numpy_to_torch",
    ]


def test_loaders_emit_nothing_when_profiling_is_disabled(capsys, tmp_path):
    # Given: profiling is disabled and equivalent loader inputs.
    coords = torch.tensor([[1, 2, 3]], dtype=torch.int32)
    path = tmp_path / "view00.npz"
    np.savez(path, coords=np.array([[1, 2, 3]], dtype=np.uint8))

    def read_vxz(_path: str, *, num_threads: int):
        assert num_threads == 1
        return coords, {
            "vertices": torch.tensor([[1]], dtype=torch.uint8),
            "intersected": torch.tensor([[5]], dtype=torch.uint8),
        }

    # When: loader helpers process the inputs without the opt-in environment flag.
    shape_payload = load_shape_vxz(tmp_path / "shape.vxz", read_vxz)
    ss_coordinates = load_ss_coordinates(path)

    # Then: outputs retain their legacy values and no profiling side effect occurs.
    assert shape_payload[0] is coords
    assert torch.equal(ss_coordinates, torch.tensor([[1, 2, 3]], dtype=torch.uint8))
    assert capsys.readouterr().out == ""
