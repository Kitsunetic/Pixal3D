from importlib import import_module
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.mark.parametrize(
    "module_name",
    (
        "data_toolkit.encode_shape_latent",
        "data_toolkit.encode_pbr_latent",
    ),
)
def test_legacy_latent_pack_narrows_coordinates_and_casts_features_on_tensor_device(
    module_name,
):
    worker = import_module(module_name)
    latent = SimpleNamespace(
        feats=torch.tensor([[1.25]], dtype=torch.float64),
        coords=torch.tensor([[0, 1, 2, 3]], dtype=torch.int64),
    )

    packed = worker._latent_pack(latent, grid_resolution=16)

    assert packed["feats"].dtype == np.float32
    assert packed["coords"].dtype == np.uint8
    np.testing.assert_array_equal(
        packed["coords"],
        np.array([[1, 2, 3]], dtype=np.uint8),
    )


@pytest.mark.parametrize(
    "module_name",
    (
        "data_toolkit.encode_shape_latent",
        "data_toolkit.encode_pbr_latent",
    ),
)
def test_legacy_latent_pack_rejects_coordinates_outside_grid(module_name):
    worker = import_module(module_name)
    latent = SimpleNamespace(
        feats=torch.tensor([[1.0]], dtype=torch.float32),
        coords=torch.tensor([[0, 16, 2, 3]], dtype=torch.int64),
    )

    with pytest.raises(ValueError, match="outside grid"):
        worker._latent_pack(latent, grid_resolution=16)


def test_legacy_shape_inputs_keep_raw_cpu_features_until_preprocessing():
    worker = import_module("data_toolkit.encode_shape_latent")
    coords = torch.tensor([[1, 2, 3]], dtype=torch.int32)
    vertices = torch.tensor([[255]], dtype=torch.uint8)
    intersected = torch.tensor([[5]], dtype=torch.uint8)

    prepared_vertices, prepared_intersected = worker._shape_encoder_inputs(
        coords,
        vertices,
        intersected,
        "cpu",
    )

    assert prepared_vertices.feats.dtype == torch.float32
    assert torch.equal(prepared_vertices.feats, torch.tensor([[1.0]]))
    assert prepared_intersected.feats.dtype == torch.bool
    assert torch.equal(
        prepared_intersected.feats,
        torch.tensor([[True, False, True]]),
    )


def test_legacy_pbr_input_normalizes_uint8_features_after_transfer():
    worker = import_module("data_toolkit.encode_pbr_latent")
    coords = torch.tensor([[1, 2, 3]], dtype=torch.int32)
    attributes = (
        torch.tensor([[0]], dtype=torch.uint8),
        torch.tensor([[255]], dtype=torch.uint8),
    )

    prepared = worker._pbr_encoder_input(coords, attributes, "cpu")

    assert prepared.feats.dtype == torch.float32
    torch.testing.assert_close(
        prepared.feats,
        torch.tensor([[-1.0, 1.0]], dtype=torch.float32),
    )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_legacy_pbr_preprocessing_returns_cuda_sparse_tensor():
    worker = import_module("data_toolkit.encode_pbr_latent")
    coords = torch.tensor([[1, 2, 3]], dtype=torch.int32)
    attributes = (torch.tensor([[255]], dtype=torch.uint8),)

    prepared = worker._pbr_encoder_input(coords, attributes, "cuda")

    assert prepared.feats.device.type == "cuda"
    assert prepared.coords.device.type == "cuda"
    assert torch.equal(prepared.feats.cpu(), torch.tensor([[1.0]]))
