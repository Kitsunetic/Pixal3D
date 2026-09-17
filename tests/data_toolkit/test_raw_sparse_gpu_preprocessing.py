from contextlib import contextmanager

import pytest
import torch

import data_toolkit.pipeline.encoder_preprocessing as preprocessing
from data_toolkit.pipeline.encoder_preprocessing import (
    prepare_pbr_sparse_batch,
    prepare_shape_sparse_batch,
)


def test_sparse_preprocessing_exposes_attributable_profile_stages(monkeypatch):
    # Given: an enabled profiler replacement that records section names.
    stages = []

    @contextmanager
    def record_stage(name, device=None):
        stages.append((name, torch.device(device) if device is not None else None))
        yield

    monkeypatch.setattr(preprocessing, "profile_stage", record_stage)
    shape_payloads = [
        (
            torch.tensor([[1, 2, 3]], dtype=torch.int32),
            torch.tensor([[255]], dtype=torch.uint8),
            torch.tensor([[5]], dtype=torch.uint8),
        )
    ]
    pbr_payloads = [
        (
            torch.tensor([[1, 2, 3]], dtype=torch.int32),
            tuple(
                torch.tensor([[value]], dtype=torch.uint8)
                for value in (0, 64, 127, 255)
            ),
        )
    ]

    # When: both sparse preprocessing paths build their CPU reference tensors.
    prepare_shape_sparse_batch(shape_payloads, "cpu")
    prepare_pbr_sparse_batch(pbr_payloads, "cpu")

    # Then: transfer, construction, and normalization can be attributed separately.
    assert [name for name, _ in stages] == [
        "shape.coordinates.transfer",
        "shape.vertices.transfer",
        "shape.intersections.transfer",
        "shape.coordinates.batch",
        "shape.features.concatenate",
        "shape.sparse.construct",
        "shape.vertices.normalize",
        "shape.intersections.unpack",
        "pbr.coordinates.transfer",
        "pbr.attribute_0.transfer",
        "pbr.attribute_1.transfer",
        "pbr.attribute_2.transfer",
        "pbr.attribute_3.transfer",
        "pbr.features.concatenate",
        "pbr.coordinates.batch",
        "pbr.sparse.construct",
        "pbr.features.normalize",
    ]


def test_shape_raw_payloads_build_batched_sparse_inputs_on_cpu():
    payloads = [
        (
            torch.tensor([[1, 2, 3]], dtype=torch.int32),
            torch.tensor([[255]], dtype=torch.uint8),
            torch.tensor([[5]], dtype=torch.uint8),
        ),
        (
            torch.tensor([[4, 5, 6]], dtype=torch.int32),
            torch.tensor([[0]], dtype=torch.uint8),
            torch.tensor([[2]], dtype=torch.uint8),
        ),
    ]

    vertices, intersected = prepare_shape_sparse_batch(payloads, "cpu")

    assert vertices.feats.device.type == "cpu"
    assert torch.equal(
        vertices.coords,
        torch.tensor([[0, 1, 2, 3], [1, 4, 5, 6]], dtype=torch.int32),
    )
    assert torch.equal(vertices.feats, torch.tensor([[1.0], [0.0]]))
    assert torch.equal(
        intersected.feats,
        torch.tensor([[True, False, True], [False, True, False]]),
    )


def test_pbr_raw_payloads_concatenate_attributes_after_transfer_on_cpu():
    payloads = [
        (
            torch.tensor([[1, 2, 3]], dtype=torch.int32),
            (
                torch.tensor([[0]], dtype=torch.uint8),
                torch.tensor([[255]], dtype=torch.uint8),
                torch.tensor([[127]], dtype=torch.uint8),
                torch.tensor([[64]], dtype=torch.uint8),
            ),
        ),
        (
            torch.tensor([[4, 5, 6]], dtype=torch.int32),
            (
                torch.tensor([[255]], dtype=torch.uint8),
                torch.tensor([[0]], dtype=torch.uint8),
                torch.tensor([[255]], dtype=torch.uint8),
                torch.tensor([[0]], dtype=torch.uint8),
            ),
        ),
    ]

    voxels = prepare_pbr_sparse_batch(payloads, "cpu")

    assert voxels.feats.device.type == "cpu"
    assert torch.equal(
        voxels.coords,
        torch.tensor([[0, 1, 2, 3], [1, 4, 5, 6]], dtype=torch.int32),
    )
    torch.testing.assert_close(
        voxels.feats,
        torch.tensor(
            [[-1.0, 1.0, -1.0 / 255.0, -127.0 / 255.0], [1.0, -1.0, 1.0, -1.0]],
        ),
    )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_shape_raw_payloads_match_cuda_preprocessing_when_available():
    payloads = [
        (
            torch.tensor([[1, 2, 3]], dtype=torch.int32),
            torch.tensor([[255]], dtype=torch.uint8),
            torch.tensor([[5]], dtype=torch.uint8),
        ),
    ]

    cpu_vertices, cpu_intersected = prepare_shape_sparse_batch(payloads, "cpu")
    cuda_vertices, cuda_intersected = prepare_shape_sparse_batch(payloads, "cuda")

    assert cuda_vertices.feats.device.type == "cuda"
    assert cuda_vertices.coords.device.type == "cuda"
    torch.testing.assert_close(cuda_vertices.feats.cpu(), cpu_vertices.feats)
    assert torch.equal(cuda_intersected.feats.cpu(), cpu_intersected.feats)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@torch.inference_mode()
def test_pbr_raw_payloads_match_cuda_preprocessing_when_available():
    payloads = [
        (
            torch.tensor([[1, 2, 3]], dtype=torch.int32),
            (
                torch.tensor([[0]], dtype=torch.uint8),
                torch.tensor([[255]], dtype=torch.uint8),
                torch.tensor([[127]], dtype=torch.uint8),
                torch.tensor([[64]], dtype=torch.uint8),
            ),
        ),
    ]

    cpu_voxels = prepare_pbr_sparse_batch(payloads, "cpu")
    cuda_voxels = prepare_pbr_sparse_batch(payloads, "cuda")

    assert cuda_voxels.feats.device.type == "cuda"
    assert cuda_voxels.coords.device.type == "cuda"
    torch.testing.assert_close(cuda_voxels.feats.cpu(), cpu_voxels.feats)
    assert torch.equal(cuda_voxels.coords.cpu(), cpu_voxels.coords)
